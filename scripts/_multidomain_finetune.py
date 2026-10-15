# -*- coding: utf-8 -*-
"""scripts/_multidomain_finetune.py — файнтюн 400M на мультидоменных семьях + Argent-replay.

Train: multidomain_train.jsonl (2250) + replay leeth_400m_train.jsonl (2000) + feedback (если есть).
Val: multidomain_val.jsonl (250) + leeth_400m_val.jsonl (400) — двойные гейты:
  1) Argent-регрессия: argent_after <= argent_before * 1.10
  2) Мультидомен: md_after <= md_before * 0.85 (обязательное улучшение)
Кандидат сохраняется в leeth_best_md_candidate.pt — промоушен ТОЛЬКО после гейтов батарей.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import DataLoader

from src.leeth.dataset_generator import LeethSample
from src.leeth.finetune import _eval_loss, _load_jsonl_samples
from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss

DATA = ROOT / "src/leeth/data_400m"
EPOCHS = 3
LR = 2e-5
BATCH = 8
ACCUM = 2
MAX_LEN = 256
REPLAY_N = 2000
SEED = 11


def log(m):
    print(m, flush=True)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    classes_path = DATA / "choice_classes.json"
    class_to_idx = json.loads(classes_path.read_text(encoding="utf-8"))
    log(f"классов: {len(class_to_idx)}")

    md_train = _load_jsonl_samples(DATA / "multidomain_train.jsonl")
    md_val = _load_jsonl_samples(DATA / "multidomain_val.jsonl")
    log(f"мультидомен: train={len(md_train)} val={len(md_val)}")

    # Argent-replay против забывания
    pool = _load_jsonl_samples(DATA / "leeth_400m_train.jsonl")
    replay = rng.sample(pool, min(REPLAY_N, len(pool)))
    log(f"argent-replay: {len(replay)} из {len(pool)}")

    # feedback (если есть) — малым добавкой
    fb_train, fb_val = [], []
    try:
        from src.leeth import feedback
        fb_train, fb_val, _ = feedback.build_training_samples(val_ratio=0.2)
        log(f"feedback: train={len(fb_train)} val={len(fb_val)}")
    except Exception as e:
        log(f"feedback недоступен ({e}) — пропускаю")

    train_samples = md_train + replay + fb_train
    rng.shuffle(train_samples)
    log(f"итого train: {len(train_samples)}")

    tokenizer = LeethTokenizer.load(DATA)

    checkpoint = torch.load(DATA / "leeth_best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, dict):
        from src.leeth.model import LeethConfig
        config = LeethConfig(**config)
    model = LeethModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    log(f"модель загружена: num_choices={config.num_choices}, device={device}")

    argent_val = _load_jsonl_samples(DATA / "leeth_400m_val.jsonl", limit=400)
    log(f"argent-val: {len(argent_val)}, md-val: {len(md_val)}")

    def eval_ce(samples, tag):
        ds = LeethTorchDataset(samples, tokenizer, max_length=MAX_LEN, class_to_idx=class_to_idx)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_leeth_batch)
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                    batch[k] = batch[k].to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outs = model(input_ids, attention_mask=mask)
                    loss, _ = compute_calibrated_loss(outs, batch)
                if not torch.isnan(loss):
                    total += loss.item() * mask.size(0)
                    n += mask.size(0)
        model.train()
        return total / max(1, n)

    argent_before = eval_ce(argent_val, "argent")
    md_before = eval_ce(md_val, "md")
    log(f"BEFORE: argent_val_ce={argent_before:.4f} md_val_ce={md_before:.4f}")

    train_ds = LeethTorchDataset(train_samples, tokenizer, max_length=MAX_LEN, class_to_idx=class_to_idx)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_leeth_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    model.train()
    micro = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_loss, nb = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outs = model(input_ids, attention_mask=mask)
                loss, _ = compute_calibrated_loss(outs, batch)
            if torch.isnan(loss) or loss.item() <= 0:
                continue
            scaler.scale(loss / ACCUM).backward()
            micro += 1
            if micro % ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            nb += 1
        log(f"epoch {epoch}/{EPOCHS} | loss={epoch_loss / max(1, nb):.4f} | batches={nb}")

    argent_after = eval_ce(argent_val, "argent")
    md_after = eval_ce(md_val, "md")
    log(f"AFTER: argent_val_ce={argent_after:.4f} md_val_ce={md_after:.4f}")

    gate_argent = argent_after <= argent_before * 1.10
    gate_md = md_after <= md_before * 0.85
    accepted = gate_argent and gate_md
    log(f"гейты: argent {'OK' if gate_argent else 'FAIL'} ({argent_after:.4f} vs <= {argent_before*1.10:.4f}); "
        f"md {'OK' if gate_md else 'FAIL'} ({md_after:.4f} vs <= {md_before*0.85:.4f})")

    payload = {
        "config": config,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "val_loss": argent_after,
        "brier_score": checkpoint.get("brier_score"),
        "finetune_meta": {
            "ts": time.time(),
            "purpose": "multidomain_v1",
            "train": len(train_samples), "md_train": len(md_train), "replay": len(replay),
            "feedback_train": len(fb_train),
            "epochs": EPOCHS, "lr": LR, "batch": BATCH, "accum": ACCUM, "max_length": MAX_LEN,
            "argent_val_before": argent_before, "argent_val_after": argent_after,
            "md_val_before": md_before, "md_val_after": md_after,
            "gate_argent": gate_argent, "gate_md": gate_md, "accepted": accepted,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    out = DATA / "leeth_best_md_candidate.pt"
    torch.save(payload, out)
    log(f"кандидат: {out} ({'ПРИНЯТ' if accepted else 'гейты не пройдены — промоушен вручную после батарей'})")
    log(f"время: {(time.time() - t0)/60:.1f} мин")


if __name__ == "__main__":
    main()