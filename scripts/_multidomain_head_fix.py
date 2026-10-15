# -*- coding: utf-8 -*-
"""scripts/_multidomain_head_fix.py — точечный дообучающий прогон ГОЛОВ при замороженном энкодере.

Диагноз (_probe_choice): после полного файнтюна choice-голова на новых сферах недообучена
(CE 2.27 vs Argent 1.365, acc 27%, коллапс на account) — 3 эпохи при lr 2e-5 не хватило
16 новым классам с init N(0,0.02).

Фикс: энкодер (embed/rotary/layers/final_norm/lm_head) заморожен; обучаются только
choice/noul/score/extract-головы + temperature на md_train (2250) + Argent-replay (2000),
lr 1e-4, 6 эпох. Backward через замороженный энкодер не идёт — шаг дешёвый.

Гейты: argent_choice_acc_after >= before - 3пп AND md_choice_acc_after >= 55%.
Кандидат: leeth_best_md2_candidate.pt (промоушен вручную после батарей).
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.leeth.finetune import _load_jsonl_samples
from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import LeethTorchDataset, collate_leeth_batch, compute_calibrated_loss

DATA = ROOT / "src/leeth/data_400m"
EPOCHS = 24
LR = 1.5e-4
BATCH = 8
MAX_LEN = 256
REPLAY_N = 2000
SEED = 11
# Стартуем с md6 (прод-конверт + расширенные шаблоны), если он есть
START_CKPT = DATA / "leeth_best.pt"
OUT_CKPT = DATA / "leeth_best_md7llm_candidate.pt"
HEAD_PREFIXES = ("choice_head.", "noul_head.", "score_head.", "extract_start_head.", "extract_end_head.", "temperature")


def log(m):
    print(m, flush=True)


def eval_choice_acc(model, samples, tokenizer, class_to_idx, device, idx_to_class):
    """acc с ограничением кандидатами (как в рантайме) + raw choice-CE."""
    ds = LeethTorchDataset(samples, tokenizer, max_length=MAX_LEN, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_leeth_batch)
    n = cand_ok = 0
    ce_sum = 0.0
    preds = Counter()
    ptr = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            bs = batch["input_ids"].size(0)
            chunk = samples[ptr:ptr + bs]
            ptr += bs
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outs = model(input_ids, attention_mask=mask)
            logits = outs["choice_logits"].float()
            probs = torch.softmax(logits, dim=-1)
            targets = batch["choice_target"]
            ce_sum += F.cross_entropy(logits, targets.to(device), reduction="sum").item()
            for j, s in enumerate(chunk):
                t = int(targets[j])
                cand = s.choices or []
                idxs = [class_to_idx[c] for c in cand if c in class_to_idx]
                if idxs:
                    cp = idxs[int(probs[j][idxs].argmax())]
                    cand_ok += cp == t
                    preds[idx_to_class[cp]] += 1
                n += 1
    model.train()
    return cand_ok / max(1, n), ce_sum / max(1, n), preds


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    class_to_idx = json.loads((DATA / "choice_classes.json").read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    log(f"классов: {len(class_to_idx)}")

    md_train = _load_jsonl_samples(DATA / "multidomain_train.jsonl")
    md_val = _load_jsonl_samples(DATA / "multidomain_val.jsonl")
    pool = _load_jsonl_samples(DATA / "leeth_400m_train.jsonl")
    replay = rng.sample(pool, min(REPLAY_N, len(pool)))
    md_choice_val = [s for s in md_val if s.question_type == "choice" and s.choices]
    arg_val = _load_jsonl_samples(DATA / "leeth_400m_val.jsonl", limit=400)
    arg_choice_val = [s for s in arg_val if s.question_type == "choice" and s.choices]
    log(f"md_train={len(md_train)} replay={len(replay)} md_choice_val={len(md_choice_val)} arg_choice_val={len(arg_choice_val)}")

    tokenizer = LeethTokenizer.load(DATA)
    start_path = START_CKPT if START_CKPT.exists() else DATA / "leeth_best.pt"
    log(f"стартовый чекпоинт: {start_path.name}")
    checkpoint = torch.load(start_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = LeethConfig(**config)
    model = LeethModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # Заморозка энкодера
    frozen = 0
    for name, p in model.named_parameters():
        if name.startswith(HEAD_PREFIXES):
            p.requires_grad = True
        else:
            p.requires_grad = False
            frozen += p.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"обучается: {trainable} параметров (головы), заморожено: {frozen}")

    acc_md_before, ce_md_before, _ = eval_choice_acc(model, md_choice_val, tokenizer, class_to_idx, device, idx_to_class)
    acc_arg_before, ce_arg_before, _ = eval_choice_acc(model, arg_choice_val, tokenizer, class_to_idx, device, idx_to_class)
    log(f"BEFORE: md_choice_acc={acc_md_before * 100:.1f}% (CE {ce_md_before:.3f}) | "
        f"argent_choice_acc={acc_arg_before * 100:.1f}% (CE {ce_arg_before:.3f})")

    train_samples = md_train + replay
    rng.shuffle(train_samples)
    ds = LeethTorchDataset(train_samples, tokenizer, max_length=MAX_LEN, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_leeth_batch)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(loader), eta_min=LR * 0.05)
    log(f"temperature = {model.temperature.item():.3f}")

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss, nb = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            for k in ("choice_target", "noul_target", "score_target", "score_levels", "start_target", "end_target"):
                batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            outs = model(input_ids, attention_mask=mask)
            loss, _ = compute_calibrated_loss(outs, batch)
            if torch.isnan(loss) or loss.item() <= 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
            sched.step()
        if epoch % 6 == 0 or epoch == EPOCHS:
            a, c, _ = eval_choice_acc(model, md_choice_val, tokenizer, class_to_idx, device, idx_to_class)
            log(f"  [контроль] md_choice_acc={a * 100:.1f}% (CE {c:.3f})")
        log(f"epoch {epoch}/{EPOCHS} | loss={epoch_loss / max(1, nb):.4f} | batches={nb}")

    acc_md_after, ce_md_after, preds_md = eval_choice_acc(model, md_choice_val, tokenizer, class_to_idx, device, idx_to_class)
    acc_arg_after, ce_arg_after, _ = eval_choice_acc(model, arg_choice_val, tokenizer, class_to_idx, device, idx_to_class)
    log(f"AFTER: md_choice_acc={acc_md_after * 100:.1f}% (CE {ce_md_after:.3f}) | "
        f"argent_choice_acc={acc_arg_after * 100:.1f}% (CE {ce_arg_after:.3f})")
    log(f"предсказания md_val: {dict(preds_md.most_common(8))}")

    gate_arg = acc_arg_after >= acc_arg_before - 0.03
    gate_md = acc_md_after >= 0.55
    log(f"гейты: argent {'OK' if gate_arg else 'FAIL'} ({acc_arg_after * 100:.1f}% >= {acc_arg_before * 100 - 3:.1f}%); "
        f"md {'OK' if gate_md else 'FAIL'} ({acc_md_after * 100:.1f}% >= 55%)")

    payload = {
        "config": config,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "val_loss": checkpoint.get("val_loss"),
        "brier_score": checkpoint.get("brier_score"),
        "finetune_meta": {
            "ts": time.time(),
            "purpose": "multidomain_v1_head_fix",
            "epochs": EPOCHS, "lr": LR, "batch": BATCH, "max_length": MAX_LEN,
            "train": len(train_samples), "md_train": len(md_train), "replay": len(replay),
            "md_choice_acc_before": acc_md_before, "md_choice_acc_after": acc_md_after,
            "argent_choice_acc_before": acc_arg_before, "argent_choice_acc_after": acc_arg_after,
            "gate_argent": gate_arg, "gate_md": gate_md, "accepted": gate_arg and gate_md,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    out = OUT_CKPT
    torch.save(payload, out)
    log(f"кандидат: {out} ({'ПРИНЯТ' if gate_arg and gate_md else 'гейты не пройдены'})")
    log(f"время: {(time.time() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()