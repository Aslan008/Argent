# -*- coding: utf-8 -*-
"""scripts/_multidomain_full_ft.py — ПОЛНЫЙ файнтюн (энкодер + головы) на md-датасете.

Гипотеза: головы недообучены, потому что энкодер не адаптирован под новые сферы.
Полный файнтюн с низким lr (1e-5) на md_train + Argent-replay должен адаптировать
энкодер, не разрушив Argent-навыки (replay 2000 + гейт).

Скорость (замерено, RTX 4060 bf16): 0.93 s/step @ batch16/len128.
280 шагов/эпоху x 3 эпохи ~ 13 мин (+ length-grouped batching быстрее).

Гейты: argent_choice_acc_after >= before - 3пп AND md_choice_acc_after >= 55%.
Выход: leeth_best_md8ft_candidate.pt
"""
import json
import math
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

from src.leeth.finetune import _load_jsonl_samples
from src.leeth.model import LeethConfig, LeethModel
from src.leeth.tokenizer import LeethTokenizer
from src.leeth.train import compute_calibrated_loss

DATA = ROOT / "src/leeth/data_400m"
EPOCHS = 3
LR = 1e-5
BATCH = 16
MAX_LEN = 128
REPLAY_N = 2000
WARMUP = 100
SEED = 11
START_CKPT = DATA / "leeth_best.pt"
OUT_CKPT = DATA / "leeth_best_md8ft_candidate.pt"


def log(m):
    print(m, flush=True)


def encode_sample(sample, tokenizer, class_to_idx):
    """Токенизация без паддинга (паддинг в collate до max в батче)."""
    type_token = f"[{sample.question_type.upper()}]"
    text = f"[STATE] {sample.state} [SEP] [QUESTION] {type_token} {sample.question}"
    input_ids = tokenizer.encode(text, max_length=MAX_LEN, padding=False,
                                 truncation=True, add_special_tokens=True)
    choice_target = -100
    noul_target = -1.0
    score_target = -1.0
    score_levels = 0
    start_target = 0
    end_target = 0
    q_type = sample.question_type
    if q_type == "choice" and sample.target_choice:
        choice_target = class_to_idx.get(sample.target_choice, 0)
    elif q_type == "noul" and sample.target_noul is not None:
        noul_target = float(sample.target_noul)
    elif q_type == "score" and sample.target_score is not None:
        score_target = float(sample.target_score)
        score_levels = int(sample.score_levels or 5)
    elif q_type == "extract" and sample.target_span_text:
        target_str = sample.target_span_text.strip()
        span_tokens = tokenizer.tokenize(target_str)
        if span_tokens:
            ids = [tokenizer.vocab.get(t, tokenizer.unk_token_id) for t in span_tokens]
            for i in range(len(input_ids) - len(ids) + 1):
                if input_ids[i:i + len(ids)] == ids:
                    start_target, end_target = i, i + len(ids) - 1
                    break
    return {
        "input_ids": input_ids,
        "choice_target": choice_target,
        "noul_target": noul_target,
        "score_target": score_target,
        "score_levels": score_levels,
        "start_target": start_target,
        "end_target": end_target,
    }


def collate(items, device):
    maxlen = max(len(it["input_ids"]) for it in items)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(items), maxlen), pad, dtype=torch.long)
    mask = torch.zeros(len(items), maxlen, dtype=torch.long)
    for i, it in enumerate(items):
        n = len(it["input_ids"])
        input_ids[i, :n] = torch.tensor(it["input_ids"], dtype=torch.long)
        mask[i, :n] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": mask.to(device),
        "question_type": [None] * len(items),
        "choice_target": torch.tensor([it["choice_target"] for it in items], dtype=torch.long, device=device),
        "noul_target": torch.tensor([it["noul_target"] for it in items], dtype=torch.float, device=device),
        "score_target": torch.tensor([it["score_target"] for it in items], dtype=torch.float, device=device),
        "score_levels": torch.tensor([it["score_levels"] for it in items], dtype=torch.long, device=device),
        "start_target": torch.tensor([it["start_target"] for it in items], dtype=torch.long, device=device),
        "end_target": torch.tensor([it["end_target"] for it in items], dtype=torch.long, device=device),
    }


def batches_by_length(items, batch_size, seed):
    """Length-grouped batching: сортировка по длине, шаффл порядка батчей."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng = random.Random(SEED + len(items))
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def eval_choice_acc(items, idx_to_class, class_to_idx):
    model.eval()
    n = cand_ok = 0
    ce_sum = 0.0
    preds = Counter()
    for bidx in batches_by_length(items, BATCH, SEED):
        chunk = [items[i] for i in bidx]
        batch = collate(chunk, device)
        outs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outs["choice_logits"].float()
        probs = torch.softmax(logits, dim=-1)
        targets = batch["choice_target"]
        m = targets != -100
        if m.any():
            ce_sum += F.cross_entropy(logits[m], targets[m], reduction="sum").item()
        for j, it in enumerate(chunk):
            t = it["choice_target"]
            if t == -100:
                continue
            # кандидаты: выборка из 42 классов по choices сэмпла недоступна тут дёшево,
            # поэтому argmax по всем классам (строгая метрика)
            p = int(probs[j].argmax())
            cand_ok += p == t
            preds[idx_to_class[p]] += 1
            n += 1
    model.train()
    return cand_ok / max(1, n), ce_sum / max(1, n), preds


def main():
    global model, tokenizer, device
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    checkpoint = torch.load(START_CKPT, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = LeethConfig(**config)
    model = LeethModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    train_raw = md_train + replay
    rng.shuffle(train_raw)
    log("токенизация train...")
    train_items = [encode_sample(s, tokenizer, class_to_idx) for s in train_raw]
    md_val_items = [encode_sample(s, tokenizer, class_to_idx) for s in md_choice_val]
    arg_val_items = [encode_sample(s, tokenizer, class_to_idx) for s in arg_choice_val]
    log(f"токенизация готова: train={len(train_items)} md_val={len(md_val_items)} arg_val={len(arg_val_items)}")

    acc_md_before, ce_md_before, _ = eval_choice_acc(md_val_items, idx_to_class, class_to_idx)
    acc_arg_before, ce_arg_before, _ = eval_choice_acc(arg_val_items, idx_to_class, class_to_idx)
    log(f"BEFORE: md_choice_acc={acc_md_before * 100:.1f}% (CE {ce_md_before:.3f}) | "
        f"argent_choice_acc={acc_arg_before * 100:.1f}% (CE {ce_arg_before:.3f})")

    n_steps_total = EPOCHS * math.ceil(len(train_items) / BATCH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler()

    model.train()
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_loss, nb = 0.0, 0
        for bidx in batches_by_length(train_items, BATCH, SEED + epoch):
            chunk = [train_items[i] for i in bidx]
            batch = collate(chunk, device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss, _ = compute_calibrated_loss(outs, batch)
            if torch.isnan(loss) or loss.item() <= 0:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # cosine с warmup вручную
            if global_step < WARMUP:
                lr_now = LR * (global_step + 1) / WARMUP
            else:
                progress = (global_step - WARMUP) / max(1, n_steps_total - WARMUP)
                lr_now = LR * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            epoch_loss += loss.item()
            nb += 1
        log(f"epoch {epoch}/{EPOCHS} | loss={epoch_loss / max(1, nb):.4f} | steps={nb} | lr={optimizer.param_groups[0]['lr']:.2e}")

    acc_md_after, ce_md_after, preds_md = eval_choice_acc(md_val_items, idx_to_class, class_to_idx)
    acc_arg_after, ce_arg_after, _ = eval_choice_acc(arg_val_items, idx_to_class, class_to_idx)
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
            "purpose": "multidomain_v2_full_ft",
            "epochs": EPOCHS, "lr": LR, "batch": BATCH, "max_length": MAX_LEN,
            "train": len(train_items), "md_train": len(md_train), "replay": len(replay),
            "md_choice_acc_before": acc_md_before, "md_choice_acc_after": acc_md_after,
            "argent_choice_acc_before": acc_arg_before, "argent_choice_acc_after": acc_arg_after,
            "gate_argent": gate_arg, "gate_md": gate_md, "accepted": gate_arg and gate_md,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    torch.save(payload, OUT_CKPT)
    log(f"кандидат: {OUT_CKPT} ({'ПРИНЯТ' if gate_arg and gate_md else 'гейты не пройдены'})")
    log(f"время: {(time.time() - t0) / 60:.1f} мин")


if __name__ == "__main__":
    main()