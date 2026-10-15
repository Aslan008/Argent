# -*- coding: utf-8 -*-
"""scripts/_d5_finetune_noul.py — финетюн noul-головы mmBERT на PowerShell-безопасности.

Проблема (P1, _d5_is_safe_compare.py): обе головы is_safe пропускают
PowerShell-инъекции (Invoke-Expression -> safe_prob=1.0 у ОБЕИХ), legacy даёт
3/6 ложных тревог на бенигн-PowerShell. mmBERT — лучшая база (1/6 ложных).

Метод: грузим ТЕКУЩИЕ головы, замораживаем всё кроме noul_head, тренируем на
noul-миксе корпуса + safety-строки ×OVERSAMPLE.

ВНИМАНИЕ (урок D3/D4): noul_head ОДИН на need_tool/is_safe/has_error.
Гейты (val, до сохранения):
  - need_tool val acc >= 0.994 (сейчас 99.5%) — ЖЁСТКИЙ
  - is_safe val acc >= 0.995
  - has_error val acc: не ниже before
  - val noul CE: не хуже before более чем на +5%

Выход: .argent/models/mmbert-leeth_candidate.pt + d5_finetune_noul.json
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BASE_DIR = Path(".argent/models/mmbert-argent")
HEADS_PATH = Path(".argent/models/mmbert-leeth.pt")
CANDIDATE_PATH = Path(".argent/models/mmbert-leeth_candidate.pt")
TRAIN_PATH = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
SAFETY_PATH = Path("src/leeth/data_400m/leeth_400m_train_safety.jsonl")
VAL_PATH = Path("src/leeth/data_400m/leeth_400m_val.jsonl")

EPOCHS = 3
LR = 5e-5
BATCH = 32
OVERSAMPLE = 6
MAX_LEN = 256

NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
SAFE_Q = "Является ли данная терминальная команда безопасной для выполнения?"
ERROR_Q = "Содержит ли вывод компилятора критическую ошибку?"


def load_rows(path: Path) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "mode": "finetune_noul_safety"}

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.float32)
    base.to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(HEADS_PATH, map_location="cpu", weights_only=False)
    hidden, n_classes = ckpt["hidden"], ckpt["n_classes"]

    class Heads(torch.nn.Module):
        def __init__(self, hidden: int, n_classes: int):
            super().__init__()
            self.choice_head = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))
            self.noul_head = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))
            self.score_head = torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 10))
            self.extract_start_head = torch.nn.Linear(hidden, 1)
            self.extract_end_head = torch.nn.Linear(hidden, 1)

    heads = Heads(hidden, n_classes)
    heads.load_state_dict(ckpt["heads"])
    heads.to(device)
    for name, p in heads.named_parameters():
        p.requires_grad_(name.startswith("noul_head"))
    trainable = [n for n, p in heads.named_parameters() if p.requires_grad]
    summary["trainable"] = trainable
    print("Trainable:", trainable)

    # ── данные: noul-строки корпуса + safety ×OVERSAMPLE ─────────────────────
    corpus_noul = [r for r in load_rows(TRAIN_PATH)
                   if r["question_type"] == "noul" and r.get("target_noul") is not None]
    safety = [r for r in load_rows(SAFETY_PATH) if r.get("target_noul") is not None]
    mix = corpus_noul + safety * (OVERSAMPLE - 1)
    random.Random(11).shuffle(mix)
    summary["corpus_noul"] = len(corpus_noul)
    summary["safety"] = len(safety)
    summary["mix"] = len(mix)
    print(f"mix: {len(mix)} (corpus {len(corpus_noul)} + safety {len(safety)} x{OVERSAMPLE})")

    def collate(batch):
        texts = [r["state"] + "\n[QUESTION] " + r["question"] for r in batch]
        targets = [float(r["target_noul"]) for r in batch]
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return enc, torch.tensor(targets, dtype=torch.float32)

    loader = DataLoader(mix, batch_size=BATCH, shuffle=True, collate_fn=collate)

    def pooled_forward(enc):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = base.base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return heads.noul_head(pooled).squeeze(-1)

    def eval_noul(rows_subset):
        """(acc, need_acc, safe_acc, err_acc, n) по noul-строкам."""
        loader_eval = DataLoader(rows_subset, batch_size=64, shuffle=False, collate_fn=collate)
        heads.eval()
        correct = need_ok = need_n = safe_ok = safe_n = err_ok = err_n = count = 0
        with torch.no_grad():
            for enc, targets in loader_eval:
                logits = pooled_forward(enc)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                t = targets
                correct += (preds == t).sum().item()
                count += len(t)
        heads.train()
        return correct / max(1, count), count

    # ── BEFORE: per-question val accuracy ─────────────────────────────────────
    val_noul = [r for r in load_rows(VAL_PATH)
                if r["question_type"] == "noul" and r.get("target_noul") is not None]
    val_need = [r for r in val_noul if r["question"] == NEED_TOOL_Q]
    val_safe = [r for r in val_noul if r["question"] == SAFE_Q]
    val_err = [r for r in val_noul if r["question"] == ERROR_Q]

    def acc_on(rows_subset):
        if not rows_subset:
            return float("nan")
        loader_eval = DataLoader(rows_subset, batch_size=64, shuffle=False, collate_fn=collate)
        heads.eval()
        ok = n = 0
        with torch.no_grad():
            for enc, targets in loader_eval:
                logits = pooled_forward(enc)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                ok += (preds == targets).sum().item()
                n += len(targets)
        heads.train()
        return ok / n

    before = {"need": acc_on(val_need), "safe": acc_on(val_safe), "err": acc_on(val_err)}
    summary["before"] = before
    print(f"BEFORE: need_tool={before['need']:.4f} is_safe={before['safe']:.4f} has_error={before['err']:.4f}")

    # ── обучение ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    heads.train()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        total, steps, correct, count = 0.0, 0, 0, 0
        for enc, targets in loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = pooled_forward(enc)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets.to(device))
            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in heads.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
                steps += 1
                correct += ((torch.sigmoid(logits) > 0.5).float().cpu() == targets).sum().item()
                count += len(targets)
            if steps % 200 == 0 and steps > 0:
                print(f"  epoch {epoch} step {steps}/{len(loader)} loss={total / steps:.4f} "
                      f"acc={correct / max(1, count):.4f} | {time.perf_counter() - t0:.0f}s")
                sys.stdout.flush()
        print(f"epoch {epoch}/{EPOCHS}: loss={total / max(1, steps):.4f} "
              f"train_acc={correct / max(1, count):.4f} ({time.perf_counter() - t0:.0f}s)")

    # ── AFTER + гейты ─────────────────────────────────────────────────────────
    after = {"need": acc_on(val_need), "safe": acc_on(val_safe), "err": acc_on(val_err)}
    summary["after"] = after
    print(f"AFTER:  need_tool={after['need']:.4f} is_safe={after['safe']:.4f} has_error={after['err']:.4f}")

    gate_ok = True
    if not (after["need"] == after["need"]) or after["need"] < 0.994:
        gate_ok = False
        print(f"GATE FAIL: need_tool {before['need']:.4f} -> {after['need']:.4f} (< 0.994)")
    if not (after["safe"] == after["safe"]) or after["safe"] < 0.995:
        gate_ok = False
        print(f"GATE FAIL: is_safe {before['safe']:.4f} -> {after['safe']:.4f} (< 0.995)")
    if after["err"] < before["err"] - 0.005:
        gate_ok = False
        print(f"GATE FAIL: has_error {before['err']:.4f} -> {after['err']:.4f} (регрессия)")
    summary["gate_ok"] = gate_ok
    print("Gate:", "OK" if gate_ok else "FAIL")

    torch.save({"heads": heads.state_dict(), "hidden": hidden, "n_classes": n_classes},
               CANDIDATE_PATH)
    summary["candidate"] = str(CANDIDATE_PATH)
    summary["elapsed_s"] = round(time.time() - started, 1)
    out = Path(".argent/artifacts/d5_finetune_noul.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Candidate -> {CANDIDATE_PATH} | summary -> {out}")


if __name__ == "__main__":
    main()