# -*- coding: utf-8 -*-
"""scripts/port_heads.py — Фаза E3: перенос 5 голов на mmBERT-базу (ДОЛГО ⏳).

Зачем: decision-головы (choice/noul/score/extract/MLM) привязаны к
размерности старой базы. На mmBERT их нужно пересоздать и обучить заново
на нашем датасете. Запускается ТОЛЬКО после успешного E2.

БЕЗ --run: dry-run. С --run: обучение голов на mmBERT-базе.

Запуск (по команде, после E2): python scripts/port_heads.py --run
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_DIR = Path(".argent/models/mmbert-argent")  # выход E2 (pretrain_mmbert.py)
TRAIN_PATH = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
VAL_PATH = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
OUT_PATH = Path(".argent/models/mmbert-leeth.pt")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in open(path, encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="запустить обучение голов (долгая операция)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=40000)
    parser.add_argument("--state-drop", type=float, default=0.0, help="вероятность дропа state (форматная аугментация)")
    parser.add_argument("--class-weights", action="store_true", help="балансировка классов choice (sklearn-balanced, cap 4)")
    parser.add_argument("--oversample", type=str, default="", help="дублирование редких классов, напр. 'UNFAMILIAR:4,NO_TOOL:2'")
    args = parser.parse_args()

    n_train = count_lines(TRAIN_PATH)
    n_val = count_lines(VAL_PATH)

    if not args.run:
        print("\n=== DRY-RUN (долгая операция НЕ запущена) ===")
        print(f"База:        {BASE_DIR} (после E2)")
        print(f"Данные:      train={n_train}, val={n_val}")
        print(f"Обучение:    5 голов (choice/noul/score/extract/MLM) с нуля на mmBERT, {args.epochs} эпох")
        print(f"Выход:       {OUT_PATH}")
        print(f"Оценка времени: ~{args.epochs * 2:.0f} ч на RTX 4060 ⏳")
        print("\nДля запуска: python scripts/port_heads.py --run")
        return

    # ── Долгая операция ⏳ ────────────────────────────────────────────────────
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.float32)
    hidden = base.config.hidden_size
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    n_classes = max(class_to_idx.values()) + 1

    class Heads(nn.Module):
        """5 decision-голов над mmBERT-представлениями (как у LeethModel)."""

        def __init__(self, hidden: int, n_classes: int):
            super().__init__()
            self.choice_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
            self.noul_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 10))
            self.extract_start_head = nn.Linear(hidden, 1)
            self.extract_end_head = nn.Linear(hidden, 1)

        def forward(self, hidden_states, pooled, attention_mask):
            start = self.extract_start_head(hidden_states).squeeze(-1)
            end = self.extract_end_head(hidden_states).squeeze(-1)
            if attention_mask is not None:
                start = start + (1.0 - attention_mask.float()) * -10000.0
                end = end + (1.0 - attention_mask.float()) * -10000.0
            return {
                "choice_logits": self.choice_head(pooled),
                "noul_logits": self.noul_head(pooled).squeeze(-1),
                "score_logits": self.score_head(pooled),
                "start_logits": start,
                "end_logits": end,
            }

    heads = Heads(hidden, n_classes).to(device)
    base.to(device)
    base.eval()  # замороженная база: dropout выключен
    for p in base.parameters():
        p.requires_grad_(False)  # база заморожена — учим только головы

    class LeethRows(Dataset):
        def __init__(self, path: Path, limit: int = 0):
            self.rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.rows.append(json.loads(line))
                    except Exception:
                        continue
                    if limit and len(self.rows) >= limit:
                        break

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    def collate(batch):
        texts, choice_t, noul_t = [], [], []
        for rec in batch:
            state = rec.get("state", "")
            if random.random() < args.state_drop:  # форматная аугментация (по умолчанию выкл.)
                state = ""
            text = state + "\n[QUESTION] " + rec.get("question", "")
            texts.append(text)
            qt = rec.get("question_type")
            if qt == "choice" and rec.get("target_choice"):
                choice_t.append(class_to_idx.get(rec["target_choice"], -100))
                noul_t.append(-1.0)
            elif qt == "noul" and rec.get("target_noul") is not None:
                choice_t.append(-100)
                noul_t.append(float(rec["target_noul"]))
            else:
                choice_t.append(-100)
                noul_t.append(-1.0)
        enc = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
        return enc, (
            torch.tensor(choice_t, dtype=torch.long),
            torch.tensor(noul_t, dtype=torch.float),
        )

    train_ds = LeethRows(TRAIN_PATH, args.max_samples)
    class_w = None
    if args.class_weights:
        # Балансировка классов choice: UNFAMILIAR в датасете ~3.6%, в прод-батарее ~24%
        counts = torch.zeros(n_classes)
        for r in train_ds.rows:
            idx = class_to_idx.get(r.get("target_choice"))
            if idx is not None:
                counts[idx] += 1
        class_w = counts.sum() / (n_classes * (counts + 1.0))  # sklearn-'balanced': средний вес ≈ 1
        class_w = class_w.clamp(max=4.0).to(device)
        print(f"class_w: min={class_w.min():.3f} max={class_w.max():.3f}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    if args.oversample:
        extra: list = []
        for spec in args.oversample.split(","):
            cls, factor = spec.split(":")
            idx = class_to_idx[cls]
            dup = [r for r in train_ds.rows if class_to_idx.get(r.get("target_choice")) == idx]
            extra.extend(dup * (int(factor) - 1))
        train_ds.rows.extend(extra)
        print(f"oversample {args.oversample}: +{len(extra)} строк")

    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    heads.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        total, steps, correct, count = 0.0, 0, 0, 0
        for enc, (choice_t, noul_t) in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            choice_t, noul_t = choice_t.to(device), noul_t.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                with torch.no_grad():  # база заморожена — граф не нужен (экономия VRAM и FLOPs MLM-головы)
                    out = base.base_model(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1]  # [bsz, seq, hidden]
                mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                logits = heads(hs, pooled, enc["attention_mask"])

                loss = torch.tensor(0.0, device=device)
                cmask = choice_t != -100
                if cmask.any():
                    loss = loss + F.cross_entropy(logits["choice_logits"][cmask], choice_t[cmask], label_smoothing=0.05, weight=class_w)
                    preds = logits["choice_logits"][cmask].argmax(dim=-1)
                    correct += (preds == choice_t[cmask]).sum().item()
                    count += cmask.sum().item()
                nmask = noul_t >= 0.0
                if nmask.any():
                    loss = loss + F.binary_cross_entropy_with_logits(logits["noul_logits"][nmask], noul_t[nmask])

            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
                steps += 1
        acc = correct / max(1, count)
        print(f"epoch {epoch}/{args.epochs}: loss={total / max(1, steps):.4f} "
              f"choice_acc={acc:.1%} ({time.perf_counter() - t0:.0f}s)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"heads": heads.state_dict(), "hidden": hidden, "n_classes": n_classes}, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")
    print("СЛЕДУЮЩИЙ ШАГ (E4): валидация на прод-батарее, затем E5 (экспорт INT8).")


if __name__ == "__main__":
    main()