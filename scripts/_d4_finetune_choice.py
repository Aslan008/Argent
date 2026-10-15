# -*- coding: utf-8 -*-
"""scripts/_d4_finetune_choice.py — финетюн choice-головы mmBERT на UNFAMILIAR-негативах.

Проблема: choice-голова хочет вызвать инструмент на составных задачах
(батарея: expected=UNFAMILIAR -> best=run_command/read_file ×12, recall 10/26).
Аудит корпуса: routing-UNFAMILIAR в train — только деструктив/OOD; составные
русские задачи отсутствуют. _d4_gen_unfamiliar_negs.py добавил 392 негатива.

Метод: грузим ТЕКУЩИЕ головы (mmbert-leeth.pt), замораживаем всё кроме
choice_head (noul_head не трогается — need_tool 99.5% в безопасности),
тренируем на choice-строках корпуса + экстра-негативы ×OVERSAMPLE.

Гейты (val, до сохранения):
  - val choice CE: деградация > +5% -> FAIL
  - val routing-choice acc (225, raw argmax): падение > 1 п.п. -> FAIL
  - val UNFAMILIAR acc: < 0.99 -> FAIL

Выход: .argent/models/mmbert-leeth_candidate.pt (прод НЕ перезаписывается —
промоушен только после экспорта и батарейного гейта).
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
EXTRA_PATH = Path("src/leeth/data_400m/leeth_400m_train_extra.jsonl")
VAL_PATH = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
P = "Incoming user request in Argent session:"

EPOCHS = 2
LR = 5e-5
BATCH = 32
OVERSAMPLE = 2
MAX_LEN = 256


def load_rows(path: Path, limit: int = 0) -> list:
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
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {"device": device, "mode": "finetune_choice_unfamiliar"}

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
        p.requires_grad_(name.startswith("choice_head"))
    trainable = [n for n, p in heads.named_parameters() if p.requires_grad]
    summary["trainable"] = trainable
    print("Trainable:", trainable)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_to_idx = json.load(f)
    unfam_idx = class_to_idx["UNFAMILIAR"]

    # ── данные: choice-строки корпуса + экстра ×OVERSAMPLE ───────────────────
    corpus_choice = [r for r in load_rows(TRAIN_PATH) if r["question_type"] == "choice" and r.get("target_choice")]
    extra = [r for r in load_rows(EXTRA_PATH) if r.get("target_choice")]
    mix = corpus_choice + extra * (OVERSAMPLE - 1)
    random.Random(7).shuffle(mix)
    summary["corpus_choice"] = len(corpus_choice)
    summary["extra"] = len(extra)
    summary["extra_oversampled"] = len(extra) * (OVERSAMPLE - 1)
    summary["mix"] = len(mix)
    print(f"mix: {len(mix)} (corpus {len(corpus_choice)} + extra {len(extra)} x{OVERSAMPLE})")

    def collate(batch):
        texts = [r["state"] + "\n[QUESTION] " + r["question"] for r in batch]
        targets = [class_to_idx.get(r["target_choice"], -100) for r in batch]
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return enc, torch.tensor(targets, dtype=torch.long)

    loader = DataLoader(mix, batch_size=BATCH, shuffle=True, collate_fn=collate)

    def pooled_forward(enc):
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = base.base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return heads.choice_head(pooled)

    def eval_choice(rows_subset):
        """(ce, acc, unfam_acc, n) по choice-строкам."""
        loader_eval = DataLoader(rows_subset, batch_size=64, shuffle=False, collate_fn=collate)
        heads.eval()
        total, nb, correct, unfam_ok, unfam_n, count = 0.0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for enc, targets in loader_eval:
                logits = pooled_forward(enc)
                targets = targets.to(device)
                m = targets != -100
                if not m.any():
                    continue
                total += F.cross_entropy(logits[m], targets[m], label_smoothing=0.05).item()
                nb += 1
                preds = logits[m].argmax(-1)
                correct += (preds == targets[m]).sum().item()
                count += m.sum().item()
                um = targets == unfam_idx
                unfam_ok += (preds[um] == targets[um]).sum().item()
                unfam_n += um.sum().item()
        heads.train()
        return total / max(1, nb), correct / max(1, count), (unfam_ok / unfam_n if unfam_n else float("nan")), count

    # ── BEFORE ────────────────────────────────────────────────────────────────
    val_choice = [r for r in load_rows(VAL_PATH) if r["question_type"] == "choice" and r.get("target_choice")]
    val_routing = [r for r in val_choice if r["question"] == TEMPLATE_Q and r["state"].startswith(P)]
    before_all = eval_choice(val_choice)
    before_rt = eval_choice(val_routing)
    summary["before"] = {"val_ce": before_all[0], "val_acc": before_all[1],
                         "val_unfam_acc": before_all[2], "routing_acc": before_rt[1]}
    print(f"BEFORE: val_ce={before_all[0]:.4f} val_acc={before_all[1]:.4f} "
          f"unfam_acc={before_all[2]:.4f} routing_acc={before_rt[1]:.4f}")

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
                targets = targets.to(device)
                m = targets != -100
                loss = F.cross_entropy(logits[m], targets[m], label_smoothing=0.05)
            if not torch.isnan(loss) and loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in heads.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item()
                steps += 1
                correct += (logits[m].argmax(-1) == targets[m]).sum().item()
                count += m.sum().item()
            if steps % 200 == 0 and steps > 0:
                print(f"  epoch {epoch} step {steps}/{len(loader)} loss={total / steps:.4f} "
                      f"acc={correct / max(1, count):.4f} | {time.perf_counter() - t0:.0f}s")
                sys.stdout.flush()
        print(f"epoch {epoch}/{EPOCHS}: loss={total / max(1, steps):.4f} "
              f"train_acc={correct / max(1, count):.4f} ({time.perf_counter() - t0:.0f}s)")

    # ── AFTER + гейты ─────────────────────────────────────────────────────────
    after_all = eval_choice(val_choice)
    after_rt = eval_choice(val_routing)
    summary["after"] = {"val_ce": after_all[0], "val_acc": after_all[1],
                        "val_unfam_acc": after_all[2], "routing_acc": after_rt[1]}
    print(f"AFTER:  val_ce={after_all[0]:.4f} val_acc={after_all[1]:.4f} "
          f"unfam_acc={after_all[2]:.4f} routing_acc={after_rt[1]:.4f}")

    gate_ok = True
    if after_all[0] > before_all[0] * 1.05:
        gate_ok = False
        print(f"GATE FAIL: val CE {before_all[0]:.4f} -> {after_all[0]:.4f} (> +5%)")
    if after_rt[1] < before_rt[1] - 0.01:
        gate_ok = False
        print(f"GATE FAIL: routing acc {before_rt[1]:.4f} -> {after_rt[1]:.4f} (> -1 п.п.)")
    if not (after_all[2] == after_all[2]) or after_all[2] < 0.99:
        gate_ok = False
        print(f"GATE FAIL: val UNFAMILIAR acc {after_all[2]} < 0.99")
    summary["gate_ok"] = gate_ok
    print("Gate:", "OK" if gate_ok else "FAIL")

    torch.save({"heads": heads.state_dict(), "hidden": hidden, "n_classes": n_classes},
               CANDIDATE_PATH)
    summary["candidate"] = str(CANDIDATE_PATH)
    summary["elapsed_s"] = round(time.time() - started, 1)
    out = Path(".argent/artifacts/d4_finetune_choice.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Candidate -> {CANDIDATE_PATH} | summary -> {out}")


if __name__ == "__main__":
    main()