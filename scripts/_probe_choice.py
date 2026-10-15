# -*- coding: utf-8 -*-
"""scripts/_probe_choice.py — диагностика choice-головы: md_val vs argent_val.

Правильное выравнивание батчей; choice-CE отдельно; распределение предсказаний.
"""
import json
import sys
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
from src.leeth.train import LeethTorchDataset, collate_leeth_batch

DATA = ROOT / "src/leeth/data_400m"

checkpoint = torch.load(DATA / "leeth_best.pt", map_location="cpu", weights_only=False)
config = checkpoint["config"]
if isinstance(config, dict):
    config = LeethConfig(**config)
model = LeethModel(config)
model.load_state_dict(checkpoint["state_dict"])
model.to("cuda").eval()

class_to_idx = json.loads((DATA / "choice_classes.json").read_text(encoding="utf-8"))
idx_to_class = {v: k for k, v in class_to_idx.items()}
tokenizer = LeethTokenizer.load(DATA)


def eval_choice(samples, label):
    ds = LeethTorchDataset(samples, tokenizer, max_length=256, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_leeth_batch)
    n = full_ok = cand_ok = 0
    ce_sum = 0.0
    full_preds, cand_preds = Counter(), Counter()
    ptr = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["input_ids"].size(0)
            chunk = samples[ptr:ptr + bs]
            ptr += bs
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            outs = model(input_ids, attention_mask=mask)
            logits = outs["choice_logits"].float()
            probs = torch.softmax(logits, dim=-1)
            targets = batch["choice_target"]
            ce_sum += F.cross_entropy(logits, targets.to("cuda"), reduction="sum").item()
            for j, s in enumerate(chunk):
                t = int(targets[j])
                fp = int(logits[j].argmax())
                full_ok += fp == t
                full_preds[idx_to_class[fp]] += 1
                cand = s.choices or []
                idxs = [class_to_idx[c] for c in cand if c in class_to_idx]
                if idxs:
                    cp = idxs[int(probs[j][idxs].argmax())]
                    cand_ok += cp == t
                    cand_preds[idx_to_class[cp]] += 1
                n += 1
    print(f"\n=== {label}: n={n} ===")
    print(f"  choice-CE: {ce_sum / n:.3f} (random ln42={3.738:.3f})")
    print(f"  argmax-42: {full_ok}/{n} = {full_ok / n * 100:.0f}%   с ограничением: {cand_ok}/{n} = {cand_ok / n * 100:.0f}%")
    print(f"  полные предсказания: {dict(full_preds.most_common(6))}")
    print(f"  среди кандидатов:    {dict(cand_preds.most_common(6))}")


md_val = _load_jsonl_samples(DATA / "multidomain_val.jsonl")
md_choice = [s for s in md_val if s.question_type == "choice" and s.choices]
eval_choice(md_choice, "md_val choice (новые сферы, с кандидатами)")

argent_val = _load_jsonl_samples(DATA / "leeth_400m_val.jsonl", limit=400)
arg_choice = [s for s in argent_val if s.question_type == "choice" and s.choices]
eval_choice(arg_choice, "argent_val choice (регрессия)")