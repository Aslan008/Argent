# -*- coding: utf-8 -*-
"""scripts/_probe_family.py — per-family choice acc на md_val (department/etype/violation)."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
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

val = _load_jsonl_samples(DATA / "multidomain_val.jsonl")
choice_val = [s for s in val if s.question_type == "choice" and s.choices]

by_q = {}
for s in choice_val:
    by_q.setdefault(s.question, []).append(s)

for qtext, samples in by_q.items():
    ds = LeethTorchDataset(samples, tokenizer, max_length=256, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_leeth_batch)
    ok = n = 0
    preds, golds = Counter(), Counter()
    ptr = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["input_ids"].size(0)
            chunk = samples[ptr:ptr + bs]
            ptr += bs
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            outs = model(input_ids, attention_mask=mask)
            probs = torch.softmax(outs["choice_logits"].float(), dim=-1)
            for j, s in enumerate(chunk):
                cand = s.choices or []
                idxs = [class_to_idx[c] for c in cand if c in class_to_idx]
                if idxs:
                    cp = idxs[int(probs[j][idxs].argmax())]
                    preds[idx_to_class[cp]] += 1
                    ok += cp == int(batch["choice_target"][j])
                    n += 1
                    golds[s.target_choice] += 1
    print(f"\n{qtext[:50]}: {ok}/{n} = {ok / max(1, n) * 100:.0f}%")
    print(f"  gold: {dict(golds)}")
    print(f"  pred: {dict(preds)}")