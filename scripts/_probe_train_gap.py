# -*- coding: utf-8 -*-
"""scripts/_probe_train_gap.py — choice-CE на train vs val: недообученность или генерализация?"""
import json
import random
import sys
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
CKPT = DATA / "leeth_best.pt"

checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
config = checkpoint["config"]
if isinstance(config, dict):
    config = LeethConfig(**config)
model = LeethModel(config)
model.load_state_dict(checkpoint["state_dict"])
model.to("cuda").eval()

class_to_idx = json.loads((DATA / "choice_classes.json").read_text(encoding="utf-8"))
tokenizer = LeethTokenizer.load(DATA)


def choice_ce(samples, label):
    ds = LeethTorchDataset(samples, tokenizer, max_length=256, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_leeth_batch)
    ce_sum, n, ok = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to("cuda")
            mask = batch["attention_mask"].to("cuda")
            outs = model(input_ids, attention_mask=mask)
            logits = outs["choice_logits"].float()
            targets = batch["choice_target"].to("cuda")
            ce_sum += F.cross_entropy(logits, targets, reduction="sum").item()
            ok += (logits.argmax(-1) == targets).sum().item()
            n += targets.size(0)
    print(f"{label}: n={n}  choice-CE={ce_sum / n:.3f}  argmax-42={ok / n * 100:.0f}%")


md_train = _load_jsonl_samples(DATA / "multidomain_train.jsonl")
md_train_choice = [s for s in md_train if s.question_type == "choice" and s.choices]
rng = random.Random(7)
choice_ce(md_train_choice[:400], "md_TRAIN choice (виденные)")
md_val = _load_jsonl_samples(DATA / "multidomain_val.jsonl")
choice_ce([s for s in md_val if s.question_type == "choice" and s.choices], "md_VAL choice (невиденные)")
arg_val = _load_jsonl_samples(DATA / "leeth_400m_val.jsonl", limit=400)
choice_ce([s for s in arg_val if s.question_type == "choice" and s.choices], "argent_VAL choice")