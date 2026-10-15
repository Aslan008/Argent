# -*- coding: utf-8 -*-
"""scripts/_tmp_fp16_drift.py — дрейф признаков/логитов при fp16-округлении весов базы.

Симуляция: fp32 -> fp16 -> fp32 (ровно то, что произойдёт при загрузке fp16-файла
с torch_dtype=float32). Сравниваем пулированные признаки и логиты choice-головы.
"""
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".argent/models/mmbert-argent"
HEADS = ROOT / ".argent/models/mmbert-leeth.pt"
DATA = ROOT / "src/leeth/data_400m"
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

tok = AutoTokenizer.from_pretrained(str(BASE))
base = AutoModelForMaskedLM.from_pretrained(str(BASE), torch_dtype=torch.float32).eval()
ckpt = torch.load(HEADS, map_location="cpu", weights_only=False)
hidden, n_classes = ckpt["hidden"], ckpt["n_classes"]


class Heads(torch.nn.Module):
    def __init__(self, hidden, n_classes):
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
heads.eval()

rows = []
with open(DATA / "docker_val.jsonl", encoding="utf-8") as f:
    rows += [json.loads(l) for l in f if l.strip()]
with open(DATA / "leeth_400m_val.jsonl", encoding="utf-8") as f:
    for line in f:
        if TEMPLATE_Q in line:
            r = json.loads(line)
            if r.get("question") == TEMPLATE_Q:
                rows.append(r)
rows = rows[:48]

texts = [r["state"] + "\n[QUESTION] " + r["question"] for r in rows]
enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")

base_fp16 = AutoModelForMaskedLM.from_pretrained(str(BASE), torch_dtype=torch.float16).float()


def pooled(model):
    with torch.no_grad():
        out = model.base_model(**enc, output_hidden_states=True)
    hs = out.hidden_states[-1]
    mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
    return (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)


with torch.no_grad():
    p32, p16 = pooled(base), pooled(base_fp16)
    l32, l16 = heads.choice_head(p32), heads.choice_head(p16)

feat_shift = (p32 - p16).abs().max().item()
logit_shift = (l32 - l16).abs().max().item()
flips = (l32.argmax(-1) != l16.argmax(-1)).sum().item()
margin32 = l32.sort(-1, descending=True).values
margin16 = l16.sort(-1, descending=True).values
top_gap32 = (margin32[:, 0] - margin32[:, 1])
top_gap16 = (margin16[:, 0] - margin16[:, 1])
gap_shift = (top_gap32 - top_gap16).abs().max().item()

print(f"rows: {len(rows)}")
print(f"max |feat shift|:  {feat_shift:.6f}")
print(f"max |logit shift|: {logit_shift:.6f}")
print(f"argmax flips:      {flips}/{len(rows)}")
print(f"max |top1-top2 gap shift|: {gap_shift:.6f}")
verdict = "SAFE" if (logit_shift < 0.01 and flips == 0) else "RISKY"
print("verdict:", verdict)