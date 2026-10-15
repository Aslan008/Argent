# -*- coding: utf-8 -*-
"""scripts/_battery_miss_diff.py — какие промахи батареи у кандидата новые vs починенные."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
tok = AutoTokenizer.from_pretrained(str(BASE))
base = AutoModelForMaskedLM.from_pretrained(str(BASE), dtype=torch.float32)
base.to("cuda" if torch.cuda.is_available() else "cpu").eval()
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
idx_to_cls = {v: k for k, v in class_to_idx.items()}
TQ = "Какой инструмент Argent должен быть вызван в первую очередь?"
cases = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))["cases"]


class Heads(nn.Module):
    def __init__(s, h, n):
        super().__init__()
        s.choice_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, n))
        s.noul_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        s.score_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 10))
        s.extract_start_head = nn.Linear(h, 1)
        s.extract_end_head = nn.Linear(h, 1)


misses = {}
for tag, path in (("prod", ".argent/models/mmbert-leeth.pt"),
                  ("cand", ".argent/models/mmbert-leeth_candidate.pt")):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = Heads(ck["hidden"], ck["n_classes"])
    h.load_state_dict(ck["heads"])
    h.to(base.device).eval()
    m = []
    with torch.no_grad():
        for c in cases:
            t = (f'Incoming user request in Argent session:\n"{c["query"]}"\n'
                 f"Working directory: C:/Users/user/Desktop/Argent\n[QUESTION] {TQ}")
            e = tok(t, truncation=True, max_length=256, return_tensors="pt")
            e = {k: v.to(base.device) for k, v in e.items()}
            out = base.base_model(**e, output_hidden_states=True)
            hs = out.hidden_states[-1]
            mask = e["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pred = idx_to_cls[int(h.choice_head(pooled)[0].argmax())]
            if pred != c["expected_tool"]:
                m.append((c["query"], c["expected_tool"], pred))
    misses[tag] = m

only_cand = [x for x in misses["cand"] if x not in misses["prod"]]
only_prod = [x for x in misses["prod"] if x not in misses["cand"]]
print(f"промахи только у КАНДИДАТА (регресс): {len(only_cand)}")
for q, w, g in only_cand:
    print(f"  {q[:58]:<60} want {w:<16} got {g}")
print(f"промахи только у ПРОДА (кандидат починил): {len(only_prod)}")
for q, w, g in only_prod:
    print(f"  {q[:58]:<60} want {w:<16} got {g}")