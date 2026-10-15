# -*- coding: utf-8 -*-
"""scripts/_battery_precompute.py — фичи для таргетированных battery-строк."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
SRC = Path(".argent/kaggle/dataset_v2/battery_rows.jsonl")
OUT = Path(".argent/kaggle/dataset_v2/feat_battery.npz")
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
print(f"строк: {len(rows)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
base.to(device).eval()

feats, targets, confs = [], [], []
B = 16
with torch.no_grad():
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        enc = tok([r["state"] + "\n[QUESTION] " + r["question"] for r in chunk],
                  truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = base.base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        feats.append(pooled.cpu())
        targets.extend(class_to_idx[r["target_choice"]] for r in chunk)
        confs.extend(class_to_idx.get(r.get("confuser_choice") or "", -100) for r in chunk)

X = torch.cat(feats).to(torch.float16).numpy()
np.savez(OUT, feats=X, targets=np.array(targets, dtype=np.int64),
         confs=np.array(confs, dtype=np.int64))
print(f"сохранено {OUT}: {X.shape}")