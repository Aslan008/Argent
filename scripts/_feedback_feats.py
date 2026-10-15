# -*- coding: utf-8 -*-
"""scripts/_feedback_feats.py — фичи для choice-сэмплов фидбека (базовый форвард).

accept+dispatch  -> target=dispatch.tool, conf=-100
partial          -> target=correction.tool, conf=ПРЕДЛОЖЕННЫЙ (неверный) инструмент
reject           -> target=NO_TOOL (если нет коррекции)
safety accept    -> ACT_Q execute_immediately; reject -> block_command
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.leeth import feedback as fb
from src.leeth.feedback import TOOL_Q, ACTION_Q

BASE_DIR = Path(".argent/models/mmbert-argent")
OUT = Path(".argent/kaggle/dataset_v2/feat_feedback.npz")
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))

rows = []  # (state, question, target_choice, confuser_choice)
for rec in fb.load_records():
    v = rec.get("verdict")
    if not v:
        continue
    state = rec.get("state") or ""
    if not state:
        continue
    prop_tool = ((rec.get("proposal") or {}).get("tool") or {}).get("choice")
    corr = rec.get("correction") or {}
    disp = rec.get("dispatch") or {}
    skip_confirmed = bool(rec.get("skip_reason")) and v == "accept"
    # 1) routing choice
    target = corr.get("tool") or disp.get("tool_name") or (
        prop_tool if v == "accept" and not skip_confirmed else None)
    if v == "reject" and not corr.get("tool"):
        target = "NO_TOOL"
    if target and target in class_to_idx:
        confuser = -100
        if v == "partial" and prop_tool and prop_tool in class_to_idx and prop_tool != target:
            confuser = class_to_idx[prop_tool]
        rows.append((state, TOOL_Q, target, confuser))
    # 2) safety action choice
    safety = rec.get("safety") or {}
    if safety.get("state") and v in ("accept", "reject"):
        act = "execute_immediately" if v == "accept" else "block_command"
        rows.append((safety["state"], ACTION_Q, act, -100))

print(f"choice-сэмплов из фидбека: {len(rows)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
base.to(device).eval()

feats, targets, confs = [], [], []
B = 16
with torch.no_grad():
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        enc = tok([s + "\n[QUESTION] " + q for s, q, _, _ in chunk],
                  truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = base.base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        feats.append(pooled.cpu())
        targets.extend(class_to_idx[t] for _, _, t, _ in chunk)
        confs.extend(c for _, _, _, c in chunk)

X = torch.cat(feats).to(torch.float16).numpy()
np.savez(OUT, feats=X, targets=np.array(targets, dtype=np.int64),
         confs=np.array(confs, dtype=np.int64))
print(f"сохранено {OUT}: {X.shape}, {OUT.stat().st_size / 1e6:.1f} МБ")