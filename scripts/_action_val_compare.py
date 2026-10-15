# -*- coding: utf-8 -*-
"""scripts/_action_val_compare.py — action-точность на val: прод vs кандидат."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

ACT_Q = "Какое действие должен предпринять агент перед запуском этой команды?"
FEATS = Path(".argent/kaggle/dataset_v2/feat_val.npz")
VAL = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
idx_to_cls = {v: k for k, v in class_to_idx.items()}

rows = [json.loads(l) for l in open(VAL, encoding="utf-8")]
choice_rows = [r for r in rows if r.get("question_type") == "choice" and r.get("target_choice")]
act_idx = [j for j, r in enumerate(choice_rows) if r.get("question") == ACT_Q]
print(f"action-строк в val (choice): {len(act_idx)}")
dist = Counter(choice_rows[j]["target_choice"] for j in act_idx)
for k, v in dist.most_common():
    print(f"  target {k}: {v}")

z = np.load(FEATS)
feats = torch.from_numpy(z["feats"].astype(np.float32))[act_idx]
targets = torch.from_numpy(z["targets"].astype(np.int64))[act_idx]


class Heads(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))


for tag, path in (("ПРОД ", ".argent/models/mmbert-leeth.pt"),
                  ("КАНД ", ".argent/models/mmbert-leeth_candidate.pt")):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    h = Heads(ckpt["hidden"], ckpt["n_classes"])
    h.load_state_dict({k: v for k, v in ckpt["heads"].items() if k.startswith("choice_head")}, strict=False)
    h.eval()
    with torch.no_grad():
        logits = h.choice_head(feats)
    acc = (logits.argmax(-1) == targets).float().mean().item()
    exec_idx = class_to_idx["execute_immediately"]
    m = targets == exec_idx
    exec_acc = (logits.argmax(-1)[m] == exec_idx).float().mean().item() if m.any() else float("nan")
    preds = Counter(idx_to_cls[int(i)] for i in logits.argmax(-1)[m])
    print(f"[{tag}] action acc={acc:.4f} | execute-строк: {int(m.sum())}, из них верно: {exec_acc:.4f}")
    for k, v in preds.most_common(4):
        print(f"     execute -> предсказал {k}: {v}")