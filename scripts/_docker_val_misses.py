# -*- coding: utf-8 -*-
"""scripts/_docker_val_misses.py — какие docker-val строки проваливаются и почему."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn

HEADS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".argent/models/mmbert-leeth_candidate.pt")
FEATS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".argent/kaggle/dataset_v2/feat_docker_val.npz")
ROWS = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".argent/kaggle/dataset_v2/docker_val.jsonl")

ckpt = torch.load(HEADS, map_location="cpu", weights_only=False)
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
idx_to_cls = {v: k for k, v in class_to_idx.items()}


class ChoiceOnly(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))


z = np.load(FEATS)
feats = torch.from_numpy(z["feats"].astype(np.float32))
targets = z["targets"]
h = ChoiceOnly(ckpt["hidden"], ckpt["n_classes"])
h.load_state_dict({k: v for k, v in ckpt["heads"].items() if k.startswith("choice_head")}, strict=False)
h.eval()
with torch.no_grad():
    logits = h.choice_head(feats)
    probs = torch.softmax(logits, -1)
rows = [json.loads(l) for l in open(ROWS, encoding="utf-8")]
ok = 0
for i, r in enumerate(rows):
    q = r["state"].split("\n")[1].strip('"')
    pred = idx_to_cls[int(logits[i].argmax())]
    want = idx_to_cls[int(targets[i])]
    conf = float(probs[i].max())
    ok += pred == want
    mark = "OK " if pred == want else "!!!"
    print(f"  {mark} {q:<46} -> {pred:<12} conf={conf:.3f} (want {want})")
print(f"итог: {ok}/{len(rows)}")