# -*- coding: utf-8 -*-
"""scripts/_battery_iter3_diag.py — по-кейсовый разбор docker_val + батарея кандидата."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

IN = Path(".argent/kaggle/dataset_v2")
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
idx_to_cls = {v: k for k, v in class_to_idx.items()}

ckpt = torch.load(".argent/models/mmbert-leeth_candidate.pt", map_location="cpu", weights_only=False)
hidden, n_classes = ckpt["hidden"], ckpt["n_classes"]


class Heads(torch.nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))


heads = Heads(hidden, n_classes)
heads.load_state_dict({k: v for k, v in ckpt["heads"].items() if k.startswith("choice_head")}, strict=False)
heads.eval()

z = np.load(IN / "feat_docker_val.npz")
Xdv = torch.from_numpy(z["feats"].astype(np.float32))
ydv = torch.from_numpy(z["targets"].astype(np.int64))

print(f"docker_val: фичи {Xdv.shape}")
with torch.no_grad():
    pred = heads.choice_head(Xdv).argmax(1)
bad = 0
for i in range(len(ydv)):
    want = idx_to_cls[int(ydv[i])]
    got = idx_to_cls[int(pred[i])]
    if got != want:
        bad += 1
        print(f"  [{i}] want {want:<18} got {got}")
print(f"docker_val: {len(ydv) - bad}/{len(ydv)}")

# батарея
zb = np.load(IN / "feat_battery.npz")
Xb = torch.from_numpy(zb["feats"].astype(np.float32))
yb = torch.from_numpy(zb["targets"].astype(np.int64))
with torch.no_grad():
    pb = heads.choice_head(Xb).argmax(1)
acc = int((pb == yb).sum())
print(f"батарея (104 таргет-строки): {(pb == yb).sum()}/{len(yb)}")