# -*- coding: utf-8 -*-
"""scripts/_cloud_verify.py — сверка облачного чекпоинта с локальным cand10."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

loc = torch.load(".argent/models/mmbert-leeth_candidate.pt", map_location="cpu", weights_only=False)
cld = torch.load(".argent/kaggle/cloud_out/mmbert-leeth_candidate.pt", map_location="cpu", weights_only=False)

max_diff, n = 0.0, 0
for k in loc["heads"]:
    a, b = loc["heads"][k].float(), cld["heads"][k].float()
    d = (a - b).abs().max().item()
    max_diff = max(max_diff, d)
    n += 1
print(f"тензорных голов: {n}, max |локальный - облачный| = {max_diff:.2e}")

# функциональная сверка на фичах
class Heads(torch.nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))


IN = Path(".argent/kaggle/dataset_v2")
for tag, ck in (("локальный", loc), ("облачный", cld)):
    h = Heads(ckpt_hidden := ck["hidden"], ck["n_classes"])
    h.load_state_dict({k: v for k, v in ck["heads"].items() if k.startswith("choice_head")}, strict=False)
    h.eval()
    with torch.no_grad():
        zb = np.load(IN / "feat_battery.npz")
        pb = h.choice_head(torch.from_numpy(zb["feats"].astype(np.float32))).argmax(1).numpy()
        zd = np.load(IN / "feat_docker_val.npz")
        pd = h.choice_head(torch.from_numpy(zd["feats"].astype(np.float32))).argmax(1).numpy()
        yb = zb["targets"]
        yd = zd["targets"]
    print(f"{tag}: battery-строки {int((pb == yb).sum())}/{len(yb)} | docker_val {int((pd == yd).sum())}/{len(yd)}")