# -*- coding: utf-8 -*-
"""scripts/_raw_state_probe.py — Leeth v3-головы на raw state (без конверта).

Полнота сравнения с Laya: её 92.9% dept были на raw state. Наши головы обучены
на конвертированных state — проверяю, как они ведут себя без конверта.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(".argent/models/mmbert-argent")
CKPT = Path(".argent/models/mmbert_dept_urgency.pt")

spec = importlib.util.spec_from_file_location("mdbattery", "scripts/_multidomain_battery.py")
mdb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdb)

DEPT_CLASSES = ["billing", "technical", "sales", "account", "other"]


class DeptHead(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class UrgencyHead(nn.Module):
    def __init__(self, hidden, n_levels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_levels))

    def forward(self, x):
        return self.net(x)


def main():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(device).eval()

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    hidden = ckpt["hidden"]
    dept_head = DeptHead(hidden, len(DEPT_CLASSES))
    dept_head.load_state_dict(ckpt["dept_head"])
    urg_head = UrgencyHead(hidden, ckpt["urgency_levels"])
    urg_head.load_state_dict(ckpt["urgency_head"])
    dept_head.to(device).eval()
    urg_head.to(device).eval()
    print(f"пул={ckpt['pool']} hidden={hidden}")

    dept_cases = [(st, next(g for q, g in qs if q == "department"))
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    urg_cases = [(st, next(g for q, g in qs if q == "urgency"))
                 for _, st, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]

    def feats(texts):
        ms, mxs = [], []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], truncation=True, max_length=1024, padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = base.base_model(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                mean = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                mx = hs.masked_fill(mask == 0, -1e4).max(dim=1).values
                ms.append(mean.cpu())
                mxs.append(torch.cat([mean, mx], dim=-1).cpu())
        return torch.cat(ms), torch.cat(mxs)

    for vname, wrap in (("конверт", mdb.prod_envelope), ("raw", lambda t: t)):
        dept_texts = [wrap(st) + "\n[QUESTION] " + mdb.Q_DEPARTMENT for st, _ in dept_cases]
        urg_texts = [wrap(st) + "\n[QUESTION] " + mdb.Q_URGENCY for st, _ in urg_cases]
        dm, dmm = feats(dept_texts)
        um, umm = feats(urg_texts)
        with torch.no_grad():
            d_pred = dept_head(dmm.to(device)).argmax(-1).cpu()
            u_pred = urg_head(umm.to(device)).argmax(-1).cpu()
        d_ok = sum(DEPT_CLASSES[p] == g for p, (_, g) in zip(d_pred.tolist(), dept_cases))
        u_ok = sum(p + 1 == int(g) for p, (_, g) in zip(u_pred.tolist(), urg_cases))
        print(f"[{vname:>8}] dept {d_ok}/14 ({d_ok / 14:.0%}) | urgency {u_ok}/13 ({u_ok / 13:.0%})")


if __name__ == "__main__":
    main()