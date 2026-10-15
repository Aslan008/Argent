# -*- coding: utf-8 -*-
"""scripts/_e5_verify_fp32.py — контроль: fp32 ONNX vs torch fp32.

Странность: fp32 ONNX давал 67.0% (E4/E5), сейчас 49.5%. Проверяем, не был ли
перезаписан mmbert_route.onnx, и сравниваем с torch fp32.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
BATTERY = Path("scripts/prod_battery.json")
CLASSES = Path("src/leeth/data_400m/choice_classes.json")
TQ = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(q: str) -> str:
    n = q[:1].lower() + q[1:] if q[:1].isalpha() else q
    return f'Incoming user request in Argent session:\n"{n}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    i2c = {v: k for k, v in json.loads(CLASSES.read_text(encoding="utf-8")).items()}
    tok = AutoTokenizer.from_pretrained(str(BASE))

    p = BASE / "mmbert_route.onnx"
    import os
    import datetime
    print("mtime:", datetime.datetime.fromtimestamp(os.path.getmtime(p)), "| size:", os.path.getsize(p) // (1024 * 1024), "MB")

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    ok = 0
    t0 = time.perf_counter()
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TQ
        enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
        logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[0][0]
        ok += int(i2c[int(logits.argmax())] == c["expected_tool"])
    wall = (time.perf_counter() - t0) / len(battery["cases"]) * 1000
    print(f"fp32 ONNX: {ok}/109 = {ok / 109:.1%} | {wall:.0f} мс/кейс")

    base = AutoModelForMaskedLM.from_pretrained(str(BASE), dtype=torch.float32).eval()
    ckpt = torch.load(BASE.parent / "mmbert-leeth.pt", map_location="cpu", weights_only=False)
    hidden, n_classes = ckpt["hidden"], ckpt["n_classes"]
    choice_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
    choice_head.load_state_dict({k[len("choice_head."):]: v for k, v in ckpt["heads"].items()
                                 if k.startswith("choice_head.")})
    ok = 0
    with torch.no_grad():
        for c in battery["cases"][:20]:
            t = build_state(c["query"]) + "\n[QUESTION] " + TQ
            enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
            out = base.base_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hs = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            logits = choice_head(pooled)
            ok += int(i2c[int(logits[0].argmax())] == c["expected_tool"])
    print(f"torch fp32 (первые 20): {ok}/20")


if __name__ == "__main__":
    main()