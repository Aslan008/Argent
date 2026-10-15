# -*- coding: utf-8 -*-
"""scripts/_e5_verify_fp32b.py — поиск причины недетерминизма fp32 ONNX (67% vs 49.5%).

Гипотезы: (1) ORT arena/потоки, (2) токенизатор, (3) порядок прогонов.
Печатаем предсказания первых 10 кейсов + длины токенов.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

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
    sess = ort.InferenceSession(str(BASE / "mmbert_route.onnx"), providers=["CPUExecutionProvider"])
    for c in battery["cases"][:10]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TQ
        enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
        n_tok = enc["input_ids"].shape[1]
        logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[0][0]
        pred = i2c[int(logits.argmax())]
        mark = "OK" if pred == c["expected_tool"] else "MISS"
        print(f"  len={n_tok:3d} {c['query'][:28]!r}: exp={c['expected_tool']}, pred={pred} {mark}")


if __name__ == "__main__":
    main()