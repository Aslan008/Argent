# -*- coding: utf-8 -*-
"""scripts/_e5_verify_fp32c.py — inline-прогон fp32 в .py-файле (проверка -c искажений).

Гипотеза: python -c "..." в PowerShell искажал f-string (кавычки/кириллица) —
вход build_state отличался, отсюда 49.5% вместо 67.0%. Прогоняем ТОТ ЖЕ код
из файла.
"""

import json
import sys
import time
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
    ok = 0
    t0 = time.perf_counter()
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TQ
        enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
        logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[0][0]
        ok += int(i2c[int(logits.argmax())] == c["expected_tool"])
    wall = (time.perf_counter() - t0) / len(battery["cases"]) * 1000
    print(f"fp32 ONNX (.py-файл): {ok}/109 = {ok / 109:.1%} | {wall:.0f} мс/кейс")


if __name__ == "__main__":
    main()