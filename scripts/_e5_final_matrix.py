# -*- coding: utf-8 -*-
"""scripts/_e5_final_matrix.py — сводная таблица всех INT8-вариантов.

Прогоняет батарею через каждый int8-артефакт на диске и печатает таблицу.
Каждая сессия ORT создаётся/закрывается отдельно (экономия памяти).
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    encs = []
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        encs.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                     "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})

    files = sorted(BASE_DIR.glob("*int8*.onnx")) + [BASE_DIR / "mmbert_route.onnx"]
    print(f"{'файл':45s} {'acc':>7s} {'мс/кейс':>8s} {'MB':>5s}")
    for f in files:
        if not f.exists():
            continue
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
            t0 = time.perf_counter()
            ok = 0
            for c, enc in zip(battery["cases"], encs):
                logits = sess.run(None, enc)[0][0]
                ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
            wall = time.perf_counter() - t0
            mb = os.path.getsize(f) // (1024 * 1024)
            print(f"{f.name:45s} {ok / 109:>7.1%} {wall / len(battery['cases']) * 1000:>7.0f} {mb:>5d}")
            del sess
        except Exception as e:
            print(f"{f.name:45s} FAIL: {str(e)[:60]}")


if __name__ == "__main__":
    import numpy as np
    main()