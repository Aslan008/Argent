# -*- coding: utf-8 -*-
"""scripts/_e5_verify_prod.py — ФИНАЛЬНАЯ верификация прод-артефакта
mmbert_route_prod_int8.onnx (.py-файл, без -c искажений)."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
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
    idx_to_class = {v: k for k, v in json.loads(CLASSES_PATH.read_text(encoding="utf-8")).items()}
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])
    ok = 0
    t0 = time.perf_counter()
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
        feed = {"input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy()}
        ok += int(idx_to_class[int(sess.run(None, feed)[0][0].argmax())] == c["expected_tool"])
    wall = (time.perf_counter() - t0) / len(battery["cases"]) * 1000
    print(f"ПРОД-АРТЕФАКТ mmbert_route_prod_int8.onnx: {ok}/109 = {ok / 109:.1%} | {wall:.0f} мс/кейс")


if __name__ == "__main__":
    main()