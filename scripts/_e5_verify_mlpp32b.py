# -*- coding: utf-8 -*-
"""scripts/_e5_verify_mlpp32b.py — inline-стиль (токенизация на лету) с печатью
первых 3 предсказаний. Диагноз: почему inline даёт 47, а verify (копия final_matrix)
— 72? Единственное отличие, которое осталось: inline токенизирует НА ЛЕТУ внутри
цикла прогона, verify — ЗАРАНЕЕ. Проверяем гипотезу: порядок токенизация/прогон."""

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
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_mlpfp32_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])
    ok = 0
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        feed = {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)}
        logits = sess.run(None, feed)[0][0]
        ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    print(f"итого (на лету): {ok}/109 = {ok / 109:.1%}")
    for c in battery["cases"][:3]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        feed = {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)}
        pred = idx_to_class[int(sess.run(None, feed)[0][0].argmax())]
        print(f"  {c['query'][:25]!r}: exp={c['expected_tool']}, pred={pred}")


if __name__ == "__main__":
    main()