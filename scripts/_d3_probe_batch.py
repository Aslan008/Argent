# -*- coding: utf-8 -*-
"""scripts/_d3_probe_batch.py — почему INT8 не батч-инвариантен?

Проверки:
1. pad_token_id токенизатора (мой тест паддил нулём — вдруг это <s>).
2. Батч из 2 ИДЕНТИЧНЫХ текстов vs соло-прогон: если инвариантно —
   диапазон активаций тот же, значит дело в динамических scale.
3. Батч из 2 РАЗНЫХ текстов (реальный pad_token_id) vs соло: флипы argmax.
"""
import json
import sys
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
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    print(f"pad_token_id={tok.pad_token_id!r} bos={tok.bos_token_id!r} cls={tok.cls_token_id!r}")

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in json.loads(CLASSES_PATH.read_text(encoding="utf-8")).items()}
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "_tmp_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])

    encs = []
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        encs.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                     "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    def solo(i):
        return sess.run(None, encs[i])[0][0]

    # 2) идентичный батч
    i = 0
    ids = encs[i]["input_ids"]
    mask = encs[i]["attention_mask"]
    out2 = sess.run(None, {"input_ids": np.concatenate([ids, ids], axis=0),
                           "attention_mask": np.concatenate([mask, mask], axis=0)})[0]
    print(f"[идентичный батч] max|diff| = {float(np.max(np.abs(out2[0] - solo(i)))):.2e}")

    # 3) разные тексты, реальный pad
    rng = np.random.default_rng(0)
    flips = 0
    checked = 0
    max_diff = 0.0
    for _ in range(30):
        a, b = rng.integers(0, len(encs), 2)
        la, lb = encs[a]["input_ids"].shape[1], encs[b]["input_ids"].shape[1]
        L = max(la, lb)
        ia = np.full((1, L), pad_id, dtype=np.int64); ia[0, :la] = encs[a]["input_ids"][0]
        ib = np.full((1, L), pad_id, dtype=np.int64); ib[0, :lb] = encs[b]["input_ids"][0]
        ma = np.zeros((1, L), dtype=np.int64); ma[0, :la] = 1
        mb = np.zeros((1, L), dtype=np.int64); mb[0, :lb] = 1
        out = sess.run(None, {"input_ids": np.concatenate([ia, ib], axis=0),
                              "attention_mask": np.concatenate([ma, mb], axis=0)})[0]
        sa, sb = solo(a), solo(b)
        max_diff = max(max_diff, float(np.max(np.abs(out[0] - sa))), float(np.max(np.abs(out[1] - sb))))
        flips += int(int(out[0].argmax()) != int(sa.argmax())) + int(int(out[1].argmax()) != int(sb.argmax()))
    print(f"[разный батч, pad={pad_id}] max|diff|={max_diff:.2e}, флипов argmax {flips}/60")


if __name__ == "__main__":
    main()