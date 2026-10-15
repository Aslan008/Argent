# -*- coding: utf-8 -*-
"""scripts/_d5_verify_fixed.py — квантование пропатченного fp32 и верификация."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

BASE = Path(".argent/models/mmbert-argent")
FIXED = BASE / "_d5_new_fp32_fixed.onnx"
INT8 = BASE / "_d5_new_int8.onnx"
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

t0 = time.perf_counter()
quantize_dynamic(str(FIXED), str(INT8), weight_type=QuantType.QInt8)
print(f"int8: {INT8.stat().st_size / 1e6:.0f} MB ({time.perf_counter() - t0:.0f}s)")

so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(str(INT8), so, providers=["CPUExecutionProvider"])

battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
cases = battery["cases"]
c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
idx_to_class = {v: k for k, v in c2i.items()}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(BASE))

texts = [f'Incoming user request in Argent session:\n"{c["query"]}"\n'
         f'Working directory: C:/Users/user/Desktop/Argent\n[QUESTION] {TEMPLATE_Q}'
         for c in cases]

# батч-инвариантность
encs = [tok(t, truncation=True, max_length=256, return_tensors="np") for t in texts[:10]]
solo = [sess.run(None, {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
                        "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)})[0][0]
        for e in encs]
pair = tok(texts[:2], truncation=True, max_length=256, padding=True, return_tensors="np")
batch_out = sess.run(None, {"input_ids": np.asarray(pair["input_ids"], dtype=np.int64),
                            "attention_mask": np.asarray(pair["attention_mask"], dtype=np.int64)})[0]
max_diff = max(float(np.max(np.abs(batch_out[i] - solo[i]))) for i in range(2))
agree = sum(int(int(batch_out[i].argmax()) == int(solo[i].argmax())) for i in range(2))
print(f"батч-инвариантность: max|diff|={max_diff:.4f}, argmax {agree}/2 "
      f"{'OK' if max_diff < 0.5 else '!!! БАТЧ-БАГ'}")

# батарея
ok = 0
for t, c in zip(texts, cases):
    enc = tok(t, truncation=True, max_length=256, return_tensors="np")
    logits = sess.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                             "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})[0][0]
    ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
print(f"батарея: {ok}/{len(cases)} = {ok / len(cases):.1%} (прод был 72/109)")