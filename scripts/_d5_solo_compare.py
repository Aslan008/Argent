# -*- coding: utf-8 -*-
"""scripts/_d5_solo_compare.py — соло-сравнение fixed vs prod + батч-тест."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(".argent/models/mmbert-argent")
so = ort.SessionOptions()
so.log_severity_level = 3
s_fixed = ort.InferenceSession(".argent/models/mmbert-argent/_d5_new_int8.onnx", so, providers=["CPUExecutionProvider"])
s_prod = ort.InferenceSession(".argent/models/mmbert-argent/mmbert_route_prod_int8.onnx", so, providers=["CPUExecutionProvider"])

t = ('Incoming user request in Argent session:\n"прочитай config.py"\n'
     "Working directory: C:/Users/user/Desktop/Argent\n"
     "[QUESTION] Какой инструмент Argent должен быть вызван в первую очередь?")
enc = tok(t, truncation=True, max_length=256, return_tensors="np")
feed = {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)}
o_fix = s_fixed.run(None, feed)[0][0]
o_prod = s_prod.run(None, feed)[0][0]
print("соло max|diff| fixed vs prod:", float(np.abs(o_fix - o_prod).max()))
print("argmax fixed:", int(o_fix.argmax()), "| prod:", int(o_prod.argmax()))

pair = tok([t, t], truncation=True, max_length=256, padding=True, return_tensors="np")
feed2 = {"input_ids": np.asarray(pair["input_ids"], dtype=np.int64),
         "attention_mask": np.asarray(pair["attention_mask"], dtype=np.int64)}
b_fix = s_fixed.run(None, feed2)[0]
b_prod = s_prod.run(None, feed2)[0]
print("батч2 vs соло fixed: max|diff| =", float(np.abs(b_fix[0] - o_fix).max()))
print("батч2 vs соло prod:  max|diff| =", float(np.abs(b_prod[0] - o_prod).max()))