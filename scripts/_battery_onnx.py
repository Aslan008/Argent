# -*- coding: utf-8 -*-
"""scripts/_battery_onnx.py — батарея 109 кейсов на произвольном fp32 ONNX (батч-1, CPU)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

model = sys.argv[1]
tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
idx_to_cls = {v: k for k, v in class_to_idx.items()}
cases = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))["cases"]

ok = 0
misses = []
for c in cases:
    t = f'Incoming user request in Argent session:\n"{c["query"]}"\nWorking directory: C:/Users/user/Desktop/Argent\n[QUESTION] {TEMPLATE_Q}'
    e = tok(t, truncation=True, max_length=256, return_tensors="np")
    outs = sess.run(None, {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
                           "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)})
    pred = idx_to_cls[int(outs[0][0].argmax())]
    ok += pred == c["expected_tool"]
    if pred != c["expected_tool"] and len(misses) < 6:
        misses.append(f"  {c['query'][:48]:<50} want {c['expected_tool']:<16} got {pred}")
print(f"{Path(model).name}: {ok}/109 = {ok/109:.1%}")
for m in misses:
    print(m)