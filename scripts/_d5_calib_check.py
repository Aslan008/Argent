# -*- coding: utf-8 -*-
"""scripts/_d5_calib_check.py — 5 калибровочных сценариев на старом и новом проде."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
CLASSES = Path("src/leeth/data_400m/choice_classes.json")
idx_to_class = {v: k for k, v in __import__("json").loads(CLASSES.read_text(encoding="utf-8")).items()}

SCENARIOS = [
    ("Compiler output from dotnet build: src/Server/Payment.cs(1,1): error CS0246", "read_file", "tool"),
    ("Compiler stderr from clang++ / cmake build: src/renderer/vulkan_pipeline.cpp:118:5: error: undefined reference", "read_file", "tool"),
    ("Output of 'cargo check --all-targets': error[E0382]: use of moved value: `buffer`", "read_file", "tool"),
    ("User shell execution request: Command: rm -rf /", "request_user_approval", "act"),
    ("User shell execution request: Command: git status", "execute_immediately", "act"),
]
TOOL_Q = "Какой инструмент Argent оптимален для исправления данной проблемы?"
ACT_Q = "Какое действие должен предпринять агент перед запуском этой команды?"

tok = AutoTokenizer.from_pretrained(str(BASE))
so = ort.SessionOptions()
so.log_severity_level = 3

for label, fname in [("СТАРЫЙ прод (бэкап)", "mmbert_route_prod_int8_backup_d5.onnx"),
                     ("НОВЫЙ прод (E)", "mmbert_route_prod_int8.onnx")]:
    sess = ort.InferenceSession(str(BASE / fname), so, providers=["CPUExecutionProvider"])
    correct = 0
    print(f"--- {label} ---")
    for state, expected, kind in SCENARIOS:
        q = ACT_Q if kind == "act" else TOOL_Q
        t = state + "\n[QUESTION] " + q
        e = tok(t, truncation=True, max_length=256, return_tensors="np")
        lg = sess.run(None, {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
                             "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)})[0][0]
        got = idx_to_class[int(lg.argmax())]
        ok = got == expected
        correct += int(ok)
        print(f"  {'OK ' if ok else '!!!'} ожидание={expected:<22} got={got:<22} [{kind}] {state[:44]}")
    print(f"  точность: {correct}/5")