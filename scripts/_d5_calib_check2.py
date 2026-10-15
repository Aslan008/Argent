# -*- coding: utf-8 -*-
"""scripts/_d5_calib_check2.py — 5 сценариев с ОГРАНИЧЕНИЕМ выбора (логика рантайма)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE = Path(".argent/models/mmbert-argent")
CLASSES = Path("src/leeth/data_400m/choice_classes.json")
c2i = json.loads(CLASSES.read_text(encoding="utf-8"))
idx_to_class = {v: k for k, v in c2i.items()}

SCENARIOS = [
    ("Compiler output from dotnet build: src/Server/Payment.cs(1,1): error CS0246", "read_file", "tool"),
    ("Compiler stderr from clang++ / cmake build: src/renderer/vulkan_pipeline.cpp:118:5: error: undefined reference", "read_file", "tool"),
    ("Output of 'cargo check --all-targets': error[E0382]: use of moved value: `buffer`", "read_file", "tool"),
    ("User shell execution request: Command: rm -rf /", "request_user_approval", "act"),
    ("User shell execution request: Command: git status", "execute_immediately", "act"),
]
TOOL_Q = "Какой инструмент Argent оптимален для исправления данной проблемы?"
TOOL_CHOICES = ["read_file", "replace_in_file", "run_command", "search_files", "execute_immediately", "block_command"]
ACT_Q = "Какое действие должен предпринять агент перед запуском этой команды?"
ACT_CHOICES = ["execute_immediately", "request_user_approval", "block_command", "sanitize_arguments"]

tok = AutoTokenizer.from_pretrained(str(BASE))
so = ort.SessionOptions()
so.log_severity_level = 3


def restricted_choice(logits, choices):
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    cand = {c: float(probs[c2i[c]]) if c in c2i and c2i[c] < len(probs) else 0.001 for c in choices}
    total = sum(cand.values()) or 1.0
    cand = {k: v / total for k, v in cand.items()}
    return max(cand, key=cand.get), cand


for label, fname in [("СТАРЫЙ прод (бэкап)", "mmbert_route_prod_int8_backup_d5.onnx"),
                     ("НОВЫЙ прод (E)", "mmbert_route_prod_int8.onnx")]:
    sess = ort.InferenceSession(str(BASE / fname), so, providers=["CPUExecutionProvider"])
    correct = 0
    print(f"--- {label} ---")
    for state, expected, kind in SCENARIOS:
        choices = ACT_CHOICES if kind == "act" else TOOL_CHOICES
        q = ACT_Q if kind == "act" else TOOL_Q
        t = state + "\n[QUESTION] " + q
        e = tok(t, truncation=True, max_length=256, return_tensors="np")
        lg = sess.run(None, {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
                             "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)})[0][0]
        got, cand = restricted_choice(lg, choices)
        ok = got == expected
        correct += int(ok)
        conf = cand[got]
        print(f"  {'OK ' if ok else '!!!'} ожидание={expected:<22} got={got:<22} conf={conf:.3f}")
    print(f"  точность: {correct}/5")