# -*- coding: utf-8 -*-
"""scripts/_d5_docker_oldprod.py — docker-запросы на СТАРОМ проде (прецедент?)."""
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
idx_to_class = {v: k for k, v in json.loads(CLASSES.read_text(encoding="utf-8")).items()}
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
CHOICES = ["read_file", "write_file", "replace_in_file", "run_command",
           "grep_search", "find_by_name", "list_directory", "browser_open",
           "browser_click", "browser_type", "browser_upload_file", "calculate",
           "ask_user_questions", "NO_TOOL", "UNFAMILIAR"]

tok = AutoTokenizer.from_pretrained(str(BASE))
so = ort.SessionOptions()
so.log_severity_level = 3

QUERIES = ["запусти docker ps -a", "docker ps", "запусти docker images"]

for label, fname in [("СТАРЫЙ прод (D3, attention int8)", "mmbert_route_prod_int8_backup_d3.onnx"),
                     ("НОВЫЙ прод (D5 v4, E-рецепт)", "mmbert_route_prod_int8.onnx")]:
    sess = ort.InferenceSession(str(BASE / fname), so, providers=["CPUExecutionProvider"])
    print(f"--- {label} ---")
    for q in QUERIES:
        state = f'Incoming user request in Argent session:\n"{q}"\nWorking directory: C:/Users/user/Desktop/Argent'
        t = state + "\n[QUESTION] " + TEMPLATE_Q
        e = tok(t, truncation=True, max_length=256, return_tensors="np")
        lg = sess.run(None, {"input_ids": np.asarray(e["input_ids"], dtype=np.int64),
                             "attention_mask": np.asarray(e["attention_mask"], dtype=np.int64)})[0][0]
        probs = np.exp(lg - lg.max())
        probs /= probs.sum()
        cand = {c: float(probs[json.loads(CLASSES.read_text(encoding="utf-8"))[c]])
                if c in idx_to_class and idx_to_class[c] < len(probs) else 0.001 for c in CHOICES}
        total = sum(cand.values()) or 1.0
        cand = {k: v / total for k, v in cand.items()}
        best = max(cand, key=cand.get)
        print(f"  {q!r:<28} -> {best:<12} conf={cand[best]:.3f}")