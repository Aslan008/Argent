# -*- coding: utf-8 -*-
"""scripts/_d3_false_detail.py — детали false-dispatch кейсов при порогах 0.65.

Печатает все dispatch-кейсы с expected != best: query, expected, best, conf,
prob(UNFAMILIAR), margin, need_p — чтобы понять, есть ли разделяющий сигнал.
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
VAL_PATH = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
DISPATCH_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]
ALLOWED_FAST_TOOLS = {"read_file", "run_command", "grep_search", "find_by_name",
                      "list_directory", "calculate", "search_files"}
ROUTING_PREFIX = "Incoming user request in Argent session:"
THR = 0.65


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def collect(sess, tok, c2i, items):
    rows = []
    for state_text, expected in items:
        t = state_text + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        outs = sess.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                               "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})
        logits = outs[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        cand = {k: float(probs[c2i[k]]) for k in DISPATCH_CHOICES if k in c2i}
        total = sum(cand.values()) or 1.0
        norm = {k: v / total for k, v in cand.items()}
        best = max(norm, key=norm.get)
        t2 = state_text + "\n[QUESTION] " + NEED_TOOL_Q
        enc2 = tok(t2, truncation=True, max_length=256, return_tensors="np")
        outs2 = sess.run(None, {"input_ids": np.asarray(enc2["input_ids"], dtype=np.int64),
                                "attention_mask": np.asarray(enc2["attention_mask"], dtype=np.int64)})
        need_p = 1.0 / (1.0 + np.exp(-float(outs2[1][0].item())))
        rows.append({"expected": expected, "best": best, "conf": norm[best],
                     "unfam": norm.get("UNFAMILIAR", 0.0), "need_p": need_p,
                     "second": sorted(norm.values())[-2]})
    return rows


def show_false(name, rows):
    print(f"\n=== {name}: false-dispatch кейсы (need_p>={THR}, conf>={THR}, best in ALLOWED) ===")
    n_false = 0
    for r in rows:
        if r["need_p"] >= THR and r["conf"] >= THR and r["best"] in ALLOWED_FAST_TOOLS \
                and r["best"] != r["expected"]:
            n_false += 1
            print(f"  expected={r['expected']:14s} best={r['best']:14s} conf={r['conf']:.2f} "
                  f"unfam={r['unfam']:.3f} margin={r['conf'] - r['unfam']:.2f} "
                  f"2nd_prob={r['second']:.2f} need_p={r['need_p']:.2f}")
    if not n_false:
        print("  (нет)")
    # и UNFAMILIAR-expected, НЕ прошедшие conf-гейт (потенциальные будущие false)
    print(f"--- UNFAMILIAR-expected с best=tool, conf<{THR} (не диспетчатся) ---")
    for r in rows:
        if r["expected"] == "UNFAMILIAR" and r["best"] in ALLOWED_FAST_TOOLS and r["conf"] < THR:
            print(f"  best={r['best']:14s} conf={r['conf']:.2f} unfam={r['unfam']:.3f} "
                  f"margin={r['conf'] - r['unfam']:.2f} need_p={r['need_p']:.2f}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    batt_items = [(build_state(c["query"]), c["expected_tool"]) for c in battery["cases"]]
    show_false("прод-батарея", collect(sess, tok, c2i, batt_items))

    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    val_items = [(r["state"], r["target_choice"]) for r in rows
                 if r["question_type"] == "choice" and r["question"] == TEMPLATE_Q
                 and r["state"].startswith(ROUTING_PREFIX)]
    show_false("val routing (held-out)", collect(sess, tok, c2i, val_items))


if __name__ == "__main__":
    main()