# -*- coding: utf-8 -*-
"""scripts/_d3_margin_calib.py — калибровка UNFAMILIAR margin-гейта.

Цель: отсечь false-dispatch на запросах с НЕИЗВЕСТНЫМ инструментом
(доминирующая ошибка choice-головы: expected=UNFAMILIAR -> best=run_command/read_file).

Гейт: prob(UNFAMILIAR) > conf(best) - margin -> отклонить (state неоднозначен).
Семантика гейтов = текущий диспетчер: need_p >= 0.65 AND conf >= 0.65 AND best in ALLOWED.
Наборы: прод-батарея (109) + val routing-choice (225, held-out).
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
THR = 0.65  # текущий confidence_threshold диспетчера


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
        conf = norm[best]
        unfam = norm.get("UNFAMILIAR", 0.0)

        t2 = state_text + "\n[QUESTION] " + NEED_TOOL_Q
        enc2 = tok(t2, truncation=True, max_length=256, return_tensors="np")
        outs2 = sess.run(None, {"input_ids": np.asarray(enc2["input_ids"], dtype=np.int64),
                                "attention_mask": np.asarray(enc2["attention_mask"], dtype=np.int64)})
        need_p = 1.0 / (1.0 + np.exp(-float(outs2[1][0].item())))
        rows.append({"expected": expected, "best": best, "conf": conf,
                     "unfam": unfam, "need_p": need_p})
    return rows


def sweep(name, rows):
    print(f"\n=== {name}: n={len(rows)} ===")

    # маржины: корректные dispatch vs UNFAMILIAR-expected (best=инструмент)
    ok_rows = [r for r in rows if r["expected"] == r["best"] and r["best"] in ALLOWED_FAST_TOOLS]
    unfam_rows = [r for r in rows if r["expected"] == "UNFAMILIAR" and r["best"] in ALLOWED_FAST_TOOLS]
    for label, grp in (("корректные (expected==best)", ok_rows), ("UNFAMILIAR-expected (best=tool)", unfam_rows)):
        if grp:
            m = sorted(r["conf"] - r["unfam"] for r in grp)
            print(f"{label}: n={len(grp)} margin min={m[0]:.2f} p25={m[len(m)//4]:.2f} "
                  f"median={m[len(m)//2]:.2f} p75={m[3*len(m)//4]:.2f} max={m[-1]:.2f}")

    print(f"\n{'margin':>6} {'dispatch':>8} {'ok':>4} {'false':>6} {'precision':>9}")
    for margin in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        d = []
        for r in rows:
            if r["need_p"] < THR or r["conf"] < THR:
                continue
            if r["best"] not in ALLOWED_FAST_TOOLS:
                continue
            if r["conf"] - r["unfam"] < margin:  # margin-гейт
                continue
            d.append(r)
        ok = sum(1 for r in d if r["best"] == r["expected"])
        false = len(d) - ok
        prec = ok / len(d) if d else 0.0
        print(f"{margin:>6.2f} {len(d):>8} {ok:>4} {false:>6} {prec:>9.1%}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    batt_items = [(build_state(c["query"]), c["expected_tool"]) for c in battery["cases"]]
    sweep("прод-батарея", collect(sess, tok, c2i, batt_items))

    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    val_items = [(r["state"], r["target_choice"]) for r in rows
                 if r["question_type"] == "choice" and r["question"] == TEMPLATE_Q
                 and r["state"].startswith(ROUTING_PREFIX)]
    print(f"\nval routing-choice строк: {len(val_items)}")
    sweep("val routing (held-out)", collect(sess, tok, c2i, val_items))


if __name__ == "__main__":
    main()