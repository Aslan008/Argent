# -*- coding: utf-8 -*-
"""scripts/_d3_threshold_calib.py — калибровка порога conf на батарее + val-сплите.

Семантика 1-в-1 с рантаймом/диспетчером:
- probs = softmax по ВСЕМ классам модели -> выборка 15 DISPATCH_CHOICES ->
  РЕНОРМАЛИЗАЦИЯ по 15 (как ChoiceResult в runtime.py) -> best/conf;
- гейт: need_p >= 0.80 (mmBERT-noul) AND conf >= thr AND best in ALLOWED
  (NO_TOOL/UNFAMILIAR отклоняются отдельными ветками).

Кривые: прод-батарея (109) + val-сплит routing-состояний (held-out).
Порог выбирается устойчивым на ОБОИХ наборах.
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


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def gate_row(need_p, conf, best, expected):
    """Точная семантика гейтов диспетчера. Возврат: (verdict, ok)."""
    if need_p < 0.80:
        return "need_tool_false", None
    if best in ("NO_TOOL", "UNFAMILIAR"):
        return ("unfamiliar" if best == "UNFAMILIAR" else "no_tool_choice"), None
    if conf < THR[0]:
        return "low_conf", None
    if best not in ALLOWED_FAST_TOOLS:
        return "tool_not_allowed", None
    return "dispatch", int(best == expected)


THR = [0.80]  # placeholder, перезаписывается в sweep


def collect(sess, tok, c2i, items):
    """items: list of (state_text, expected_tool). Возврат rows с conf/best/need_p."""
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

        t2 = state_text + "\n[QUESTION] " + NEED_TOOL_Q
        enc2 = tok(t2, truncation=True, max_length=256, return_tensors="np")
        outs2 = sess.run(None, {"input_ids": np.asarray(enc2["input_ids"], dtype=np.int64),
                                "attention_mask": np.asarray(enc2["attention_mask"], dtype=np.int64)})
        need_p = 1.0 / (1.0 + np.exp(-float(outs2[1][0].item())))
        rows.append({"expected": expected, "best": best, "conf": conf, "need_p": need_p})
    return rows


def sweep(name, rows):
    print(f"\n=== {name}: n={len(rows)} ===")
    print(f"{'порог':>6} {'dispatch':>8} {'ok':>4} {'false':>6} {'precision':>9}")
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        THR[0] = thr
        d = [gate_row(r["need_p"], r["conf"], r["best"], r["expected"]) for r in rows]
        disp = [(v, ok) for v, ok in d if v == "dispatch"]
        ok = sum(o for _, o in disp)
        false = len(disp) - ok
        prec = ok / len(disp) if disp else 0.0
        print(f"{thr:>6.2f} {len(disp):>8} {ok:>4} {false:>6} {prec:>9.1%}")
    # где застревают low_conf при 0.80
    THR[0] = 0.80
    low = [r for r in rows if r["need_p"] >= 0.80 and r["conf"] < 0.80
           and r["best"] not in ("NO_TOOL", "UNFAMILIAR") and r["best"] in ALLOWED_FAST_TOOLS]
    ok_low = sum(1 for r in low if r["best"] == r["expected"])
    print(f"low_conf при 0.80: {len(low)}, из них top-1 верен: {ok_low} ({ok_low / max(len(low), 1):.0%})")
    from collections import Counter
    bad = Counter((r["expected"], r["best"]) for r in low if r["best"] != r["expected"])
    for (exp, best), n in bad.most_common(5):
        print(f"  [BAD] expected={exp:16s} best={best:16s} x{n}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])

    # 1) прод-батарея
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    batt_items = [(build_state(c["query"]), c["expected_tool"]) for c in battery["cases"]]
    sweep("прод-батарея", collect(sess, tok, c2i, batt_items))

    # 2) val-сплит: choice-вопрос про инструмент на routing-состояниях
    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    val_items = [(r["state"], r["target_choice"]) for r in rows
                 if r["question_type"] == "choice" and r["question"] == TEMPLATE_Q
                 and r["state"].startswith(ROUTING_PREFIX)]
    print(f"\nval routing-choice строк: {len(val_items)}")
    sweep("val routing (held-out)", collect(sess, tok, c2i, val_items))


if __name__ == "__main__":
    main()