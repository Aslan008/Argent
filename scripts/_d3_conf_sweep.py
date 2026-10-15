# -*- coding: utf-8 -*-
"""scripts/_d3_conf_sweep.py — свип порога conf для гейта диспетчера.

Для каждого кейса батареи: choice-пробы mmBERT (соло) + need_tool (mmBERT).
Строим precision/recall по порогу conf: сколько из отклонённых low_tool_confidence
случаев choice-голова угадывает (argmax == expected_tool).
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
NEED_TOOL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
DISPATCH_CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]
ALLOWED_FAST_TOOLS = {"read_file", "run_command", "grep_search", "find_by_name",
                      "list_directory", "calculate", "search_files"}


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])

    rows = []
    for c in battery["cases"]:
        state = build_state(c["query"])
        t = state + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        outs = sess.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                               "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})
        logits = outs[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        tool_probs = {k: float(probs[v]) for k, v in c2i.items() if k in DISPATCH_CHOICES}
        t2 = state + "\n[QUESTION] " + NEED_TOOL_Q
        enc2 = tok(t2, truncation=True, max_length=256, return_tensors="np")
        outs2 = sess.run(None, {"input_ids": np.asarray(enc2["input_ids"], dtype=np.int64),
                                "attention_mask": np.asarray(enc2["attention_mask"], dtype=np.int64)})
        need_p = 1.0 / (1.0 + np.exp(-float(outs2[1][0].item())))

        # нормировка как у диспетчера: softmax по ВСЕМ 15 choices
        best = max(tool_probs, key=lambda k: tool_probs[k])
        conf = tool_probs[best]  # softmax уже нормирован по всем choices
        rows.append({
            "query": c["query"], "expected": c["expected_tool"],
            "need_p": need_p, "best": best, "conf": conf,
            "correct": int(best == c["expected_tool"]),
        })

    # базовое разбиение при пороге 0.80
    disp = [r for r in rows if r["need_p"] >= 0.5 and r["conf"] >= 0.80 and r["best"] in ALLOWED_FAST_TOOLS]
    low = [r for r in rows if r["need_p"] >= 0.5 and r["conf"] < 0.80]
    print(f"порог 0.80: dispatch={len(disp)} (ok={sum(r['correct'] for r in disp)}), "
          f"low_conf={len(low)} (из них угадано top-1: {sum(r['correct'] for r in low)})")

    print("\n=== свип порога conf (need_p>=0.80 — как у диспетчера, best in allowed) ===")
    print(f"{'порог':>6} {'dispatch':>8} {'ok':>4} {'false':>6} {'precision':>9} {'покрытие':>8}")
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        d = [r for r in rows if r["need_p"] >= 0.80 and r["conf"] >= thr and r["best"] in ALLOWED_FAST_TOOLS]
        ok = sum(r["correct"] for r in d)
        false = len(d) - ok
        prec = ok / len(d) if d else 0.0
        print(f"{thr:>6.2f} {len(d):>8} {ok:>4} {false:>6} {ok / 109:>9.1%} {len(d) / 109:>8.1%}")

    # разброс low_conf: где застревают
    print("\n=== low_conf (0.5<=conf<0.8): expected vs best ===")
    from collections import Counter
    pairs = Counter((r["expected"], r["best"]) for r in low)
    for (exp, best), n in pairs.most_common(12):
        mark = "OK " if exp == best else "BAD"
        print(f"  [{mark}] expected={exp:16s} best={best:16s} x{n}")


if __name__ == "__main__":
    main()