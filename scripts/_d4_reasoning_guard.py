# -*- coding: utf-8 -*-
"""scripts/_d4_reasoning_guard.py — измерение эффекта reasoning-гейта (оффлайн).

Гипотеза: составные/рассуждающие запросы («составь план», «напиши тесты»,
«запусти тесты и исправь падения») не должны fast-dispatch'иться вообще —
они требуют LLM. Явный guard в диспетчере надёжнее финетюна (2 попытки
финетюна обрушили батарею: 66% -> 33/36%).

Измеряем на батарее + val routing: guard-паттерны -> сколько false отсечено,
сколько корректных dispatch потеряно.
"""
import json
import re
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

# reasoning-паттерны: запрос требует рассуждения/нескольких шагов, не одного инструмента
REASONING_RE = re.compile(
    r"^(составь|спроектируй|проведи|сделай ревью|проанализируй|исследуй|сравни|"
    r"оптимизируй|улучши|ускорь|задокументируй|напиши тесты|добавь тесты|"
    r"напиши документацию|разберись|найди причину|почему|подготовь roadmap|"
    r"write tests|add tests|review|analyze|research|compare|optimize|refactor|document)\b",
    re.IGNORECASE)
# составные: инструментальный глагол + « и » + второй глагол
COMPOUND_RE = re.compile(
    r"^(прочитай|открой|запусти|проверь|покажи|найди|посмотри|выполни)\b.+\bи\s+"
    r"(исправ|удали|обнови|найди|перепиши|почини|закрой|составь)",
    re.IGNORECASE)


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                providers=["CPUExecutionProvider"])

    def run(items, label):
        rows = []
        for state_text, expected, query in items:
            t1 = state_text + "\n[QUESTION] " + TEMPLATE_Q
            enc = tok(t1, truncation=True, max_length=256, return_tensors="np")
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
            rows.append({"query": query, "expected": expected, "best": best,
                         "conf": norm[best], "need_p": need_p})

        for guard_on in (False, True):
            disp = []
            for r in rows:
                if r["need_p"] < THR or r["conf"] < THR or r["best"] not in ALLOWED_FAST_TOOLS:
                    continue
                if guard_on and (REASONING_RE.search(r["query"]) or COMPOUND_RE.search(r["query"])):
                    continue
                disp.append(r)
            ok = sum(1 for r in disp if r["best"] == r["expected"])
            false = len(disp) - ok
            print(f"  guard={'ON ' if guard_on else 'OFF'}: dispatch={len(disp)} ok={ok} false={false} "
                  f"precision={ok / max(1, len(disp)):.1%}")
            if guard_on:
                lost = [r for r in rows
                        if r["need_p"] >= THR and r["conf"] >= THR and r["best"] in ALLOWED_FAST_TOOLS
                        and r["best"] == r["expected"]
                        and (REASONING_RE.search(r["query"]) or COMPOUND_RE.search(r["query"]))]
                print(f"    потеряно корректных dispatch из-за guard: {len(lost)}")
                for r in lost:
                    print(f"      - {r['query'][:60]!r} -> {r['best']}")

    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    batt = [(build_state(c["query"]), c["expected_tool"], c["query"]) for c in battery["cases"]]
    print(f"=== прод-батарея (n={len(batt)}) ===")
    run(batt, "батарея")

    rows_v = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    val_items = []
    for r in rows_v:
        if r["question_type"] == "choice" and r["question"] == TEMPLATE_Q and r["state"].startswith(ROUTING_PREFIX):
            m = re.search(r'"(.+)"', r["state"])
            val_items.append((r["state"], r["target_choice"], m.group(1) if m else ""))
    print(f"\n=== val routing (n={len(val_items)}) ===")
    run(val_items, "val")


if __name__ == "__main__":
    main()