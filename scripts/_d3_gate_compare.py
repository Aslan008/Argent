# -*- coding: utf-8 -*-
"""scripts/_d3_gate_compare.py — сравнение legacy vs mmBERT на уровне гейтов диспетчера.

Симулирует точную логику гейтов LeethDispatcher (need_tool >= 0.80, tool conf >= 0.80
с candidate-restriction, UNFAMILIAR -> delegate, ALLOWED_FAST_TOOLS) для ОБОИХ
моделей на прод-батарее (109 кейсов) + на тестовых запросах диспетчера.

Метрики на модель:
- dispatch: гейты пройдены (инструмент разрешён)
- dispatch_ok: dispatch и choice == expected_tool
- false_dispatch: dispatch и choice != expected_tool (ОПАСНО)
- missed_dispatch: не dispatch, а expected ∈ ALLOWED_FAST_TOOLS (потеря скорости)
- delegate_ok: не dispatch и expected ∈ {UNFAMILIAR, NO_TOOL} (правильно делегировали)
- wrong_delegate: не dispatch, а expected — обычный инструмент
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(".argent/models/mmbert-argent")
LEGACY_DIR = Path("src/leeth/data_400m")
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
                      "search_files", "list_directory", "calculate"}
CONF_THRESHOLD = 0.80


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


class LegacyBackend:
    """Старая модель: leeth_int8.onnx + LeethTokenizer, формат [STATE]..[SEP]..[TYPE]."""

    def __init__(self):
        from src.leeth.tokenizer import LeethTokenizer
        self.tok = LeethTokenizer.load(LEGACY_DIR)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(LEGACY_DIR / "leeth_int8.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        with open(LEGACY_DIR / "choice_classes.json", encoding="utf-8") as f:
            self.c2i = json.load(f)
        self.i2c = {v: k for k, v in self.c2i.items()}

    def _encode(self, state: str, question: str, qtype: str, max_len: int = 512):
        text = f"[STATE] {state} [SEP] [QUESTION] [{qtype.upper()}] {question}"
        ids = self.tok_encode(text, max_len)
        return ids

    def tok_encode(self, text: str, max_len: int):
        return self.tok.encode(text, max_length=max_len, padding=False,
                               truncation=True, add_special_tokens=True)

    def run_pair(self, state: str):
        texts = [
            f"[STATE] {state} [SEP] [QUESTION] [NOUL] {NEED_TOOL_Q}",
            f"[STATE] {state} [SEP] [QUESTION] [CHOICE] {TEMPLATE_Q}",
        ]
        ids = [self.tok_encode(t, 512) for t in texts]
        max_len = max(len(x) for x in ids)
        pad_id = self.tok.pad_token_id
        input_ids = np.array([x + [pad_id] * (max_len - len(x)) for x in ids], dtype=np.int64)
        mask = np.array([[1 if t != pad_id else 0 for t in row] for row in input_ids.tolist()], dtype=np.int64)
        outs = self.sess.run(None, {"input_ids": input_ids, "attention_mask": mask})
        noul_prob = float(outs[4][0].item())
        choice_logits = outs[2][0]
        probs = np.exp(choice_logits - choice_logits.max())
        probs = probs / probs.sum()
        return noul_prob, {c: float(probs[self.c2i[c]]) for c in DISPATCH_CHOICES if c in self.c2i}


class MmBERTBackend:
    """Новая модель: mmbert_route_prod_int8.onnx + HF-токенизатор, формат state+\\n[QUESTION]."""

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = __import__("transformers").AutoTokenizer.from_pretrained(str(BASE_DIR))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        with open(CLASSES_PATH, encoding="utf-8") as f:
            self.c2i = json.load(f)
        self.i2c = {v: k for k, v in self.c2i.items()}

    def run_pair(self, state: str):
        # Рантайм-режим: каждый вопрос СОЛО (батч-1) — динамические scale INT8
        # зависят от min/max батча, смешанный батч меняет предсказания
        # (см. _d3_probe_batch.py: 13/60 флипов). Батч-1 == верифицированное
        # поведение прод-батареи (66.1%).
        texts = [
            state + "\n[QUESTION] " + NEED_TOOL_Q,
            state + "\n[QUESTION] " + TEMPLATE_Q,
        ]
        noul_prob = None
        choice_probs = None
        for t in texts:
            enc = self.tok(t, truncation=True, max_length=256, return_tensors="np")
            feed = {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                    "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)}
            outs = self.sess.run(None, feed)
            if noul_prob is None:
                noul_prob = 1.0 / (1.0 + np.exp(-float(outs[1][0].item())))
            else:
                logits = outs[0][0]
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                choice_probs = {c: float(probs[self.c2i[c]]) for c in DISPATCH_CHOICES if c in self.c2i}
        return noul_prob, choice_probs


def gate_decision(need_tool_prob: float, choice_probs: dict):
    """Точная реплика гейтов диспетчера (без extract-головы)."""
    if need_tool_prob < CONF_THRESHOLD:
        return ("need_tool_false", None, need_tool_prob)
    total = sum(choice_probs.values()) or 1.0
    norm = {k: v / total for k, v in choice_probs.items()}
    best = max(norm, key=norm.get)
    conf = norm[best]
    if best == "UNFAMILIAR":
        return ("unfamiliar", best, conf)
    if conf < CONF_THRESHOLD:
        return ("low_tool_confidence", best, conf)
    if best not in ALLOWED_FAST_TOOLS:
        return ("tool_not_allowed", best, conf)
    return ("dispatch", best, conf)


def evaluate(backend, cases):
    stats = {"dispatch": 0, "dispatch_ok": 0, "false_dispatch": 0,
             "delegate_ok": 0, "wrong_delegate": 0, "need_tool_false": 0,
             "unfamiliar": 0, "low_conf": 0, "tool_not_allowed": 0}
    per_tool = {}
    t0 = time.perf_counter()
    for c in cases:
        state = build_state(c["query"])
        noul_p, choice_probs = backend.run_pair(state)
        reason, tool, conf = gate_decision(noul_p, choice_probs)
        stats[reason] = stats.get(reason, 0) + 1
        exp = c["expected_tool"]
        if reason == "dispatch":
            ok = tool == exp
            stats["dispatch_ok"] += int(ok)
            if not ok:
                stats["false_dispatch"] += 1
            b = per_tool.setdefault(exp, {"n": 0, "ok": 0, "false": 0})
            b["n"] += 1
            b["ok"] += int(ok)
            b["false"] += int(not ok)
        elif exp in ("UNFAMILIAR", "NO_TOOL"):
            stats["delegate_ok"] += 1
        else:
            stats["wrong_delegate"] += 1
            b = per_tool.setdefault(exp, {"n": 0, "ok": 0, "false": 0})
            b["n"] += 1
    wall = time.perf_counter() - t0
    n = len(cases)
    print(f"\n=== {type(backend).__name__} === ({wall / n * 1000:.0f} мс/кейс, 2 форварда)")
    print(f"dispatch: {stats['dispatch']}/{n} (ok {stats['dispatch_ok']}, FALSE {stats['false_dispatch']})")
    print(f"делегирование: правильное {stats['delegate_ok']}, ошибочное {stats['wrong_delegate']}")
    print(f"причины отказа: need_tool_false={stats['need_tool_false']}, unfamiliar={stats['unfamiliar']}, "
          f"low_conf={stats['low_conf']}, not_allowed={stats['tool_not_allowed']}")
    return stats, per_tool


def label(backend):
    return type(backend).__name__


def main() -> None:
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]

    legacy = LegacyBackend()
    stats_l, per_tool_l = evaluate(legacy, cases)

    mmbert = MmBERTBackend()
    stats_m, per_tool_m = evaluate(mmbert, cases)

    print("\n=== Сравнение по инструментам (dispatch_ok/false из n) ===")
    tools = sorted(set(per_tool_l) | set(per_tool_m), key=lambda t: -(per_tool_l.get(t, per_tool_m.get(t)) or {}).get("n", 0))
    print(f"{'tool':22s} {'legacy':>16s} | {'mmBERT':>16s}")
    for t in tools:
        bl = per_tool_l.get(t, {"n": 0, "ok": 0, "false": 0})
        bm = per_tool_m.get(t, {"n": 0, "ok": 0, "false": 0})
        print(f"{t:22s} {bl['ok']:>3}/{bl['n']:<3} F{bl['false']:<3} | {bm['ok']:>3}/{bm['n']:<3} F{bm['false']:<3}")


if __name__ == "__main__":
    main()