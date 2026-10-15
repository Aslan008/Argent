# -*- coding: utf-8 -*-
"""scripts/_d3_hybrid_gates.py — гейты диспетчера на батарее для гибридного роутинга.

Сравнивает 3 конфигурации noul-роутинга (choice всегда mmBERT):
1. need_tool → legacy (текущий прод)
2. need_tool → mmBERT (новый: 99.5% val acc vs 88.1%)
Метрики: dispatch count / ok / false, причины отказов.
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


def gate_decision(need_tool_p, tool_probs, allowed):
    if need_tool_p < 0.50:
        return "need_tool_false", None, 0.0
    if tool_probs is None:
        return "no_choice", None, 0.0
    cand = [c for c in allowed if c in tool_probs]
    if not cand:
        return "no_candidates", None, 0.0
    best = max(cand, key=lambda c: tool_probs[c])
    total = sum(tool_probs[c] for c in cand)
    if total <= 0:
        return "no_candidates", None, 0.0
    conf = tool_probs[best] / total
    if best in ("NO_TOOL", "UNFAMILIAR"):
        return ("unfamiliar" if best == "UNFAMILIAR" else "no_tool_choice"), None, conf
    if conf < 0.80:
        return "low_tool_confidence", None, conf
    if best not in ALLOWED_FAST_TOOLS:
        return "tool_not_allowed", best, conf
    return "dispatch", best, conf


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    c2i = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess_m = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                  providers=["CPUExecutionProvider"])
    sess_l = ort.InferenceSession(str(Path("src/leeth/data_400m/leeth_int8.onnx")), so,
                                  providers=["CPUExecutionProvider"])
    ltok_mod = None

    # legacy noul через рантайм (родной токенизатор/формат)
    import importlib
    import src.leeth.runtime as rt_mod
    os.environ["LEETH_ROUTER"] = "legacy"
    importlib.reload(rt_mod)
    leeth = rt_mod.leeth
    leeth.reload(force_device="onnx")
    from src.leeth.primitives import NoulQuestion

    stats = {}
    for c in battery["cases"]:
        state = build_state(c["query"])
        # mmBERT: choice (соло, батч-1)
        t = state + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        outs = sess_m.run(None, {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                                 "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})
        logits = outs[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        tool_probs = {k: float(probs[v]) for k, v in c2i.items() if k in DISPATCH_CHOICES}
        # need_tool: ДВА плеча — legacy (текущий прод) и mmBERT (кандидат)
        t2 = state + "\n[QUESTION] " + NEED_TOOL_Q
        enc2 = tok(t2, truncation=True, max_length=256, return_tensors="np")
        outs2 = sess_m.run(None, {"input_ids": np.asarray(enc2["input_ids"], dtype=np.int64),
                                  "attention_mask": np.asarray(enc2["attention_mask"], dtype=np.int64)})
        need_p_mm = 1.0 / (1.0 + np.exp(-float(outs2[1][0].item())))
        batch_l = leeth.decide(state, [NoulQuestion(id="n", question=NEED_TOOL_Q)])
        need_p_lg = float(batch_l.results["n"].probability)

        for tag, need_p in (("legacy_noul", need_p_lg), ("mmbert_noul", need_p_mm)):
            reason, tool, conf = gate_decision(need_p, tool_probs, DISPATCH_CHOICES)
            tag_stats = stats.setdefault(tag, {})
            tag_stats[reason] = tag_stats.get(reason, 0) + 1
            if reason == "dispatch":
                ok = int(tool == c["expected_tool"])
                key = f"dispatch_{'ok' if ok else 'false'}"
                tag_stats[key] = tag_stats.get(key, 0) + 1
                if not ok:
                    tag_stats[f"false_{c['expected_tool']}"] = tag_stats.get(f"false_{c['expected_tool']}", 0) + 1
    for tag in ("legacy_noul", "mmbert_noul"):
        print(f"=== гибрид: choice mmBERT + need_tool {tag} ===")
        for k, v in sorted(stats[tag].items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import os
    main()