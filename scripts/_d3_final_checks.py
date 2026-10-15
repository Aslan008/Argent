# -*- coding: utf-8 -*-
"""scripts/_d3_final_checks.py — чистые замеры: латентность с интерливом + noul legacy.

1. Латентность: dynamic prod vs static qdq_u8, интерлив (A,B,A,B) после
   прогрева каждого — отсекает эффекты нагрузки машины.
2. Noul legacy на val-сплите через РОДНОЙ токенизатор рантайма
   (mmBERT-токенизатор даёт ids вне вокаба legacy — 3560 токенов).
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
VAL_PATH = Path("src/leeth/data_400m/leeth_400m_val.jsonl")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def lat(sess, enc, n=10):
    sess.run(None, enc)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, enc)
        out.append((time.perf_counter() - t0) * 1000)
    out.sort()
    return out[0], statistics.median(out)


def main() -> None:
    so = ort.SessionOptions()
    so.log_severity_level = 3

    # ── 1) латентность с интерливом ─────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))
    encs = []
    for c in battery["cases"][:20]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        encs.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                     "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})

    sess_dyn = ort.InferenceSession(str(BASE_DIR / "mmbert_route_prod_int8.onnx"), so,
                                    providers=["CPUExecutionProvider"])
    sess_st = ort.InferenceSession(str(BASE_DIR / "_tmp_static_qdq_u8.onnx"), so,
                                   providers=["CPUExecutionProvider"])
    for s in (sess_dyn, sess_st):
        for e in encs[:5]:
            s.run(None, e)  # прогрев обоих

    print("=== латентность батч-1 (интерлив dyn/st, warm) ===")
    for i in range(3):
        mn_d, md_d = lat(sess_dyn, encs[i])
        mn_s, md_s = lat(sess_st, encs[i])
        print(f"раунд {i}: dynamic min={mn_d:.0f} med={md_d:.0f} | static min={mn_s:.0f} med={md_s:.0f} мс")

    # ── 2) noul: mmBERT vs legacy на val-сплите, разбивка по вопросу ────────
    import os
    import src.leeth.runtime as rt_mod

    os.environ["LEETH_ROUTER"] = "legacy"
    import importlib
    importlib.reload(rt_mod)
    leeth = rt_mod.leeth
    leeth.reload(force_device="onnx")
    from src.leeth.primitives import NoulQuestion

    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    noul_rows = [r for r in rows if r["question_type"] == "noul"]
    by_q = {}
    for r in noul_rows:
        by_q.setdefault(r["question"], []).append(r)
    print(f"\n=== noul на val-сплите: {len(noul_rows)} строк, {len(by_q)} вопросов ===")

    for q, q_rows in sorted(by_q.items()):
        labels = [float(r["target_noul"]) for r in q_rows]
        print(f"\n--- {q[:60]} ({len(q_rows)} строк, {sum(labels):.0f} положительных) ---")
        # mmBERT: сырая сессия + mmBERT-токенизатор
        probs_m = []
        for r in q_rows:
            t = r["state"] + "\n[QUESTION] " + r["question"]
            enc = tok(t, truncation=True, max_length=256, return_tensors="np")
            feed = {"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                    "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)}
            out = sess_dyn.run(None, feed)[1]
            probs_m.append(1.0 / (1.0 + np.exp(-float(out[0].item()))))
        # legacy: через рантайм (родной токенизатор и формат)
        probs_l = []
        for r in q_rows:
            batch = leeth.decide(r["state"], [NoulQuestion(id="n", question=r["question"])])
            probs_l.append(float(batch.results["n"].probability))
        for name, probs in (("mmBERT ", probs_m), ("legacy ", probs_l)):
            preds = [int(p >= 0.5) for p in probs]
            acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
            brier = sum((p - l) ** 2 for p, l in zip(probs, labels)) / len(labels)
            tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1.0)
            fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0.0)
            fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1.0)
            print(f"  [{name}] acc={acc:.1%} brier={brier:.4f} | TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()