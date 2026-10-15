# -*- coding: utf-8 -*-
"""scripts/_d3_noul_and_latency.py — финальные данные для двух решений D3.

1. Чистое сравнение латентности в ОДНОМ процессе (dynamic prod vs static
   qdq_u8 vs static qop_u8): батч-1 и батч-2, после прогрева.
2. Валидация noul-головы на held-out val-сплите (836 строк, target_noul):
   mmBERT noul vs legacy noul — accuracy@0.5 + Brier.
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


def make_sess(name: str, so):
    return ort.InferenceSession(str(BASE_DIR / name), so, providers=["CPUExecutionProvider"])


def lat_measure(sess, enc, n=12):
    sess.run(None, enc)  # warm
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, enc)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return lat[0], statistics.median(lat), lat[-2]


def lat_measure_batch2(sess, encs, n=12):
    a, b = encs[0], encs[1]
    la, lb = a["input_ids"].shape[1], b["input_ids"].shape[1]
    L = max(la, lb)
    ia = np.full((2, L), 0, dtype=np.int64); ia[0, :la] = a["input_ids"][0]; ia[1, :lb] = b["input_ids"][0]
    ma = np.zeros((2, L), dtype=np.int64); ma[0, :la] = 1; ma[1, :lb] = 1
    feed = {"input_ids": ia, "attention_mask": ma}
    sess.run(None, feed)
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return lat[0], statistics.median(lat), lat[-2]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    so = ort.SessionOptions()
    so.log_severity_level = 3

    # ── 1) латентность: dynamic vs static ───────────────────────────────────
    battery = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))
    encs = []
    for c in battery["cases"][:20]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        encs.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                     "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})

    print("=== латентность форварда (warm, батч-1 / батч-2) ===")
    for name in ("mmbert_route_prod_int8.onnx", "_tmp_static_qdq_u8.onnx", "_tmp_static_qop_u8.onnx"):
        sess = make_sess(name, so)
        mn, md, mx = lat_measure(sess, encs[0])
        mn2, md2, mx2 = lat_measure_batch2(sess, encs)
        print(f"{name:32s} батч-1: min={mn:.0f} med={md:.0f} мс | батч-2: min={mn2:.0f} med={md2:.0f} мс")

    # ── 2) noul на val-сплите ───────────────────────────────────────────────
    rows = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines()]
    noul_rows = [r for r in rows if r["question_type"] == "noul"]
    print(f"\n=== noul на val-сплите: {len(noul_rows)} строк ===")

    enc_n = []
    labels = []
    for r in noul_rows:
        t = r["state"] + "\n[QUESTION] " + r["question"]
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        enc_n.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                      "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})
        labels.append(float(r["target_noul"]))

    for name, kind in (("mmbert_route_prod_int8.onnx", "mmbert"),
                       ("_tmp_static_qdq_u8.onnx", "mmbert-static"),
                       ("mmbert_route_prod_int8.onnx", "legacy")):
        if kind == "legacy":
            # legacy-модель живёт в data_400m
            sess = ort.InferenceSession(str(Path("src/leeth/data_400m/leeth_int8.onnx")), so,
                                        providers=["CPUExecutionProvider"])
            outs_idx = 4  # noul_probs
        else:
            sess = make_sess(name, so)
            outs_idx = 1  # noul_logits (raw)
        preds, probs_list = [], []
        t0 = time.perf_counter()
        for enc in enc_n:
            out = sess.run(None, enc)[outs_idx]
            if kind == "mmbert":
                p = 1.0 / (1.0 + np.exp(-float(out[0].item())))
            else:
                p = float(out[0].item())
            probs_list.append(p)
            preds.append(int(p >= 0.5))
        wall = (time.perf_counter() - t0) / len(enc_n) * 1000
        acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
        brier = sum((p - l) ** 2 for p, l in zip(probs_list, labels)) / len(labels)
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1.0)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0.0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1.0)
        print(f"[{kind:13s}] acc={acc:.1%} brier={brier:.4f} | TP={tp} FP={fp} FN={fn} | {wall:.0f} мс/кейс")


if __name__ == "__main__":
    main()