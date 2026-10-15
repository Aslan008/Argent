# -*- coding: utf-8 -*-
"""scripts/_d3_static_quant.py — статическая квантизация mmBERT (батч-инвариантность).

Проблема dynamic-INT8: scale активаций считается из min/max батча → смешанный
батч меняет предсказания (13/60 флипов, scripts/_d3_probe_batch.py) → рантайм
вынужден гонять батч-1 соло на вопрос.

Статическая квантизация фиксирует scale по калибровочному сету → батч
становится свободным: все choice-вопросы состояния — ОДИН форвард.

Рецепт (как у прод-артефакта): bias4_fixed (fp32, батч-безопасный граф) +
MLP-узлы слоёв 11-21 fp32, остальное INT8 per-channel. Формат QDQ
(рекомендован для CPU EP), активации QInt8 MinMax-калибровка.

Калибровочный сет: прод-батарея (109) + стратифицированная выборка тренировочного
корпуса (все форматы состояний: shell/webhook/code review/compiler/разговорные).
"""
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quant_pre_process,
                                      quantize_static)
from transformers import AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
FP32_PATH = BASE_DIR / "mmbert_route_bias4_fixed.onnx"
OUT_PATH = BASE_DIR / "_tmp_static_int8.onnx"
PREPRO_PATH = BASE_DIR / "_tmp_preproc.onnx"
PROD_PATH = BASE_DIR / "mmbert_route_prod_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TRAIN_PATH = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
FP32_EXCLUDE_LAYERS = range(11, 22)


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def calibration_texts() -> list:
    """Прод-батарея + стратифицированная выборка корпуса (все форматы состояний)."""
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    texts = []
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    for c in battery["cases"]:
        texts.append(build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q)

    rows = [json.loads(l) for l in TRAIN_PATH.read_text(encoding="utf-8").splitlines()]
    rng = random.Random(42)
    rng.shuffle(rows)
    # стратификация по формату состояния (первая строка) — до 6 вопросов на формат
    by_fmt = {}
    for r in rows:
        fmt = r["state"].split("\n", 1)[0]
        by_fmt.setdefault(fmt, []).append(r)
    for fmt, fmt_rows in sorted(by_fmt.items()):
        for r in fmt_rows[:6]:
            texts.append(r["state"] + "\n[QUESTION] " + r["question"])
    print(f"калибровка: {len(texts)} текстов ({len(by_fmt)} форматов состояний)")
    return texts, tok


class Reader(CalibrationDataReader):
    def __init__(self, texts, tok):
        self.encs = []
        for t in texts:
            enc = tok(t, truncation=True, max_length=256, return_tensors="np")
            self.encs.append({
                "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
            })
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.encs):
            return None
        e = self.encs[self._idx]
        self._idx += 1
        return e

    def rewind(self):
        self._idx = 0


def battery_acc(sess, idx_to_class, encs, cases):
    ok = 0
    t0 = time.perf_counter()
    for c, enc in zip(cases, encs):
        logits = sess.run(None, enc)[0][0]
        ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    return ok, (time.perf_counter() - t0) / len(encs) * 1000


def batch_invariance(sess, encs, n_pairs=30):
    """Батч из 2 разных текстов (реальный pad) == 2 отдельных прогона."""
    rng = np.random.default_rng(0)
    pad_id = 0
    max_diff, flips, checked = 0.0, 0, 0
    for _ in range(n_pairs):
        a, b = rng.integers(0, len(encs), 2)
        la, lb = encs[a]["input_ids"].shape[1], encs[b]["input_ids"].shape[1]
        L = max(la, lb)
        ia = np.full((1, L), pad_id, dtype=np.int64); ia[0, :la] = encs[a]["input_ids"][0]
        ib = np.full((1, L), pad_id, dtype=np.int64); ib[0, :lb] = encs[b]["input_ids"][0]
        ma = np.zeros((1, L), dtype=np.int64); ma[0, :la] = 1
        mb = np.zeros((1, L), dtype=np.int64); mb[0, :lb] = 1
        out = sess.run(None, {"input_ids": np.concatenate([ia, ib], axis=0),
                              "attention_mask": np.concatenate([ma, mb], axis=0)})[0]
        sa = sess.run(None, encs[a])[0][0]
        sb = sess.run(None, encs[b])[0][0]
        max_diff = max(max_diff, float(np.max(np.abs(out[0] - sa))),
                       float(np.max(np.abs(out[1] - sb))))
        flips += int(int(out[0].argmax()) != int(sa.argmax()))
        flips += int(int(out[1].argmax()) != int(sb.argmax()))
        checked += 2
    return max_diff, flips, checked


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in json.loads(CLASSES_PATH.read_text(encoding="utf-8")).items()}
    encs = []
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        encs.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                     "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})

    # ── 1) пре-процессинг (shape inference + оптимизация графа) ───────────
    # Без него ORT ругается и QDQ-оптимизации работают хуже (замер: 56.9%,
    # 485 мс/кейс).
    t0 = time.perf_counter()
    quant_pre_process(str(FP32_PATH), str(PREPRO_PATH), skip_optimization=False)
    print(f"quant_pre_process: {time.perf_counter() - t0:.0f} с")

    # ── 2) статическая квантизация ──────────────────────────────────────────
    model = onnx.load(str(PREPRO_PATH))
    g = model.graph
    exclude = [n.name for n in g.node
               if n.op_type in ("MatMul", "Gemm")
               and any(n.name.startswith(f"/base/layers.{L}/mlp/") for L in FP32_EXCLUDE_LAYERS)]
    print(f"исключено из квантизации: {len(exclude)} MLP-узлов (слои 11-21)")

    texts, tok_cal = calibration_texts()
    reader = Reader(texts, tok_cal)

    t0 = time.perf_counter()
    quantize_static(str(PREPRO_PATH), str(OUT_PATH), calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    nodes_to_exclude=exclude,
                    op_types_to_quantize=["MatMul", "Gemm"])
    print(f"quantize_static: {time.perf_counter() - t0:.0f} с, "
          f"size={os.path.getsize(OUT_PATH) / (1024 * 1024):.0f} MB")

    # ── 2) верификация ──────────────────────────────────────────────────────
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(OUT_PATH), so, providers=["CPUExecutionProvider"])
    ok, ms = battery_acc(sess, idx_to_class, encs, battery["cases"])
    print(f"[static int8] батарея батч-1: {ok}/109 = {ok / 109:.1%} | {ms:.0f} мс/кейс")
    md, flips, checked = batch_invariance(sess, encs)
    print(f"[static int8] батч-инвариантность: max|diff|={md:.2e}, флипов {flips}/{checked}")

    # ── 3) замена прод-артефакта при успехе ─────────────────────────────────
    if ok / 109 >= 0.65 and md < 1e-4:
        os.replace(OUT_PATH, PROD_PATH)
        print(f"прод-артефакт заменён: {PROD_PATH.name}")
    else:
        print("гейт не пройден — прод-артефакт не тронут")


if __name__ == "__main__":
    main()