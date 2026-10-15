# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment2.py — прицельная INT8-квантизация mmBERT.

v1: dynamic per-channel, квантизуем ТОЛЬКО FFN (mlp/Wi, mlp/Wo) — attention fp32.
v2: static QDQ per-channel с калибровкой (Percentile 99.999) на 609 текстах.
v3: static QDQ per-channel с MinMax-калибровкой.
"""

import json
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
                                      QuantFormat, QuantType, quantize_static,
                                      quantize_dynamic)

BASE_DIR = Path(".argent/models/mmbert-argent")
FP32_PATH = BASE_DIR / "mmbert_route.onnx"
TRAIN_PATH = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def make_texts() -> list:
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in battery["cases"]]
    rows = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000: break
            rows.append(line)
    random.seed(42)
    for line in random.sample(rows, 500):
        r = json.loads(line)
        texts.append((r.get("state") or "") + "\n[QUESTION] " + (r.get("question") or TEMPLATE_Q))
    return texts


class Reader(CalibrationDataReader):
    def __init__(self, texts, tokenizer):
        self.recs = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=256, return_tensors="np")
            self.recs.append({"input_ids": enc["input_ids"].astype(np.int64),
                              "attention_mask": enc["attention_mask"].astype(np.int64)})
        self.i = 0

    def get_next(self):
        if self.i >= len(self.recs):
            return None
        self.i += 1
        return self.recs[self.i - 1]

    def rewind(self):
        self.i = 0


def battery_acc(sess) -> tuple:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    correct = 0
    t0 = time.perf_counter()
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
        logits = sess.run(None, {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})[0][0]
        correct += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    wall = time.perf_counter() - t0
    n = len(battery["cases"])
    return correct, n, wall / n * 1000


def main() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))

    # v1: dynamic per-channel, только FFN-узлы (attention fp32)
    g = onnx.load(str(FP32_PATH)).graph
    ffn_nodes = [n.name for n in g.node
                 if n.op_type == "MatMul" and "/mlp/" in n.name]
    out1 = BASE_DIR / "mmbert_route_int8_ffn.onnx"
    quantize_dynamic(str(FP32_PATH), str(out1), weight_type=QuantType.QInt8,
                     per_channel=True, nodes_to_quantize=ffn_nodes)
    print(f"[ffn-only] quantized, size={out1.stat().st_size / (1024 * 1024):.0f} MB, "
          f"узлов квантизовано: {len(ffn_nodes)}")
    sess = ort.InferenceSession(str(out1), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[ffn-only] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    # v1b: dynamic per-channel, FFN + головы, БЕЗ attention
    attn_nodes = [n.name for n in g.node
                  if n.op_type in ("MatMul", "Gemm") and ("/attn/" in n.name or "rotary" in n.name)]
    out1b = BASE_DIR / "mmbert_route_int8_ffnhd.onnx"
    quantize_dynamic(str(FP32_PATH), str(out1b), weight_type=QuantType.QInt8,
                     per_channel=True, nodes_to_exclude=attn_nodes)
    print(f"[ffn+heads] size={out1b.stat().st_size / (1024 * 1024):.0f} MB, attn исключён: {len(attn_nodes)}")
    sess = ort.InferenceSession(str(out1b), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[ffn+heads] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    # v1c: dynamic per-channel, только Wqkv+Wo (QKV/O проекции), rotary/softmax fp32
    qkv_nodes = [n.name for n in g.node
                 if n.op_type in ("MatMul", "Gemm") and ("Wqkv" in n.name or "/Wo/" in n.name
                                                          or "/mlp/" in n.name)]
    out1c = BASE_DIR / "mmbert_route_int8_qkv.onnx"
    quantize_dynamic(str(FP32_PATH), str(out1c), weight_type=QuantType.QInt8,
                     per_channel=True, nodes_to_quantize=qkv_nodes)
    print(f"[qkv+mlp] size={out1c.stat().st_size / (1024 * 1024):.0f} MB, узлов: {len(qkv_nodes)}")
    sess = ort.InferenceSession(str(out1c), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[qkv+mlp] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    # v2/v3: static QDQ с калибровкой (Mul не трогаем — scalar-scale баг ORT)
    texts = make_texts()
    print(f"калибровка: {len(texts)} текстов")
    for name, calib_method, extra in [
            ("static_pct", CalibrationMethod.Percentile, {"extra_options": {"CalibPercentile": 99.999}}),
            ("static_mm", CalibrationMethod.MinMax, {})]:
        out = BASE_DIR / f"mmbert_route_{name}.onnx"
        t0 = time.perf_counter()
        try:
            quantize_static(
                str(FP32_PATH), str(out),
                calibration_data_reader=Reader(texts, tokenizer),
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=["MatMul"],
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=calib_method,
                **extra,
            )
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            continue
        print(f"[{name}] quantized за {time.perf_counter() - t0:.0f}s, "
              f"size={out.stat().st_size / (1024 * 1024):.0f} MB")
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        ok, n, ms = battery_acc(sess)
        print(f"[{name}] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    # v2b: static QDQ только на MatMul (Mul не трогаем — scalar-scale баг ORT)
    out = BASE_DIR / "mmbert_route_static_mm2.onnx"
    t0 = time.perf_counter()
    try:
        quantize_static(
            str(FP32_PATH), str(out),
            calibration_data_reader=Reader(texts, tokenizer),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["MatMul"],
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
        )
        print(f"[static_mm2] quantized за {time.perf_counter() - t0:.0f}s, "
              f"size={out.stat().st_size / (1024 * 1024):.0f} MB")
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        ok, n, ms = battery_acc(sess)
        print(f"[static_mm2] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")
    except Exception as e:
        print(f"[static_mm2] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()