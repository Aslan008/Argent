# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment.py — подбор параметров INT8-квантизации mmBERT.

Проблема: quantize_dynamic (per-tensor, QInt8) роняет батарею с 67% до 40.4%.
Эксперимент: (1) fp32 ONNX на батарее — контроль, (2) per_channel=True,
(3) per_channel + исключение голов choice/noul из квантизации.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

BASE_DIR = Path(".argent/models/mmbert-argent")
FP32_PATH = BASE_DIR / "mmbert_route.onnx"
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


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
    variants = [
        ("int8_pc", dict(weight_type=QuantType.QInt8, per_channel=True), None),
        ("int8_pc_nohd", dict(weight_type=QuantType.QInt8, per_channel=True), "heads"),
    ]
    for name, kwargs, exclude in variants:
        out = BASE_DIR / f"mmbert_route_{name}.onnx"
        if exclude == "heads":
            # исключаем Gemm голов choice/noul (последние 4 Gemm-узла графа)
            import onnx
            g = onnx.load(str(FP32_PATH)).graph
            gemms = [n.name for n in g.node if n.op_type == "Gemm"]
            kwargs = dict(kwargs, nodes_to_exclude=gemms[-4:])
        t0 = time.perf_counter()
        quantize_dynamic(str(FP32_PATH), str(out), **kwargs)
        print(f"[{name}] quantized за {time.perf_counter() - t0:.0f}s, "
              f"size={out.stat().st_size / (1024 * 1024):.0f} MB")
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        ok, n, ms = battery_acc(sess)
        print(f"[{name}] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    sess32 = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess32)
    print(f"[fp32-ONNX] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()