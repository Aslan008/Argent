# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment13.py — точечное исключение massive-узлов из INT8.

Диагноз: massive-активации живут в MLP-ветках слоёв 11/12/18:
  - слой 11: канал 924 (константа на <bos>) — вынесен в bias4
  - слой 12: канал 957 (пары '/'->'ms', ' '->'2') — вынесен в bias4
  - слой 18: канал 370 — massive на 218 токенах с РАЗНЫМИ значениями (100-1431),
    вынос невозможен; SmoothQuant сглаживает вход Wi, но Wo-out (1287) остаётся.

Финальный ход: bias4 + исключение из квантизации MLP-узлов слоёв 11/12/18
(Wi и Wo остаются fp32) + smooth входа Wi18 (канал 370, s=8).
Остальные 19 слоёв квантизуются полностью.

Проверки: (1) fp32-паритет, (2) int8 точность (цель ≥64%), (3) размер/скорость.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

BASE_DIR = Path(".argent/models/mmbert-argent")
BIAS4_PATH = BASE_DIR / "mmbert_route_bias4.onnx"
MIXED_PATH = BASE_DIR / "mmbert_route_mixed.onnx"
MIXED_INT8_PATH = BASE_DIR / "mmbert_route_mixed_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

S = 8.0
SMOOTH_SPECS = [("/base/layers.18/mlp/Wi/MatMul", 370, S)]
EXCLUDE_LAYERS = [11, 12, 18]  # MLP этих слоёв остаётся fp32


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
    model = onnx.load(str(BIAS4_PATH))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes = {n.name: n for n in g.node}

    # smooth входа Wi18 (канал 370)
    for node_name, channel, s in SMOOTH_SPECS:
        node = nodes[node_name]
        w_name = node.input[1]
        w = numpy_helper.to_array(inits[w_name]).copy()
        w[channel, :] *= s
        inits[w_name].CopyFrom(numpy_helper.from_array(w, w_name))
        ln_out = node.input[0]
        scale_name = f"{node_name}/smooth_scale"
        scale = np.ones(768, dtype=np.float32)
        scale[channel] = 1.0 / s
        g.initializer.append(numpy_helper.from_array(scale, scale_name))
        mul_out = f"{node_name}/smooth_out"
        mul = onnx.helper.make_node("Mul", [ln_out, scale_name], [mul_out],
                                    name=f"{node_name}/smooth_mul")
        idx = list(g.node).index(node)
        g.node.insert(idx, mul)
        node.input[0] = mul_out
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(MIXED_PATH))
    print(f"mixed-модель сохранена: {MIXED_PATH}")

    # исключаем MLP-узлы слоёв 11/12/18 из квантизации
    exclude = []
    for layer in EXCLUDE_LAYERS:
        for n in g.node:
            if n.name.startswith(f"/base/layers.{layer}/mlp/") and n.op_type in ("MatMul", "Gemm"):
                exclude.append(n.name)
    print(f"исключено из квантизации: {len(exclude)} узлов (слои {EXCLUDE_LAYERS})")

    out_q = BASE_DIR / "mmbert_route_mixed_int8.onnx"
    quantize_dynamic(str(MIXED_PATH), str(out_q), weight_type=QuantType.QInt8,
                     per_channel=True, nodes_to_exclude=exclude)
    print(f"[mixed int8] size={out_q.stat().st_size / (1024 * 1024):.0f} MB")
    sess_q = ort.InferenceSession(str(out_q), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess_q)
    print(f"[mixed int8] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()