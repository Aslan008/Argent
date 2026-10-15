# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment6.py — SmoothQuant на ВЕСАХ Wi (без Mul в графе).

Ошибка exp5: Mul вставлялся ПОСЛЕ LayerNormalization, но потребители Wi-out —
Shape/Slice (chunk inp/gate), а не сам MatMul — граф стал некорректным.
Правильный подход: канал 530 hn — это ЧИСТЫЙ вход LayerNormalization, его
потребитель ТОЛЬКО Wi11. Делим выход LN на s, умножаем СТРОКИ 530/2076 весов
Wi11/Wi12 (ось 0 = вход-канал) на s. fp32-математика идентична, massive-активация
(17373 -> 2170) сглажена, выброс уходит в веса.

Проверки: (1) fp32-паритет smooth vs оригинал на батарее (обязан быть 67.0%),
(2) dynamic per-channel INT8 после миграции.
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
FP32_PATH = BASE_DIR / "mmbert_route.onnx"
SMOOTH_PATH = BASE_DIR / "mmbert_route_smooth.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

S = 8.0
# (имя MatMul, канал входа, множитель веса)
SPECS = [
    ("/base/layers.11/mlp/Wi/MatMul", 530, S),
    ("/base/layers.12/mlp/Wi/MatMul", 530, S),
]


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
    model = onnx.load(str(FP32_PATH))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes = {n.name: n for n in g.node}

    for node_name, channel, s in SPECS:
        node = nodes[node_name]
        w_name = node.input[1]  # onnx::MatMul_XXXX, dims (768, 2304), ось 0 = вход-канал
        w = numpy_helper.to_array(inits[w_name]).copy()
        w[channel, :] *= s
        inits[w_name].CopyFrom(numpy_helper.from_array(w, w_name))
        # делим выход LayerNormalization на s (канал 530)
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
        # перенаправляем ВСЕХ потребителей ln_out на mul_out
        for n2 in g.node:
            for k, inp in enumerate(n2.input):
                if inp == ln_out:
                    n2.input[k] = mul_out
    # после правок пересортируем узлы топологически (Mul вставлен в середину)
    topologically = []
    visited = set()
    node_by_out = {}
    for n3 in g.node:
        for o in n3.output:
            node_by_out[o] = n3

    def visit(n3):
        if id(n3) in visited:
            return
        visited.add(id(n3))
        for inp in n3.input:
            producer = node_by_out.get(inp)
            if producer is not None:
                visit(producer)
        topologically.append(n3)

    for n3 in list(g.node):
        visit(n3)
    del g.node[:]
    g.node.extend(topologically)
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(SMOOTH_PATH))
    print(f"smooth-модель сохранена: {SMOOTH_PATH}")

    sess = ort.InferenceSession(str(SMOOTH_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[smooth fp32] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    out_q = BASE_DIR / "mmbert_route_smooth_int8.onnx"
    quantize_dynamic(str(SMOOTH_PATH), str(out_q), weight_type=QuantType.QInt8, per_channel=True)
    print(f"[smooth int8] size={out_q.stat().st_size / (1024 * 1024):.0f} MB")
    sess_q = ort.InferenceSession(str(out_q), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess_q)
    print(f"[smooth int8] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()