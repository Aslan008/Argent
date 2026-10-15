# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment9.py — вынос massive-каналов ВСЕХ слоёв (каскад).

После выноса канала 924 слоя 11 выброс сместился в слой 12 (канал 957, токен 24
'/', -2483.7) -> слои 13-21 держат ~2900. Каскад: для каждого слоя 11..21 ищем
канал с max|act| на его MLP-выходе и выносим его в константный Add (только для
токенов, где |act| > порога — упрощение: константа на ВСЕХ токенах, где канал
выстреливает, вычисляем маску по порогу 100).

Упрощение против exp8: маска не по <bos>, а по факту |act| > threshold —
но в ONNX это сложнее. Компромисс: выносим константой на ВСЕ токены
(act у остальных токенов на этом канале ~0 — проверено probe2: канал 957
на остальных токенах < 8), поэтому Add константы на все токены ≈ Add на bos.
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
BIAS_PATH = BASE_DIR / "mmbert_route_bias2.onnx"
BIAS_INT8_PATH = BASE_DIR / "mmbert_route_bias2_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

# (слой, канал act, среднее act) — из _e5_massive_probe3 (среднее по 20 текстам)
SPECS = [
    (11, 924, 16709.5),
    (12, 957, -2570.5),
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

    for layer, act_ch, mean_act in SPECS:
        wo_node = nodes[f"/base/layers.{layer}/mlp/Wo/MatMul"]
        wo_w = inits[wo_node.input[1]]
        wo_arr = numpy_helper.to_array(wo_w).copy()  # (1152, 768)
        const_vec = wo_arr[act_ch, :] * mean_act
        wo_arr[act_ch, :] = 0.0
        inits[wo_w.name].CopyFrom(numpy_helper.from_array(wo_arr, wo_w.name))
        const_name = f"/base/layers.{layer}/mlp/Wo/bias_const"
        g.initializer.append(numpy_helper.from_array(const_vec.astype(np.float32), const_name))
        add = onnx.helper.make_node("Add", [wo_node.output[0], const_name],
                                    [wo_node.output[0] + "/bias_out"], name=f"/base/layers.{layer}/mlp/Wo/bias_add")
        add_idx = list(g.node).index(wo_node) + 1
        g.node.insert(add_idx, add)
        wo_out = wo_node.output[0]
        for n2 in g.node:
            for k, inp in enumerate(n2.input):
                if inp == wo_out and n2.name != wo_node.name and n2.name != add.name:
                    n2.input[k] = wo_out + "/bias_out"
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(BIAS_PATH))
    print(f"каскадная модель сохранена: {BIAS_PATH} ({len(SPECS)} выносов)")

    sess = ort.InferenceSession(str(BIAS_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[каскад fp32] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    quantize_dynamic(str(BIAS_PATH), str(BIAS_INT8_PATH), weight_type=QuantType.QInt8, per_channel=True)
    print(f"[каскад int8] size={BIAS_INT8_PATH.stat().st_size / (1024 * 1024):.0f} MB")
    sess_q = ort.InferenceSession(str(BIAS_INT8_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess_q)
    print(f"[каскад int8] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()