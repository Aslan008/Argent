# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment10.py — вынос с маской <bos> для слоя 11 + маска
«токен выстреливает» для слоя 12 (канал 957 выстреливает на РАЗНЫХ токенах
23-31 — это, вероятно, спец-токены/знаки '/'; маска по input_ids в множестве).

Упрощение: для слоя 12 канал 957 выстреливает на токенах, чьи input_ids входят
в маленький набор (знаки '/'). Собираем этот набор по 20 текстам и делаем маску
Equal(input_ids, X) для каждого токена из набора -> Or -> Cast -> Unsqueeze -> Mul.

Если набор токенов большой — fallback: константа на все токены НЕ подходит,
тогда остаёмся на exp8 (только слой 11, 56.0% int8) или fp32.
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
BIAS_PATH = BASE_DIR / "mmbert_route_bias3.onnx"
BIAS_INT8_PATH = BASE_DIR / "mmbert_route_bias3_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

SPECS = [
    (11, 924, 16709.5, "bos"),
    (12, 957, -2570.5, "slash"),  # маска: input_ids in {ids слэша/знаков}
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


def collect_slash_ids() -> list:
    """ID токенов, на которых канал 957 слоя 12 выстреливает (из probe3: 23-31)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    ids = set()
    for s in ["/", "//", " / ", "C:/", "scripts/", "src/", "  ", "\n", " ", ":", "."]:
        enc = tok(s, add_special_tokens=False)
        ids.update(enc["input_ids"])
    return sorted(ids)


def main() -> None:
    model = onnx.load(str(FP32_PATH))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    nodes = {n.name: n for n in g.node}
    slash_ids = collect_slash_ids()
    print(f"маска слоя 12: {len(slash_ids)} токенов")

    for layer, act_ch, mean_act, mask_kind in SPECS:
        wo_node = nodes[f"/base/layers.{layer}/mlp/Wo/MatMul"]
        wo_w = inits[wo_node.input[1]]
        wo_arr = numpy_helper.to_array(wo_w).copy()
        const_vec = wo_arr[act_ch, :] * mean_act
        wo_arr[act_ch, :] = 0.0
        inits[wo_w.name].CopyFrom(numpy_helper.from_array(wo_arr, wo_w.name))
        const_name = f"/base/layers.{layer}/mlp/Wo/bias_const"
        g.initializer.append(numpy_helper.from_array(const_vec.astype(np.float32), const_name))

        if mask_kind == "bos":
            bos_id = collect_bos_id()
            mask_ids = [bos_id]
        else:
            mask_ids = slash_ids
        # маска: Or(Equal(input_ids, id1), Equal(input_ids, id2), ...)
        # узлы НЕ вставляем по ходу — собираем в список, вставляем одной цепочкой ниже
        eq_nodes = []
        eq_outputs = []
        for j, tid in enumerate(mask_ids):
            id_name = f"/base/layers.{layer}/mlp/Wo/mask_id_{j}"
            g.initializer.append(numpy_helper.from_array(np.array([tid], dtype=np.int64), id_name))
            eq = onnx.helper.make_node("Equal", ["input_ids", id_name],
                                       [f"/base/layers.{layer}/mlp/Wo/eq_{j}"],
                                       name=f"/base/layers.{layer}/mlp/Wo/eq_{j}")
            eq_outputs.append(f"/base/layers.{layer}/mlp/Wo/eq_{j}")
            eq_nodes.append(eq)
        # Or-цепочка
        or_nodes = []
        acc_name = eq_outputs[0]
        for j in range(1, len(eq_outputs)):
            or_name = f"/base/layers.{layer}/mlp/Wo/or_{j}"
            or_node = onnx.helper.make_node("Or", [acc_name, eq_outputs[j]], [or_name],
                                            name=f"/base/layers.{layer}/mlp/Wo/or_{j}")
            or_nodes.append(or_node)
            acc_name = or_name
        cast_f = onnx.helper.make_node("Cast", [acc_name],
                                       [f"/base/layers.{layer}/mlp/Wo/mask_f"], to=onnx.TensorProto.FLOAT,
                                       name=f"/base/layers.{layer}/mlp/Wo/mask_cast")
        g.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64),
                                                     f"/base/layers.{layer}/mlp/Wo/axes"))
        unsq = onnx.helper.make_node("Unsqueeze", [f"/base/layers.{layer}/mlp/Wo/mask_f",
                                                   f"/base/layers.{layer}/mlp/Wo/axes"],
                                     [f"/base/layers.{layer}/mlp/Wo/mask_u"],
                                     name=f"/base/layers.{layer}/mlp/Wo/mask_unsq")
        mul_mask = onnx.helper.make_node("Mul", [f"/base/layers.{layer}/mlp/Wo/mask_u", const_name],
                                         [f"/base/layers.{layer}/mlp/Wo/bias_term"],
                                         name=f"/base/layers.{layer}/mlp/Wo/bias_mul")
        add = onnx.helper.make_node("Add", [wo_node.output[0], f"/base/layers.{layer}/mlp/Wo/bias_term"],
                                    [wo_node.output[0] + "/bias_out"], name=f"/base/layers.{layer}/mlp/Wo/bias_add")
        # вставляем ВСЮ цепочку ПЕРЕД потребителем Wo (Add_1), а не сразу после Wo:
        # иначе eq-узлы (Equal по input_ids) окажутся ПОСЛЕ своих потребителей.
        wo_idx = list(g.node).index(wo_node)
        consumer_idx = min(i for i, n2 in enumerate(g.node)
                           if wo_node.output[0] in n2.input and n2.name != wo_node.name)
        insert_at = consumer_idx
        chain = [eq_nodes, or_nodes, cast_f, unsq, mul_mask, add]
        for node in [x for grp in chain for x in (grp if isinstance(grp, list) else [grp])][::-1]:
            g.node.insert(insert_at, node)
        wo_out = wo_node.output[0]
        for n2 in g.node:
            for k, inp in enumerate(n2.input):
                if inp == wo_out and n2.name != wo_node.name and n2.name != add.name:
                    n2.input[k] = wo_out + "/bias_out"
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(BIAS_PATH))
    print(f"модель сохранена: {BIAS_PATH}")

    sess = ort.InferenceSession(str(BIAS_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[маскированный fp32] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    quantize_dynamic(str(BIAS_PATH), str(BIAS_INT8_PATH), weight_type=QuantType.QInt8, per_channel=True)
    print(f"[маскированный int8] size={BIAS_INT8_PATH.stat().st_size / (1024 * 1024):.0f} MB")
    sess_q = ort.InferenceSession(str(BIAS_INT8_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess_q)
    print(f"[маскированный int8] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


def collect_bos_id() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    return tok.bos_token_id


if __name__ == "__main__":
    main()