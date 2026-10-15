# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment8.py — вынос massive-компоненты в константный Add.

act[<bos>, ch924] слоя 11 = 16709 ± 5% по текстам — почти константа.
Компонент Wo[:,924] * act924 на токене <bos> ведёт себя как константный вектор.

Фикс: (1) обнуляем колонку 924 веса Wo11 (и колонку 530 Wi11 — источник),
(2) добавляем константный Add (Wo11[:,924] * MEAN_ACT) только к токену <bos>.
Математика: Wo@act = Wo'@act' + const на bos. Выброс исчезает из активаций.

Проверки: (1) fp32-паритет на батарее (допуск ±1 кейс — константа усреднена),
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
BIAS_PATH = BASE_DIR / "mmbert_route_bias.onnx"
BIAS_INT8_PATH = BASE_DIR / "mmbert_route_bias_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

# из _e5_massive_probe: среднее act[<bos>, ch924] слоя 11 по 20 текстам
ACT924_MEAN = 16709.5


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

    wi_node = nodes["/base/layers.11/mlp/Wi/MatMul"]
    wo_node = nodes["/base/layers.11/mlp/Wo/MatMul"]
    wi_w = inits[wi_node.input[1]]
    wo_w = inits[wo_node.input[1]]
    wo_arr = numpy_helper.to_array(wo_w).copy()  # (1152 in, 768 out): MatMul(x, W)

    # 1) константный вектор: Wo[924, :] * mean(act924) — добавляется к Wo-out на <bos>
    const_vec = wo_arr[924, :] * ACT924_MEAN
    # 2) обнуляем: строку 924 в Wo (канал act) и КОЛОНКУ 2076 в Wi (gate-канал 924)
    #    Wi (768, 2304): inp = колонки 0:1152, gate = 1152:2304 -> gate-канал 924 = колонка 2076
    wi_arr = numpy_helper.to_array(wi_w).copy()
    wi_arr[:, 924 + 1152] = 0.0
    wo_arr[924, :] = 0.0
    inits[wi_w.name].CopyFrom(numpy_helper.from_array(wi_arr, wi_w.name))
    inits[wo_w.name].CopyFrom(numpy_helper.from_array(wo_arr, wo_w.name))

    # 3) Add: wo_out + const_vec * is_bos
    #    маска <bos>: Equal(input_ids, bos_id) -> Cast(float) -> Unsqueeze(axis 2) -> Mul(const) -> Add
    bos_id = tok_bos_id()
    const_name = "/base/layers.11/mlp/Wo/bias_const"
    g.initializer.append(numpy_helper.from_array(const_vec.astype(np.float32), const_name))
    g.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "/base/layers.11/mlp/Wo/axes"))
    g.initializer.append(numpy_helper.from_array(np.array([bos_id], dtype=np.int64), "bos_id"))
    eq = onnx.helper.make_node("Equal", ["input_ids", "bos_id"], ["/base/layers.11/mlp/Wo/bos_eq"],
                               name="/base/layers.11/mlp/Wo/bos_eq")
    cast_f = onnx.helper.make_node("Cast", ["/base/layers.11/mlp/Wo/bos_eq"],
                                   ["/base/layers.11/mlp/Wo/bos_mask_f"], to=onnx.TensorProto.FLOAT,
                                   name="/base/layers.11/mlp/Wo/bos_cast_f")
    unsq2 = onnx.helper.make_node("Unsqueeze", ["/base/layers.11/mlp/Wo/bos_mask_f", "/base/layers.11/mlp/Wo/axes"],
                                  ["/base/layers.11/mlp/Wo/bos_mask_u"], name="/base/layers.11/mlp/Wo/bos_unsq")
    mul_mask = onnx.helper.make_node("Mul", ["/base/layers.11/mlp/Wo/bos_mask_u", const_name],
                                     ["/base/layers.11/mlp/Wo/bias_term"], name="/base/layers.11/mlp/Wo/bias_mul")
    add = onnx.helper.make_node("Add", [wo_node.output[0], "/base/layers.11/mlp/Wo/bias_term"],
                                [wo_node.output[0] + "/bias_out"], name="/base/layers.11/mlp/Wo/bias_add")
    # вставляем цепочку сразу ПОСЛЕ Wo, НО Add должен идти ПОСЛЕ потребителя?
    # НЕТ: Add вставляем ПОСЛЕ Wo, а потребителя Wo (Add_1) переключаем на bias_out
    add_idx = list(g.node).index(wo_node) + 1
    for node in [eq, cast_f, unsq2, mul_mask, add][::-1]:
        g.node.insert(add_idx, node)
    # перенаправляем потребителей wo_out на новый выход (Add_1 стоит позже — ок)
    wo_out = wo_node.output[0]
    for n2 in g.node:
        for k, inp in enumerate(n2.input):
            if inp == wo_out and n2.name != wo_node.name and n2.name != "/base/layers.11/mlp/Wo/bias_add":
                n2.input[k] = wo_out + "/bias_out"
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(BIAS_PATH))
    print(f"bias-модель сохранена: {BIAS_PATH}")

    sess = ort.InferenceSession(str(BIAS_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[bias fp32] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")

    quantize_dynamic(str(BIAS_PATH), str(BIAS_INT8_PATH), weight_type=QuantType.QInt8, per_channel=True)
    print(f"[bias int8] size={BIAS_INT8_PATH.stat().st_size / (1024 * 1024):.0f} MB")
    sess_q = ort.InferenceSession(str(BIAS_INT8_PATH), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess_q)
    print(f"[bias int8] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


def tok_bos_id() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    return tok.bos_token_id


if __name__ == "__main__":
    main()