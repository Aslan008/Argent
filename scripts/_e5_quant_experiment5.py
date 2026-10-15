# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment5.py — SmoothQuant-миграция massive-активации.

Корень: слой 11, канал 924 gate-половины Wi имеет строку веса std=0.605 (13× медианы),
на токене <bos> даёт act=17373 -> Wo-out ~7000 -> h12 max 6714. RMSNorm съедает
выброс в fp32 (модель работает, 67%), но INT8-шкала активаций рушится.

Фикс (SmoothQuant): делим вход-канал 530 hn на s=8 (веса Wi-строк 530/2076 умножаем
на 8). fp32-математика идентична, активации сглажены, выброс уходит в веса
(per-channel квантизация их переживает).

Проверки: (1) fp32-паритет до/после миграции на батарее, (2) dynamic per-channel
INT8 после миграции, (3) static MinMax после миграции.
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
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_dynamic,
                                      quantize_static)

BASE_DIR = Path(".argent/models/mmbert-argent")
FP32_PATH = BASE_DIR / "mmbert_route.onnx"
SMOOTH_PATH = BASE_DIR / "mmbert_route_smooth.onnx"
TRAIN_PATH = Path("src/leeth/data_400m/leeth_400m_train.jsonl")
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

# (номер слоя, канал hn, сила s) — из диагностики _e5_anomaly_probe
SMOOTH_SPECS = [(11, 530, 8.0), (12, None, 8.0)]  # 12-й уточняется по факту


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[1:].isalpha() or query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def make_texts(n_train: int = 200) -> list:
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    texts = [build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q for c in battery["cases"]]
    rows = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            rows.append(line)
    random.seed(42)
    for line in random.sample(rows, n_train):
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


def smooth_onnx(model: onnx.ModelProto, specs) -> onnx.ModelProto:
    """SmoothQuant-миграция: вход канала c делится на s, колонки весов MatMul умножаются на s.

    specs: [(node_name, channel, s)] — для каждого MatMul с именем node_name.
    Вставляет Mul(scale-вектор) перед входом MatMul и умножает соответствующие
    колонки initializer-веса.
    """
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    nodes = {n.name: n for n in graph.node}
    tensor_to_init = {}
    for i in graph.initializer:
        tensor_to_init[i.name] = i
    # карта: выход узла -> узел (для отслеживания входов)
    for node_name, channel, s in specs:
        node = nodes[node_name]
        w_name = None
        for inp in node.input:
            if inp in tensor_to_init and len(inits[inp].dims) == 2:
                w_name = inp
        if w_name is None:
            raise RuntimeError(f"{node_name}: 2D-вес не найден среди входов {node.input}")
        w = onnx.numpy_helper.to_array(inits[w_name]).copy()
        # экспорт HF: MatMul(x, W) с W (in, out) — вход-канал = ось 0
        axis = 0
        w[channel, :] *= s
        new_w = onnx.numpy_helper.from_array(w, inits[w_name].name)
        inits[w_name].CopyFrom(new_w)
        # вставляем Mul перед входом x (первый вход, не вес)
        x_name = [i for i in node.input if i not in inits][0]
        scale_name = f"{node_name}/smooth_scale"
        scale = np.ones(w.shape[0], dtype=np.float32)  # вход-каналы = ось 0 веса
        scale[channel] = 1.0 / s
        graph.initializer.append(onnx.numpy_helper.from_array(scale, scale_name))
        mul_out = f"{node_name}/smooth_out"
        mul = onnx.helper.make_node("Mul", [x_name, scale_name], [mul_out],
                                    name=f"{node_name}/smooth_mul")
        graph.node.insert(list(graph.node).index(node), mul)
        node.input[0] = mul_out
    onnx.checker.check_model(model, full_check=False)
    return model


def main() -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))

    model = onnx.load(str(FP32_PATH))
    g = model.graph
    wi11 = "/base/layers.11/mlp/Wi/MatMul"
    wi12 = "/base/layers.12/mlp/Wi/MatMul"
    specs = [(wi11, 530, 8.0), (wi12, 530, 8.0)]
    smooth_onnx(model, specs)
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