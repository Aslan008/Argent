# -*- coding: utf-8 -*-
"""scripts/_e5_quant_experiment4.py — одиночный вариант квантизации (argv[1]).

Варианты:
  attn_dyn  — dynamic per-channel, ТОЛЬКО attention (Wqkv/Wo) + головы; MLP fp32
              (обоснование: massive-активации живут в MLP-ветке, у attention
              входы после RMSNorm ограничены ~±9)
  pct999    — static QDQ MatMul-only, Percentile 99.9 (агрессивный клиппинг выброса)
  pct995    — static QDQ MatMul-only, Percentile 99.5 (ещё агрессивнее)

Запуск: python scripts/_e5_quant_experiment4.py <variant>
Каждый вариант — в отдельном процессе (калибровка OOM-ила при повторном использовании сессии).
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


def make_texts(n_train: int = 300) -> list:
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


def main() -> None:
    from transformers import AutoTokenizer
    variant = sys.argv[1] if len(sys.argv) > 1 else "attn_dyn"
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    g = onnx.load(str(FP32_PATH)).graph

    if variant == "attn_dyn":
        nodes = [n.name for n in g.node if n.op_type in ("MatMul", "Gemm")
                 and ("Wqkv" in n.name or "/Wo/" in n.name or "choice_head" in n.name or "noul_head" in n.name)]
        out = BASE_DIR / "mmbert_route_int8_attn.onnx"
        quantize_dynamic(str(FP32_PATH), str(out), weight_type=QuantType.QInt8,
                         per_channel=True, nodes_to_quantize=nodes)
    else:
        pct = {"pct999": 99.9, "pct995": 99.5}[variant]
        out = BASE_DIR / f"mmbert_route_{variant}.onnx"
        quantize_static(
            str(FP32_PATH), str(out),
            calibration_data_reader=Reader(make_texts(), tokenizer),
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["MatMul"],
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"CalibPercentile": pct},
        )

    print(f"[{variant}] size={out.stat().st_size / (1024 * 1024):.0f} MB")
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ok, n, ms = battery_acc(sess)
    print(f"[{variant}] батарея: {ok}/{n} = {ok / n:.1%} | {ms:.0f} мс/кейс")


if __name__ == "__main__":
    main()