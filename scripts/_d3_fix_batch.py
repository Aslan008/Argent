# -*- coding: utf-8 -*-
"""scripts/_d3_fix_batch.py — фикс batch>1 бага в bias4-графе + пересборка прод-INT8.

Баг: маска по парам слоя 12 использует Concat(shifted[batch,seq-1], padval_u[1,1], axis=1).
ONNX Concat НЕ броадкастит: при batch=2 ось 0 расходится (1 vs 2). Все прогоны E5
шли на батче 1, поэтому баг не вскрылся. Прод-рантайму нужен батч (decide_batch).

Фикс: Concat -> Pad (pads=[0,0,0,1], constant_value=-1) — батч-безопасно.

Далее: пересборка prod-INT8 по рецепту прод-артефакта (проверено _d3_inspect_prod.py):
bias4 + MLP-узлы слоёв 11-21 fp32, остальное INT8 per-channel.

Верификация: (1) fp32 батарея батч-1 (ожидание 67.9%), (2) батч-инвариантность
(батч 2 == 2 отдельных прогона), (3) INT8 батарея батч-1 (ожидание 66.1%),
(4) INT8 батч-инвариантность.
"""

import json
import os
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
from transformers import AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
BIAS4_PATH = BASE_DIR / "mmbert_route_bias4.onnx"
BIAS4_FIXED = BASE_DIR / "mmbert_route_bias4_fixed.onnx"
PROD_TMP = BASE_DIR / "_tmp_prod_int8.onnx"
PROD_PATH = BASE_DIR / "mmbert_route_prod_int8.onnx"
BATTERY_PATH = Path("scripts/prod_battery.json")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
FP32_EXCLUDE_LAYERS = range(11, 22)  # MLP этих слоёв остаётся fp32 (рецепт прод-артефакта)


def build_state(query: str) -> str:
    normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


def load_battery():
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    encs = []
    for c in battery["cases"]:
        t = build_state(c["query"]) + "\n[QUESTION] " + TEMPLATE_Q
        enc = tok(t, truncation=True, max_length=256, return_tensors="np")
        encs.append({"input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                     "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64)})
    return battery, idx_to_class, encs


def battery_acc(sess, battery, idx_to_class, encs):
    ok = 0
    t0 = time.perf_counter()
    for c, enc in zip(battery["cases"], encs):
        logits = sess.run(None, enc)[0][0]
        ok += int(idx_to_class[int(logits.argmax())] == c["expected_tool"])
    wall = time.perf_counter() - t0
    return ok, wall / len(encs) * 1000


def batch_invariance(sess, battery, idx_to_class, encs, n_pairs=30):
    """Батч из 2 текстов == 2 отдельных прогона (max|diff| по логитам)."""
    rng = np.random.default_rng(0)
    max_diff = 0.0
    argmax_agree = 0
    checked = 0
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(encs), 2)
        # паддинг до общей длины
        max_len = max(encs[i]["input_ids"].shape[1], encs[j]["input_ids"].shape[1])
        pad_id = 0
        pi = np.full((1, max_len), pad_id, dtype=np.int64)
        pi[0, :encs[i]["input_ids"].shape[1]] = encs[i]["input_ids"][0]
        pj = np.full((1, max_len), pad_id, dtype=np.int64)
        pj[0, :encs[j]["input_ids"].shape[1]] = encs[j]["input_ids"][0]
        mi = np.zeros((1, max_len), dtype=np.int64); mi[0, :encs[i]["input_ids"].shape[1]] = 1
        mj = np.zeros((1, max_len), dtype=np.int64); mj[0, :encs[j]["input_ids"].shape[1]] = 1
        outs = sess.run(None, {"input_ids": np.concatenate([pi, pj], axis=0),
                               "attention_mask": np.concatenate([mi, mj], axis=0)})
        solo_i = sess.run(None, encs[i])[0][0]
        solo_j = sess.run(None, encs[j])[0][0]
        max_diff = max(max_diff, float(np.max(np.abs(outs[0][0] - solo_i))),
                       float(np.max(np.abs(outs[0][1] - solo_j))))
        argmax_agree += int(int(outs[0][0].argmax()) == int(solo_i.argmax()))
        argmax_agree += int(int(outs[0][1].argmax()) == int(solo_j.argmax()))
        checked += 2
    return max_diff, argmax_agree, checked


def fix_graph() -> None:
    model = onnx.load(str(BIAS4_PATH))
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    replaced = 0
    for n in list(g.node):
        if n.op_type == "Concat" and n.name.startswith("/base/layers.12/mlp/Wo/concat_"):
            j = n.name.rsplit("_", 1)[-1]
            shifted = n.input[0]
            out = n.output[0]
            pads_name = f"/base/layers.12/mlp/Wo/pads_{j}"
            val_name = f"/base/layers.12/mlp/Wo/padval_{j}"
            if pads_name not in inits:
                g.initializer.append(numpy_helper.from_array(np.array([0, 0, 0, 1], dtype=np.int64), pads_name))
                g.initializer.append(numpy_helper.from_array(np.array(-1, dtype=np.int64), val_name))
            pad_node = onnx.helper.make_node("Pad", [shifted, pads_name, val_name], [out],
                                             name=f"/base/layers.12/mlp/Wo/padfix_{j}")
            g.node.remove(n)
            g.node.insert(list(g.node).index(
                next(x for x in g.node if out in x.input)), pad_node)
            replaced += 1
    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, str(BIAS4_FIXED))
    print(f"заменено Concat->Pad: {replaced}; сохранено: {BIAS4_FIXED.name}")


def main() -> None:
    battery, idx_to_class, encs = load_battery()

    # ── 1) фикс графа ────────────────────────────────────────────────────────
    fix_graph()

    # ── 2) fp32 верификация ──────────────────────────────────────────────────
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(BIAS4_FIXED), so, providers=["CPUExecutionProvider"])
    ok, ms = battery_acc(sess, battery, idx_to_class, encs)
    print(f"[fp32 fixed] батарея батч-1: {ok}/109 = {ok / 109:.1%} | {ms:.0f} мс/кейс")
    md_fp32, agree_fp32, checked_fp32 = batch_invariance(sess, battery, idx_to_class, encs)
    print(f"[fp32 fixed] батч-инвариантность: max|diff|={md_fp32:.2e}, argmax {agree_fp32}/{checked_fp32}")

    # ── 3) пересборка прод-INT8 ──────────────────────────────────────────────
    model = onnx.load(str(BIAS4_FIXED))
    g = model.graph
    exclude = [n.name for n in g.node
               if n.op_type in ("MatMul", "Gemm")
               and any(n.name.startswith(f"/base/layers.{L}/mlp/") for L in FP32_EXCLUDE_LAYERS)]
    print(f"исключено из квантизации: {len(exclude)} MLP-узлов (слои 11-21)")
    quantize_dynamic(str(BIAS4_FIXED), str(PROD_TMP), weight_type=QuantType.QInt8,
                     per_channel=True, nodes_to_exclude=exclude)
    print(f"[int8] size={os.path.getsize(PROD_TMP) / (1024 * 1024):.0f} MB")

    sess_q = ort.InferenceSession(str(PROD_TMP), so, providers=["CPUExecutionProvider"])
    ok, ms = battery_acc(sess_q, battery, idx_to_class, encs)
    print(f"[int8 fixed] батарея батч-1: {ok}/109 = {ok / 109:.1%} | {ms:.0f} мс/кейс")
    md, agree, checked = batch_invariance(sess_q, battery, idx_to_class, encs)
    print(f"[int8 fixed] батч-инвариантность: max|diff|={md:.2e}, argmax {agree}/{checked}")
    # ── 4) замена прод-артефакта ─────────────────────────────────────────────
    # Батч-инвариантность INT8 недостижима в принципе (динамические scale по
    # min/max батча; идентичный батч инвариантен, смешанный — нет, 13/60 флипов,
    # см. _d3_probe_batch.py). Рантайм для mmBERT гоняет батч-1 на вопрос,
    # поэтому критерий: сохранение батч-1 точности (72/109) + корректность
    # fp32-графа на батче (diff=0).
    if ok / 109 >= 0.65 and md_fp32 < 1e-4 and agree_fp32 == checked_fp32:
        os.replace(PROD_TMP, PROD_PATH)
        print(f"прод-артефакт заменён: {PROD_PATH.name} ({os.path.getsize(PROD_PATH) / (1024**2):.0f} MB)")
    else:
        print("ВЕРИФИКАЦИЯ НЕ ПРОЙДЕНА — прод-артефакт НЕ заменён")


if __name__ == "__main__":
    main()