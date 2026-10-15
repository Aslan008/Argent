# -*- coding: utf-8 -*-
"""scripts/_d5_patch_batch_fix.py — вставка масочной ветки (D3-рецепт) в свежий fp32-граф.

Прод-артефакт содержит масочную ветку у слоёв 11/12 (Wo), которой нет в свежем
экспорте — отсюда батч-баг (max|diff|=2.86). Рецепт снят с прода:

Слой 11: eq_0 = Equal(input_ids, mask_id=2) -> mask_cast(Cast->f32) -> mask_unsq(Unsqueeze,axes=[2])
         -> bias_mul(mask_u * bias_const[768]) -> bias_term
         bias_add = Add(MatMul_output_0, bias_term); потребитель MatMul -> bias_add.
Слой 12: eqc_0/1 = Equal(input_ids, cur_0/1); slice_0/1 = Slice(input_ids, starts=1, ends=INT_MAX, axes=[1]);
         padfix_0/1 = Pad(shifted, pads=[0,0,0,1], cv=-1); eqn_0/1 = Equal(shifted_pad, nxt_0/1);
         and_0/1 = And(eqc, eqn); or_1 = Or(and_0, and_1); далее как слой 11.

Токены: слой 11 — mask_id=2 (</s>); слой 12 — пары (235248,235284),(235283,999).
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

BASE = Path(".argent/models/mmbert-argent")
FRESH = BASE / "_d5_new_fp32.onnx"
OUT = BASE / "_d5_new_fp32_fixed.onnx"

m = onnx.load(str(FRESH))
g = m.graph
inits = {i.name: i for i in g.initializer}

# bias_const берём из прода (они относятся к базовой модели, не к головам)
mp = onnx.load(str(BASE / "mmbert_route_prod_int8.onnx"))
bias_const = {
    11: numpy_helper.to_array(next(i for i in mp.graph.initializer
                                   if i.name == "/base/layers.11/mlp/Wo/bias_const")),
    12: numpy_helper.to_array(next(i for i in mp.graph.initializer
                                   if i.name == "/base/layers.12/mlp/Wo/bias_const")),
}

new_nodes = []
new_inits = []

def add_init(name, arr):
    if name in inits:
        return name  # уже есть в графе — не дублируем
    t = numpy_helper.from_array(np.asarray(arr), name)
    new_inits.append(t)
    inits[name] = t
    return name

for L in (11, 12):
    p = f"/base/layers.{L}/mlp/Wo"
    if L == 11:
        add_init(f"{p}/mask_id", np.array([2], dtype=np.int64))
        new_nodes_L = [
            helper.make_node("Equal", ["input_ids", f"{p}/mask_id"], [f"{p}/eq_0"], name=f"{p}/eq_0"),
            helper.make_node("Cast", [f"{p}/eq_0"], [f"{p}/mask_f"], to=TensorProto.FLOAT, name=f"{p}/mask_cast"),
        ]
    else:
        add_init(f"{p}/cur_0", np.array([235248], dtype=np.int64))
        add_init(f"{p}/cur_1", np.array([235283], dtype=np.int64))
        add_init(f"{p}/nxt_0", np.array([235284], dtype=np.int64))
        add_init(f"{p}/nxt_1", np.array([999], dtype=np.int64))
        add_init(f"{p}/starts", np.array([1], dtype=np.int64))
        add_init(f"{p}/ends", np.array([np.iinfo(np.int64).max], dtype=np.int64))
        add_init(f"{p}/axes_s", np.array([1], dtype=np.int64))
        add_init(f"{p}/axes_u", np.array([2], dtype=np.int64))
        add_init(f"{p}/pads_0", np.array([0, 0, 0, 1], dtype=np.int64))
        add_init(f"{p}/padval_0", np.array(-1, dtype=np.int64))
        add_init(f"{p}/pads_1", np.array([0, 0, 0, 1], dtype=np.int64))
        add_init(f"{p}/padval_1", np.array(-1, dtype=np.int64))
        new_nodes_L = [
            helper.make_node("Equal", ["input_ids", f"{p}/cur_0"], [f"{p}/eqc_0"], name=f"{p}/eqc_0"),
            helper.make_node("Equal", ["input_ids", f"{p}/cur_1"], [f"{p}/eqc_1"], name=f"{p}/eqc_1"),
            helper.make_node("Slice", ["input_ids", f"{p}/starts", f"{p}/ends", f"{p}/axes_s"],
                             [f"{p}/shifted_0"], name=f"{p}/slice_0"),
            helper.make_node("Slice", ["input_ids", f"{p}/starts", f"{p}/ends", f"{p}/axes_s"],
                             [f"{p}/shifted_1"], name=f"{p}/slice_1"),
            helper.make_node("Pad", [f"{p}/shifted_0", f"{p}/pads_0", f"{p}/padval_0"],
                             [f"{p}/shifted_pad_0"], name=f"{p}/padfix_0"),
            helper.make_node("Pad", [f"{p}/shifted_1", f"{p}/pads_1", f"{p}/padval_1"],
                             [f"{p}/shifted_pad_1"], name=f"{p}/padfix_1"),
            helper.make_node("Equal", [f"{p}/shifted_pad_0", f"{p}/nxt_0"], [f"{p}/eqn_0"], name=f"{p}/eqn_0"),
            helper.make_node("Equal", [f"{p}/shifted_pad_1", f"{p}/nxt_1"], [f"{p}/eqn_1"], name=f"{p}/eqn_1"),
            helper.make_node("And", [f"{p}/eqc_0", f"{p}/eqn_0"], [f"{p}/and_0"], name=f"{p}/and_0"),
            helper.make_node("And", [f"{p}/eqc_1", f"{p}/eqn_1"], [f"{p}/and_1"], name=f"{p}/and_1"),
            helper.make_node("Or", [f"{p}/and_0", f"{p}/and_1"], [f"{p}/or_1"], name=f"{p}/or_1"),
            helper.make_node("Cast", [f"{p}/or_1"], [f"{p}/mask_f"], to=TensorProto.FLOAT, name=f"{p}/mask_cast"),
        ]
    add_init(f"{p}/axes_u", np.array([2], dtype=np.int64))
    add_init(f"{p}/bias_const", bias_const[L])
    # у слоя 12 axes_u уже добавлен выше — пропустится дедупом
    nodes_L = new_nodes_L + [
        helper.make_node("Unsqueeze", [f"{p}/mask_f", f"{p}/axes_u"], [f"{p}/mask_u"], name=f"{p}/mask_unsq"),
        helper.make_node("Mul", [f"{p}/mask_u", f"{p}/bias_const"], [f"{p}/bias_term"], name=f"{p}/bias_mul"),
        helper.make_node("Add", [f"{p}/MatMul_output_0", f"{p}/bias_term"], [f"{p}/bias_add_out"], name=f"{p}/bias_add"),
    ]
    # перенаправляем потребителя MatMul_output_0 на bias_add_out
    mm_out = f"{p}/MatMul_output_0"
    redirected = 0
    for n in g.node:
        for k, inp in enumerate(n.input):
            if inp == mm_out:
                n.input[k] = f"{p}/bias_add_out"
                redirected += 1
    assert redirected == 1, f"слой {L}: ожидался 1 потребитель MatMul, найдено {redirected}"
    # вставляем узлы ПЕРЕД потребителем (Add_1) — проще: в конец графа не годится (топология).
    # ONNX требует топологический порядок: вставим сразу после MatMul-узла слоя L.
    idx = next(i for i, n in enumerate(g.node) if n.name == f"{p}/MatMul")
    for j, nn in enumerate(nodes_L):
        g.node.insert(idx + 1 + j, nn)
    print(f"слой {L}: вставлено {len(nodes_L)} узлов, потребитель MatMul перенаправлен")

for t in new_inits:
    g.initializer.append(t)

onnx.checker.check_model(m)
onnx.save(m, str(OUT))
print("сохранено:", OUT, f"({OUT.stat().st_size/1e6:.0f} MB)")