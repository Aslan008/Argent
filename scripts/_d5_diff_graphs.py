# -*- coding: utf-8 -*-
"""scripts/_d5_diff_graphs.py — сверка прод-артефакта (с D3-фиксом) и свежего fp32.

Ищет Pad-узлы в проде и их контекст, затем соответствующий участок в свежем fp32.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx

BASE = Path(".argent/models/mmbert-argent")
PROD = BASE / "mmbert_route_prod_int8.onnx"
FRESH = BASE / "_d5_new_fp32.onnx"

mp = onnx.load(str(PROD))
gp = mp.graph
print("=== ПРОД (int8, с фиксом): Pad-узлы ===")
for n in gp.node:
    if n.op_type == "Pad":
        attrs = []
        for a in n.attribute:
            if a.name == "pads":
                attrs.append(f"pads={list(a.ints)}")
            elif a.name == "constant_value":
                attrs.append(f"cv={a.f if a.type == 1 else a.i}")
            else:
                attrs.append(f"{a.name}={a.type}")
        # контекст: кто ест выход Pad
        consumers = [c.op_type + ":" + c.name for c in gp.node if n.output[0] in c.input]
        producers = [p.op_type + ":" + p.name for p in gp.node for o in p.output if o in n.input]
        print(f"  {n.name}: in={list(n.input)} {attrs}")
        print(f"    from: {producers}")
        print(f"    to: {consumers}")

print("\n=== ПРОД: Concat-узлы рядом с Pad ===")
pad_outputs = {o for n in gp.node if n.op_type == "Pad" for o in n.output}
for n in gp.node:
    if n.op_type == "Concat" and any(i in pad_outputs for i in n.input):
        print(f"  {n.name}: inputs={list(n.input)}")

print("\n=== СВЕЖИЙ fp32: узлы вокруг маскировки (MaskedFill/Where/Equal рядом с Concat) ===")
mf = onnx.load(str(FRESH))
gf = mf.graph
inits_f = {i.name: i for i in gf.initializer}
# ищем Concat, чьи входы — результат Sub/Equal/MaskedFill (масочная цепочка)
for n in gf.node:
    if n.op_type == "Concat":
        src_ops = [p.op_type for p in gf.node for o in p.output if o in n.input]
        if any(op in ("Sub", "Equal", "MaskedFill", "Where") for op in src_ops):
            const_inputs = [(i, list(inits_f[i].dims)) for i in n.input if i in inits_f]
            print(f"  {n.name}: src={src_ops} consts={const_inputs} out={list(n.output)}")