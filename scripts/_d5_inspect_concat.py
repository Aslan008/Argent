# -*- coding: utf-8 -*-
"""scripts/_d5_inspect_concat.py — поиск дефектного Concat(shifted, padval[1,1]) в fp32-графе."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx

FP32 = Path(".argent/models/mmbert-argent/_d5_new_fp32.onnx")
m = onnx.load(str(FP32))
g = m.graph
inits = {i.name: i for i in g.initializer}

print("Concat-узлы с Constant-входом формы [1,1] (или скаляром):")
for n in g.node:
    if n.op_type != "Concat":
        continue
    for inp in n.input:
        if inp in inits:
            t = inits[inp]
            dims = list(t.dims)
            if dims in ([1, 1], [1]) or (len(dims) <= 2 and sum(dims) <= 2):
                vals = onnx.numpy_helper.to_array(t).flatten().tolist()[:4]
                print(f"  {n.name}: input={inp} shape={dims} vals={vals} "
                      f"axis={next((a.name for a in n.attribute if a.name == 'axis'), '?')} "
                      f"outputs={list(n.output)}")
                break

print("\nВсе Pad-узлы (для сверки с прод-артефактом):")
for n in g.node:
    if n.op_type == "Pad":
        print(f"  {n.name}: inputs={list(n.input)} attrs={[(a.name, list(a.ints) if a.ints else a.i) for a in n.attribute]}")