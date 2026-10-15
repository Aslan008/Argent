# -*- coding: utf-8 -*-
"""scripts/_d3_inspect_prod.py — какой рецепт у прод-артефакта: какие MatMul/Gemm НЕ квантизованы."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx
from onnx import numpy_helper

BASE = Path(".argent/models/mmbert-argent")
model = onnx.load(str(BASE / "mmbert_route_prod_int8.onnx"))
g = model.graph
inits = {i.name: i for i in g.initializer}

# квантизованный узел: вес проходит через DequantizeLinear
deq_inputs = set()
for n in g.node:
    if n.op_type == "DequantizeLinear":
        deq_inputs.add(n.input[0])

fp32_matmul, q_matmul = [], []
for n in g.node:
    if n.op_type in ("MatMul", "Gemm") and len(n.input) >= 2:
        w = n.input[1]
        if w in inits:
            arr = numpy_helper.to_array(inits[w])
            (q_matmul if (w in deq_inputs or arr.dtype.name.startswith(("int8", "uint8"))) else fp32_matmul).append(n.name)

print(f"fp32 MatMul/Gemm: {len(fp32_matmul)}, int8: {len(q_matmul)}")
print("\n--- fp32 (исключены из квантизации) ---")
for name in fp32_matmul:
    print(" ", name)