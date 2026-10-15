# -*- coding: utf-8 -*-
"""scripts/_d5_check_weights.py — сверка весов Wo слоя 12: прод (де-квант) vs свежий fp32."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import onnx

mp = onnx.load(".argent/models/mmbert-argent/mmbert_route_prod_int8.onnx")
inits_p = {i.name: i for i in mp.graph.initializer}
mf = onnx.load(".argent/models/mmbert-argent/_d5_new_fp32.onnx")
inits_f = {i.name: i for i in mf.graph.initializer}

for n in mp.graph.node:
    if "layers.12/mlp/Wo" in n.name and n.op_type == "MatMul":
        print("MatMul:", n.name, list(n.input))
        B = n.input[1]
        break
else:
    raise SystemExit("MatMul слоя 12 Wo не найден")

w_p = onnx.numpy_helper.to_array(inits_p[B])
w_f = onnx.numpy_helper.to_array(inits_f["onnx::MatMul_4179"])
print("прод:", w_p.shape, w_p.dtype, "свежий:", w_f.shape, w_f.dtype)
if w_p.shape != w_f.shape and w_p.T.shape == w_f.shape:
    w_p = w_p.T
print("max|diff|:", float(np.abs(w_p.astype(np.float32) - w_f.astype(np.float32)).max()))
print("первые прод:", w_p.flatten()[:3].tolist())
print("первые свежий:", w_f.flatten()[:3].tolist())