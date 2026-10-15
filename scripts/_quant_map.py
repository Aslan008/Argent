# -*- coding: utf-8 -*-
"""scripts/_quant_map.py — какие узлы прод-модели остались fp32 (рецепт прода)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import onnx
from onnx import TensorProto

m = onnx.load(sys.argv[1] if len(sys.argv) > 1 else ".argent/models/mmbert-argent/mmbert_route_prod_int8.onnx")
inits = {i.name: i for i in m.graph.initializer}
fp32_nodes, int8_nodes = [], []
for n in m.graph.node:
    if n.op_type not in ("MatMul", "Gemm"):
        continue
    w_name = n.input[1] if len(n.input) > 1 else ""
    w = inits.get(w_name)
    if w is None:
        continue
    (fp32_nodes if w.data_type == TensorProto.FLOAT else int8_nodes).append(n.name)

print(f"fp32-узлов: {len(fp32_nodes)} | int8-узлов: {len(int8_nodes)}")
groups: dict = {}
for name in fp32_nodes:
    parts = name.split("/")
    key = "/".join(parts[:4]) if name.startswith("/base/layers") else name
    bucket = groups = None
    if "/mlp/" in name:
        layer = name.split("/")[3]
        key = f"mlp слой {layer}"
    elif "/self_attn/" in name:
        layer = name.split("/")[3]
        key = f"attn слой {layer}"
    elif "choice_head" in name:
        key = "choice_head"
    elif "noul_head" in name:
        key = "noul_head"
    else:
        key = name[:60]
    bucket = fp32_nodes
    d = groups.setdefault(key, []) if (groups := __import__("collections").defaultdict(list)) else None
print("--- fp32 (не квантизованы) ---")
import collections
cnt = collections.Counter()
for name in fp32_nodes:
    if "/mlp/" in name:
        cnt[f"mlp {name.split('/')[3]}"] += 1
    elif "/self_attn/" in name:
        cnt[f"attn {name.split('/')[3]}"] += 1
    elif "choice_head" in name:
        cnt["choice_head"] += 1
    elif "noul_head" in name:
        cnt["noul_head"] += 1
    else:
        cnt[name[:50]] += 1
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")