# -*- coding: utf-8 -*-
"""scripts/_quantize_int8.py — INT8-квантование FP32 ONNX отдельным процессом."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.export import quantize_onnx_model

out = quantize_onnx_model(Path("src/leeth/data_400m/leeth.onnx"), output_filename="leeth_int8.onnx")
print("INT8 EXPORTED:", out, out.stat().st_size, flush=True)