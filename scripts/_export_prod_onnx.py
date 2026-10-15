"""One-off: export the production 400M checkpoint to fp32 ONNX + INT8 quantized.

The runtime (src/leeth/runtime.py) prefers the INT8 ONNX CPU backend; without
it every leeth.reload() in tests falls back to torch.load of the 1.4 GB
checkpoint (~3 GB peak per reload) and exhausts laptop memory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.export import export_to_onnx, quantize_onnx_model

MODEL_DIR = "src/leeth/data_400m"

fp32 = export_to_onnx(MODEL_DIR, weights_filename="leeth_best.pt",
                      output_filename="leeth.onnx")
int8 = quantize_onnx_model(fp32, output_filename="leeth_int8.onnx")
print("EXPORTED:", int8)