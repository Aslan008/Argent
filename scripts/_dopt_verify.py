"""Разовая верификация DoPT ONNX (экспорт корректен + эквивалентен кернел-пути)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leeth.dopt_export import verify_dopt_onnx

ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".argent/models/mmbert_dynamic_opt_v1.pt")
onnx = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".argent/models/dopt/dopt_v1.onnx")
m = verify_dopt_onnx(ckpt, Path(".argent/models/mmbert-argent"), onnx)
sys.exit(0 if m["pass"] else 1)