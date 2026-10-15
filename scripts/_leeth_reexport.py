"""Re-exports ONNX + INT8 from the current leeth_best.pt with atomic replace."""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth.export import export_to_onnx, quantize_onnx_model  # noqa: E402

DATA_DIR = Path("src/leeth/data")


def _atomic_replace(src: Path, dst: Path, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)


if __name__ == "__main__":
    stamp = int(time.time())
    tmp_onnx = export_to_onnx(
        str(DATA_DIR), weights_filename="leeth_best.pt",
        output_filename=f"leeth.onnx.tmp{stamp}",
    )
    _atomic_replace(tmp_onnx, DATA_DIR / "leeth.onnx")
    tmp_int8 = quantize_onnx_model(DATA_DIR / "leeth.onnx", output_filename=f"leeth_int8.onnx.tmp{stamp}")
    _atomic_replace(tmp_int8, DATA_DIR / "leeth_int8.onnx")
    print("Re-export OK: leeth.onnx + leeth_int8.onnx now match leeth_best.pt")