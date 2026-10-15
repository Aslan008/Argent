# -*- coding: utf-8 -*-
"""scripts/_e5_cleanup.py — чистка экспериментальных ONNX-артефактов E5.

Оставляем: mmbert_route.onnx (fp32, 67.0%), mmbert_route_prod_int8.onnx (66.1%),
mmbert_route_bias4.onnx (основа рецепта). Остальное — эксперименты (~15 GB)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(".argent/models/mmbert-argent")
KEEP = {"mmbert_route.onnx", "mmbert_route_prod_int8.onnx", "mmbert_route_bias4.onnx"}


def main() -> None:
    removed = 0
    freed = 0
    for f in BASE.glob("mmbert_route*.onnx"):
        if f.name in KEEP:
            continue
        sz = os.path.getsize(f)
        f.unlink()
        removed += 1
        freed += sz
    for f in BASE.glob("_tmp_*.onnx"):
        sz = os.path.getsize(f)
        f.unlink()
        removed += 1
        freed += sz
    print(f"удалено {removed} файлов, освобождено {freed / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()