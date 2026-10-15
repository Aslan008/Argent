# -*- coding: utf-8 -*-
"""scripts/_multidomain_export.py — реэкспорт ONNX из промоутнутого leeth_best.pt.

fp32 -> leeth.onnx, int8 -> leeth_int8.onnx (атомарная замена), затем верификация
выравнивания torch/ONNX/int8 (verify_exported_model).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.export import export_to_onnx, quantize_onnx_model, verify_exported_model

DATA = ROOT / "src/leeth/data_400m"


def main():
    print("1/4 экспорт fp32 ONNX (во временный файл)...", flush=True)
    tmp_fp32 = export_to_onnx(str(DATA), weights_filename="leeth_best.pt", output_filename="leeth.onnx.tmp")
    print(f"   fp32 готов: {tmp_fp32.name} ({tmp_fp32.stat().st_size / 1e6:.0f} МБ)", flush=True)

    print("2/4 атомарная замена leeth.onnx...", flush=True)
    os.replace(tmp_fp32, DATA / "leeth.onnx")

    print("3/4 квантизация int8 (во временный файл)...", flush=True)
    tmp_int8 = quantize_onnx_model(DATA / "leeth.onnx", output_filename="leeth_int8.onnx.tmp")
    print(f"   int8 готов: {tmp_int8.name} ({tmp_int8.stat().st_size / 1e6:.0f} МБ)", flush=True)
    os.replace(tmp_int8, DATA / "leeth_int8.onnx")
    print("   атомарная замена leeth_int8.onnx выполнена", flush=True)

    print("4/4 верификация выравнивания torch/ONNX/int8...", flush=True)
    metrics = verify_exported_model(str(DATA))
    for k, v in metrics.items():
        print(f"   {k}: {v}", flush=True)
    print("ГОТОВО: рантайм теперь обслуживает новые 42-классовые веса", flush=True)


if __name__ == "__main__":
    main()