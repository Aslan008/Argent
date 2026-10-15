# -*- coding: utf-8 -*-
"""scripts/_kaggle_fp16_base.py — fp16-конвертация mmBERT-базы для заливки на Kaggle.

fp32 1.9 GB -> fp16 962 MB: быстрее заливка, кернел апкастит в fp32 при загрузке
(проверено в battery-треке: расхождение фичей 2.87e-06).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(".argent/models/mmbert-argent")
DST = Path(".argent/kaggle/du_dataset/mmbert-argent-fp16")


def main():
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model = AutoModelForMaskedLM.from_pretrained(str(SRC), dtype=torch.float16)
    DST.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(DST))
    tok = AutoTokenizer.from_pretrained(str(SRC))
    tok.save_pretrained(str(DST))
    size = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file()) / 1e6
    print(f"fp16 база: {DST} ({size:.0f} MB)")


if __name__ == "__main__":
    main()