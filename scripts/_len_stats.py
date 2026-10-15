# -*- coding: utf-8 -*-
"""scripts/_len_stats.py — распределение длин токенов в 400m-датасете."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.tokenizer import LeethTokenizer

DATA = ROOT / "src/leeth/data_400m"
tok = LeethTokenizer.load(DATA)
lens = []
with open(DATA / "leeth_400m_train.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        s = json.loads(line)
        text = f'[STATE] {s["state"]} [SEP] [QUESTION] [{s["question_type"].upper()}] {s["question"]}'
        lens.append(len(tok.encode(text, max_length=1024, padding=False, truncation=True, add_special_tokens=True)))
lens.sort()
print(f"n={len(lens)} median={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} p99={lens[int(len(lens)*0.99)]} max={lens[-1]}")