"""Accurate compound-family share in the regenerated corpora."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dataset_generator import LeethDatasetGenerator

MARKS = LeethDatasetGenerator.META_PREFIXES + LeethDatasetGenerator.META_SUFFIXES

for name in ["leeth_train.jsonl", "leeth_val.jsonl", "leeth_train_v7backup.jsonl"]:
    p = Path(__file__).resolve().parent.parent / "src/leeth/data" / name
    n_total, n_compound, targets = 0, 0, Counter()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            n_total += 1
            if s["question_type"] == "choice" and any(m in s["state"] for m in MARKS):
                n_compound += 1
                targets[s.get("target_choice")] += 1
    print(f"{name}: {n_total} samples, compound choice-samples: {n_compound} "
          f"({100 * n_compound / max(n_total, 1):.1f}%)")
    if targets:
        print(f"  targets: {dict(targets)}")