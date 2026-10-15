"""Regenerate the finetune corpora (train/val) and the full dataset with the
compound-meta family included. Backs up the v7-era files first."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dataset_generator import LeethDatasetGenerator

DATA = Path(__file__).resolve().parent.parent / "src/leeth/data"

# ── Backup v7-era corpora ─────────────────────────────────────────────────────
for name in ["leeth_train.jsonl", "leeth_val.jsonl"]:
    src = Path(__file__).resolve().parent.parent / "src/leeth/data" / name
    dst = src.with_name(name.replace(".jsonl", "_v7backup.jsonl"))
    if not dst.exists():
        shutil.copy(src, dst)
        print(f"backup: {dst.name}")

# ── Regenerate with the same seeds/counts as train_leeth uses ────────────────
train_gen = LeethDatasetGenerator(seed=42)
n_train = train_gen.generate_and_save(
    Path(__file__).resolve().parent.parent / "src/leeth/data/leeth_train.jsonl", 5000)
val_gen = LeethDatasetGenerator(seed=9999)
n_val = val_gen.generate_and_save(
    Path(__file__).resolve().parent.parent / "src/leeth/data/leeth_val.jsonl", 500)

# ── Verify compound presence ──────────────────────────────────────────────────
def compound_share(path: Path) -> tuple[int, Counter]:
    """Count samples whose state contains a known meta prefix (compound family)."""
    meta_marks = ("pytest крутит другой агент", "параллельно идёт сборка",
                  "не обращай внимания на опечатки", "кстати,", "короче,",
                  "слушай,", "Заранее спасибо", "Не объясняй, просто сделай")
    n, targets = 0, Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["question_type"] == "choice" and any(m in s["state"] for m in meta_marks):
                n += 1
                targets[s.get("target_choice")] += 1
    return n, targets

for name, total in [("leeth_train.jsonl", n_train), ("leeth_val.jsonl", n_val)]:
    p = Path(__file__).resolve().parent.parent / "src/leeth/data" / name
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    n_compound, targets = compound_share(Path(str(p)))
    print(f"{name}: {len(lines)} samples, compound-family ~{n_compound if (n_compound := n_compound) else n_compound}")
    print(f"  compound targets: {dict(targets)}")