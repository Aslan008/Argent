"""Generate the compound-meta family standalone, validate it, print stats."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dataset_generator import LeethDatasetGenerator, LeethSample

gen = LeethDatasetGenerator(seed=42)
samples = gen.generate_compound_meta_samples()

# ── Validation ────────────────────────────────────────────────────────────────
# Every positive extract target must be a substring of its wrapped query.
bad = []
for s in samples:
    if s.question_type == "extract" and s.target_span_text:
        query = s.state.split('"')[1] if '"' in s.state else ""
        if s.target_span_text.strip().lower() not in query.lower():
            bad.append((s.target_span_text, s.state[:120]))
print(f"param-in-query violations: {len(bad)}")
for p, st in bad[:5]:
    print(f"  BAD: param={p!r} state={st!r}")

# ── Stats ─────────────────────────────────────────────────────────────────────
by_type = Counter(s.question_type for s in samples)
by_choice = Counter(s.target_choice for s in samples if s.question_type == "choice")
noul = Counter(round(s.target_noul) for s in samples if s.question_type == "noul")
print(f"total samples: {len(samples)}")
print(f"by question_type: {dict(by_type)}")
print(f"choice targets: {dict(by_choice)}")
print(f"noul balance (0=нет, 1=да): {dict(noul)}")

# ── Preview: a few wrapped positives ─────────────────────────────────────────
print("\n--- примеры обёрнутых запросов ---")
seen = set()
for s in samples:
    if s.question_type == "choice" and s.target_choice not in seen:
        q = s.state.split('"')[1]
        print(f"  [{s.target_choice:14s}] {q}")
        seen.add(s.target_choice)

# ── Save standalone preview file ──────────────────────────────────────────────
out = Path(__file__).resolve().parent.parent / "src/leeth/data/leeth_compound_preview.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps({
            "state": s.state, "question_type": s.question_type, "question": s.question,
            "choices": s.choices, "target_choice": s.target_choice,
            "target_noul": s.target_noul, "target_span_text": s.target_span_text,
        }, ensure_ascii=False) + "\n")
print(f"\nsaved preview: {out}")