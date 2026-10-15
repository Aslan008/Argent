"""Surgical purge of poisoned feedback records.

Drops ONLY: verdict=reject, no correction.tool, destructive query.
These are auto-logged LLM verdicts from live sessions that teach
need=0/NO_TOOL for destructive queries (the self-reinforcing loop).
The correct label for destructive queries is UNFAMILIAR (need=1.0).
"""
import json
import shutil
from pathlib import Path

FB = Path("src/leeth/data/leeth_feedback.jsonl")
BACKUP = Path("src/leeth/data/leeth_feedback_v10backup.jsonl")

DESTRUCTIVE_MARKS = ("удали", "очисти", "rm -rf", "снеси", "сотри",
                     "форматни", "убей", "стереть")

shutil.copy(FB, BACKUP)
kept, dropped = [], []
with open(FB, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        q = str(r.get("query", "")).lower()
        is_destructive = any(m in q for m in DESTRUCTIVE_MARKS)
        is_poisoned = (r.get("verdict") == "reject"
                       and not (r.get("correction") or {}).get("tool")
                       and is_destructive)
        if is_poisoned:
            dropped.append(f"{r.get('query', '')[:50]} | {r.get('reason', '')[:60]}")
        else:
            kept.append(line)

with open(FB, "w", encoding="utf-8") as f:
    f.writelines(kept)

print(f"dropped {len(dropped)} poisoned records:")
for d in dropped:
    print(f"  - {d}")
print(f"kept {len(kept)} records (backup: {BACKUP.name})")