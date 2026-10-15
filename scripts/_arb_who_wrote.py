"""Check shadow log: whose records are these (test runs vs live traffic)?"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG = Path(".argent/artifacts/arbitrator_shadow.jsonl")
recs = [json.loads(l) for l in io.open(LOG, encoding="utf-8").read().strip().splitlines()]

by_minute = {}
for r in recs:
    by_minute[r["ts"][:16]] = by_minute.get(r["ts"][:16], 0) + 1

print(f"total: {len(recs)} records")
print("records per minute:")
for minute in sorted(by_minute):
    print(f"  {minute}  x{by_minute[minute]}")
print("\nlast 3 queries:")
for r in recs[-3:]:
    print(f"  [{r['ts']}] gates={r['gates']} tool={r['tool']} :: {r['query'][:60]}")