"""Verify the shadow JSONL: encoding, parseability, field completeness."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG = Path(".argent/artifacts/arbitrator_shadow.jsonl")
lines = io.open(LOG, encoding="utf-8").read().strip().splitlines()
recs = [json.loads(l) for l in lines]
print(f"{len(recs)} records, all parse OK")
print("sample query:", recs[-1]["query"])
print("fields complete:", all(
    all(k in r for k in ("ts", "query", "tool", "sat", "core", "latency_ms", "gates", "enforce"))
    for r in recs
))
cyrillic_ok = any("прочитай" in r["query"] for r in recs)
print("cyrillic intact (utf-8):", cyrillic_ok)