# -*- coding: utf-8 -*-
"""scripts/_flywheel_unique.py — уникальные запросы pending-маховика для ревью."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import feedback as fb

recs = fb.load_records()
pend = [r for r in recs if not r.get("verdict")]
valuable = [r for r in pend
            if r.get("dispatch") or r.get("skip_reason") in ("unfamiliar", "low_tool_confidence")]
seen: dict = {}
for r in valuable:
    q = (r.get("query") or "").strip()
    key = q.lower()
    if key not in seen:
        seen[key] = {
            "id": r["id"], "q": q,
            "skip": r.get("skip_reason") or "DISPATCH",
            "tool": (r.get("proposal") or {}).get("tool", {}).get("choice", "?"),
            "conf": (r.get("proposal") or {}).get("tool", {}).get("confidence", 0),
            "n": 0,
        }
    seen[key]["n"] += 1

print(f"уникальных запросов: {len(seen)} (записей: {len(valuable)})")
for s in sorted(seen.values(), key=lambda x: -x["n"]):
    print(f"{s['n']}x|{s['id']}|{s['skip']}|{s['tool']}|{s['conf']:.2f}|{s['q'][:80]}")