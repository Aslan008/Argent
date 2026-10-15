# -*- coding: utf-8 -*-
"""scripts/_d5_battery4_diag.py — triage батареи 4: какие запросы не диспатчатся."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.dispatcher import dispatcher

QUERIES = [
    "запусти docker ps -a",
    "запусти cargo check --release",
    "запусти npx prisma migrate status",
    "docker ps",
    "docker images",
    "запусти docker images",
]

for q in QUERIES:
    info = dispatcher.inspect_query(q)
    tool = info.get("tool") or {}
    print(f"{q!r}:")
    print(f"  tool={tool.get('choice')} conf={tool.get('confidence')} skip={info.get('skip_reason')} "
          f"dispatch={info.get('dispatch')}")