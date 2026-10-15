# -*- coding: utf-8 -*-
"""scripts/_d3_legacy_confident.py — какие battery-запросы LEGACY диспетчит с conf >= 0.80."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts._d3_gate_compare import LegacyBackend, build_state, gate_decision

battery = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))
legacy = LegacyBackend()
for c in battery["cases"]:
    if c["expected_tool"] not in ("read_file", "run_command", "calculate"):
        continue
    state = build_state(c["query"])
    noul_p, probs = legacy.run_pair(state)
    reason, tool, conf = gate_decision(noul_p, probs)
    if reason == "dispatch":
        print(f"[{c['expected_tool']:12s}] conf={conf:.2f}  {c['query'][:70]}")