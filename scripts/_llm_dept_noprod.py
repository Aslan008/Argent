# -*- coding: utf-8 -*-
"""scripts/_llm_dept_noprod.py — LLM-кейсы через ГОЛЫЙ текст (без прод-конверта)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import leeth
from src.leeth.primitives import ChoiceQuestion
from _multidomain_battery import Q_DEPARTMENT, DEPT_CHOICES

leeth.initialize(force_device="cuda")

cases = []
for f in ["_llm_dept_batch1.json", "_llm_dept_batch2.json"]:
    cases.extend(json.loads((ROOT / "scripts" / f).read_text(encoding="utf-8")))

q = ChoiceQuestion(id="department", question=Q_DEPARTMENT, choices=DEPT_CHOICES)
ok = 0
for c in cases:
    b = leeth.decide(c["text"], [q])
    r = b.results["department"]
    ok += r.choice == c["dept"]
print(f"LLM-кейсы department (голый текст): {ok}/{len(cases)} = {ok / len(cases) * 100:.0f}%")