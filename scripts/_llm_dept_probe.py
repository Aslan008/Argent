# -*- coding: utf-8 -*-
"""scripts/_llm_dept_probe.py — проверка md6 на LLM-сгенерированных department-кейсах (80 шт)."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import leeth
from src.leeth.primitives import ChoiceQuestion
from _multidomain_battery import Q_DEPARTMENT, DEPT_CHOICES, prod_envelope

leeth.initialize(force_device="cuda")

cases = []
for f in ["_llm_dept_batch1.json", "_llm_dept_batch2.json"]:
    cases.extend(json.loads((ROOT / "scripts" / f).read_text(encoding="utf-8")))

q = ChoiceQuestion(id="department", question=Q_DEPARTMENT, choices=DEPT_CHOICES)
ok = 0
per_dept = {}
preds = Counter()
for c in cases:
    b = leeth.decide(prod_envelope(c["text"]), [q])
    r = b.results["department"]
    d = per_dept.setdefault(c["dept"], {"n": 0, "ok": 0})
    d["n"] += 1
    d["ok"] += r.choice == c["dept"]
    ok += r.choice == c["dept"]
    preds[r.choice] += 1

print(f"LLM-кейсы department: {ok}/{len(cases)} = {ok / len(cases) * 100:.0f}%")
for dept, d in sorted(per_dept.items()):
    print(f"  {dept:<10}: {d['ok']}/{d['n']}")
print(f"предсказания: {dict(preds)}")