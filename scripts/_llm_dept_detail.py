# -*- coding: utf-8 -*-
"""scripts/_llm_dept_detail.py — детальный разбор предсказаний на LLM-кейсах."""
import json
import sys
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
for c in cases:
    b = leeth.decide(prod_envelope(c["text"]), [q])
    r = b.results["department"]
    top2 = sorted(r.probabilities.items(), key=lambda kv: -kv[1])[:2]
    mark = "OK  " if r.choice == c["dept"] else "FAIL"
    print(f"{mark} gold={c['dept']:<10} pred={r.choice:<10} top2={[(k, round(v, 3)) for k, v in top2]} | {c['text'][:70]}")