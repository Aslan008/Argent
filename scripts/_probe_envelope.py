# -*- coding: utf-8 -*-
"""scripts/_probe_envelope.py — изоляция гипотезы конвертов на choice-семье department."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
from src.leeth.primitives import ChoiceQuestion
from _multidomain_battery import Q_DEPARTMENT, DEPT_CHOICES, CASES

leeth.initialize(force_device="cuda")
print("device:", leeth.device, "| классов в маппинге:", len(leeth.class_to_idx), flush=True)

dept_cases = [(s, qs[0][1]) for sp, s, qs in CASES if sp == "support" and qs[0][0] == "department"]


def run(states, label):
    ok = 0
    preds = []
    for text, gold in states:
        q = ChoiceQuestion(id="department", question=Q_DEPARTMENT, choices=DEPT_CHOICES)
        batch = leeth.decide(text, [q])
        pred = batch.results["department"].choice
        preds.append(pred)
        ok += pred == gold
    print(f"{label}: {ok}/{len(states)} = {ok / len(states) * 100:.0f}%  предсказания: {preds[:6]}", flush=True)


def prod_envelope(text):
    normalized = text[:1].lower() + text[1:] if text[:1].isalpha() else text
    return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'


run(dept_cases, "голый текст (как в батарее)      ")
run([(f'Incoming support ticket:\n"{t}"\nClient: ООО Ромашка, plan: Pro', g) for t, g in dept_cases],
    "мой тренировочный конверт          ")
run([(prod_envelope(t), g) for t, g in dept_cases],
    "прод-конверт Argent                ")