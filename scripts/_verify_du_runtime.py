# -*- coding: utf-8 -*-
"""scripts/_verify_du_runtime.py — проверка dept/urgency через прод-runtime.

Прогоняет рукописную батарею (_multidomain_battery.CASES) через leeth.decide
(тот же путь, что прод-диспетчер) и сверяет с torch-оценкой голов
(dept 100%, urgency 84.6%) и routing-батареей 91/109.
"""
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
from src.leeth.runtime import (
    MMBERT_DEPT_CLASSES,
    MMBERT_DEPT_QUESTION,
    MMBERT_URGENCY_QUESTION,
)

spec = importlib.util.spec_from_file_location("mdbattery", "scripts/_multidomain_battery.py")
mdb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mdb)

from src.leeth.primitives import ChoiceQuestion, ScoreQuestion

if not leeth.initialize():
    print(f"init error: {leeth._init_error}")
    sys.exit(1)
print(f"mmbert_active={leeth.mmbert_active} du_available={leeth.mmbert_du_available} device={leeth.device}")

dept_cases = [(st, next(g for q, g in qs if q == "department"))
              for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
urg_cases = [(st, next(g for q, g in qs if q == "urgency"))
             for _, st, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]

# ── dept через runtime ───────────────────────────────────────────────────────
ok = 0
t0 = time.perf_counter()
for state, gold in dept_cases:
    batch = leeth.decide(mdb.prod_envelope(state), [
        ChoiceQuestion(id="department", question=MMBERT_DEPT_QUESTION, choices=list(MMBERT_DEPT_CLASSES))])
    res = batch.results["department"]
    hit = res.choice == gold
    ok += hit
    mark = "OK " if hit else "MISS"
    print(f"[{mark}] {state[:55]!r} gold={gold} pred={res.choice} conf={res.confidence:.2f}")
dept_ms = (time.perf_counter() - t0) * 1000 / len(dept_cases)
print(f"DEPT runtime: {ok}/{len(dept_cases)} ({ok / len(dept_cases):.1%}), {dept_ms:.1f} мс/кейс")

# ── urgency через runtime ────────────────────────────────────────────────────
ok = 0
t0 = time.perf_counter()
for state, gold in urg_cases:
    batch = leeth.decide(mdb.prod_envelope(state), [
        ScoreQuestion(id="urgency", question=MMBERT_URGENCY_QUESTION, levels=3, rubric=mdb.URGENCY_RUBRIC)])
    res = batch.results["urgency"]
    hit = res.level == int(gold)
    ok += hit
    mark = "OK " if hit else "MISS"
    print(f"[{mark}] {state[:55]!r} gold={gold} pred={res.level} conf={max(res.probabilities.values()):.2f} score={res.score:.0f}")
urg_ms = (time.perf_counter() - t0) * 1000 / len(urg_cases)
print(f"URGENCY runtime: {ok}/{len(urg_cases)} ({ok / len(urg_cases):.1%}), {urg_ms:.1f} мс/кейс")

# ── смешанный батч: dept + urgency в одном decide ────────────────────────────
state = dept_cases[0][0]
batch = leeth.decide(mdb.prod_envelope(state), [
    ChoiceQuestion(id="department", question=MMBERT_DEPT_QUESTION, choices=list(MMBERT_DEPT_CLASSES)),
    ScoreQuestion(id="urgency", question=MMBERT_URGENCY_QUESTION, levels=3, rubric=mdb.URGENCY_RUBRIC),
])
print(f"mixed batch: dept={batch.results['department'].choice} urgency_level={batch.results['urgency'].level} "
      f"({batch.elapsed_ms:.1f} мс, device={batch.device})")

# ── routing-регресс: 3 кейса прод-батареи ────────────────────────────────────
import json
routing = json.loads(Path("scripts/prod_battery.json").read_text(encoding="utf-8"))["cases"][:5]
tool_classes = list(json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8")).keys())
r_ok = 0
for c in routing:
    batch = leeth.decide(mdb.prod_envelope(c["query"]), [
        ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?",
                       choices=tool_classes)])
    pred = batch.results["tool"].choice
    r_ok += pred == c["expected_tool"]
    print(f"[{'OK ' if pred == c['expected_tool'] else 'MISS'}] {c['query'][:50]!r} exp={c['expected_tool']} pred={pred}")
print(f"ROUTING smoke: {r_ok}/5")