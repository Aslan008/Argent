"""Diagnose: period separator + capitalization effects."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ScoreQuestion, ExtractQuestion, INTENT_COUNT_QUESTION, FIRST_INTENT_QUESTION

decomp_q = [
    ScoreQuestion(id="intent_count", question=INTENT_COUNT_QUESTION, levels=3),
]

print("=== COUNT by separator ===")
for q in [
    "Что в файле main.py. Сколько будет 2-3?",
    "что в файле main.py и сколько будет 2-3?",
    "что в файле main.py. сколько будет 2-3?",
    "прочитай config.py. запусти pytest",
    "прочитай config.py и запусти pytest",
]:
    state = dispatcher._build_state(q)
    batch = dispatcher._runtime.decide(state, decomp_q)
    c = batch.results.get("intent_count")
    print(f"  count={c.level} conf={c.confidence:.2f} | {q}")

print()
print("=== SPAN dispatch by capitalization ===")
for span in [
    "Что в файле main.py",
    "что в файле main.py",
    "что в main.py",
    "Что в main.py",
    "Сколько будет 2-3?",
    "сколько будет 2-3",
]:
    info = dispatcher.inspect_query(span)
    tool = (info.get("tool") or {})
    d = dispatcher.try_fast_dispatch(span, _depth=1)
    print(f"  '{span}' -> {tool.get('choice')} conf={tool.get('confidence'):.3f} | {'None' if d is None else d.tool_name + ' ' + str(d.arguments)}")