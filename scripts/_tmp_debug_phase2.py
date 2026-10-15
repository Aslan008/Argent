"""Debug the phase-2 decomposition triage: what do the new heads say?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.runtime import leeth
from src.leeth.primitives import (
    ScoreQuestion, ExtractQuestion, INTENT_COUNT_QUESTION, FIRST_INTENT_QUESTION,
)
from src.leeth.dispatcher import dispatcher

QUERIES = [
    "что находится в файле main.py и сколько будет 10-5+2?",
    "прочитай config.py и запусти pytest",
    "настрой nginx reverse proxy и скажи сколько будет 2+2",
    "удали папку build",
    "сколько будет 2 + 2 * 22",
]

print("runtime available:", leeth.is_available, "| device:", leeth.device)
q = [
    ScoreQuestion(id="intent_count", question=INTENT_COUNT_QUESTION, levels=3),
    ExtractQuestion(id="first_intent", question=FIRST_INTENT_QUESTION, target_type="task"),
]
for query in QUERIES:
    state = dispatcher._build_state(query)
    batch = leeth.decide(state, q)
    c = batch.results.get("intent_count")
    s = batch.results.get("first_intent")
    print(f"Q: {query[:60]}")
    print(f"  count: level={getattr(c, 'level', None)} score={getattr(c, 'score', None):.1f} conf={getattr(c, 'confidence', None):.3f} probs={getattr(c, 'probabilities', None)}")
    print(f"  span:  '{getattr(s, 'text', None)}' conf={getattr(s, 'confidence', None):.3f}")
    print()