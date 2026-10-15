"""Step-by-step trace of the decomposition chain loop."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ScoreQuestion, ExtractQuestion, INTENT_COUNT_QUESTION, FIRST_INTENT_QUESTION

Q = "что находится в файле main.py и сколько будет 10-5+2?"

# phase-2 directly
decomp_q = [
    ScoreQuestion(id="intent_count", question=INTENT_COUNT_QUESTION, levels=3),
    ExtractQuestion(id="first_intent", question=FIRST_INTENT_QUESTION, target_type="task"),
]
state = dispatcher._build_state(Q)
batch = dispatcher._runtime.decide(state, decomp_q)
count_res = batch.results.get("intent_count")
span_res = batch.results.get("first_intent")
print(f"count: level={count_res.level} conf={count_res.confidence:.3f}")
raw_span = (span_res.text or "").strip().strip("'\"`")
first_span = dispatcher._clean_span_to_query(raw_span, Q)
print(f"raw_span: '{raw_span}'")
print(f"clean_span: '{first_span}'")

# simulate the loop manually
steps = []
remaining = Q
span = first_span
print(f"\n-- depth 0: span='{span}'")
d0 = dispatcher.try_fast_dispatch(span, _depth=1)
print(f"   dispatch: {'None' if d0 is None else d0.tool_name}")
if d0:
    steps.append(d0)
    remaining = dispatcher._strip_leading_span(remaining, span)
    print(f"   remaining: '{remaining}'")

    sub = dispatcher.inspect_query(remaining)
    sub_tool = (sub.get("tool") or {}).get("choice", "")
    print(f"\n-- depth 1: remaining triage tool={sub_tool} conf={(sub.get('tool') or {}).get('confidence')}")
    span2 = remaining.strip()
    print(f"   span='{span2}'")
    d2 = dispatcher.try_fast_dispatch(span2, _depth=1)
    print(f"   dispatch: {'None' if d2 is None else d2.tool_name + ' ' + str(d2.arguments)}")
    if d2:
        steps.append(d2)

print(f"\nTOTAL STEPS: {len(steps)}")
for s in steps:
    print(f"  {s.tool_name} {s.arguments}")