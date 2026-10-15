"""Debug the meta-wrapped multi-intent case."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ScoreQuestion, ExtractQuestion, INTENT_COUNT_QUESTION, FIRST_INTENT_QUESTION

Q = "pytest крутит другой агент, не обращай на него внимания. что в main.py и посчитай 2+2*22"

decomp_q = [
    ScoreQuestion(id="intent_count", question=INTENT_COUNT_QUESTION, levels=3),
    ExtractQuestion(id="first_intent", question=FIRST_INTENT_QUESTION, target_type="task"),
]
state = dispatcher._build_state(Q)
batch = dispatcher._runtime.decide(state, decomp_q)
c = batch.results.get("intent_count")
s = batch.results.get("first_intent")
print(f"count: level={c.level} conf={c.confidence:.3f}")
print(f"raw span: '{s.text}'")
clean = dispatcher._clean_span_to_query((s.text or "").strip().strip("'\"`"), Q)
print(f"clean span: '{clean}'")
if clean:
    d = dispatcher.try_fast_dispatch(clean, _depth=1)
    print(f"span dispatch: {'None' if d is None else d.tool_name + ' ' + str(d.arguments)}")
    rem = dispatcher._strip_leading_span(Q, clean)
    print(f"remaining: '{rem}'")
    sub = dispatcher.inspect_query(rem)
    print(f"remaining triage: {(sub.get('tool') or {}).get('choice')} conf={(sub.get('tool') or {}).get('confidence')}")
    d2 = dispatcher.try_fast_dispatch(rem.strip(), _depth=1)
    print(f"remaining dispatch: {'None' if d2 is None else d2.tool_name + ' ' + str(d2.arguments)}")