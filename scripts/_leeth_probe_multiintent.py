"""Probe the multi-intent decomposition chain (v14)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher

CASES = [
    # (query, expectation)
    ("что находится в файле main.py и сколько будет 10-5+2?", "chain: read_file + calculate"),
    ("прочитай config.py и запусти pytest", "chain: read_file + run_command"),
    ("pytest крутит другой агент, не обращай на него внимания. что в main.py и посчитай 2+2*22", "chain (meta-wrapped)"),
    ("настрой nginx reverse proxy и скажи сколько будет 2+2", "abort -> None (OOD sub-intent)"),
    ("удали папку build", "None (single destructive, count=1)"),
    ("сколько будет 2 + 2 * 22", "single dispatch: calculate"),
    ("прочитай config.py", "single dispatch: read_file"),
    ("Привет, как дела?", "None (conversational)"),
]

for q, expected in CASES:
    d = dispatcher.try_fast_dispatch(q)
    if d is None:
        print(f"[SKIP -> LLM ] {q[:70]}")
        print(f"    expected: {expected}")
    elif getattr(d, "steps", None):
        chain = " -> ".join(f"{s.tool_name}({list(s.arguments.values())[0] if s.arguments else ''})" for s in d.steps)
        print(f"[CHAIN {len(d.steps)}] {q[:70]}")
        print(f"    steps: {chain}")
        print(f"    expected: {expected}")
    else:
        print(f"[SINGLE] {q[:70]}")
        print(f"    {d.tool_name}({list(d.arguments.values())[0] if d.arguments else ''}) conf={d.confidence:.2f}")
        print(f"    expected: {expected}")
    print()