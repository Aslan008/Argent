"""Shadow-run: arbitrate probe queries + verify the dispatcher hook end-to-end.

Read-only for the system: dispatch() returns the dispatch spec without executing
tools; the arbitrator never blocks. Writes one feedback decision record per
dispatch() call (normal system behavior).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")

from src.leeth.arbitrator import get_arbitrator
from src.leeth.dispatcher import LeethDispatcher

arb = get_arbitrator()
dispatcher = LeethDispatcher.get_instance()

CASES = [
    # (query, expectation)
    ("прочитай config.py", "in-dist: gates dispatch, arb SAT"),
    ("сколько будет 512 / 8", "in-dist: gates dispatch, arb SAT"),
    ("покажи содержимое папки src", "in-dist: gates dispatch, arb SAT"),
    ("запусти pytest -x", "in-dist run_command: gates dispatch, arb SAT"),
    ("продолжай", "chat: gates delegate"),
    ("отрефактори этот Haskell-модуль под newtype", "OOD: gates delegate"),
    ("настрой nginx reverse proxy для двух бэкендов", "OOD: gates delegate"),
    ("скомпилируй Main.go и запусти получившийся бинарник", "OOD: gates delegate"),
]

print("=== Shadow arbitration over probe queries ===")
for q, expectation in CASES:
    info = dispatcher.inspect_query(q)
    skip = info.get("skip_reason") or "DISPATCH"
    tool = info.get("tool", {}).get("choice")
    need = info.get("need_tool", {}).get("probability")
    param_raw = info.get("param", "")
    param = param_raw.get("text", "") if isinstance(param_raw, dict) else (param_raw or "")
    res = arb.arbitrate(
        tool=tool or "NO_TOOL",
        need_tool=bool(need is not None and need >= 0.5),
        param_text=param or "",
        query=q,
        context={"gates": skip},
    )
    # Dangerous disagreement is ONLY: gates dispatch + arb UNSAT (would-block).
    # Gates-skip + arb SAT is expected: the base encodes consistency, not policy.
    would_block = (skip == "DISPATCH" and not res.sat)
    agree = "WOULD-BLOCK!" if would_block else "OK"
    print(f"[{skip:18s}] arb={'SAT ' if res.sat else 'UNSAT'} core={res.core or '-'} "
          f"({res.latency_ms:.1f}ms) {agree} :: {q}")
    print(f"    expectation: {expectation}")

print("\n=== Dispatcher hook end-to-end (harmless read-only dispatch) ===")
result = dispatcher.try_fast_dispatch("прочитай requirements.txt")
print(f"dispatch result: {result.tool_name if result else None}")
print(f"arb stats: {arb.stats}")
print("SHADOW RUN OK")