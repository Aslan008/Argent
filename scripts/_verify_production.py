"""Verify the PRODUCTION path uses the 400M model: singleton + dispatcher + config."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
print(f"model_dir: {leeth.model_dir}")
ok = leeth.initialize()
print(f"init: {ok}, device={leeth.device}", flush=True)
if not ok:
    print(f"init error: {leeth._init_error}")
    sys.exit(1)

from src.leeth.dispatcher import LeethDispatcher
d = LeethDispatcher.get_instance()

from config import get_fast_dispatch_backend
print(f"fast_dispatch_backend: {get_fast_dispatch_backend()}", flush=True)

# Live production-path smoke via try_fast_dispatch (the real agent-turn entry)
# Expectations follow the dispatcher design:
#   - routine tool + conf >= gate -> FastDispatchResult
#   - NO_TOOL (need_tool=False)   -> None (chit-chat goes to the main agent)
#   - UNFAMILIAR                  -> None (explicit delegation to the LLM)
#   - low-choice-confidence       -> None (fallback to the LLM)
CASES = [
    ("прочитай main.py", "read_file"),  # dispatcher pipeline: conf 0.93 >= gate 0.80
    ("сколько будет 2+2*22", "calculate"),
    ("привет, как дела?", None),  # NO_TOOL -> main agent handles
    ("настрой nginx reverse proxy", None),  # UNFAMILIAR -> delegate
    ("запусти pytest", "run_command"),
]
n_pass = 0
for i, (query, expected) in enumerate(CASES):
    r = d.try_fast_dispatch(query)
    got = r.tool_name if r else None
    passed = got == expected
    n_pass += passed
    detail = f"got={got!r}" + (f" conf={getattr(r, 'confidence', 0):.2f}" if r else " (None)")
    print(f"[{'PASS' if passed else 'FAIL'}] {query!r} -> {detail} | expected={expected!r}", flush=True)

print(f"\n{n_pass}/{len(CASES)} production-path dispatches correct", flush=True)