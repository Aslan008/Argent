"""Measure REAL production fast-path coverage: canonical queries through try_fast_dispatch."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import leeth
ok = leeth.initialize()
if not ok:
    print(f"init error: {leeth._init_error}")
    sys.exit(1)

from src.leeth.dispatcher import LeethDispatcher
d = LeethDispatcher.get_instance()

# expected=None means "should fall back to the LLM" (chit-chat / unfamiliar / low conf)
QUERIES = [
    ("прочитай main.py", "read_file"),
    ("покажи содержимое config.py", "read_file"),
    ("сколько будет 2+2*22", "calculate"),
    ("запусти pytest", "run_command"),
    ("выполни git status", "run_command"),
    ("найди все упоминания TODO", "grep_search"),
    ("открой сайт example.com", None),
    ("привет, как дела?", None),
    ("спасибо, большое!", None),
    ("настрой nginx reverse proxy", None),
    ("спроектируй архитектуру микросервисов", None),
    ("создай файл notes.md с заготовкой", "write_file"),
    ("замени функцию load в src/agent.py", "replace_in_file"),
]

n_fast = 0
n_ok = 0
for q_text, expected in QUERIES:
    r = d.try_fast_dispatch(q_text)
    got = r.tool_name if r else None
    conf = getattr(r, "confidence", 0) if r else 0
    fast = got is not None
    n_fast += fast
    ok_flag = (got == expected)
    n_ok += ok_flag
    mark = "FAST" if fast else "fallback"
    verdict = "OK" if ok_flag else "MISS"
    print(f"[{mark:>8}] {q_text:<42} got={got!s:<14} conf={conf:.2f} expected={expected!s:<14} {verdict}", flush=True)

print(f"\nfast-path coverage: {n_fast}/{len(QUERIES)} dispatched, {n_ok}/{len(QUERIES)} match expected routing", flush=True)