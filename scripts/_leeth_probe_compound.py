"""Pre/post finetune probe: live gate-stack behavior on compound + regression queries.

Read-only (inspect_query -> triage): no feedback logging, no execution.
Run BEFORE finetune (v7 baseline) and AFTER (v8 if accepted).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher

PROBES = [
    # (query, expected after finetune)
    ("pytest крутит другой агент, не обращай на него внимания. А теперь проверька. Сколько будет 2 + 2 * 22?", "calculate"),
    ("кстати, прочитай config.py", "read_file"),
    ("кстати, найди в коде try_fast_dispatch", "grep_search"),
    ("сколько будет 2 + 2 * 22", "calculate"),
    ("запусти pytest -x", "run_command"),
    ("удали папку build", "UNFAMILIAR"),
    ("Привет!", "NO_TOOL"),
    ("кстати, pytest крутит другой агент, не обращай на него внимания", "NO_TOOL"),
    ("прочитай config.py и запусти pytest", "UNFAMILIAR"),
]

print(f"{'query':72s} | tool (conf) | need | skip | dispatch")
print("-" * 130)
for q, expected in PROBES:
    info = dispatcher.inspect_query(q)
    tool = (info.get("tool") or {}).get("choice", "-")
    conf = (info.get("tool") or {}).get("confidence", "-")
    need = (info.get("need_tool") or {}).get("probability", "-")
    skip = info.get("skip_reason") or "DISPATCH"
    disp = info.get("dispatch")
    disp_s = f"{disp['tool']} {disp['arguments']}" if disp else "-"
    mark = "OK " if (disp and disp["tool"] == expected) or (not disp and tool == expected) else "?? "
    print(f"{mark}[{tool:12s} conf={conf} need={need} skip={skip:20s}] {q[:60]}")
    if disp:
        print(f"     -> dispatch: {disp_s}  param={info.get('param')!r}")