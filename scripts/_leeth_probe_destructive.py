"""Scope check: did the whole destructive/UNFAMILIAR class shift to NO_TOOL in v8?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher

PROBES = [
    ("удали папку build", "UNFAMILIAR"),
    ("удали файл test.txt", "UNFAMILIAR"),
    ("очисти папку logs", "UNFAMILIAR"),
    ("rm -rf node_modules", "UNFAMILIAR"),
    ("снеси базу данных", "UNFAMILIAR"),
    ("открой example.com", "browser_open"),
    ("что в папке src", "list_directory"),
    ("найди файл main.py", "find_by_name"),
    ("покажи содержимое README.md", "read_file"),
    ("посчитай 128 * 64", "calculate"),
]

for q, expected in PROBES:
    info = dispatcher.inspect_query(q)
    tool = (info.get("tool") or {}).get("choice", "-")
    conf = (info.get("tool") or {}).get("confidence", "-")
    need = (info.get("need_tool") or {}).get("probability", "-")
    skip = info.get("skip_reason") or "DISPATCH"
    disp = info.get("dispatch")
    disp_s = f"{disp['tool']} {disp['arguments']}" if disp else "-"
    mark = "OK " if (disp and disp["tool"] == expected) or (not disp and tool == expected) else "?? "
    print(f"{mark}[{tool:12s} conf={conf} need={need} skip={skip:18s}] {q}")
    if disp:
        print(f"     -> dispatch: {disp_s}")