"""Isolate: which calculate phrasings trip the UNFAMILIAR verdict?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher

QUERIES = [
    "сколько будет 10-5+2?",
    "сколько будет 10 - 5 + 2?",
    "сколько будет 2 + 2 * 22?",
    "посчитай 10-5+2",
    "сколько будет 128 * 64?",
    "сколько будет 1024 / 8?",
]

for q in QUERIES:
    info = dispatcher.inspect_query(q)
    tool = (info.get("tool") or {})
    d = dispatcher.try_fast_dispatch(q, _depth=1)
    print(f"'{q}' -> {tool.get('choice')} conf={tool.get('confidence'):.3f} | dispatch={'None' if d is None else d.tool_name}")