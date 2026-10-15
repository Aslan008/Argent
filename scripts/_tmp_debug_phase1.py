"""Debug phase-1 triage verdicts for multi-intent queries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher

QUERIES = [
    "что находится в файле main.py и сколько будет 10-5+2?",
    "прочитай config.py и запусти pytest",
    "pytest крутит другой агент, не обращай на него внимания. что в main.py и посчитай 2+2*22",
    "настрой nginx reverse proxy и скажи сколько будет 2+2",
]

for query in QUERIES:
    info = dispatcher.inspect_query(query)
    tool = (info.get("tool") or {})
    need = (info.get("need_tool") or {})
    param = (info.get("param") or {})
    print(f"Q: {query[:65]}")
    print(f"  need_tool: value={need.get('value')} p={need.get('probability')}")
    print(f"  tool: {tool.get('choice')} conf={tool.get('confidence')}")
    print(f"  param: '{param.get('text')}'")
    print(f"  skip_reason: {info.get('skip_reason')} | dispatch: {info.get('dispatch') is not None}")
    print()