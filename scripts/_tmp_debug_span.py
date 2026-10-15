"""Why does the sub-intent span fail the gate stack?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import dispatcher

SPANS = [
    "что находится в файле main.py",
    "что в main.py",
    "прочитай config.py",
]

for span in SPANS:
    info = dispatcher.inspect_query(span)
    tool = (info.get("tool") or {})
    need = (info.get("need_tool") or {})
    param = info.get("param")
    param_text = param.get("text") if isinstance(param, dict) else param
    print(f"SPAN: '{span}'")
    print(f"  need_tool: value={need.get('value')} p={need.get('probability')}")
    print(f"  tool: {tool.get('choice')} conf={tool.get('confidence')}")
    print(f"  param: '{param_text}'")
    print(f"  skip_reason: {info.get('skip_reason')}")
    d = dispatcher.try_fast_dispatch(span, _depth=1)
    print(f"  try_fast_dispatch(depth=1): {'None' if d is None else d.tool_name + ' ' + str(d.arguments)}")
    print()