"""Prints the extracted param for a specific query (debug)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth import leeth
from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ExtractQuestion

query = "выполни shutdown /s /t 0"
state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'
batch = leeth.decide(state, [ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)")])
print("extract head raw:", repr(batch.results["param"].text))

disp = dispatcher.try_fast_dispatch(query)
print("dispatcher:", disp.tool_name if disp else None, "| args:", disp.arguments if disp else None)