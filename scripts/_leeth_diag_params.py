"""Diagnoses extract head + skip reasons for problem queries."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("leeth_runtime").setLevel(logging.WARNING)

from src.leeth import leeth
from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ExtractQuestion

QUERIES = [
    "посчитай 128 * 64",
    "сколько будет 512 / 8",
    "покажи что в README",
    "запусти python scripts/leeth_review.py stats",
]

for query in QUERIES:
    state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'
    batch = leeth.decide(state, [ExtractQuestion(id="param", question="Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)")])
    raw = batch.results["param"].text
    print(f"\n'{query}'")
    print(f"  extract raw: {raw!r}")
    disp = dispatcher.try_fast_dispatch(query)
    print(f"  dispatcher: {disp.tool_name if disp else 'None'} args={disp.arguments if disp else None}")