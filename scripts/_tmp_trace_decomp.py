"""Trace where _try_decomposition bails (DEBUG logging for the dispatcher)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(name)s: %(message)s")
for noisy in ("urllib3", "matplotlib", "transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from src.leeth.dispatcher import dispatcher

d = dispatcher.try_fast_dispatch("что находится в файле main.py и сколько будет 10-5+2?")
print("RESULT:", "None" if d is None else ("CHAIN " + str(len(d.steps)) if d.steps else d.tool_name))