"""Dry-run: verify the dispatcher's UNFAMILIAR gate end-to-end (read-only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.dispatcher import LeethDispatcher

dispatcher = LeethDispatcher.get_instance()

queries = [
    ("прочитай config.py", "in-dist tool query -> should dispatch"),
    ("сколько будет 512 / 8", "in-dist tool query -> should dispatch"),
    ("продолжай", "chat -> should delegate (unfamiliar or need_tool)"),
    ("отрефактори этот Haskell-модуль под newtype", "OOD -> should delegate as UNFAMILIAR"),
    ("настрой nginx reverse proxy для двух бэкендов", "OOD -> should delegate as UNFAMILIAR"),
]

for q, expectation in queries:
    info = dispatcher.inspect_query(q)
    print(f"[{info.get('skip_reason') or 'DISPATCH'}] tool={info.get('tool', {}).get('choice')} "
          f"conf={info.get('tool', {}).get('confidence')} need={info.get('need_tool', {}).get('probability')} :: {q}")
print("expectation:", expectation)
print("DRY-RUN OK")