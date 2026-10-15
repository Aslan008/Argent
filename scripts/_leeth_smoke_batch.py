"""
scripts/_leeth_smoke_batch.py — Smoke-test batch for the Leeth feedback loop.

Runs a set of realistic queries through the fast-path dispatcher ONLY to
generate decision records for review. Nothing is executed: try_fast_dispatch
only proposes; execution happens solely in agent.py's turn loop.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.leeth.dispatcher import dispatcher  # noqa: E402

QUERIES = [
    "прочитай requirements.txt",
    "запусти pytest tests/ -x",
    "сколько будет 512 / 8",
    "найди файл tokenizer.py",
    "покажи содержимое папки src",
    "найди все упоминания log_decision в коде",
    "а что ты умеешь?",
    "удали папку build",
    "объясни как работает рантайм Leeth",
    "открой config.py",
    "посчитай sqrt(2)",
    "выполни git diff HEAD~1",
]


def main() -> None:
    for q in QUERIES:
        r = dispatcher.try_fast_dispatch(q)
        print(("DISPATCH " + r.tool_name) if r else "SKIP", "|", q)


if __name__ == "__main__":
    main()