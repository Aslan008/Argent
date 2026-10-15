"""Training batch #3 — targeted at the 4 problem queries found after cycle 3."""

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
    "посчитай 128 * 64",                        # expect calculate proposal (low conf) -> partial
    "сколько будет 512 / 8",                    # expect need_tool_false skip -> partial
    "запусти python scripts/leeth_review.py stats",  # expect safety skip -> accept (safe command)
    "покажи что в README",                      # expect empty_param skip -> partial (param=README)
]


def main() -> None:
    for q in QUERIES:
        r = dispatcher.try_fast_dispatch(q)
        print(("DISPATCH " + r.tool_name) if r else "SKIP", "|", q)


if __name__ == "__main__":
    main()