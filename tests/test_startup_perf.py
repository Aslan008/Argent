"""Guard: Argent's cold start must not drag in the heavy ML stack.

Importing sentence-transformers at module top pulled torch+transformers into
every startup (~25s). They are now lazy-imported only when deep-research
reranking actually runs. This test fails if an eager import creeps back in.
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def test_startup_does_not_import_heavy_ml_stack():
    code = (
        "import sys, main\n"
        "heavy = [m for m in ('sentence_transformers', 'torch', 'transformers') "
        "if m in sys.modules]\n"
        "print('HEAVY:' + ','.join(heavy))\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, (
        "Heavy ML modules imported at startup (should be lazy):\n"
        f"{r.stdout}\n{r.stderr}"
    )
