"""The one-minute subset: contracts, semantics, docs. Run it between edits.

The full suite is about thirteen minutes here. Knowing that price changed how I worked:
I batched changes and got thirty-seven failures at once, which is harder to read than
three waves of twelve. This runs the suites that answer "did I break the shape of the
tool surface" in around a minute; the full run stays for the moment before a commit.

    .venv\\Scripts\\python.exe scripts/test_fast.py
    .venv\\Scripts\\python.exe scripts/test_fast.py -k permissions    # any pytest args

Exit code is pytest's, so it can gate a script.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Chosen by subject, not by speed: everything that fails when a tool's name, argument,
#: result or documentation drifts. Deliberately no model, desktop or network tests.
SUITES = [
    "tests/test_tool_contracts.py",
    "tests/test_tool_coverage.py",
    "tests/test_tool_registry_consistency.py",
    "tests/test_tools_doc.py",
    "tests/test_command_sections.py",
    "tests/test_docs_match_reality.py",
    "tests/test_tool_result_semantics.py",
    "tests/test_tool_calls_end_to_end.py",
    "tests/test_env_honesty.py",
    "tests/test_arg_coercion.py",
    "tests/test_permissions.py",
    "tests/test_command_diagnostics.py",
    "tests/test_run_command_admin.py",
    "tests/test_file_ops_read.py",
    "tests/test_file_ops_write.py",
    "tests/test_file_ops_replace.py",
]


def main(argv: list[str]) -> int:
    missing = [s for s in SUITES if not (ROOT / s).is_file()]
    if missing:
        print("missing suites (renamed or removed?):", ", ".join(missing))
        return 2

    cmd = [sys.executable, "-m", "pytest", *SUITES, "-q", "-p", "no:cacheprovider", *argv]
    started = time.monotonic()
    completed = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.monotonic() - started

    verdict = "ok" if completed.returncode == 0 else "FAILED"
    print(f"\nfast run {verdict} in {elapsed:.1f}s — full suite before committing")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
