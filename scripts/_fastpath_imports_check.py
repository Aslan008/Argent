"""Verify the fast-path tool imports resolve and calculate returns the right answer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.misc_tools import calculate as _calc
print("calculate('2 + 2 * 22') ->", repr(_calc(expression="2 + 2 * 22")))

# Validate every module used by the fast-path execution block (agent.py ~1030-1054)
from tools.search_ops import grep_search, search_files  # noqa: F401
from tools.file_ops import list_directory  # noqa: F401
from src.agent.shell import run_text  # noqa: F401
print("all fast-path imports OK")