"""Every tool is named by at least one test file.

Deliberately coarse: this catches "added a tool, forgot the tests", not "tested
thoroughly". The map lives in `tests/tool_coverage.json` and is rebuilt by
`scratch/build_coverage.py` — a tool that appears in the registry but not in the map
fails here, so the map cannot silently fall behind.

Known gap, tracked in `.argent/artifacts/tools-plan.md` (phase 5): `run_deep_linter`
is the only tool nobody exercises. The number below is a ratchet — it may not grow.
"""

import json
from pathlib import Path

from argent.tools.schemas import TOOL_SCHEMAS

MAP_PATH = Path(__file__).with_name("tool_coverage.json")
KNOWN_UNCOVERED = {"run_deep_linter"}


def _coverage():
    return json.loads(MAP_PATH.read_text(encoding="utf-8"))


def test_the_map_covers_exactly_the_registered_tools():
    declared = {entry["function"]["name"] for entry in TOOL_SCHEMAS}
    mapped = set(_coverage())

    missing = sorted(declared - mapped)
    stale = sorted(mapped - declared)

    assert not missing, (
        f"в карте нет инструментов: {missing}. Добавь их и пересобери карту: "
        f"`.venv/Scripts/python.exe scratch/build_coverage.py`"
    )
    assert not stale, f"в карте есть уже удалённые инструменты: {stale}. Пересобери карту."


def test_only_known_tools_lack_a_test():
    uncovered = {name for name, files in _coverage().items() if not files}
    unknown = sorted(uncovered - KNOWN_UNCOVERED)
    assert not unknown, (
        f"новые инструменты без единого теста: {unknown}. "
        f"Тест на поведение обязателен для всего, что трогает файлы, процессы или чужие окна."
    )
