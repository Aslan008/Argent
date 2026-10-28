"""`docs/TOOLS.md` must describe the schemas that exist, not the ones that used to.

A hand-written tool page is wrong within a release: names change, parameters move into
a neighbour, a tool stops being advertised. So the page is generated, and this test is
the lock — it re-renders from `argent/tools/schemas.py` and compares with the file on
disk. Editing the schemas without regenerating turns the suite red, which is the only
thing that has ever kept documentation honest here.

Run the fix it suggests:
    .venv\\Scripts\\python.exe scripts/gen_tools_doc.py
"""

import importlib.util
from pathlib import Path

import pytest

from argent.tools import OPTIONAL_TOOLS, TOOL_GROUPS, group_of
from argent.tools.schemas import TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "TOOLS.md"


def _generator():
    spec = importlib.util.spec_from_file_location("gen_tools_doc", ROOT / "scripts" / "gen_tools_doc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page():
    return DOC.read_text(encoding="utf-8")


class TestThePageMatchesTheCode:
    def test_the_file_exists(self):
        assert DOC.exists(), "docs/TOOLS.md is missing — run scripts/gen_tools_doc.py"

    def test_it_is_not_stale(self, page):
        expected = _generator().render()
        assert page == expected, (
            "docs/TOOLS.md is out of date. Regenerate it:\n"
            "  .venv\\Scripts\\python.exe scripts/gen_tools_doc.py"
        )

    def test_every_advertised_tool_is_listed_once(self, page):
        names = [t["function"]["name"] for t in TOOL_SCHEMAS if t["function"]["name"] not in OPTIONAL_TOOLS]
        for name in names:
            rows = [ln for ln in page.splitlines() if ln.startswith(f"| `{name}` |")]
            assert len(rows) == 1, f"{name} appears {len(rows)} times in docs/TOOLS.md"

    def test_no_tool_is_invented(self, page):
        known = {t["function"]["name"] for t in TOOL_SCHEMAS}
        mentioned = set()
        for line in page.splitlines():
            if line.startswith("| `"):
                mentioned.add(line.split("`")[1])
        assert not (mentioned - known), f"docs/TOOLS.md names tools that do not exist: {sorted(mentioned - known)}"

    def test_the_summary_counts_are_real(self, page):
        advertised = sum(1 for t in TOOL_SCHEMAS if t["function"]["name"] not in OPTIONAL_TOOLS)
        assert f"{advertised} tools in {len(TOOL_GROUPS)} groups" in page

    def test_the_long_form_is_pointed_at(self, page):
        assert "skills/argent-tools-reference/SKILL.md" in page


class TestTheTaxonomyIsUsable:
    def test_every_advertised_tool_has_a_group(self):
        missing = sorted(t["function"]["name"] for t in TOOL_SCHEMAS
                         if t["function"]["name"] not in OPTIONAL_TOOLS
                         and not group_of(t["function"]["name"]))
        assert not missing, f"tools outside the taxonomy (docs would be ungrouped): {missing}"

    def test_group_names_are_readable_not_shouting(self):
        assert all(g == g.lower() for g in TOOL_GROUPS), sorted(TOOL_GROUPS)

    def test_a_one_line_table_entry_stays_short(self):
        """The page exists to be scanned; a paragraph per row defeats it."""
        gen = _generator()
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            if fn["name"] in OPTIONAL_TOOLS:
                continue
            line = gen._one_line(fn.get("description"))
            assert len(line) <= 160, f"{fn['name']} renders {len(line)} characters"
