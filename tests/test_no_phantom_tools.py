"""Source that names a tool must name a tool that exists.

The Obsidian tools were deleted from the registry, but four lists still spelled
write_obsidian_note. Nothing failed and nothing was logged — the name simply
never matched, so:

  * salvage.py could not rescue a truncated Obsidian write (fine, moot), but
    its path regex still preferred note_path over file_path;
  * parser.py's malformed-JSON recovery carried a dead branch;
  * orchestrator.py handed a project three tool names that resolve to nothing,
    while asking the user whether to "enable Obsidian integration".

A tool name in a string literal has no compiler behind it, so the check is
here: any literal collection where most entries ARE real tool names is treated
as a tool list, and every entry has to exist.
"""

import ast
from pathlib import Path

import pytest

from tools.schemas import AVAILABLE_TOOLS

_ROOT = Path(__file__).resolve().parents[1]
# semantic_search is registered only when RAG is on; call_mcp_tool proxies MCP.
_REAL = set(AVAILABLE_TOOLS) | {"semantic_search"}
# tests and scripts legitimately name tools that must NOT exist (bad input,
# removed-tool regressions); the rule is about the program itself.
_SKIP = {".git", "venv_pyqt6", "node_modules", "__pycache__", ".pytest_cache",
         ".claude", "tests", "scripts", "desktop", "visuals", "skills", "plugins"}


def _string_elements(node):
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None
    out = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            out.append(element.value)
        else:
            return None
    return out


def _drifted():
    hits = []
    for path in _ROOT.rglob("*.py"):
        if any(part in _SKIP for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = _string_elements(node)
            if not names or len(names) < 2:
                continue
            known = [n for n in names if n in _REAL]
            if len(known) * 2 < len(names):      # not a tool list
                continue
            ghosts = [n for n in names if n not in _REAL]
            if ghosts:
                hits.append(f"{path.relative_to(_ROOT)}:{node.lineno} -> {ghosts}")
    return hits


class TestEveryNamedToolExists:
    def test_no_list_names_a_tool_that_was_removed(self):
        drifted = _drifted()
        assert not drifted, (
            "эти списки называют инструменты, которых нет в реестре — "
            f"совпадения не будет, и никто об этом не узнает: {drifted}")

    @pytest.mark.parametrize("name", ["write_obsidian_note", "search_obsidian_notes",
                                      "update_obsidian_properties"])
    def test_the_ones_that_bit_us_are_gone(self, name):
        assert name not in _REAL
        for module in ("src/agent/salvage.py", "src/agent/parser.py",
                       "src/project/orchestrator.py"):
            text = (_ROOT / module).read_text(encoding="utf-8")
            assert name not in text, f"{module} всё ещё ссылается на {name}"


class TestTheDetectorWorks:
    def test_it_recognises_a_tool_list_with_a_ghost(self, tmp_path, monkeypatch):
        """A check that cannot fail proves nothing."""
        module = tmp_path / "fake_module.py"
        module.write_text('TOOLS = ["read_file", "write_file", "fly_to_the_moon"]\n',
                          encoding="utf-8")
        monkeypatch.setattr("tests.test_no_phantom_tools._ROOT", tmp_path)
        hits = _drifted()
        assert len(hits) == 1 and "fly_to_the_moon" in hits[0]

    def test_it_ignores_a_list_that_is_not_about_tools(self, tmp_path, monkeypatch):
        module = tmp_path / "fake_module.py"
        module.write_text('COLOURS = ["red", "green", "read_file"]\n', encoding="utf-8")
        monkeypatch.setattr("tests.test_no_phantom_tools._ROOT", tmp_path)
        assert _drifted() == []
