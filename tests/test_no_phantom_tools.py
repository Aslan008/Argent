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
# .argent is the data directory (models, caches, training staging), not the program.
_SKIP = {".git", ".venv", "venv", "venv_pyqt6", "site-packages", "node_modules",
         "__pycache__", ".pytest_cache", ".argent",
         ".claude", "tests", "scripts", "desktop", "visuals", "skills", "plugins", "scratch"}


# Not tool lists, though they look like them to the heuristic below:
#   * tools/lazy_registry.py pairs a tool with the module that implements it, and
#     a module path is not a tool name;
#   * src/leeth and src/micromodels are the class vocabularies of trained routers
#     (browser_click, find_by_name, NO_TOOL, UNFAMILIAR, ...). A label is mapped
#     to a real tool when it is dispatched (find_by_name -> search_files); the
#     vocabulary itself is frozen into the model and cannot follow a rename;
#   * src/leeth2/data/argent.py holds the same legacy vocabulary: the hand-written
#     routing battery and the old training files are labelled with it, and Leeth 2
#     is evaluated against exactly those labels.
_NOT_TOOL_LISTS = ("tools/lazy_registry.py",)
_LABEL_VOCABULARIES = ("src/leeth/", "src/micromodels/", "src/leeth2/data/argent.py")


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
        rel = path.relative_to(_ROOT).as_posix()
        if rel in _NOT_TOOL_LISTS or rel.startswith(_LABEL_VOCABULARIES):
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
        for module in ("src/agent/salvage.py", "src/agent/parser.py"):
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


class TestTranslationTables:
    """Tables that translate someone else's tool names into ours."""

    def test_the_ecosystem_skill_map_points_at_real_tools(self):
        """`MultiEdit -> multi_replace_in_file` and `TodoWrite -> add_work_task`
        sat here after both targets were gone: a skill written for Claude Code
        was told to call something that does not exist."""
        from skill_manager import _TOOL_NAME_MAP
        ghosts = {k: v for k, v in _TOOL_NAME_MAP.items() if v not in _REAL}
        assert not ghosts, ghosts
