"""Tool schema ↔ implementation contract tests.

These tests verify that the conventions declared in ``tools/schemas.py`` match
the actual implementation behavior. The LSP line-indexing bug was exactly this
class of issue: schemas declared "1-indexed" but ``lsp_manager`` passed lines
directly to multilspy (0-indexed). No test verified the contract.

These tests are pure Python (no LSP server needed) and run fast.
"""

import json
import textwrap

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Schema declarations: verify the contracts are what we expect
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaDeclarations:
    """The schemas in schemas.py must declare the correct indexing conventions.
    If a schema silently drops the '1-indexed' description, a future developer
    has no way to know the expected convention."""

    @pytest.fixture
    def schemas_by_name(self):
        from argent.tools.schemas import TOOL_SCHEMAS
        return {t["function"]["name"]: t["function"] for t in TOOL_SCHEMAS}

    def _find_param(self, fn_schema, param_name):
        props = fn_schema.get("parameters", {}).get("properties", {})
        return props.get(param_name, {})

    def test_lsp_tools_declare_1_indexed_lines(self, schemas_by_name):
        """Every LSP navigation tool with a 'line' parameter must declare it as 1-indexed."""
        lsp_tools = ["grep_search"]
        for name in lsp_tools:
            schema = schemas_by_name.get(name)
            assert schema is not None, f"Tool '{name}' must be registered in TOOL_SCHEMAS"
            line_param = self._find_param(schema, "line")
            assert line_param, f"Tool '{name}' must have a 'line' parameter"
            desc = line_param.get("description", "")
            assert "1-indexed" in desc, (
                f"Tool '{name}' line param must say '1-indexed', got: '{desc}'"
            )

    def test_lsp_tools_declare_0_indexed_columns(self, schemas_by_name):
        """Every LSP navigation tool with a 'column' parameter must declare it as 0-indexed."""
        lsp_tools = ["grep_search"]
        for name in lsp_tools:
            schema = schemas_by_name.get(name)
            assert schema is not None, f"Tool '{name}' must be registered in TOOL_SCHEMAS"
            col_param = self._find_param(schema, "column")
            assert col_param, f"Tool '{name}' must have a 'column' parameter"
            desc = col_param.get("description", "")
            assert "0-indexed" in desc, (
                f"Tool '{name}' column param must say '0-indexed', got: '{desc}'"
            )

    def test_read_file_declares_1_indexed(self, schemas_by_name):
        """read_file start_line/end_line must be declared 1-indexed."""
        schema = schemas_by_name["read_file"]
        for param in ("start_line", "end_line"):
            desc = self._find_param(schema, param).get("description", "")
            assert "1-indexed" in desc, (
                f"read_file.{param} must say '1-indexed', got: '{desc}'"
            )

    def test_multi_replace_in_file_chunk_declares_lines(self, schemas_by_name):
        """multi_replace_in_file_chunk uses start_line/end_line — they must
        be declared with line-number semantics."""
        schema = schemas_by_name.get("multi_replace_in_file_chunk")
        if not schema:
            pytest.skip("multi_replace_in_file_chunk not in schemas")
        # The schema uses a JSON string for changes, so we check the description
        desc = schema.get("description", "")
        assert "line" in desc.lower(), (
            "multi_replace_in_file_chunk description should mention line numbers"
        )


# ══════════════════════════════════════════════════════════════════════════════
# read_file: 1-indexed start_line/end_line
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFileContract:
    """Verify read_file actually treats start_line/end_line as 1-indexed."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from argent.tools import file_ops
        self.file_ops = file_ops
        self.tmp_path = tmp_path

    def test_start_line_2_returns_from_line_2(self):
        """start_line=2 should skip line 1 and return from line 2."""
        (self.tmp_path / "f.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        out = self.file_ops.read_file("f.py", start_line=2)
        # Strip line-number gutter
        from argent.tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "line1" not in plain, "start_line=2 should skip line 1"
        assert "line2" in plain

    def test_end_line_2_returns_lines_1_and_2(self):
        """end_line=2 should return lines 1-2 (inclusive), not 1-3."""
        (self.tmp_path / "f.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        out = self.file_ops.read_file("f.py", end_line=2)
        from argent.tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "line1" in plain
        assert "line2" in plain
        assert "line3" not in plain, "end_line=2 should not include line 3"

    def test_start_line_1_returns_from_first_line(self):
        """start_line=1 should return from line 1 (not skip it)."""
        (self.tmp_path / "f.py").write_text("first\nsecond\n", encoding="utf-8")
        out = self.file_ops.read_file("f.py", start_line=1)
        from argent.tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "first" in plain


# ══════════════════════════════════════════════════════════════════════════════
# multi_replace_in_file_chunk: 1-indexed start_line/end_line
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiReplaceContract:
    """Verify multi_replace_in_file_chunk treats start_line/end_line as
    1-indexed and inclusive."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from argent.tools import file_ops
        self.file_ops = file_ops
        self.tmp_path = tmp_path

    def _make_file(self, name="m.txt", content=None):
        if content is None:
            content = "line1\nline2\nline3\nline4\nline5\n"
        (self.tmp_path / name).write_text(content, encoding="utf-8")
        return str(self.tmp_path / name)

    def test_start_line_1_replaces_first_line(self):
        """start_line=1 should replace line 1, not line 2."""
        fp = self._make_file()
        changes = json.dumps([{
            "start_line": 1, "end_line": 1,
            "target_content": "line1",
            "replacement_content": "REPLACED",
        }])
        self.file_ops.multi_replace_in_file_chunk(fp, changes)
        result = (self.tmp_path / "m.txt").read_text(encoding="utf-8")
        assert result.startswith("REPLACED"), f"Line 1 not replaced: {result}"
        assert "line2" in result, "Line 2 should be untouched"

    def test_end_line_2_replaces_lines_1_and_2(self):
        """end_line=2 should replace lines 1-2 (inclusive), not 1-3."""
        fp = self._make_file()
        changes = json.dumps([{
            "start_line": 1, "end_line": 2,
            "target_content": "line1\nline2",
            "replacement_content": "REPLACED",
        }])
        self.file_ops.multi_replace_in_file_chunk(fp, changes)
        result = (self.tmp_path / "m.txt").read_text(encoding="utf-8")
        assert "REPLACED" in result
        assert "line3" in result, "Line 3 should be untouched"
        assert "line1" not in result, "Line 1 should be replaced"
        assert "line2" not in result, "Line 2 should be replaced"

    def test_single_line_replaces_only_that_line(self):
        """start_line=end_line=3 should replace only line 3."""
        fp = self._make_file()
        changes = json.dumps([{
            "start_line": 3, "end_line": 3,
            "target_content": "line3",
            "replacement_content": "REPLACED",
        }])
        self.file_ops.multi_replace_in_file_chunk(fp, changes)
        result = (self.tmp_path / "m.txt").read_text(encoding="utf-8")
        lines = result.strip().split("\n")
        assert lines[0] == "line1"
        assert lines[1] == "line2"
        assert lines[2] == "REPLACED"
        assert lines[3] == "line4"
        assert lines[4] == "line5"


# ══════════════════════════════════════════════════════════════════════════════
# Shape of the API: names, width, description budget
# ══════════════════════════════════════════════════════════════════════════════
#
# Every constant here is a RATCHET set to the value measured on 2026-10-27 and may only
# go down. Lowering one after a cleanup is the point; raising one has to be a deliberate
# edit with a reason in the diff. The plan these numbers serve:
# `.argent/artifacts/tools-plan.md`.

# Phase 2 lowered these from 21_870 / 2_017 by moving the long form into
# `skills/argent-tools-reference/SKILL.md`. The plan's 9_000 turned out to be arithmetic
# that does not survive contact with 60 tools: the remaining descriptions each hold the
# line that prevents a known mistake (the read_file gutter, the write_file 150-line
# guard, "the image arrives in the NEXT message"), and cutting them to ~90 characters
# would delete exactly that. Next milestone: 13_000.
BUDGET_TOTAL_DESCRIPTION_CHARS = 16_810
BUDGET_PER_TOOL_DESCRIPTION_CHARS = 651          # `wait_heartbeat`
PATH_LIKE = {"file_path", "path", "directory", "dir_path", "directory_path"}
BUDGET_DISTINCT_PATH_NAMES = 1                      # `path` — phase 1 done
BUDGET_PARAMETERS_PER_TOOL = 16                    # `manage_automations`
BUDGET_UNDESCRIBED_PARAMETERS = 15

#: Tools that appear only on machines where an optional subsystem is installed
#: (`semantic_search` arrives with the RAG engine). They are exercised by their own
#: tests and must not make the shipped budget or the handler map fail elsewhere.
from argent.tools import OPTIONAL_TOOLS   # a product fact: docs and budgets share it


def _all_functions():
    from argent.tools.schemas import TOOL_SCHEMAS
    return [entry["function"] for entry in TOOL_SCHEMAS]


def _shipped_functions():
    """What every install has. Optional subsystems are measured by their own tests."""
    return [fn for fn in _all_functions() if fn["name"] not in OPTIONAL_TOOLS]


def _props(fn):
    return (fn.get("parameters") or {}).get("properties") or {}


class TestApiSurface:
    """What a caller has to hold in their head to use a tool once, without reading the
    description twice."""

    def test_tool_names_are_unique(self):
        from collections import Counter

        names = [fn["name"] for fn in _all_functions()]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        assert not duplicates, f"имя инструмента повторяется: {duplicates}"

    def test_every_schema_has_a_handler_and_no_handler_is_orphaned(self):
        """A schema without a handler fails on the first call; a handler without a schema
        is invisible to the model. Neither shows up until somebody tries it."""
        from argent.tools.lazy_registry import DEFAULT_TOOL_LOADERS

        names = {fn["name"] for fn in _all_functions()}
        unhandled = sorted(names - set(DEFAULT_TOOL_LOADERS) - OPTIONAL_TOOLS)
        invisible = sorted(set(DEFAULT_TOOL_LOADERS) - names - OPTIONAL_TOOLS)
        assert not unhandled, f"объявлены, но не реализованы: {unhandled}"
        assert not invisible, f"реализованы, но не объявлены модели: {invisible}"

    def test_descriptions_stay_within_the_context_budget(self):
        fns = _shipped_functions()
        total = sum(len(fn.get("description") or "") for fn in fns)
        assert total <= BUDGET_TOTAL_DESCRIPTION_CHARS, (
            f"описания выросли до {total} символов (потолок "
            f"{BUDGET_TOTAL_DESCRIPTION_CHARS}). Это платится в КАЖДОМ запросе. Самые "
            f"длинные: "
            + ", ".join(
                f"{name} {size}"
                for size, name in sorted(
                    ((len(fn.get("description") or ""), fn["name"]) for fn in fns),
                    reverse=True,
                )[:3]
            )
        )

        over = [(fn["name"], len(fn.get("description") or "")) for fn in fns
                if len(fn.get("description") or "") > BUDGET_PER_TOOL_DESCRIPTION_CHARS]
        assert not over, (
            f"длиннее потолка на инструмент ({BUDGET_PER_TOOL_DESCRIPTION_CHARS}): {over}"
        )

    def test_parameters_are_named_predictably(self):
        used = set()
        for fn in _all_functions():
            used.update(_props(fn))
        path_names = sorted(used & PATH_LIKE)
        assert len(path_names) <= BUDGET_DISTINCT_PATH_NAMES, (
            f"имён для пути стало {len(path_names)}: {path_names}. "
            f"Каноническое — `path`, остальные обязаны быть алиасами."
        )

    def test_no_tool_is_wider_than_a_person_can_hold(self):
        wide = sorted(((len(_props(fn)), fn["name"]) for fn in _shipped_functions()), reverse=True)
        over = [pair for pair in wide if pair[0] > BUDGET_PARAMETERS_PER_TOOL]
        assert not over, (
            f"шире потолка ({BUDGET_PARAMETERS_PER_TOOL}): {over}. "
            f"Связанные параметры надо свернуть в объект с собственными описаниями."
        )

    def test_every_parameter_says_what_it_is(self):
        missing = [
            (fn["name"], param)
            for fn in _all_functions()
            for param, spec in _props(fn).items()
            if not (spec or {}).get("description")
        ]
        assert len(missing) <= BUDGET_UNDESCRIBED_PARAMETERS, (
            f"параметров без описания стало {len(missing)} "
            f"(потолок {BUDGET_UNDESCRIBED_PARAMETERS}): {missing[:8]}"
        )

    def test_every_tool_belongs_to_exactly_one_group(self):
        """The taxonomy is what `/help` and the docs group by. A tool outside it is a tool
        nobody looks up, and a duplicate makes the map lie."""
        from argent.tools import TOOL_GROUPS

        names = [fn["name"] for fn in _all_functions()]
        seen = {}
        for group, members in TOOL_GROUPS.items():
            for name in members:
                assert name not in seen, f"{name} в двух группах: {seen[name]} и {group}"
                seen[name] = group

        missing = sorted(set(names) - set(seen))
        extra = sorted(set(seen) - set(names) - OPTIONAL_TOOLS)
        assert not missing, f"инструменты вне таксономии: {missing}"
        assert not extra, f"в таксономии есть несуществующие инструменты: {extra}"

    def test_required_parameters_are_declared(self):
        """A rename that touches `properties` but not `required` produces a schema no
        model can satisfy: it is told to pass a name that does not exist."""
        broken = []
        for fn in _all_functions():
            props = set(_props(fn))
            for required in (fn.get("parameters") or {}).get("required") or []:
                if required not in props:
                    broken.append((fn["name"], required))
        assert not broken, f"'required' ссылается на отсутствующее свойство: {broken}"

    def test_the_schema_and_the_signature_agree_on_the_name(self):
        """There is no translation layer any more: what the schema advertises is what
        the function declares. A tool that drifted apart would fail on its first call
        as "missing required argument", which looks like the model's mistake."""
        import inspect

        from argent.tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS

        mismatched = []
        for schema in TOOL_SCHEMAS:
            fn = schema["function"]
            impl = AVAILABLE_TOOLS.get(fn["name"])
            if impl is None:
                continue
            declared = set((fn.get("parameters") or {}).get("properties") or {})
            missing = declared - set(inspect.signature(impl).parameters)
            if missing:
                mismatched.append((fn["name"], sorted(missing)))
        assert not mismatched, f"схема обещает параметры, которых нет в функции: {mismatched}"