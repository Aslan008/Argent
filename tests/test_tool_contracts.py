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
        from tools.schemas import TOOL_SCHEMAS
        return {t["function"]["name"]: t["function"] for t in TOOL_SCHEMAS}

    def _find_param(self, fn_schema, param_name):
        props = fn_schema.get("parameters", {}).get("properties", {})
        return props.get(param_name, {})

    def test_lsp_tools_declare_1_indexed_lines(self, schemas_by_name):
        """Every LSP tool with a 'line' parameter must declare it as 1-indexed."""
        lsp_tools = ["find_definition", "find_references", "get_hover",
                     "find_implementations", "get_call_hierarchy"]
        for name in lsp_tools:
            schema = schemas_by_name.get(name)
            if not schema:
                continue
            line_param = self._find_param(schema, "line")
            if line_param:
                desc = line_param.get("description", "")
                assert "1-indexed" in desc, (
                    f"Tool '{name}' line param must say '1-indexed', got: '{desc}'"
                )

    def test_lsp_tools_declare_0_indexed_columns(self, schemas_by_name):
        """Every LSP tool with a 'column' parameter must declare it as 0-indexed."""
        lsp_tools = ["find_definition", "find_references", "get_hover",
                     "find_implementations", "get_call_hierarchy"]
        for name in lsp_tools:
            schema = schemas_by_name.get(name)
            if not schema:
                continue
            col_param = self._find_param(schema, "column")
            if col_param:
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
        from tools import file_ops
        self.file_ops = file_ops
        self.tmp_path = tmp_path

    def test_start_line_2_returns_from_line_2(self):
        """start_line=2 should skip line 1 and return from line 2."""
        (self.tmp_path / "f.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        out = self.file_ops.read_file("f.py", start_line=2)
        # Strip line-number gutter
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "line1" not in plain, "start_line=2 should skip line 1"
        assert "line2" in plain

    def test_end_line_2_returns_lines_1_and_2(self):
        """end_line=2 should return lines 1-2 (inclusive), not 1-3."""
        (self.tmp_path / "f.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
        out = self.file_ops.read_file("f.py", end_line=2)
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "line1" in plain
        assert "line2" in plain
        assert "line3" not in plain, "end_line=2 should not include line 3"

    def test_start_line_1_returns_from_first_line(self):
        """start_line=1 should return from line 1 (not skip it)."""
        (self.tmp_path / "f.py").write_text("first\nsecond\n", encoding="utf-8")
        out = self.file_ops.read_file("f.py", start_line=1)
        from tools._helpers import _strip_read_line_numbers
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
        from tools import file_ops
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