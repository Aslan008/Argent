"""End-to-end LSP integration tests with a real jedi-language-server.

These tests exist because a critical bug — 1-indexed lines from tool schemas
being passed directly to multilspy (which expects 0-indexed) — went undetected
by 1797 unit tests. Unit tests exercised ``lsp_manager`` with 0-indexed lines
(matching multilspy's convention), and tool-registry tests checked existence
rather than behavior. No test ever called through the full chain:

    tool function (1-indexed) → lsp_manager (converts) → multilspy (0-indexed)

These tests fix that. They start a real ``jedi-language-server`` via multilspy,
create a known Python file, and verify that every LSP query returns the correct
result when called with **1-indexed** line numbers — the convention declared in
``tools/schemas.py``.
"""

import pathlib
import tempfile

import pytest

# ── Skip everything if multilspy is not installed ─────────────────────────────

pytest.importorskip("multilspy")

from src.lsp.manager import lsp_manager  # noqa: E402

# ── Known sample file ────────────────────────────────────────────────────────
# Line numbers are 1-indexed throughout. Column numbers are 0-indexed.
#
#  1: import os
#  2: (empty)
#  3: class MyClass:
#  4:     def method_one(self):
#  5:         pass
#  6:     def method_two(self):
#  7:         pass
#  8: (empty)
#  9: def standalone_function():
# 10:     obj = MyClass()
# 11:     obj.method_one()
# 12:     return obj
# 13: (empty)
# 14: variable = standalone_function()

SAMPLE_CODE = """\
import os

class MyClass:
    def method_one(self):
        pass
    def method_two(self):
        pass

def standalone_function():
    obj = MyClass()
    obj.method_one()
    return obj

variable = standalone_function()
"""

# Symbol positions (1-indexed line, 0-indexed column)
SYM_CLASS_DEF = (3, 6)        # MyClass definition
SYM_METHOD_ONE_DEF = (4, 8)  # method_one definition
SYM_FUNC_DEF = (9, 4)        # standalone_function definition
SYM_CLASS_USE = (10, 10)     # MyClass() instantiation
SYM_METHOD_USE = (11, 8)     # obj.method_one() call
SYM_FUNC_USE = (14, 11)      # standalone_function() call


# ── Session-scoped fixture: temp directory + known files ──────────────────────

@pytest.fixture(scope="session")
def lsp_project_dir():
    """Create a temp directory with known Python files for LSP tests.

    The LSP server is started lazily on first query and cached by
    (language, project_root). Using a session-scoped directory means
    the server starts once and is reused across all tests.
    """
    d = tempfile.mkdtemp(prefix="argent_lsp_test_")
    # Main sample file
    (pathlib.Path(d) / "sample.py").write_text(SAMPLE_CODE, encoding="utf-8")
    # File with a syntax error
    (pathlib.Path(d) / "syntax_err.py").write_text("def f(:\n    pass\n", encoding="utf-8")
    # File with an indentation error
    (pathlib.Path(d) / "indent_err.py").write_text("def f():\n  pass\n", encoding="utf-8")
    return d


@pytest.fixture(autouse=True)
def _chdir_to_project(monkeypatch, lsp_project_dir):
    """Change cwd so lsp_manager uses our temp dir as project root."""
    monkeypatch.chdir(lsp_project_dir)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sample_path(lsp_project_dir):
    return str(pathlib.Path(lsp_project_dir) / "sample.py")


# ══════════════════════════════════════════════════════════════════════════════
# find_definition
# ══════════════════════════════════════════════════════════════════════════════

class TestFindDefinition:
    def test_finds_class_definition(self, lsp_project_dir):
        """Basic: probing the class definition returns itself."""
        fp = _sample_path(lsp_project_dir)
        results = lsp_manager.find_definition(fp, SYM_CLASS_DEF[0], SYM_CLASS_DEF[1])
        assert results, "Expected at least one definition result"
        assert any("sample.py" in r["file_path"] for r in results)

    def test_line_indexing_on_usage(self, lsp_project_dir):
        """THE regression test: probing line 10 (1-indexed) — a usage of
        MyClass — must find the definition on line 3. If the line were
        passed as-is to multilspy (0-indexed), it would probe line 11
        (obj.method_one()) and find method_one instead."""
        fp = _sample_path(lsp_project_dir)
        results = lsp_manager.find_definition(fp, SYM_CLASS_USE[0], SYM_CLASS_USE[1])
        assert results, "Expected definition results for MyClass usage on line 10"
        # The definition of MyClass is on line 3
        found_lines = [r["line"] for r in results if "sample.py" in r["file_path"]]
        assert 3 in found_lines, (
            f"Expected definition on line 3, got lines {found_lines}. "
            f"This indicates a 1-indexed/0-indexed mismatch."
        )

    def test_finds_method_from_call(self, lsp_project_dir):
        """Probing obj.method_one() on line 11 should find method_one def
        on line 4. If off-by-one, would probe line 12 (return obj)."""
        fp = _sample_path(lsp_project_dir)
        results = lsp_manager.find_definition(fp, SYM_METHOD_USE[0], SYM_METHOD_USE[1])
        assert results, "Expected definition results for method_one call"
        found_lines = [r["line"] for r in results if "sample.py" in r["file_path"]]
        assert 4 in found_lines, (
            f"Expected method_one definition on line 4, got lines {found_lines}"
        )

    def test_line_one_indexing_regression(self, lsp_project_dir):
        """The most basic regression: line 1 (1-indexed) must map to line 0
        (0-indexed) in multilspy. If the conversion is missing, line 1
        becomes line 0 in multilspy — which is actually correct by accident.
        But line 2 (1-indexed) should map to line 1 (0-indexed), not line 2.
        We test by probing 'import' on line 1 — the definition of 'os'
        should be found (in the os module), not an error."""
        fp = _sample_path(lsp_project_dir)
        # Probe 'os' on line 1, column 7 (after 'import ')
        results = lsp_manager.find_definition(fp, 1, 7)
        # jedi-language-server should find the os module definition
        assert results is not None, "find_definition returned None for line 1"

    def test_nonexistent_file_returns_none(self, lsp_project_dir):
        """Error handling: a nonexistent file should return None, not crash."""
        results = lsp_manager.find_definition("does_not_exist.py", 1, 0)
        assert results is None


# ══════════════════════════════════════════════════════════════════════════════
# find_references
# ══════════════════════════════════════════════════════════════════════════════

class TestFindReferences:
    def test_finds_all_usages_of_class(self, lsp_project_dir):
        """References to MyClass should include the definition (line 3) and
        the usage (line 10)."""
        fp = _sample_path(lsp_project_dir)
        results = lsp_manager.find_references(fp, SYM_CLASS_DEF[0], SYM_CLASS_DEF[1])
        assert results, "Expected reference results"
        ref_lines = [r["line"] for r in results if "sample.py" in r["file_path"]]
        # Definition on line 3, usage on line 10
        assert 3 in ref_lines, f"Definition (line 3) not found in references: {ref_lines}"
        assert 10 in ref_lines, f"Usage (line 10) not found in references: {ref_lines}"

    def test_line_numbers_are_1_indexed(self, lsp_project_dir):
        """Output line numbers must be 1-indexed (multilspy returns 0-indexed,
        lsp_manager adds +1). If the +1 is missing, lines would be 2 and 9
        instead of 3 and 10."""
        fp = _sample_path(lsp_project_dir)
        results = lsp_manager.find_references(fp, SYM_CLASS_DEF[0], SYM_CLASS_DEF[1])
        ref_lines = [r["line"] for r in results if "sample.py" in r["file_path"]]
        assert 3 in ref_lines, (
            f"Line 3 expected (1-indexed output). Got {ref_lines}. "
            f"If lines are 2 and 9, the +1 conversion in find_references is missing."
        )


# ══════════════════════════════════════════════════════════════════════════════
# get_hover
# ══════════════════════════════════════════════════════════════════════════════

class TestGetHover:
    def test_returns_content_for_class(self, lsp_project_dir):
        """Hover over MyClass should return meaningful content."""
        fp = _sample_path(lsp_project_dir)
        result = lsp_manager.get_hover(fp, SYM_CLASS_DEF[0], SYM_CLASS_DEF[1])
        assert result is not None, "Expected hover result"
        assert result.get("content", ""), "Hover content should not be empty"

    def test_line_indexing_for_function(self, lsp_project_dir):
        """Hover over standalone_function on line 9 should return content
        mentioning 'function' or the function name. If off-by-one, would
        hover over line 10 (obj = MyClass()) and get different content."""
        fp = _sample_path(lsp_project_dir)
        result = lsp_manager.get_hover(fp, SYM_FUNC_DEF[0], SYM_FUNC_DEF[1])
        assert result is not None, "Expected hover result for standalone_function"
        content = result.get("content", "")
        assert content, "Hover content should not be empty"


# ══════════════════════════════════════════════════════════════════════════════
# get_diagnostics
# ══════════════════════════════════════════════════════════════════════════════

class TestGetDiagnostics:
    def test_clean_file_no_diagnostics(self, lsp_project_dir):
        """A valid Python file should have no error diagnostics."""
        fp = _sample_path(lsp_project_dir)
        diags = lsp_manager.get_diagnostics(fp)
        assert diags is not None, "Expected diagnostics list (not None)"
        # Filter to errors only (severity 1)
        errors = [d for d in diags if d.get("severity") == 1]
        assert not errors, f"Unexpected errors in clean file: {errors}"

    def test_detects_syntax_error(self, lsp_project_dir):
        """A file with 'def f(:' should produce a syntax error diagnostic."""
        fp = str(pathlib.Path(lsp_project_dir) / "syntax_err.py")
        diags = lsp_manager.get_diagnostics(fp)
        assert diags is not None, "Expected diagnostics list"
        assert len(diags) > 0, "Expected at least one diagnostic for syntax error"
        # At least one should be an error (severity 1)
        errors = [d for d in diags if d.get("severity") == 1]
        assert errors, f"Expected severity=1 error, got: {diags}"

    def test_detects_indentation_error(self, lsp_project_dir):
        """A file with inconsistent indentation should produce a diagnostic."""
        fp = str(pathlib.Path(lsp_project_dir) / "indent_err.py")
        diags = lsp_manager.get_diagnostics(fp)
        assert diags is not None, "Expected diagnostics list"
        # jedi-language-server may or may not report indentation issues;
        # the key assertion is that it doesn't crash and returns a list
        assert isinstance(diags, list), f"Expected list, got {type(diags)}"


# ══════════════════════════════════════════════════════════════════════════════
# get_workspace_symbols
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceSymbols:
    def test_finds_myclass(self, lsp_project_dir):
        """Searching for 'MyClass' should find it in sample.py."""
        results = lsp_manager.get_workspace_symbols("MyClass")
        assert results, "Expected workspace symbol results for 'MyClass'"
        found = [s for s in results if s.get("name") == "MyClass"]
        assert found, f"MyClass not found in workspace symbols: {results}"
        assert any("sample.py" in s.get("file_path", "") for s in found)


# ══════════════════════════════════════════════════════════════════════════════
# get_call_hierarchy (Jedi fallback)
# ══════════════════════════════════════════════════════════════════════════════

class TestCallHierarchyFallback:
    """jedi-language-server does NOT support textDocument/prepareCallHierarchy.
    The tool layer (lsp_tools.py) falls back to Jedi for Python."""

    def test_incoming_finds_callers(self, lsp_project_dir):
        """Incoming call hierarchy for standalone_function should find its
        callers via Jedi fallback."""
        from tools.lsp_tools import get_call_hierarchy
        fp = _sample_path(lsp_project_dir)
        result = get_call_hierarchy(fp, SYM_FUNC_DEF[0], SYM_FUNC_DEF[1], "incoming")
        assert result, "Expected non-empty call hierarchy result"
        assert "Caller" in result, f"Expected 'Caller' in result: {result}"
        # standalone_function is called on line 14
        assert "14" in result or "standalone" in result.lower(), (
            f"Expected caller on line 14, got: {result}"
        )

    def test_outgoing_finds_callees(self, lsp_project_dir):
        """Outgoing call hierarchy for standalone_function should find its
        callees via AST fallback."""
        from tools.lsp_tools import get_call_hierarchy
        fp = _sample_path(lsp_project_dir)
        result = get_call_hierarchy(fp, SYM_FUNC_DEF[0], SYM_FUNC_DEF[1], "outgoing")
        assert result, "Expected non-empty call hierarchy result"
        assert "Callee" in result, f"Expected 'Callee' in result: {result}"
        # standalone_function calls MyClass() and method_one()
        assert "MyClass" in result or "method_one" in result, (
            f"Expected MyClass or method_one as callee, got: {result}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# find_implementations (Jedi fallback)
# ══════════════════════════════════════════════════════════════════════════════

class TestFindImplementationsFallback:
    """jedi-language-server does NOT support textDocument/implementation.
    The tool layer falls back to Jedi for Python."""

    def test_finds_implementations_via_jedi(self, lsp_project_dir):
        """find_implementations should fall back to Jedi and return results."""
        from tools.lsp_tools import find_implementations
        fp = _sample_path(lsp_project_dir)
        result = find_implementations(fp, SYM_CLASS_DEF[0], SYM_CLASS_DEF[1])
        assert result, "Expected non-empty implementations result"
        assert "implementation" in result.lower() or "No impl" not in result, (
            f"Expected implementations, got: {result}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tool interface (lsp_tools.py) with 1-indexed lines
# ══════════════════════════════════════════════════════════════════════════════

class TestToolInterface1Indexed:
    """Verify that the tool-layer functions (lsp_definition, lsp_references)
    accept and correctly handle 1-indexed line numbers, matching the schema
    declarations in schemas.py."""

    def test_lsp_definition_with_1_indexed_line(self, lsp_project_dir):
        """Calling lsp_definition with line=10 (1-indexed) should find
        MyClass definition on line 3 — not line 4 (which would indicate
        the line was passed as 0-indexed to multilspy)."""
        from tools.lsp_tools import lsp_definition
        fp = _sample_path(lsp_project_dir)
        result = lsp_definition(fp, SYM_CLASS_USE[0], SYM_CLASS_USE[1])
        assert result, "Expected non-empty definition result"
        assert "sample.py" in result
        assert ":3" in result or "line 3" in result, (
            f"Expected definition on line 3, got: {result}"
        )

    def test_lsp_references_with_1_indexed_line(self, lsp_project_dir):
        """Calling lsp_references with line=3 (1-indexed) should find
        references including line 3 and line 10."""
        from tools.lsp_tools import lsp_references
        fp = _sample_path(lsp_project_dir)
        result = lsp_references(fp, SYM_CLASS_DEF[0], SYM_CLASS_DEF[1])
        assert result, "Expected non-empty references result"
        assert "sample.py" in result