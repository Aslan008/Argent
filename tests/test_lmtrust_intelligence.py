"""Blind-spot tests for intelligence.py — CodeIntelligence.

Covers:
  1. CodeIntelligence init with a project path
  2. find_definitions returns results for a known symbol
  3. find_definitions returns empty for non-existent symbol
  4. find_references returns results for a known symbol
  5. find_references returns empty for non-existent symbol
  6. get_symbol_at returns symbol at given position
  7. get_symbol_at returns None (empty string) for empty position
"""
import sys
import os
from pathlib import Path

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence import CodeIntelligence  # noqa: E402


# Sample source used across tests
SAMPLE_SOURCE = "x = 42\ndef foo():\n    return x\n"


@pytest.fixture
def intel_project(tmp_path):
    """Create a temp Python file and init CodeIntelligence pointed at it."""
    test_file = tmp_path / "sample_mod.py"
    test_file.write_text(SAMPLE_SOURCE, encoding="utf-8")
    ci = CodeIntelligence(project_path=str(tmp_path))
    return ci, str(test_file)


# ---------------------------------------------------------------------------
# 1. Init
# ---------------------------------------------------------------------------
def test_codeintelligence_init_with_project_path(tmp_path):
    """CodeIntelligence should accept and resolve a project path."""
    ci = CodeIntelligence(project_path=str(tmp_path))
    assert ci.project_path == tmp_path.resolve()
    assert ci.project_path.exists()


# ---------------------------------------------------------------------------
# 2 & 3. find_definitions
# ---------------------------------------------------------------------------
def test_find_definitions_returns_results_for_known_symbol(intel_project):
    """find_definitions on a real symbol (the 'x' inside foo's return) should
    locate the definition of x on line 1."""
    ci, file_path = intel_project
    # 'x' appears in "    return x" at line 3; column points at 'x'
    line_no = SAMPLE_SOURCE.splitlines().index("    return x") + 1
    col = "    return x".index("x")
    results = ci.find_definitions(file_path, line_no, col)
    assert isinstance(results, list)
    assert len(results) >= 1
    # No error entries expected
    assert not any("error" in r for r in results)
    # The definition should reference 'x'
    names = [r.get("name") for r in results]
    assert "x" in names


def test_find_definitions_empty_for_nonexistent_symbol(intel_project):
    """find_definitions at a position with no symbol (whitespace) should
    return an empty list (or a list with no error-free results)."""
    ci, file_path = intel_project
    # Column 0 on the blank-ish area — line 1 col 0 is 'x', so pick a column
    # well past end of line 1 ("x = 42" is 6 chars) to target empty space.
    results = ci.find_definitions(file_path, 1, 100)
    assert isinstance(results, list)
    # Either empty, or only error entries; no valid definition dicts
    valid = [r for r in results if "error" not in r]
    assert valid == []


# ---------------------------------------------------------------------------
# 4 & 5. find_references
# ---------------------------------------------------------------------------
def test_find_references_returns_results_for_known_symbol(intel_project):
    """find_references on 'x' should return at least the assignment and the
    usage inside foo()."""
    ci, file_path = intel_project
    line_no = SAMPLE_SOURCE.splitlines().index("    return x") + 1
    col = "    return x".index("x")
    results = ci.find_references(file_path, line_no, col)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert not any("error" in r for r in results)
    names = [r.get("name") for r in results]
    assert "x" in names


def test_find_references_empty_for_nonexistent_symbol(intel_project):
    """find_references at a whitespace column should yield no valid refs."""
    ci, file_path = intel_project
    results = ci.find_references(file_path, 1, 100)
    assert isinstance(results, list)
    valid = [r for r in results if "error" not in r]
    assert valid == []


# ---------------------------------------------------------------------------
# 6 & 7. get_symbol_at
# ---------------------------------------------------------------------------
def test_get_symbol_at_returns_symbol_at_position(intel_project):
    """get_symbol_at should extract the word at the given column."""
    ci, file_path = intel_project
    # Line 1: "x = 42"  -> 'x' starts at column 0
    symbol = ci.get_symbol_at(file_path, 1, 0)
    assert symbol == "x"
    # Line 2: "def foo():" -> 'foo' starts at column 4
    symbol_foo = ci.get_symbol_at(file_path, 2, 4)
    assert symbol_foo == "foo"
    # Line 3: "    return x" -> 'return' starts at column 4
    symbol_ret = ci.get_symbol_at(file_path, 3, 4)
    assert symbol_ret == "return"
    # Line 3: "    return x" -> 'x' starts at column 11
    symbol_x = ci.get_symbol_at(file_path, 3, 11)
    assert symbol_x == "x"


def test_get_symbol_at_returns_empty_for_empty_position(intel_project):
    """get_symbol_at on a line/column with no word should return ''."""
    ci, file_path = intel_project
    # Column 100 is past end of line 1 — no word there
    assert ci.get_symbol_at(file_path, 1, 100) == ""
    # Line 0 is out of the valid 1..len range
    assert ci.get_symbol_at(file_path, 0, 0) == ""
    # Line beyond file length
    assert ci.get_symbol_at(file_path, 999, 0) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])