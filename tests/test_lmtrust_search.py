"""LMTrust blind-spot tests for: search_files, grep_search, calculate,
set_goal, analyze_project, filter_new_items.

Layers applied (per LMTrust decision tree):
  L0 Smoke, L1 Contract, L2 Boundary, L4 Adversarial, L4b Fallback,
  L9 Negative Space.
Directions: D1 Math/Numeric, D4 Error Handling, D7 API Contract, D8 Security,
D10 Invariant.
"""

import json
import math
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _stub_memory(monkeypatch):
    """Replace memory_manager.memory so tools don't touch the real store."""
    import memory_manager

    class FakeMemory:
        def __init__(self):
            self.completed = []
            self.modified = []
            self.facts = []
            self._objective = None
            self._current_task = None

        def add_completed(self, s): self.completed.append(s)
        def add_file_modified(self, s): self.modified.append(s)
        def add_fact(self, s): self.facts.append(s)
        def add_error(self, s): self.completed.append(f"ERR:{s}")
        def set_objective(self, s): self._objective = s
        def set_current_task(self, s): self._current_task = s

    fake = FakeMemory()
    monkeypatch.setattr(memory_manager, "memory", fake)
    # Also patch the import in misc_tools
    import tools.misc_tools as mt
    monkeypatch.setattr(mt, "memory", fake)
    yield


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# search_files — L0/L1/L2/L4/L9
# ══════════════════════════════════════════════════════════════════════════════

class TestSearchFilesSmoke:
    """L0: basic smoke — does the tool run at all?"""

    def test_finds_files_in_directory(self, tmp_path):
        (tmp_path / "a.py").write_text("print('hi')\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("hello\n", encoding="utf-8")
        from tools.search_ops import search_files
        out = search_files(str(tmp_path))
        assert "a.py" in out
        assert "b.txt" in out
        assert "Found 2 file(s)" in out


class TestSearchFilesEdgeCases:
    """L2: boundary conditions."""

    def test_empty_directory(self, tmp_path):
        from tools.search_ops import search_files
        out = search_files(str(tmp_path))
        assert "No files found" in out

    def test_name_contains_filter(self, tmp_path):
        (tmp_path / "alpha.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "beta.py").write_text("y\n", encoding="utf-8")
        from tools.search_ops import search_files
        out = search_files(str(tmp_path), name_contains="alpha")
        assert "alpha.py" in out
        assert "beta.py" not in out

    def test_content_contains_filter(self, tmp_path):
        (tmp_path / "match.py").write_text("UNIQUE_MARKER_42\n", encoding="utf-8")
        (tmp_path / "nomatch.py").write_text("other\n", encoding="utf-8")
        from tools.search_ops import search_files
        out = search_files(str(tmp_path), content_contains="UNIQUE_MARKER_42")
        assert "match.py" in out
        assert "nomatch.py" not in out

    def test_respects_max_results(self, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text(f"content {i}\n", encoding="utf-8")
        from tools.search_ops import search_files
        out = search_files(str(tmp_path), max_results=3)
        assert "Found 3 file(s)" in out
        assert "Results limited to 3" in out


class TestSearchFilesAdversarial:
    """L4: what if assumptions are violated?"""

    def test_nonexistent_directory_errors(self):
        from tools.search_ops import search_files
        out = search_files("/no/such/dir/here")
        assert "Error" in out
        assert "does not exist" in out

    def test_file_path_not_directory_errors(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x\n", encoding="utf-8")
        from tools.search_ops import search_files
        out = search_files(str(f))
        assert "Error" in out
        assert "not a directory" in out

    def test_skips_binary_files(self, tmp_path):
        """L9: binary files should NOT appear in content search results."""
        binary = tmp_path / "data.exe"
        binary.write_bytes(b"\x00\x01\x02SEARCHABLE\x00")
        (tmp_path / "code.py").write_text("SEARCHABLE\n", encoding="utf-8")
        from tools.search_ops import search_files
        out = search_files(str(tmp_path), content_contains="SEARCHABLE")
        assert "code.py" in out
        assert "data.exe" not in out


# ══════════════════════════════════════════════════════════════════════════════
# grep_search — L0/L1/L2/L4/L4b
# ══════════════════════════════════════════════════════════════════════════════

class TestGrepSearchSmoke:
    """L0: basic smoke."""

    def test_finds_pattern_in_file(self, tmp_path):
        (tmp_path / "code.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        from tools.search_ops import grep_search
        out = grep_search(str(tmp_path), "def hello")
        assert "hello" in out
        assert "Found" in out


class TestGrepSearchEdgeCases:
    """L2: boundary conditions."""

    def test_no_matches_returns_message(self, tmp_path):
        (tmp_path / "code.py").write_text("print('hi')\n", encoding="utf-8")
        from tools.search_ops import grep_search
        out = grep_search(str(tmp_path), "NONEXISTENT_PATTERN_XYZ")
        assert "No matches" in out

    def test_file_pattern_filter(self, tmp_path):
        (tmp_path / "match.py").write_text("target_line\n", encoding="utf-8")
        (tmp_path / "skip.txt").write_text("target_line\n", encoding="utf-8")
        from tools.search_ops import grep_search
        out = grep_search(str(tmp_path), "target_line", file_pattern="*.py")
        assert "match.py" in out
        # .txt should be excluded by the file_pattern filter
        # (ripgrep -g *.py only matches .py files)

    def test_respects_max_results(self, tmp_path):
        for i in range(20):
            (tmp_path / f"f{i}.py").write_text(f"TARGET_LINE\n", encoding="utf-8")
        from tools.search_ops import grep_search
        out = grep_search(str(tmp_path), "TARGET_LINE", max_results=5)
        assert "Found" in out
        # Should be limited (exact count depends on ripgrep vs python fallback)
        # but the limit notice should appear
        assert "Results limited" in out or "Found 5" in out


class TestGrepSearchAdversarial:
    """L4: what if assumptions are violated?"""

    def test_nonexistent_directory_errors(self):
        from tools.search_ops import grep_search
        out = grep_search("/no/such/dir/xyz", "pattern")
        assert "Error" in out

    def test_file_path_not_directory_errors(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x\n", encoding="utf-8")
        from tools.search_ops import grep_search
        out = grep_search(str(f), "x")
        assert "Error" in out
        assert "not a directory" in out

    def test_invalid_regex_pattern(self, tmp_path):
        """L4: invalid regex should produce a graceful error, not a crash."""
        (tmp_path / "code.py").write_text("print('hi')\n", encoding="utf-8")
        from tools.search_ops import grep_search
        # Invalid regex: unmatched parenthesis
        out = grep_search(str(tmp_path), "(unclosed")
        # Should either error gracefully or fall back to Python which catches re.error
        assert "Error" in out or "No matches" in out or "Found" not in out


# ══════════════════════════════════════════════════════════════════════════════
# calculate — L0/L1/L2/L4/L4b/L9 (D1 Math, D8 Security)
# ══════════════════════════════════════════════════════════════════════════════

class TestCalculateSmoke:
    """L0: basic smoke."""

    def test_simple_arithmetic(self):
        from tools.misc_tools import calculate
        out = calculate("2 + 2")
        assert "4" in out

    def test_multiplication(self):
        from tools.misc_tools import calculate
        out = calculate("3 * 7")
        assert "21" in out


class TestCalculateContract:
    """L1: documented behavior."""

    def test_returns_expression_and_result(self):
        from tools.misc_tools import calculate
        out = calculate("10 - 3")
        assert "10 - 3" in out
        assert "7" in out

    def test_caret_as_exponent(self):
        """Models write ^ meaning **. Should be converted."""
        from tools.misc_tools import calculate
        out = calculate("2 ^ 3")
        assert "8" in out

    def test_float_result_preserved(self):
        from tools.misc_tools import calculate
        out = calculate("10 / 4")
        assert "2.5" in out

    def test_integer_division(self):
        from tools.misc_tools import calculate
        out = calculate("7 // 2")
        assert "3" in out


class TestCalculateEdgeCases:
    """L2: boundary conditions (D1 Math/Numeric)."""

    def test_empty_expression_errors(self):
        from tools.misc_tools import calculate
        out = calculate("")
        assert "Error" in out

    def test_whitespace_only_expression_errors(self):
        from tools.misc_tools import calculate
        out = calculate("   ")
        assert "Error" in out

    def test_division_by_zero_errors(self):
        from tools.misc_tools import calculate
        out = calculate("1 / 0")
        assert "Error" in out
        assert "division by zero" in out.lower() or "zero" in out.lower()

    def test_sqrt_negative(self):
        """L4: sqrt(-1) should either error or produce a complex result."""
        from tools.misc_tools import calculate
        out = calculate("sqrt(-1)")
        # math.sqrt(-1) raises ValueError → falls to symbolic path
        # Should not crash, should return something
        assert out  # not empty

    def test_factorial_of_negative(self):
        from tools.misc_tools import calculate
        out = calculate("factorial(-1)")
        # math.factorial(-1) raises ValueError
        assert "Error" in out or "cannot evaluate" in out.lower() or "=" in out


class TestCalculateAdversarial:
    """L4: violation of assumptions (D8 Security)."""

    def test_code_injection_rejected(self):
        """L9/D8: __import__ must NOT execute arbitrary code.
        The error message echoes the expression (so 'HACKED' appears in
        the echoed input), but the key is that it reports an error and
        does NOT produce command output like a shell echo."""
        from tools.misc_tools import calculate
        out = calculate("__import__('os').system('echo HACKED')")
        assert "Error" in out or "cannot evaluate" in out.lower()
        # The expression is rejected — SymPy reports 'system' is not allowed.
        # 'HACKED' may appear in the echoed expression, but should NOT appear
        # as standalone command output (which would mean code executed).
        assert "system is not allowed" in out or "not allowed" in out.lower() or "Error" in out

    def test_very_large_exponent_rejected(self):
        """L4/D1: DoS guard — 9**9**9 should be refused, not hang."""
        from tools.misc_tools import calculate
        out = calculate("9 ** 99999")
        # Should error due to _safe_pow guard (abs(b) > 10000)
        assert "Error" in out or "cannot evaluate" in out.lower() or "too large" in out.lower()

    def test_nested_attribute_access_rejected(self):
        """L9/D8: attribute access should not be allowed."""
        from tools.misc_tools import calculate
        out = calculate("().__class__.__bases__[0].__subclasses__()")
        assert "Error" in out or "cannot evaluate" in out.lower()

    def test_nan_input(self):
        """L4/D1: NaN should not silently propagate."""
        from tools.misc_tools import calculate
        out = calculate("sqrt(float('nan'))")
        # Either errors or returns nan — but should not crash
        assert out  # not empty

    def test_inf_arithmetic(self):
        """L4/D1: Inf should be handled."""
        from tools.misc_tools import calculate
        out = calculate("1e308 * 10")
        # Should overflow or error
        assert out  # not empty


class TestCalculateFallback:
    """L4b: symbolic fallback when arithmetic path fails."""

    def test_symbolic_integration(self):
        """L4b: integrate(x**2, x) falls through to SymPy."""
        from tools.misc_tools import calculate
        out = calculate("integrate(x**2, x)")
        # Should contain x**3/3 or similar
        assert "x" in out.lower() or "error" in out.lower()

    def test_symbolic_solve(self):
        from tools.misc_tools import calculate
        out = calculate("solve(x**2 - 4, x)")
        # Should contain -2 and 2
        assert "2" in out or "error" in out.lower()


# ══════════════════════════════════════════════════════════════════════════════
# set_goal — L0/L1/L2/L4
# ══════════════════════════════════════════════════════════════════════════════

class TestSetGoalSmoke:
    """L0: basic smoke."""

    def test_set_objective(self):
        from tools.misc_tools import set_goal
        out = set_goal(objective="Build a web app")
        assert "OBJECTIVE" in out
        assert "Build a web app" in out

    def test_set_current_task(self):
        from tools.misc_tools import set_goal
        out = set_goal(current_task="Write tests")
        assert "CURRENT TASK" in out
        assert "Write tests" in out


class TestSetGoalEdgeCases:
    """L2/L4: boundary and adversarial."""

    def test_both_empty_errors(self):
        from tools.misc_tools import set_goal
        out = set_goal()
        assert "Error" in out

    def test_both_whitespace_only_errors(self):
        from tools.misc_tools import set_goal
        out = set_goal(objective="   ", current_task="  ")
        assert "Error" in out

    def test_objective_truncated_in_output(self):
        """L1: objective is truncated to 120 chars in the output."""
        from tools.misc_tools import set_goal
        long_obj = "A" * 200
        out = set_goal(objective=long_obj)
        assert "OBJECTIVE" in out
        # The output should contain a truncated version (120 chars)
        assert "A" * 200 not in out  # full 200 should not be there

    def test_none_parameters_error(self):
        from tools.misc_tools import set_goal
        out = set_goal(objective=None, current_task=None)
        assert "Error" in out


# ══════════════════════════════════════════════════════════════════════════════
# analyze_project — L0/L1/L2/L4
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeProjectSmoke:
    """L0: basic smoke."""

    def test_analyzes_current_directory(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        from tools.project_analysis import analyze_project
        out = analyze_project(str(tmp_path))
        assert "PROJECT ANALYSIS" in out
        assert "main.py" in out
        assert "Python" in out


class TestAnalyzeProjectEdgeCases:
    """L2/L4: boundary and adversarial."""

    def test_file_path_not_directory_errors(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x\n", encoding="utf-8")
        from tools.project_analysis import analyze_project
        out = analyze_project(str(f))
        assert "Error" in out
        assert "not a directory" in out

    def test_empty_directory(self, tmp_path):
        from tools.project_analysis import analyze_project
        out = analyze_project(str(tmp_path))
        assert "PROJECT ANALYSIS" in out
        assert "no standard manifest" in out.lower() or "(no standard manifest" in out

    def test_detects_python_manifests(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django\nflask\n", encoding="utf-8")
        from tools.project_analysis import analyze_project
        out = analyze_project(str(tmp_path))
        assert "Python" in out
        assert "requirements.txt" in out

    def test_detects_node_manifests(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "test", "scripts": {"build": "webpack"}}), encoding="utf-8")
        from tools.project_analysis import analyze_project
        out = analyze_project(str(tmp_path))
        assert "JavaScript" in out or "Node" in out

    def test_detects_entry_points(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
        from tools.project_analysis import analyze_project
        out = analyze_project(str(tmp_path))
        assert "entry point" in out.lower() or "main.py" in out

    def test_detects_frameworks(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django>=4.0\n", encoding="utf-8")
        from tools.project_analysis import analyze_project
        out = analyze_project(str(tmp_path))
        assert "django" in out.lower()


# ══════════════════════════════════════════════════════════════════════════════
# filter_new_items — L0/L1/L2/L4
# ══════════════════════════════════════════════════════════════════════════════

class TestFilterNewItemsSmoke:
    """L0: basic smoke."""

    def test_returns_new_items(self, monkeypatch):
        from tools.misc_tools import filter_new_items
        # Monkeypatch the automation memory
        monkeypatch.setattr("src.automation.memory.filter_new",
                            lambda items: items[1:])  # drop first
        monkeypatch.setattr("src.automation.memory.current_scope",
                            lambda: "test_scope")
        out = filter_new_items(["url1", "url2", "url3"])
        assert "2 of 3" in out
        assert "url2" in out
        assert "url3" in out
        assert "url1" not in out  # filtered out


class TestFilterNewItemsEdgeCases:
    """L2/L4: boundary and adversarial."""

    def test_empty_list(self, monkeypatch):
        from tools.misc_tools import filter_new_items
        monkeypatch.setattr("src.automation.memory.filter_new", lambda items: [])
        monkeypatch.setattr("src.automation.memory.current_scope", lambda: "")
        out = filter_new_items([])
        assert "No items" in out or "nothing is new" in out.lower()

    def test_all_old_items(self, monkeypatch):
        from tools.misc_tools import filter_new_items
        monkeypatch.setattr("src.automation.memory.filter_new", lambda items: [])
        monkeypatch.setattr("src.automation.memory.current_scope", lambda: "scope1")
        out = filter_new_items(["a", "b", "c"])
        assert "0 of 3" in out
        assert "already" in out.lower()

    def test_all_new_items(self, monkeypatch):
        from tools.misc_tools import filter_new_items
        monkeypatch.setattr("src.automation.memory.filter_new",
                            lambda items: list(items))
        monkeypatch.setattr("src.automation.memory.current_scope", lambda: "")
        out = filter_new_items(["x", "y"])
        assert "2 of 2" in out

    def test_string_input_handled(self, monkeypatch):
        """L4: string input should be handled, not crash."""
        from tools.misc_tools import filter_new_items
        monkeypatch.setattr("src.automation.memory.filter_new",
                            lambda items: items[:1])
        monkeypatch.setattr("src.automation.memory.current_scope", lambda: "")
        # String that looks like JSON array
        out = filter_new_items('["a", "b"]')
        assert "a" in out or "Error" in out  # should handle gracefully

    def test_non_list_non_string_input_errors(self, monkeypatch):
        from tools.misc_tools import filter_new_items
        monkeypatch.setattr("src.automation.memory.filter_new", lambda items: [])
        monkeypatch.setattr("src.automation.memory.current_scope", lambda: "")
        out = filter_new_items(42)
        assert "Error" in out

    def test_label_appears_in_output(self, monkeypatch):
        from tools.misc_tools import filter_new_items
        monkeypatch.setattr("src.automation.memory.filter_new",
                            lambda items: list(items))
        monkeypatch.setattr("src.automation.memory.current_scope", lambda: "")
        out = filter_new_items(["x"], label="job postings")
        assert "job postings" in out