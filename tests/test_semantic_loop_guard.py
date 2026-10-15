"""Tests for SemanticLoopGuard in Argent."""

import os
from pathlib import Path
import pytest

from src.agent.semantic_loop_guard import (
    SemanticLoopGuard,
    normalize_error_signature,
    build_semantic_loop_note,
)


class TestSemanticNormalization:
    """Verifies that volatile error details are collapsed to stable canonical signatures."""

    def test_pytest_failed_test_name_extracted(self):
        output = """
=================================== FAILURES ===================================
FAILED tests/test_math.py::test_addition - AssertionError: assert 1 == 2
FAILED tests/test_solver.py::test_run - ValueError: invalid param
========================= 2 failed in 0.42s =========================
"""
        sig = normalize_error_signature(output)
        assert sig == "pytest_failed:tests/test_math.py::test_addition,tests/test_solver.py::test_run"

    def test_python_exception_volatile_tokens_normalized(self):
        output = """
Traceback (most recent call last):
  File "C:\\Users\\dev\\project\\calc.py", line 42, in solve
    result = compute(0x7ffd1234, "2026-09-05 19:30:00")
ZeroDivisionError: division by zero in 15.3s
"""
        sig = normalize_error_signature(output)
        assert "py_exc:ZeroDivisionError" in sig
        assert "<elapsed>" in sig or "division by zero" in sig

    def test_c_cpp_error_signature_extracted(self):
        output = """
Physics.cpp(45): error C2065: 'gravity': undeclared identifier
Physics.cpp(50): fatal error C1083: Cannot open include file: 'glm.h': No such file or directory
"""
        sig = normalize_error_signature(output)
        assert "c_error:" in sig
        assert "error C2065" in sig or "C1083" in sig


class TestSemanticOscillationDetection:
    """Verifies state oscillation and ping-pong loop detection."""

    def test_edit_revert_oscillation_detected(self, tmp_path):
        guard = SemanticLoopGuard(window=8)
        f = tmp_path / "code.py"

        # State A (step 1)
        f.write_text("def run(): return 1\n", encoding="utf-8")
        assert guard.record_file_edit(str(f)) is None

        # State B (step 2)
        f.write_text("def run(): return 2\n", encoding="utf-8")
        assert guard.record_file_edit(str(f)) is None

        # State A (step 3 - revert back to A)
        f.write_text("def run(): return 1\n", encoding="utf-8")
        assert guard.record_file_edit(str(f)) is None  # first revert -> count 1

        # State B (step 4 - revert back to B)
        f.write_text("def run(): return 2\n", encoding="utf-8")
        level = guard.record_file_edit(str(f))  # second revert -> count 2 -> warn
        assert level == "warn"

        note = build_semantic_loop_note(level, "edit_revert")
        assert "SEMANTIC REGRESSION GUARD" in note
        assert "edit-revert loop" in note

    def test_error_ping_pong_detected(self):
        guard = SemanticLoopGuard(window=8)

        err_a = "FAILED tests/test_a.py::test_one - AssertionError"
        err_b = "FAILED tests/test_b.py::test_two - AssertionError"

        # Step 1: Error A
        assert guard.record_command_result("pytest", err_a, exit_code=1) is None
        # Step 2: Error B
        assert guard.record_command_result("pytest", err_b, exit_code=1) is None
        # Step 3: Error A repeats (first ping pong)
        assert guard.record_command_result("pytest", err_a, exit_code=1) is None
        # Step 4: Error B repeats (second ping pong -> triggers warn)
        level = guard.record_command_result("pytest", err_b, exit_code=1)
        assert level == "warn"

        note = build_semantic_loop_note(level, "error_ping_pong")
        assert "SEMANTIC REGRESSION GUARD" in note
        assert "error ping-pong cycle" in note

    def test_reset_clears_history(self, tmp_path):
        guard = SemanticLoopGuard(window=8)
        f = tmp_path / "test.py"

        f.write_text("x = 1\n", encoding="utf-8")
        guard.record_file_edit(str(f))
        f.write_text("x = 2\n", encoding="utf-8")
        guard.record_file_edit(str(f))

        guard.reset()
        assert len(guard._file_state_history) == 0
        assert len(guard._error_signature_history) == 0
        assert guard._revert_count == 0
