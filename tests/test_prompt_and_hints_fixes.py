"""Regression tests verifying fixes for agent prompts, diagnostics, loop guards, and match hints."""

import pytest
from src.agent.command_diagnostics import diagnose_command_error
from src.agent.healing import build_healing_hint
from argent.tools._helpers import _build_match_hint
from src.agent.loop_guard import build_loop_note
from src.agent import plan_mode


class TestCommandDiagnosticsFixes:
    def test_python_file_not_found_not_misdiagnosed_as_cmake(self):
        cmd = "python script.py"
        output = "Error: [Errno 2] No such file or directory: 'config.json'"
        hint = diagnose_command_error(cmd, output, 1)
        assert hint is not None
        assert "CMake" not in hint
        assert "target_include_directories" not in hint

    def test_taskkill_non_rust_does_not_mention_cargo(self):
        cmd = "taskkill /F /IM python.exe"
        output = "ERROR: The process 'python.exe' not found."
        hint = diagnose_command_error(cmd, output, 1)
        assert hint is not None
        assert "cargo" not in hint.lower()
        assert "rustc" not in hint.lower()
        assert ".cargo-lock" not in hint

    def test_powershell_cmdlet_not_mentioning_csharp(self):
        cmd = "git log | Select-Object -First 5"
        output = "'Select-Object' is not recognized as an internal or external command"
        hint = diagnose_command_error(cmd, output, 1)
        assert hint is not None
        assert "C#" not in hint
        assert "NOT the problem" in hint

    def test_python_traceback_with_file_not_found_diagnosed_as_python(self):
        cmd = "python main.py"
        output = (
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 12, in <module>\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'settings.json'"
        )
        hint = diagnose_command_error(cmd, output, 1)
        assert hint is not None
        assert "Python raised an exception" in hint
        assert "A file or directory in the command was not found" not in hint


class TestMatchHintOffsetFix:
    def test_match_hint_does_not_prepend_preceding_characters(self):
        content = "prefix_junk_before_line\nTARGET_LINE = 42\nother_code = 100\n"
        target = "TARGET_LINE = 42\n"
        hint = _build_match_hint(target, content)
        assert hint != ""
        # The snippet must NOT include prefix_junk_before_line
        assert "prefix_junk_before_line" not in hint
        assert "TARGET_LINE = 42" in hint


class TestHealingHintFix:
    def test_replace_in_file_does_not_mention_pytest(self):
        hint = build_healing_hint(attempt=1, func_name="replace_in_file")
        assert "Pytest AssertionError" not in hint
        assert "If you just ran tests" not in hint
        assert "replace_in_file" in hint


class TestLoopGuardNudgeFix:
    def test_loop_guard_nudge_does_not_hallucinate_browser(self):
        hint = build_loop_note(level="nudge")
        assert "browser_action" not in hint
        assert "wait_for" not in hint


class TestPlanModeApprovalInterception:
    def test_request_user_approval_in_plan_mode_redirects_to_exit_plan_mode(self):
        plan_mode.PLAN.enter()
        try:
            violation = plan_mode.violation("request_user_approval", {"message": "Please approve"})
            assert violation is not None
            assert "exit_plan_mode" in violation
        finally:
            plan_mode.PLAN.leave()
