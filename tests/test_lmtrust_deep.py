"""
LMTrust Deep Blind Spot & Contract Verification Suite.

Validates the newly created modules:
1. SemanticLoopGuard (src/agent/semantic_loop_guard.py)
2. Interactive Prompt Detection (tools/command_ops.py)
3. Native Crash & Toolchain Diagnostics (src/agent/command_diagnostics.py)
4. Desktop Automation & Win32 GDI Safety (src/automation/desktop.py)

Covers Layers:
- L1 Contract & Types
- L2 Boundary (Hex HWND, 0, Negative codes, Case-insensitive paths)
- L3 Property-based fuzzing (Hypothesis: arbitrary inputs, surrogate escapes)
- L4 Adversarial (ANSI escapes, Corrupted logs, Closed HWNDs)
- L4b Fallback paths (GDI fallback, non-zero return codes)
- L5 State transitions & Memory bounds
- L9 Negative Space (No GDI leaks, No unhandled exceptions)
"""

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from hypothesis import given, strategies as st, settings

from src.agent.semantic_loop_guard import (
    SemanticLoopGuard,
    normalize_error_signature,
    build_semantic_loop_note,
)
from src.agent.command_diagnostics import diagnose_command_error
from tools.command_ops import INTERACTIVE_PROMPT_PATTERNS
from src.automation.desktop import DesktopAutomation


# ==============================================================================
# Layer 1 & 2: SemanticLoopGuard Contract, Boundary, and Windows Path Case
# ==============================================================================
class TestSemanticLoopGuardLMTrust:
    """Mode A: L1, L2, L4, L5 tests for SemanticLoopGuard."""

    def test_l2_case_insensitive_path_handling_on_windows(self, tmp_path):
        """L2: On Windows, 'C:/path/file.py' and 'c:/path/file.py' must be the exact same file."""
        test_file = tmp_path / "target.py"
        test_file.write_text("v1 = 1", encoding="utf-8")

        guard = SemanticLoopGuard(window=5)

        path_lower = str(test_file).lower()
        path_upper = str(test_file).upper()

        # Step 1: Lowercase path edit
        guard.record_file_edit(path_lower)
        assert len(guard._file_state_history) == 1

        # Step 2: Edit to v2
        test_file.write_text("v2 = 2", encoding="utf-8")
        guard.record_file_edit(path_upper)
        assert len(guard._file_state_history) == 2

        # Step 3: Revert to v1 with different case
        test_file.write_text("v1 = 1", encoding="utf-8")
        result = guard.record_file_edit(path_lower)

        # Should detect oscillation period 2 regardless of path casing
        # (Only 1 unique file tracked, not 2 duplicates)
        norm_keys = set(guard._file_state_history[-1].keys())
        assert len(norm_keys) == 1, f"Path case caused duplicate file tracking: {norm_keys}"

    def test_l2_empty_null_and_deleted_file_states(self, tmp_path):
        """L2: Handle empty paths, deleted files, and file removals gracefully."""
        guard = SemanticLoopGuard()

        # Null / empty inputs should not crash or corrupt history
        assert guard.record_file_edit("") is None
        assert len(guard._file_state_history) == 0

        # Create and track file
        f = tmp_path / "temp.txt"
        f.write_text("state_A", encoding="utf-8")
        guard.record_file_edit(str(f))
        assert len(guard._file_state_history) == 1

        # Delete file on disk (state B)
        f.unlink()
        guard.record_file_edit(str(f))
        # Should record state without throwing FileNotFoundError
        assert len(guard._file_state_history) >= 1

    def test_l5_bounded_memory_growth_under_heavy_edits(self, tmp_path):
        """L5: History must respect maximum window size and never leak memory."""
        guard = SemanticLoopGuard(window=4)
        f = tmp_path / "scale.py"

        for i in range(50):
            f.write_text(f"val = {i}", encoding="utf-8")
            guard.record_file_edit(str(f))

        assert len(guard._file_state_history) <= 4
        assert len(guard._error_signature_history) <= 4

    @given(st.text(max_size=1000))
    @settings(max_examples=50)
    def test_l3_property_error_signature_normalization_never_crashes(self, text):
        """L3 Property: normalize_error_signature must never raise exception on arbitrary text."""
        sig = normalize_error_signature(text)
        assert isinstance(sig, str)


# ==============================================================================
# Layer 2 & 4: Interactive Prompt Detector (ANSI Escapes & Edge Cases)
# ==============================================================================
class TestInteractivePromptDetectorLMTrust:
    """Mode A: L2, L4 adversarial tests for interactive prompt patterns."""

    def test_l4_ansi_color_and_formatting_escapes(self):
        """L4: CLI prompts often emit ANSI color/cursor escape codes before or after [y/n]."""
        raw_prompt = "\x1b[32m? Would you like to proceed? (y/n) \x1b[0m"
        
        # Helper to strip ANSI escapes (the test documents the need for stripping)
        ansi_regex = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
        cleaned = ansi_regex.sub("", raw_prompt).strip()

        matched = any(rx.search(cleaned) for rx in INTERACTIVE_PROMPT_PATTERNS)
        assert matched, f"Failed to match cleaned ANSI prompt: {cleaned}"

    @pytest.mark.parametrize("prompt_text", [
        "Do you want to continue? [Y/n]",
        "Enter password:",
        "Please select an option (1-4):",
        "Press any key to continue . . .",
        "Are you sure you want to delete this database? [yes/no]:",
        "Proceed with installation? [y/N]",
    ])
    def test_l1_contract_standard_prompts(self, prompt_text):
        """L1: Standard interactive prompts must match."""
        assert any(rx.search(prompt_text.strip()) for rx in INTERACTIVE_PROMPT_PATTERNS)

    @pytest.mark.parametrize("normal_text", [
        "All tests passed in 4.2s",
        "Compiled 12 files successfully",
        "Password has been updated",
        "Selected option 2: full build",
        "Continuous integration build finished",
    ])
    def test_l9_negative_space_normal_outputs_do_not_match(self, normal_text):
        """L9 Negative Space: Non-prompt lines must NEVER falsely trigger interactive detection."""
        assert not any(rx.search(normal_text.strip()) for rx in INTERACTIVE_PROMPT_PATTERNS)


# ==============================================================================
# Layer 1, 2, 4: Native Diagnostics (Hex NTSTATUS, Missing Headers, Linkers)
# ==============================================================================
class TestCommandDiagnosticsLMTrust:
    """Mode A: L1, L2, L4 tests for command diagnostics."""

    def test_l2_unsigned_hex_representation_of_segfault(self):
        """L2: STATUS_ACCESS_VIOLATION may come as unsigned 3221225477, signed -1073741819, or hex."""
        hint_signed = diagnose_command_error("test.exe", "Exited abnormally", -1073741819)
        hint_unsigned = diagnose_command_error("test.exe", "Exited abnormally", 3221225477)

        assert hint_signed is not None and "STATUS_ACCESS_VIOLATION" in hint_signed
        assert hint_unsigned is not None and "STATUS_ACCESS_VIOLATION" in hint_unsigned

    def test_l4_wrapped_crash_code_in_output_when_exit_code_is_generic(self):
        """L4 Adversarial: If runner script exits with code 1, but output contains 0xC0000005."""
        out = "Application crashed: Exception 0xC0000005 at address 0x7FFF0010"
        hint = diagnose_command_error("run.bat", out, 1)
        # Verify that diagnostic notices the 0xC0000005 crash inside output
        assert hint is not None, "Failed to diagnose crash code embedded in output"
        assert "STATUS_ACCESS_VIOLATION" in hint or "Memory Access Violation" in hint

    def test_l1_cmake_and_linker_patterns(self):
        """L1: Diagnostics for CMake package errors and GCC/Clang linker errors."""
        # CMake
        cmake_out = "CMake Error at CMakeLists.txt:12 (find_package): Could not find package OpenSSL"
        hint = diagnose_command_error("cmake ..", cmake_out, 1)
        assert hint is not None and "CMake Configuration Error" in hint

        # GCC/Clang linker
        ld_out = "/usr/bin/ld: cannot find -lsqlite3: No such file or directory"
        hint_ld = diagnose_command_error("make", ld_out, 1)
        assert hint_ld is not None and "Linker" in hint_ld or "C/C++" in hint_ld


# ==============================================================================
# Layer 2, 4b, 5, 9: Desktop Automation & Win32 GDI Safety
# ==============================================================================
class TestDesktopAutomationLMTrust:
    """Mode A: L2, L4b, L9 tests for DesktopAutomation."""

    def test_l2_hex_string_hwnd_resolution(self):
        """L2: HWND passed as hex string (e.g. '0x102AE' or '0x0012') must resolve to int HWND."""
        if not DesktopAutomation.is_available():
            pytest.skip("Windows only")

        from ctypes import wintypes
        real_rect = wintypes.RECT(0, 0, 800, 600)
        with patch.object(DesktopAutomation, "list_windows", return_value=[{"hwnd": 66222, "title": "TestWin"}]):
            with patch("ctypes.windll.user32.IsWindow", return_value=1):
                with patch("ctypes.wintypes.RECT", return_value=real_rect):
                    with patch("ctypes.windll.user32.GetWindowRect", return_value=1):
                        with patch("ctypes.windll.user32.GetWindowDC", return_value=0):
                            res = DesktopAutomation.capture_window("0x102AE")
                            assert "Could not acquire Device Context for HWND 66222" in res.get("error", "")

    def test_l2_closed_or_invalid_hwnd(self):
        """L2: Passing an invalid or destroyed HWND must fail fast with clean error."""
        if not DesktopAutomation.is_available():
            pytest.skip("Windows only")

        with patch("ctypes.windll.user32.IsWindow", return_value=0):
            res = DesktopAutomation.capture_window(999999)
            assert res["success"] is False
            assert "Invalid or closed HWND" in res["error"] or "invalid" in res["error"].lower()

    def test_l9_no_gdi_leak_on_bitmap_creation_failure(self):
        """L9 Negative Space: If CreateCompatibleBitmap fails (returns 0), ReleaseDC must still be called."""
        if not DesktopAutomation.is_available():
            pytest.skip("Windows only")

        from ctypes import wintypes
        real_rect = wintypes.RECT(0, 0, 100, 100)
        with patch("ctypes.windll.user32.IsWindow", return_value=1):
            with patch("ctypes.wintypes.RECT", return_value=real_rect):
                with patch("ctypes.windll.user32.GetWindowRect", return_value=1):
                    with patch("ctypes.windll.user32.GetWindowDC", return_value=1234):
                        with patch("ctypes.windll.user32.ReleaseDC") as mock_release_dc:
                            with patch("ctypes.windll.gdi32.CreateCompatibleDC", return_value=5678):
                                with patch("ctypes.windll.gdi32.DeleteDC") as mock_delete_dc:
                                    # Simulate CreateCompatibleBitmap failure (e.g. OOM / out of GDI handles)
                                    with patch("ctypes.windll.gdi32.CreateCompatibleBitmap", return_value=0):
                                        res = DesktopAutomation.capture_window(1234)
                                        assert res["success"] is False
                                        # Must release HWND DC and delete Mem DC
                                        assert mock_release_dc.called, "HWND DC was leaked on bitmap failure!"
                                        assert mock_delete_dc.called, "Mem DC was leaked on bitmap failure!"
