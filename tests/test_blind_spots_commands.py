"""LMTrust blind spot test suite for Argent command execution tools (command_ops.py).
Covers L1 (Contract/CWD/Signatures), L1 (Encoding/Admin UTF-16LE), L4 (Stdin closure),
L5 (Background process slot leaks), L8 (Timeout wait deadlock), L9 (ANSI cleanup).
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tools.command_ops as cmd_ops
from tools.command_ops import (
    run_command,
    run_admin_command,
    start_background_command,
    read_background_command,
    stop_background_command,
    list_background_commands,
    ACTIVE_PROCESSES,
    ACTIVE_PROCESSES_LOCK,
)


@pytest.fixture(autouse=True)
def auto_approve(monkeypatch):
    """Auto-approve all commands in tests."""
    monkeypatch.setattr("tools.command_ops.request_approval", lambda *a, **k: True)
    monkeypatch.setattr("tools.command_ops.assess_command_risk", lambda c: ("safe", []))
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.clear()
    yield
    with ACTIVE_PROCESSES_LOCK:
        for pid, info in list(ACTIVE_PROCESSES.items()):
            try:
                info["process"].kill()
            except Exception:
                pass
        ACTIVE_PROCESSES.clear()


class TestCommandBlindSpotsLMTrust:
    """LMTrust Matrix tests for command execution reliability and capability coverage."""

    # 1. L1 x D7: run_command MUST support cwd parameter for scoping to subdirectories
    def test_l1_run_command_supports_cwd(self, tmp_path):
        sub_dir = tmp_path / "subproject"
        sub_dir.mkdir()
        marker = sub_dir / "marker.txt"
        marker.write_text("found me", encoding="utf-8")

        # Running python to check current directory
        py_cmd = "python -c \"import os; print('CWD:', os.path.basename(os.getcwd()))\""
        out = run_command(py_cmd, cwd=str(sub_dir))
        assert "Exit code: 0" in out
        assert "CWD: subproject" in out

    # 2. L1 x D7: run_command with non-existent cwd MUST return clear error, not crash
    def test_l1_run_command_nonexistent_cwd_returns_error(self, tmp_path):
        bad_dir = tmp_path / "non_existent_folder_12345"
        out = run_command("python -c \"print('hi')\"", cwd=str(bad_dir))
        assert "Error" in out or "not exist" in out.lower()

    # 3. L1 x D3: run_admin_command MUST decode UTF-16LE PowerShell redirection output
    def test_l1_run_admin_command_decodes_utf16le(self, monkeypatch):
        import base64
        import re

        # Simulate elevated command output with UTF-16LE BOM as PowerShell '>' writes
        test_text = "Успешная операция админа: OK"
        utf16_bytes = test_text.encode("utf-16le")

        def fake_shellexecute(hwnd, verb, file, params, cwd, show):
            m = re.search(r"-EncodedCommand (\S+)", params)
            script = base64.b64decode(m.group(1)).decode("utf-16-le")
            captured_path = re.search(r"> '([^']+)'", script).group(1)
            Path(captured_path).write_bytes(b"\xff\xfe" + utf16_bytes)
            return 42

        monkeypatch.setattr(
            cmd_ops.ctypes.windll.shell32,
            "ShellExecuteW",
            fake_shellexecute,
            raising=False,
        )

        result = run_admin_command("Write-Host 'test'")
        assert "Успешная операция админа: OK" in result
        assert "\x00" not in result  # Must not contain raw null bytes from bad UTF-8 decode

    # 4. L4 x D5: run_command MUST close stdin so commands expecting EOF don't hang
    def test_l4_run_command_closes_stdin_promptly(self):
        # Python script that reads stdin until EOF
        py_script = (
            "python -c \"import sys; data = sys.stdin.read(); "
            "print('Read:', repr(data))\""
        )
        t0 = time.time()
        out = run_command(py_script)
        elapsed = time.time() - t0
        assert "Exit code: 0" in out
        assert "Read: ''" in out
        # Must finish almost instantly with EOF, not hang for 180s silence limit
        assert elapsed < 5.0

    # 5. L5 x D5: start_background_command MUST auto-reap exited processes to avoid slot exhaustion
    def test_l5_background_command_auto_reaps_exited_processes(self):
        # Start and let 10 fast processes exit without reading them
        for i in range(cmd_ops.MAX_BACKGROUND_PROCESSES):
            res = start_background_command("python -c \"import sys; sys.exit(0)\"")
            assert "Started background process with PID:" in res

        # Wait for processes to exit (PowerShell processes take ~1.5-2.5s to finish startup & exit on Windows)
        for _ in range(50):
            if any(info["process"].poll() is not None for info in ACTIVE_PROCESSES.values()):
                break
            time.sleep(0.1)

        # 11th process should succeed if exited slots are auto-reaped
        res11 = start_background_command("python -c \"print('process 11')\"")
        assert "Maximum number of background processes" not in res11, \
            "Exited background processes must not permanently block the MAX_BACKGROUND_PROCESSES limit"
        assert "Started background process with PID:" in res11

    # 6. L9 x D3: Negative Space - run_command output MUST strip ANSI escape codes
    def test_l9_run_command_strips_ansi_escape_codes(self):
        # Emit colored text using ANSI escapes: \x1b[32mGREEN\x1b[0m
        py_cmd = "python -c \"print('\\x1b[32mSUCCESS_COLOR\\x1b[0m')\""
        out = run_command(py_cmd)
        assert "SUCCESS_COLOR" in out
        assert "\x1b[" not in out, "ANSI escape codes should be stripped from command output"

    # 7. L2 x D1: run_command supports custom timeout parameter
    def test_l2_run_command_supports_custom_timeout(self):
        # Sleep for 10s with 1s timeout
        py_cmd = "python -c \"import time; time.sleep(10)\""
        t0 = time.time()
        out = run_command(py_cmd, timeout=1)
        elapsed = time.time() - t0
        assert "terminated" in out.lower() or "timeout" in out.lower()
        assert elapsed < 5.0, "Command should terminate around custom timeout of 1s"

    # 8. L1 x D7: schemas.py for run_command documents cwd and timeout
    def test_l1_schema_documents_cwd_and_timeout(self):
        from tools.schemas import TOOL_SCHEMAS
        schema = next(t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == "run_command")
        props = schema["parameters"]["properties"]
        assert "cwd" in props, "run_command schema should document 'cwd' parameter"
        assert "timeout" in props, "run_command schema should document 'timeout' parameter"
