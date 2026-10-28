"""Elevation: one flag on `run_command`, and the temp-file discipline behind it.

`run_admin_command` used to be a tool of its own, advertised next to `run_command`.
Two nearly identical choices, and a model that picked the elevated one for an ordinary
command got a UAC prompt for nothing. It is now `run_command(admin=true)` — the name is
gone from the API, and these tests hold both halves: the routing and the plumbing that
made the old tool safe (a unique temp file, always cleaned up).

`ShellExecuteW` is mocked, so this runs without a UAC prompt.
"""

import base64
import re
from pathlib import Path

import argent.tools.command_ops as co


class _FakeShell32:
    captured_path = None

    def ShellExecuteW(self, hwnd, verb, file, params, cwd, show):
        # params is now: -NoProfile -EncodedCommand <base64 of UTF-16LE script>
        m = re.search(r"-EncodedCommand (\S+)", params)
        script = base64.b64decode(m.group(1)).decode("utf-16-le")
        _FakeShell32.captured_path = re.search(r"> '([^']+)'", script).group(1)
        Path(_FakeShell32.captured_path).write_text("ADMIN OUTPUT OK", encoding="utf-8")
        return 42  # > 32 => ShellExecute success


class _FakeWindll:
    shell32 = _FakeShell32()


def test_admin_true_on_run_command_takes_the_elevated_path(monkeypatch):
    """One tool, one flag: the model no longer chooses between two commands that differ
    only in a UAC prompt — and an ordinary command cannot be sent to UAC by accident."""
    seen = {}

    def fake_elevated(command):
        seen["command"] = command
        return "elevated ok"

    monkeypatch.setattr(co, "_run_elevated", fake_elevated)

    assert co.run_command("net stop spooler", admin=True) == "elevated ok"
    assert seen["command"] == "net stop spooler"


def test_there_is_no_second_admin_tool():
    from argent.tools import AVAILABLE_TOOLS
    from argent.tools.schemas import TOOL_SCHEMAS

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "run_admin_command" not in names, "старое имя не должно предлагаться модели"
    assert "run_admin_command" not in AVAILABLE_TOOLS, "и не должно быть вызываемым"
    assert "admin" in {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}[
        "run_command"]["parameters"]["properties"]


def test_unique_temp_and_cleanup(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(co.ctypes, "windll", _FakeWindll(), raising=False)
    result = co.run_command("whoami", admin=True)
    path = _FakeShell32.captured_path
    assert "ADMIN OUTPUT OK" in result
    assert path != "C:/Windows/Temp/argent_admin_out.txt"  # not the hardcoded path
    assert "argent_admin_" in path                          # unique per-run name
    assert not Path(path).exists()                          # cleaned up in finally


def test_denied_by_user(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: False)
    assert "NOT run" in co.run_command("whoami", admin=True)


def test_uac_denied(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)

    class _S:
        def ShellExecuteW(self, *a):
            return 5  # <= 32 => failure / UAC denied

    class _W:
        shell32 = _S()

    monkeypatch.setattr(co.ctypes, "windll", _W(), raising=False)
    assert "denied or execution failed" in co.run_command("whoami", admin=True)
