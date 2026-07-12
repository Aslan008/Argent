"""run_admin_command must use a unique temp file (not a shared hardcoded path)
and always clean it up. ShellExecuteW is mocked so this runs without UAC."""

import base64
import re
from pathlib import Path

import tools.command_ops as co


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


def test_unique_temp_and_cleanup(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(co.ctypes, "windll", _FakeWindll(), raising=False)
    result = co.run_admin_command("whoami")
    path = _FakeShell32.captured_path
    assert "ADMIN OUTPUT OK" in result
    assert path != "C:/Windows/Temp/argent_admin_out.txt"  # not the hardcoded path
    assert "argent_admin_" in path                          # unique per-run name
    assert not Path(path).exists()                          # cleaned up in finally


def test_denied_by_user(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: False)
    assert "NOT run" in co.run_admin_command("whoami")


def test_uac_denied(monkeypatch):
    monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)

    class _S:
        def ShellExecuteW(self, *a):
            return 5  # <= 32 => failure / UAC denied

    class _W:
        shell32 = _S()

    monkeypatch.setattr(co.ctypes, "windll", _W(), raising=False)
    assert "denied or execution failed" in co.run_admin_command("whoami")
