"""Blind-spot tests for tools/system_ops.py.

Covers read_event_logs, get_process_info, query_registry, search_system_files,
and the internal _run_powershell helper.
"""

import os
import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from tools.system_ops import (
    _HIVES,
    _MAX_FILE_RESULTS,
    _run_powershell,
    get_process_info,
    query_registry,
    read_event_logs,
    search_system_files,
)


def _completed(stdout="", stderr="", returncode=0):
    """Build a minimal stand-in for subprocess.CompletedProcess."""
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    # MagicMock stringifies fine; .strip() works because it's a str
    return cp


# ---------------------------------------------------------------------------
# read_event_logs
# ---------------------------------------------------------------------------


class TestReadEventLogs:
    def test_windows_returns_json(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(
                stdout='[{"ID":1,"EntryType":"Error","Message":"test"}]'
            ),
        )
        result = read_event_logs()
        assert '"ID":1' in result
        assert "Error" in result

    def test_windows_empty_stdout(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(stdout="   "),
        )
        result = read_event_logs()
        # Empty stdout with no specific "No events" stderr -> generic error/empty
        assert "Error or empty result" in result or "No matching events" in result

    def test_windows_no_events_stderr(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(
                stdout="", stderr="No events were found that match the selection",
            ),
        )
        result = read_event_logs()
        assert result == "No matching events found in the log."

    def test_non_windows_platform(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        result = read_event_logs()
        assert "only supported on Windows" in result

    def test_log_name_parameter_passed(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        captured = {}

        def fake_ps(script, extra_env=None):
            captured["env"] = extra_env
            return _completed(stdout='[{"ID":1}]')

        monkeypatch.setattr("tools.system_ops._run_powershell", fake_ps)
        read_event_logs(log_name="System")
        assert captured["env"]["ARGENT_LOG"] == "System"

    def test_invalid_log_name(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        result = read_event_logs(log_name="App;Evil")
        assert "invalid log name" in result

    def test_entry_type_filter_passed(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        captured = {}

        def fake_ps(script, extra_env=None):
            captured["env"] = extra_env
            return _completed(stdout='[{"ID":1}]')

        monkeypatch.setattr("tools.system_ops._run_powershell", fake_ps)
        read_event_logs(entry_type="Critical,Error")
        assert "Critical" in captured["env"]["ARGENT_LEVELS"]
        assert "Error" in captured["env"]["ARGENT_LEVELS"]

    def test_invalid_entry_type(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        result = read_event_logs(entry_type="Bogus")
        assert "unknown entry type" in result

    def test_timeout(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")

        def fake_ps(script, extra_env=None):
            raise subprocess.TimeoutExpired(cmd="powershell", timeout=30)

        monkeypatch.setattr("tools.system_ops._run_powershell", fake_ps)
        result = read_event_logs()
        assert "timed out" in result

    def test_newest_clamped(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        captured = {}

        def fake_ps(script, extra_env=None):
            captured["env"] = extra_env
            return _completed(stdout='[{"ID":1}]')

        monkeypatch.setattr("tools.system_ops._run_powershell", fake_ps)
        read_event_logs(newest=99999)
        assert int(captured["env"]["ARGENT_MAX"]) <= 500

    def test_nonzero_returncode(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(
                stdout="", stderr="something broke", returncode=1
            ),
        )
        result = read_event_logs()
        assert "Error or empty result" in result


# ---------------------------------------------------------------------------
# get_process_info
# ---------------------------------------------------------------------------


class TestGetProcessInfo:
    def test_windows_returns_json(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(
                stdout='[{"Name":"chrome","Id":123}]'
            ),
        )
        result = get_process_info()
        assert "chrome" in result

    def test_windows_empty_stdout(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(stdout=""),
        )
        result = get_process_info()
        assert result == "Process not found or error occurred."

    def test_non_windows_platform(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        result = get_process_info()
        assert "only supported on Windows" in result

    def test_process_name_filter_passed(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        captured = {}

        def fake_ps(script, extra_env=None):
            captured["env"] = extra_env
            return _completed(stdout='[{"Name":"chrome"}]')

        monkeypatch.setattr("tools.system_ops._run_powershell", fake_ps)
        get_process_info(process_name="chrome")
        assert captured["env"]["ARGENT_PROC"] == "chrome"

    def test_timeout(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")

        def fake_ps(script, extra_env=None):
            raise subprocess.TimeoutExpired(cmd="powershell", timeout=30)

        monkeypatch.setattr("tools.system_ops._run_powershell", fake_ps)
        result = get_process_info()
        assert "timed out" in result

    def test_nonzero_returncode(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            "tools.system_ops._run_powershell",
            lambda script, extra_env=None: _completed(
                stdout="", stderr="err", returncode=1
            ),
        )
        result = get_process_info()
        assert result == "Process not found or error occurred."


# ---------------------------------------------------------------------------
# query_registry
# ---------------------------------------------------------------------------


class TestQueryRegistryPathParsing:
    """Tests that don't require winreg to actually function — they exercise the
    path-parsing and root-resolution logic which runs before OpenKey."""

    def _make_winreg(self):
        """Create a fake winreg module with the hive constants."""
        wr = types.ModuleType("winreg")
        wr.HKEY_LOCAL_MACHINE = 0x80000002
        wr.HKEY_CURRENT_USER = 0x80000001
        wr.HKEY_CLASSES_ROOT = 0x80000000
        wr.HKEY_USERS = 0x80000003
        wr.HKEY_CURRENT_CONFIG = 0x80000005
        return wr

    def test_non_windows_platform(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        result = query_registry("HKLM:\\Software")
        assert "only available on Windows" in result

    def test_empty_path(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        result = query_registry("")
        assert "registry path is required" in result

    def test_whitespace_only_path(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        result = query_registry("   ")
        assert "registry path is required" in result

    def test_unknown_root_lists_available(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKXX:\\Software")
        assert "unknown registry root" in result
        for hive in sorted(_HIVES):
            assert hive in result

    def test_forward_slashes_converted(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()
        # OpenKey should receive backslashes
        opened = {}

        def fake_openkey(hive, sub, **kw):
            opened["sub"] = sub
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryInfoKey = lambda key: (0, 0, 0)
        monkeypatch.setitem(sys.modules, "winreg", wr)
        query_registry("HKLM:/Software/Microsoft")
        assert "/" not in opened["sub"]
        assert "\\" in opened["sub"]

    def test_hklm_drive_form_resolves(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()
        opened = {}

        def fake_openkey(hive, sub, **kw):
            opened["hive"] = hive
            opened["sub"] = sub
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryInfoKey = lambda key: (0, 0, 0)
        monkeypatch.setitem(sys.modules, "winreg", wr)
        query_registry("HKLM:\\Software\\Microsoft")
        assert opened["hive"] == wr.HKEY_LOCAL_MACHINE
        assert opened["sub"] == "Software\\Microsoft"

    def test_full_hive_name_resolves(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()
        opened = {}

        def fake_openkey(hive, sub, **kw):
            opened["hive"] = hive
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryInfoKey = lambda key: (0, 0, 0)
        monkeypatch.setitem(sys.modules, "winreg", wr)
        query_registry("HKEY_LOCAL_MACHINE\\Software")
        assert opened["hive"] == wr.HKEY_LOCAL_MACHINE


class TestQueryRegistryValues:
    def _make_winreg(self):
        wr = types.ModuleType("winreg")
        wr.HKEY_LOCAL_MACHINE = 0x80000002
        wr.HKEY_CURRENT_USER = 0x80000001
        wr.HKEY_CLASSES_ROOT = 0x80000000
        wr.HKEY_USERS = 0x80000003
        wr.HKEY_CURRENT_CONFIG = 0x80000005
        return wr

    def test_name_parameter_returns_value(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()

        def fake_openkey(hive, sub, **kw):
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryValueEx = lambda key, name: ("1.0", 1)
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKLM:\\Software\\Microsoft", name="Version")
        assert "Version" in result
        assert "1.0" in result

    def test_name_not_found(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()

        def fake_openkey(hive, sub, **kw):
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryValueEx = lambda key, name: (_ for _ in ()).throw(FileNotFoundError())
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKLM:\\Software\\Microsoft", name="Missing")
        assert "not found" in result

    def test_key_not_found(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()

        def fake_openkey(hive, sub, **kw):
            raise FileNotFoundError()

        wr.OpenKey = fake_openkey
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKLM:\\NonExistent")
        assert "not found" in result

    def test_permission_denied(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()

        def fake_openkey(hive, sub, **kw):
            raise PermissionError()

        wr.OpenKey = fake_openkey
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKLM:\\Software")
        assert "access denied" in result

    def test_key_with_no_values(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()

        def fake_openkey(hive, sub, **kw):
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryInfoKey = lambda key: (0, 0, 0)  # 0 values
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKLM:\\Software\\Empty")
        assert "no values" in result

    def test_key_enumerates_values(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        wr = self._make_winreg()

        def fake_openkey(hive, sub, **kw):
            ctx = MagicMock()
            ctx.__enter__ = lambda self: ctx
            ctx.__exit__ = lambda self, *a: None
            return ctx

        wr.OpenKey = fake_openkey
        wr.QueryInfoKey = lambda key: (0, 2, 0)  # 2 values
        wr.EnumValue = lambda key, i: [(None, "dataA", 1), ("ValB", "dataB", 1)][i]
        monkeypatch.setitem(sys.modules, "winreg", wr)
        result = query_registry("HKLM:\\Software\\Test")
        assert "(Default)" in result
        assert "dataA" in result
        assert "ValB" in result


# ---------------------------------------------------------------------------
# search_system_files
# ---------------------------------------------------------------------------


class TestSearchSystemFiles:
    def test_basic_pattern_returns_all(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.log").write_text("world")
        result = search_system_files(str(tmp_path))
        assert "a.txt" in result
        assert "b.log" in result

    def test_filter_pattern_txt_only(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.log").write_text("world")
        result = search_system_files(str(tmp_path), filter_pattern="*.txt")
        assert "a.txt" in result
        assert "b.log" not in result

    def test_output_includes_size(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")  # 5 bytes
        result = search_system_files(str(tmp_path), filter_pattern="*.txt")
        assert "bytes" in result
        assert "5" in result

    def test_recursive_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")
        result = search_system_files(str(tmp_path), filter_pattern="*.txt")
        assert "deep.txt" in result

    def test_max_results_limit(self, tmp_path):
        # Create more files than _MAX_FILE_RESULTS
        for i in range(_MAX_FILE_RESULTS + 10):
            (tmp_path / f"file_{i}.txt").write_text("x")
        result = search_system_files(str(tmp_path), filter_pattern="*.txt")
        assert "stopped at" in result
        assert str(_MAX_FILE_RESULTS) in result

    def test_non_existent_path(self):
        result = search_system_files("C:/nonexistent_path_xyz_123")
        assert "does not exist" in result

    def test_file_not_dir(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("x")
        result = search_system_files(str(f))
        assert "not a directory" in result

    def test_empty_path(self):
        result = search_system_files("")
        assert "search path is required" in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        result = search_system_files(str(tmp_path), filter_pattern="*.nonexist")
        assert "No files found" in result

    def test_whitespace_path(self):
        result = search_system_files("   ")
        assert "search path is required" in result

    def test_empty_filter_defaults_to_star(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        result = search_system_files(str(tmp_path), filter_pattern="")
        assert "a.txt" in result


# ---------------------------------------------------------------------------
# _run_powershell
# ---------------------------------------------------------------------------


class TestRunPowershell:
    def test_constructs_proper_command(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _completed(stdout="ok")

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        _run_powershell("Write-Output test")
        assert captured["cmd"][0] == "powershell"
        assert "-NoProfile" in captured["cmd"]
        assert "-NonInteractive" in captured["cmd"]
        assert "-Command" in captured["cmd"]
        assert captured["cmd"][-1] == "Write-Output test"

    def test_extra_env_merged(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            return _completed(stdout="ok")

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        monkeypatch.setattr(os, "environ", {"EXISTING": "1"})
        _run_powershell("script", extra_env={"ARGENT_LOG": "System"})
        assert captured["env"]["EXISTING"] == "1"
        assert captured["env"]["ARGENT_LOG"] == "System"

    def test_extra_env_values_stringified(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            return _completed(stdout="ok")

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        _run_powershell("script", extra_env={"ARGENT_MAX": 50})
        assert captured["env"]["ARGENT_MAX"] == "50"
        assert isinstance(captured["env"]["ARGENT_MAX"], str)

    def test_encoding_utf8(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["kwargs"] = kwargs
            return _completed(stdout="ok")

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        _run_powershell("script")
        assert captured["kwargs"].get("encoding") == "utf-8"
        assert captured["kwargs"].get("errors") == "replace"

    def test_timeout_value(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["kwargs"] = kwargs
            return _completed(stdout="ok")

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        _run_powershell("script")
        assert captured["kwargs"].get("timeout") == 30

    def test_capture_output(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["kwargs"] = kwargs
            return _completed(stdout="ok")

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        _run_powershell("script")
        assert captured["kwargs"].get("capture_output") is True
        assert captured["kwargs"].get("text") is True

    def test_timeout_propagates(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=30)

        monkeypatch.setattr("tools.system_ops.subprocess.run", fake_run)
        with pytest.raises(subprocess.TimeoutExpired):
            _run_powershell("script")