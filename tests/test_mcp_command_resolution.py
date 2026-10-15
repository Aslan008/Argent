"""An MCP server's `command` is found through the user's PATH, not through the
directory of whichever interpreter hosts Argent.

Regression: the sarper server (`command: python`) started from the CLI but failed
under the Rust TUI with "Failed to start MCP server", because Windows resolves a
bare name next to the *calling* executable first. The TUI runs from
`.venv\\Scripts\\python.exe`, so `python` became the bare interpreter behind that
venv, which lacks the server's dependencies.
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock

import mcp_client


def _start_and_capture_argv(monkeypatch, command, env=None):
    popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", popen)
    t = mcp_client.StdioTransport(command, ["-u", "server.py"], env=env)
    t.send_request = MagicMock(return_value={"result": {"serverInfo": {}}})
    t.send_notification = MagicMock()
    assert t.start() is True
    return popen.call_args.args[0]


def test_command_is_resolved_through_path(tmp_path, monkeypatch):
    name = "argent_fake_server.cmd" if os.name == "nt" else "argent_fake_server"
    exe = tmp_path / name
    exe.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))

    argv = _start_and_capture_argv(monkeypatch, "argent_fake_server")

    assert os.path.normcase(argv[0]) == os.path.normcase(str(exe))
    assert argv[1:] == ["-u", "server.py"]


def test_server_env_path_wins_over_the_process_path(tmp_path, monkeypatch):
    """A server entry that sets its own PATH is resolved against that PATH."""
    name = "argent_fake_server.cmd" if os.name == "nt" else "argent_fake_server"
    exe = tmp_path / name
    exe.write_text("")
    exe.chmod(0o755)

    argv = _start_and_capture_argv(
        monkeypatch, "argent_fake_server", env={"PATH": str(tmp_path)}
    )

    assert os.path.normcase(argv[0]) == os.path.normcase(str(exe))


def test_unknown_command_is_passed_through_unchanged(monkeypatch):
    """Nothing on PATH: keep the name so Popen raises its usual, readable error."""
    argv = _start_and_capture_argv(monkeypatch, "definitely-not-installed-anywhere")
    assert argv[0] == "definitely-not-installed-anywhere"


def test_python_resolves_to_the_path_interpreter_not_the_host(monkeypatch):
    """The exact failure: with a venv interpreter hosting Argent, `python` must
    still mean the first `python` on PATH."""
    import shutil

    expected = shutil.which("python")
    if expected is None:
        return
    argv = _start_and_capture_argv(monkeypatch, "python")
    assert os.path.normcase(argv[0]) == os.path.normcase(expected)
    assert argv[0] != sys.executable or expected == sys.executable
