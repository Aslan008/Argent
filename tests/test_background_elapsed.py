"""Polling a running background process must yield DISTINCT results (elapsed
time) so the loop guard doesn't force-stop a legitimate wait."""

import queue

import tools.command_ops as command_ops


class _FakeProc:
    def __init__(self, code=None):
        self._code = code

    def poll(self):
        return self._code


def _register(pid, code=None, started=100.0):
    command_ops.ACTIVE_PROCESSES[pid] = {
        "process": _FakeProc(code),
        "out_queue": queue.Queue(),
        "err_queue": queue.Queue(),
        "command": "build",
        "started": started,
    }


def test_running_status_has_elapsed_and_varies(monkeypatch):
    pid = "tst-run"
    _register(pid, code=None, started=100.0)
    clock = iter([103.0, 108.0])
    monkeypatch.setattr(command_ops.time, "time", lambda: next(clock))
    try:
        r1 = command_ops.read_background_command(pid)
        r2 = command_ops.read_background_command(pid)
    finally:
        command_ops.ACTIVE_PROCESSES.pop(pid, None)

    assert "RUNNING for 3s" in r1
    assert "RUNNING for 8s" in r2
    assert r1 != r2                     # loop guard sees distinct signatures


def test_list_shows_running_elapsed(monkeypatch):
    pid = "tst-list"
    _register(pid, code=None, started=50.0)
    monkeypatch.setattr(command_ops.time, "time", lambda: 62.0)
    try:
        out = command_ops.list_background_commands()
    finally:
        command_ops.ACTIVE_PROCESSES.pop(pid, None)
    assert "RUNNING 12s" in out
