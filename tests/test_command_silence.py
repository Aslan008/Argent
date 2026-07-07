"""run_command must not hang forever on a silent/interactive command."""

import threading

import tools.command_ops as command_ops
from tools.command_ops import run_command


class _Pipe:
    """A fake stdout: yields queued lines, then blocks (simulating a program
    waiting on stdin) until terminate() releases it with EOF."""

    def __init__(self, lines, block_at_end=True):
        self._lines = list(lines)
        self._block = block_at_end
        self._released = threading.Event()

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        if self._block:
            self._released.wait()  # block like a process awaiting input
        return b""                 # EOF: the command finished

    def close(self):
        pass


class _Proc:
    def __init__(self, pipe):
        self.stdout = pipe
        self.returncode = 0
        self._done = False

    def poll(self):
        return None if not self._done else self.returncode

    def terminate(self):
        self._done = True
        self.returncode = -15
        self.stdout._released.set()   # unblock the reader -> EOF

    def wait(self):
        return self.returncode


def _approve_all(monkeypatch):
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(command_ops, "assess_command_risk", lambda c: ("safe", []))


def test_silent_interactive_command_is_terminated(monkeypatch):
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 0.3)
    # Prints a prompt line, then blocks forever waiting for input.
    proc = _Proc(_Pipe([b"Package name:\n"]))
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("npm init")
    assert "awaits interactive input" in out
    assert "start_background_command" in out
    assert proc._done                      # actually terminated, no eternal hang


def test_normal_command_completes_without_watchdog(monkeypatch):
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 5)
    proc = _Proc(_Pipe([b"hello\n", b"world\n"], block_at_end=False))   # finishes -> EOF
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("echo hello")
    assert "hello" in out and "world" in out
    assert "awaits interactive input" not in out
