"""run_command must not hang on a silent command — and must SHOW the prompt.

The watchdog worked, but the question never reached the screen: an interactive
prompt is written WITHOUT a trailing newline ("Ok to proceed? (y) ") because
the program expects the answer on the same line, and readline blocks until a
newline that never comes. The user saw a package being installed, then
silence, then a killed command — never the y/n.

The doubles here are real OS pipes on purpose. A fake exposing only readline()
cannot express "bytes arrived without a newline", which is the entire bug.
"""

import os
import threading

import tools.command_ops as command_ops
from tools.command_ops import run_command


class _Proc:
    """A process whose stdout is a real pipe the test writes into."""

    def __init__(self, chunks, close_at_end=True):
        read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb")
        self.returncode = 0
        self._done = False
        self._closed = False

        def _write():
            try:
                for chunk in chunks:
                    os.write(self._write_fd, chunk)
                if close_at_end:
                    self._close_write()
            except OSError:
                pass

        threading.Thread(target=_write, daemon=True).start()

    def _close_write(self):
        if not self._closed:
            self._closed = True
            try:
                os.close(self._write_fd)
            except OSError:
                pass

    def poll(self):
        return None if not self._done else self.returncode

    def terminate(self):
        self._done = True
        self.returncode = -15
        self._close_write()          # unblock the reader -> EOF

    def wait(self):
        return self.returncode


def _approve_all(monkeypatch):
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(command_ops, "assess_command_risk", lambda c: ("safe", []))


def test_silent_interactive_command_is_terminated(monkeypatch):
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 0.3)
    proc = _Proc([b"Package name:\n"], close_at_end=False)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("npm init")
    assert "start_background_command" in out
    assert proc._done                      # actually terminated, no eternal hang


def test_normal_command_completes_without_watchdog(monkeypatch):
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 5)
    proc = _Proc([b"hello\n", b"world\n"])
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("echo hello")
    assert "hello" in out and "world" in out
    assert "awaits interactive input" not in out


def test_a_prompt_without_a_newline_is_shown(monkeypatch):
    """The reported failure: npx asks "Ok to proceed? (y)" on an unterminated
    line, and the user saw nothing at all."""
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 0.3)
    proc = _Proc([b"Need to install the following packages:\n  cowsay@1.0\n",
                  b"Ok to proceed? (y) "], close_at_end=False)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("npx cowsay hi")
    assert "Ok to proceed? (y)" in out
    assert "cowsay@1.0" in out


def test_the_diagnosis_quotes_what_it_is_waiting_on(monkeypatch):
    """"No output for 5s" is true and unhelpful; the pending text says which
    question went unanswered."""
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 0.3)
    proc = _Proc([b"Ok to proceed? (y) "], close_at_end=False)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("npx cowsay hi")
    assert "waiting for an answer to" in out and "Ok to proceed?" in out


def test_output_without_a_trailing_newline_is_not_lost(monkeypatch):
    """A command whose last line lacks \\n used to drop that line entirely."""
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 5)
    proc = _Proc([b"first\n", b"last line without newline"])
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("echo test")
    assert "last line without newline" in out


def test_a_line_split_across_chunks_is_reassembled(monkeypatch):
    """os.read returns whatever has arrived, so one line can span reads."""
    _approve_all(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 5)
    proc = _Proc([b"hel", b"lo wor", b"ld\n"])
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("echo hello world")
    assert "hello world" in out
