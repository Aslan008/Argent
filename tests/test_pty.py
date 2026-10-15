"""Commands on a pseudo-terminal: a program that asks a question can be answered."""

import sys
import time

import pytest

import tools.command_ops as command_ops
from src.agent import pty_process
from src.agent.pty_process import PtyPopen, PtyUnavailable, TerminalText

needs_pty = pytest.mark.skipif(not pty_process.available(), reason="no pseudo-terminal support here")

ASK = ("import sys\n"
       "name = input('Name: ')\n"
       "print('hi', name)\n"
       "sys.exit(3)\n")


def _read_until(process, needle: str, seconds: float = 20.0) -> str:
    """Collect what the terminal printed until `needle` shows up (or the time is out)."""
    import threading
    text, done = [], threading.Event()

    def pump():
        buffer = TerminalText()
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            text.append(buffer.feed(chunk.decode("utf-8", "replace")))
            if needle in "".join(text):
                break
        done.set()

    threading.Thread(target=pump, daemon=True).start()
    done.wait(seconds)
    return "".join(text)


# --- plain text out of terminal output ---------------------------------------------------------------------

class TestTerminalText:
    def test_colours_and_cursor_moves_go(self):
        raw = "\x1b[1;32mOK\x1b[0m done\x1b[2K\x1b[1;7H"
        assert TerminalText().feed(raw) == "OK done"

    def test_title_sequences_go(self):
        assert TerminalText().feed("\x1b]0;my title\x07text\x1b]8;;http://x\x1b\\link") == "textlink"

    def test_a_sequence_cut_between_chunks_is_held(self):
        t = TerminalText()
        first = t.feed("abc\x1b[3")
        second = t.feed("1mred\x1b[0m")
        assert first + second == "abcred"

    def test_a_redrawn_line_keeps_its_last_version(self):
        assert TerminalText().feed("10%\r50%\r100%\r\nready\r\n") == "100%\nready\n"

    def test_plain_text_is_untouched(self):
        assert TerminalText().feed("Name: Anna\nПривет\n") == "Name: Anna\nПривет\n"

    def test_flush_returns_what_was_held(self):
        t = TerminalText()
        t.feed("tail\x1b[")
        assert t.flush() == ""


# --- the process -------------------------------------------------------------------------------------------

@needs_pty
class TestPtyPopen:
    def test_a_question_is_asked_and_answered(self):
        process = pty_process.spawn([sys.executable, "-c", ASK])
        try:
            assert "Name:" in _read_until(process, "Name:")
            assert process.poll() is None
            process.stdin.write(b"Anna\n")
            assert "hi Anna" in _read_until(process, "hi Anna")
            assert process.wait(timeout=20) == 3
            assert process.poll() == 3
        finally:
            process.close()

    def test_wait_times_out_and_kill_ends_it(self):
        import subprocess
        process = pty_process.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.3)
            process.kill()
            assert process.wait(timeout=20) is not None
        finally:
            process.close()

    def test_stderr_is_empty_and_ends_at_once(self):
        process = pty_process.spawn([sys.executable, "-c", "print('x')"])
        try:
            assert process.stderr.read(10) == b""
        finally:
            process.close()

    def test_unavailable_is_reported(self, monkeypatch):
        class Boom:
            @staticmethod
            def spawn(*a, **k):
                raise OSError("no console")

        import types
        fake = types.ModuleType("winpty")
        fake.PtyProcess = Boom
        monkeypatch.setitem(sys.modules, "winpty", fake)
        if sys.platform.startswith("win"):
            with pytest.raises(PtyUnavailable):
                PtyPopen([sys.executable, "-c", "pass"])


# --- through the background-command tools ----------------------------------------------------------------

def _command(tmp_path, name: str, source: str) -> str:
    """A command line that runs `source` as a script file: no quoting question for any shell."""
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return f'"{sys.executable}" "{path}"' if not sys.platform.startswith("win") else f'& "{sys.executable}" "{path}"'


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(command_ops, "is_destructive_command", lambda *a, **k: False)
    monkeypatch.setattr(command_ops, "command_grant_key", lambda *a: "test")
    with command_ops.ACTIVE_PROCESSES_LOCK:
        command_ops.ACTIVE_PROCESSES.clear()
        command_ops.COMPLETED_PROCESSES.clear()
    yield command_ops
    command_ops.stop_all_background_processes()


def _pid(started: str) -> str:
    return started.split("PID:")[1].split()[0].rstrip(".")


def _read_until_tool(ops, pid: str, needle: str, seconds: float = 25.0) -> str:
    deadline = time.time() + seconds
    seen = ""
    while time.time() < deadline:
        seen += ops.read_background_command(pid)
        if needle in seen:
            return seen
        time.sleep(0.2)
    return seen


@needs_pty
class TestThroughTheTools:
    def test_the_whole_dialogue(self, ops, tmp_path):
        started = ops.start_background_command(_command(tmp_path, "ask.py", ASK), pty=True)
        assert "on a terminal" in started
        pid = _pid(started)
        assert "Name:" in _read_until_tool(ops, pid, "Name:")
        assert ops.send_background_command(pid, "Anna") == f"Sent input to PID {pid}."
        out = _read_until_tool(ops, pid, "EXITED")
        import re
        # the shell around the program reports its own status (1 for a failed native command): 1 or 3
        assert "hi Anna" in out and re.search(r"EXITED with code [13]", out)
        assert "\x1b" not in out                       # plain text, not terminal control codes

    def test_a_secret_prompt_is_flagged_and_not_echoed(self, ops, capsys, tmp_path):
        code = "import getpass\ngetpass.getpass('Password: ')\n"
        pid = _pid(ops.start_background_command(_command(tmp_path, "secret.py", code), pty=True))
        out = _read_until_tool(ops, pid, "secret")
        assert "asks for a secret" in out
        ops.send_background_command(pid, "hunter2")
        assert "hunter2" not in capsys.readouterr().out

    def test_without_pty_nothing_changes(self, ops, tmp_path):
        started = ops.start_background_command(_command(tmp_path, "one.py", "print(1)\n"))
        assert "on a terminal" not in started
        info = ops.ACTIVE_PROCESSES[_pid(started)]
        assert info["pty"] is False

    def test_falls_back_to_pipes_and_says_so(self, ops, monkeypatch, tmp_path):
        def refuse(*a, **k):
            raise PtyUnavailable("pywinpty is not installed")
        monkeypatch.setattr(pty_process, "spawn", refuse)
        started = ops.start_background_command(_command(tmp_path, "one.py", "print(1)\n"), pty=True)
        assert "Started background process" in started
        assert "without one" in started and "pywinpty" in started
        assert ops.ACTIVE_PROCESSES[_pid(started)]["pty"] is False


def test_the_message_for_a_command_that_waits_names_the_terminal_mode():
    import inspect
    source = inspect.getsource(command_ops.run_command)
    assert "pty=true" in source
