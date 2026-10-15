"""A command on a pseudo-terminal, shaped like the Popen the background-command code already uses.

Why. A program that asks a question (`npm init`, `git commit` without -m, `ssh` meeting a new host, `python -i`)
checks that it talks to a terminal; behind plain pipes it either refuses or prints nothing. The pipes were
enough for servers and builds; for a dialogue the program needs a terminal of its own.

What. `spawn()` returns a `PtyPopen`: `.stdout.read(n)` gives bytes as they arrive, `.stdin.write(bytes)` types
them, `.poll() / .wait() / .kill() / .pid / .returncode` behave as on `subprocess.Popen`, so the reader threads,
`read_background_command`, `send_background_command` and `kill_process_tree` need no second code path.

  Windows   ConPTY through the `pywinpty` package. A fact learned the hard way: current conhost asks the
            terminal "who are you" (`ESC [ c`) before it lets the program run, and waits for the answer. A host
            that does not answer sees only the start-up sequences and no program output, ever. `PtyPopen`
            answers.
  POSIX     the standard `pty` module.

`PtyUnavailable` is raised when there is no way to make a terminal; the caller falls back to pipes and says so.
`TerminalText` turns what a terminal prints (colours, cursor moves, redrawn lines) back into plain lines.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from typing import Optional

from logger import get_logger

log = get_logger("pty")

ROWS, COLUMNS = 30, 120

_DEVICE_ATTRIBUTES_QUERY = "\x1b[c"
_DEVICE_ATTRIBUTES_REPLY = "\x1b[?1;0c"          # a plain VT100 with no options


class PtyUnavailable(RuntimeError):
    """No pseudo-terminal can be made here (the package is missing, the OS is too old)."""


def available() -> bool:
    try:
        if os.name == "nt":
            import winpty  # noqa: F401
        else:
            import pty  # noqa: F401
        return True
    except Exception:
        return False


# --- file-likes the shared reader code uses ---------------------------------------------------------------------

class _Output:
    """`.read(n)` blocks until bytes arrive; b"" means the terminal closed."""

    def __init__(self):
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._closed = False

    def feed(self, data: Optional[bytes]) -> None:
        self._q.put(data)

    def read(self, n: int = -1) -> bytes:
        if self._closed:
            return b""
        chunk = self._q.get()
        if chunk is None:
            self._q.put(None)               # later reads see the end too
            return b""
        return chunk

    def close(self) -> None:
        self._closed = True
        self._q.put(None)


class _NoOutput:
    """The terminal has one stream; stderr is empty and the reader thread that watches it ends at once."""

    def read(self, n: int = -1) -> bytes:
        return b""

    def close(self) -> None:
        pass


class _Input:
    def __init__(self, send):
        self._send = send
        self.closed = False

    def write(self, data) -> int:
        text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
        # Enter on a terminal is a carriage return; "\n" alone is "line feed" and many programs ignore it.
        text = re.sub(r"\r?\n", "\r", text)
        self._send(text)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


# --- the process -----------------------------------------------------------------------------------------------

class PtyPopen:
    """Popen-shaped handle of a process running on a pseudo-terminal."""

    def __init__(self, command, cwd: Optional[str] = None, env: Optional[dict] = None,
                 rows: int = ROWS, cols: int = COLUMNS):
        self.args = command
        self.returncode: Optional[int] = None
        self.stdout = _Output()
        self.stderr = _NoOutput()
        self._lock = threading.Lock()
        self._backend = None
        self._posix_proc: Optional[subprocess.Popen] = None
        self._master: Optional[int] = None
        if os.name == "nt":
            self._start_windows(command, cwd, env, rows, cols)
        else:
            self._start_posix(command, cwd, env, rows, cols)
        self.stdin = _Input(self._send)
        threading.Thread(target=self._pump, daemon=True, name="ArgentPtyReader").start()

    # -- starting --------------------------------------------------------------------------------------------

    def _start_windows(self, command, cwd, env, rows, cols) -> None:
        try:
            from winpty import PtyProcess
        except Exception as e:
            raise PtyUnavailable(f"the 'pywinpty' package is not installed ({e})") from e
        argv = command if isinstance(command, (list, tuple)) else str(command)
        try:
            self._backend = PtyProcess.spawn(argv, cwd=cwd, env=env, dimensions=(rows, cols))
        except Exception as e:
            raise PtyUnavailable(f"ConPTY did not start: {e}") from e
        self.pid = int(self._backend.pid)

    def _start_posix(self, command, cwd, env, rows, cols) -> None:
        try:
            import fcntl
            import pty
            import struct
            import termios
            master, slave = pty.openpty()
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except Exception as e:
            raise PtyUnavailable(f"no pseudo-terminal: {e}") from e
        try:
            shell = not isinstance(command, (list, tuple))
            self._posix_proc = subprocess.Popen(command, shell=shell, cwd=cwd, env=env, stdin=slave, stdout=slave,
                                                stderr=slave, start_new_session=True, close_fds=True)
        except Exception as e:
            os.close(master)
            raise PtyUnavailable(f"could not start: {e}") from e
        finally:
            try:
                os.close(slave)
            except OSError:
                pass
        self._master = master
        self.pid = self._posix_proc.pid

    # -- the terminal ----------------------------------------------------------------------------------------

    def _send(self, text: str) -> None:
        if self._backend is not None:
            self._backend.write(text)
        elif self._master is not None:
            os.write(self._master, text.encode("utf-8"))

    def _pump(self) -> None:
        try:
            if self._backend is not None:
                while True:
                    text = self._backend.read(4096)
                    if not text:
                        if not self._backend.isalive():
                            break
                        time.sleep(0.01)
                        continue
                    if _DEVICE_ATTRIBUTES_QUERY in text:
                        try:
                            self._backend.write(_DEVICE_ATTRIBUTES_REPLY)
                        except Exception:
                            pass
                    self.stdout.feed(text.encode("utf-8", "replace"))
            elif self._master is not None:
                while True:
                    try:
                        data = os.read(self._master, 4096)
                    except OSError:
                        break
                    if not data:
                        break
                    self.stdout.feed(data)
        except EOFError:
            pass
        except Exception as e:                      # the terminal closed under the reader
            log.debug("pty reader ended: %s", e)
        finally:
            self.stdout.feed(None)

    # -- Popen surface ---------------------------------------------------------------------------------------

    def poll(self) -> Optional[int]:
        with self._lock:
            if self.returncode is not None:
                return self.returncode
            if self._backend is not None:
                if self._backend.isalive():
                    return None
                status = self._backend.exitstatus
                self.returncode = int(status) if status is not None else -1
            elif self._posix_proc is not None:
                code = self._posix_proc.poll()
                if code is None:
                    return None
                self.returncode = code
            return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.03)

    def kill(self) -> None:
        try:
            if self._backend is not None:
                self._backend.terminate(force=True)
            elif self._posix_proc is not None:
                self._posix_proc.kill()
        except Exception as e:
            log.debug("pty kill: %s", e)

    terminate = kill

    def send_signal(self, sig) -> None:
        self.kill()

    def close(self) -> None:
        self.kill()
        self.stdout.close()
        if self._master is not None:
            try:
                os.close(self._master)
            except OSError:
                pass
            self._master = None
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass


def spawn(command, cwd: Optional[str] = None) -> PtyPopen:
    """Start `command` (a string for the shell, or an argv list) on a new terminal.

    The environment keeps pagers away: `git log` on a terminal would open `less` and wait for a key forever.
    """
    env = dict(os.environ)
    env.setdefault("GIT_PAGER", "cat")
    env.setdefault("PAGER", "cat")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return PtyPopen(command, cwd=cwd, env=env)


# --- what a terminal prints, as plain text --------------------------------------------------------------------

_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ESC_TWO = re.compile(r"\x1b[()#][0-9A-Za-z]|\x1b[=>78DEHMNOZc]")
# The start of a sequence cut by the end of the chunk: wait for the next chunk before deciding.
_INCOMPLETE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*|\][^\x07\x1b]*\x1b?|[()#])?$")


class TerminalText:
    """Stateful: feed what the terminal printed, get plain text back. A sequence split between two chunks is
    held until it is whole; a line the program redraws in place (`\\r`) keeps its last version."""

    def __init__(self):
        self._pending = ""

    def feed(self, text: str) -> str:
        text = self._pending + (text or "")
        self._pending = ""
        m = _INCOMPLETE.search(text)
        if m and m.group(0):
            self._pending = m.group(0)
            text = text[:m.start()]
        text = _OSC.sub("", text)
        text = _CSI.sub("", text)
        text = _ESC_TWO.sub("", text)
        text = text.replace("\x1b", "").replace("\x07", "")
        text = text.replace("\r\n", "\n")
        if "\r" in text:
            lines = []
            for line in text.split("\n"):
                if "\r" in line:
                    parts = [p for p in line.split("\r") if p.strip()]
                    line = parts[-1] if parts else ""
                lines.append(line)
            text = "\n".join(lines)
        return text

    def flush(self) -> str:
        """The end of the output: a sequence that never completed is control noise, not text."""
        self._pending = ""
        return ""
