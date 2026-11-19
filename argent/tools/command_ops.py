import os
import subprocess
import queue
import threading
import ctypes
import time
import re
import atexit
from pathlib import Path
from typing import Any
from argent.approval import (
    request_approval, is_destructive_command, command_grant_key, assess_command_risk,
    RUN_COMMAND_ACTION, BACKGROUND_COMMAND_ACTION,
)
from src.agent.shell import (
    choose_shell, build_command_argv, run_text,
    kill_process_tree as _kill_process_tree,
)
from src.agent.command_diagnostics import diagnose_command_error
from argent.memory_manager import memory
from argent.ui import console
from argent.logger import get_logger

log = get_logger("tools")


def _normalize_pid(pid: Any) -> str:
    """Normalize numeric, float, or whitespace PID inputs to a canonical string."""
    if pid is None:
        return ""
    s = str(pid).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s



# ── cancellation ────────────────────────────────────────────────────────────
#
# Ctrl+C used to set a flag that only the agent loop read, between tools. A
# `run_command` already running never looked at it, so stopping during a build
# meant waiting for the build: the turn stayed busy for minutes after the user
# asked it to stop. The output loop below already wakes every 0.1 s, so it is
# the natural place to notice.
#
# A generation counter, not a boolean: a flag has to be cleared somewhere, and
# any clear races with a cancel that arrives just before the next command
# starts. A number only ever grows, so "has anyone asked to stop since I began?"
# needs no reset and cannot be missed.
_CANCEL_GENERATION = 0
_CANCEL_LOCK = threading.Lock()


def bump_cancel_generation() -> int:
    """Record a stop request. Called from the turn's cancel path."""
    global _CANCEL_GENERATION
    with _CANCEL_LOCK:
        _CANCEL_GENERATION += 1
        return _CANCEL_GENERATION


def cancel_generation() -> int:
    """The current stop generation; a running command compares it to its own."""
    with _CANCEL_LOCK:
        return _CANCEL_GENERATION


def _spawn(command: str, cwd: str | None = None, **popen_kwargs):
    """Start a process, picking the right shell on Windows (PowerShell vs cmd)
    so PowerShell-only commands work and cmd-style && / || chains still run.
    On POSIX, fall back to the default shell."""
    if os.name == "nt":
        argv = build_command_argv(command, choose_shell(command))
        return subprocess.Popen(argv, cwd=cwd, **popen_kwargs)
    # Its own process group, so killing the tree can actually kill the tree: a
    # child that shares our group makes `killpg` suicide, `kill_process_tree`
    # falls back to `process.kill()` — which reaches only /bin/sh — and the real
    # work (a build, a test run) survives as an orphan nobody can stop.
    popen_kwargs.setdefault("start_new_session", True)
    return subprocess.Popen(command, shell=True, cwd=cwd, **popen_kwargs)

def _runs_under_powershell(command: str) -> bool:
    """Whether this command will be handed to PowerShell — see src/agent/shell.py.

    The note about exit codes is only true where it is true: cmd and native tools
    that own their exit codes (cargo) are not given it.
    """
    if os.name != "nt":
        return False
    try:
        from src.agent.shell import choose_shell
        return choose_shell(command) == "powershell"
    except Exception:
        return False


ACTIVE_PROCESSES = {}
ACTIVE_PROCESSES_LOCK = threading.Lock()
COMPLETED_PROCESSES = {}
MAX_BACKGROUND_PROCESSES = 10

# How much of a background process's output is kept beyond the queue.
# read_background_command drains the queue, so without a separate copy a finished
# process can no longer be asked what it printed — and the tail is exactly the part
# that says why something failed.
OUTPUT_TAIL_BYTES = 16 * 1024
TAILS_KEPT = 10       # only the newest completed processes keep their output
COMPLETED_KEPT = 50   # how many completed entries are remembered at all

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def _decode_output(data: bytes) -> str:
    """Bytes from a child process, in whichever encoding it decided to use."""
    if not data:
        return ""
    for codec in ("utf-8", "cp866"):
        try:
            return data.decode(codec)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1251", errors="replace")


def _drop_partial_leading_char(data: bytes) -> bytes:
    """Start `data` on a character boundary when it is UTF-8.

    Trimming the tail to its byte budget can cut a multi-byte symbol in half, and
    a tail that begins mid-character makes the whole buffer fail to decode — the
    fallback then turns all 16 KB of log into mojibake. Only a leading
    continuation byte (0x80-0xBF) can mean that, and in a legacy code page those
    same bytes are ordinary characters, so nothing is dropped unless the remainder
    really decodes as UTF-8.
    """
    if not data or (data[0] & 0xC0) != 0x80:
        return data
    for skip in range(1, min(4, len(data))):
        try:
            data[skip:].decode("utf-8")
            return data[skip:]
        except UnicodeDecodeError:
            continue
    return data


class _OutputTail:
    """The last `limit` bytes a background process printed.

    Chunks arrive from the reader thread and are dropped from the left once the
    limit is reached, so a chatty process costs a fixed few kilobytes and the end
    of its output is always kept.
    """

    def __init__(self, limit: int = OUTPUT_TAIL_BYTES):
        self._limit = limit
        self._chunks: list = []
        self._size = 0
        self._lock = threading.Lock()

    def add(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._chunks.append(data)
            self._size += len(data)
            while self._size > self._limit and self._chunks:
                excess = self._size - self._limit
                first = self._chunks[0]
                if len(first) <= excess:
                    self._chunks.pop(0)
                    self._size -= len(first)
                else:
                    head = _drop_partial_leading_char(first[excess:])
                    self._chunks[0] = head
                    self._size -= len(first) - len(head)

    def text(self) -> str:
        with self._lock:
            data = b"".join(self._chunks)
        return _decode_output(data)


def _record_completed(pid, info: dict, exit_code, stopped: bool = False) -> None:
    """File a process under COMPLETED_PROCESSES, carrying its output tail.

    The caller holds ACTIVE_PROCESSES_LOCK. Only the newest TAILS_KEPT entries keep
    their output: fifty finished commands must not hold fifty buffers.
    """
    if len(COMPLETED_PROCESSES) >= COMPLETED_KEPT:
        COMPLETED_PROCESSES.pop(next(iter(COMPLETED_PROCESSES)), None)
    tail = info.get("tail")
    entry = {
        "exit_code": exit_code,
        "command": info.get("command", ""),
        "started": info.get("started"),
        "finished": time.time(),
        "pty": bool(info.get("pty")),
        "output": tail.text() if isinstance(tail, _OutputTail) else "",
    }
    if stopped:
        entry["stopped"] = True
    COMPLETED_PROCESSES[pid] = entry
    if len(COMPLETED_PROCESSES) > TAILS_KEPT:
        for old_pid in list(COMPLETED_PROCESSES.keys())[:-TAILS_KEPT]:
            COMPLETED_PROCESSES[old_pid]["output"] = ""

# After this many consecutive reads with no new output, inject a hint
# telling the model to switch to wait_heartbeat instead of busy-polling.
IDLE_POLL_HINT_THRESHOLD = 3

COMMAND_SILENCE_LIMIT = 180
FAST_INTERACTIVE_SILENCE_LIMIT = 3.0

from argent.text_safety import strip_terminal_escapes as _strip_escapes


def _collapse_carriage_returns(text: str) -> str:
    """Keep the last frame of a progress-bar redraw, and leave CRLF alone.

    A lone ``\\r`` means "overwrite this row" (``10%\\r50%\\r100%``), so only what
    follows the last one survives. CRLF is a plain Windows line ending, not an
    overwrite — treating it as one and stripping each piece ate the leading
    whitespace of every line but the last, so `git diff`, a Python listing or a
    YAML manifest reached the model flattened to column zero.
    """
    if "\r" not in text:
        return text
    lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if "\r" in line:
            frames = [frame for frame in line.split("\r") if frame]
            line = frames[-1] if frames else ""
        lines.append(line)
    return "\n".join(lines)


INTERACTIVE_PROMPT_PATTERNS = [
    re.compile(r"(?:\[y/n\]|\(y/n\)|\[yes/no\]|\(yes/no\))\s*$", re.IGNORECASE),
    re.compile(r"(?:password|passphrase|token)\s*:\s*$", re.IGNORECASE),
    re.compile(r"(?:select|choose)\s+(?:an?\s+)?(?:option|number|\d+)?\s*(?:\([^\)]+\)|\[[^\]]+\])?\s*[:?]?\s*$", re.IGNORECASE),
    re.compile(r"(?:press\s+any\s+key|hit\s+enter|press\s+enter)\s*(?:to\s+continue)?\s*(?:[\.!…]|\.\s*)+\s*$", re.IGNORECASE),
    re.compile(r"(?:are\s+you\s+sure\b.*|confirm(?:\s+action)?)\s*[:?]\s*$", re.IGNORECASE),
    re.compile(r"(?:do\s+you\s+want\s+to\s+continue|proceed)\s*[:?]?\s*$", re.IGNORECASE),
]

def _guard_refusal(command: str) -> str | None:
    """The `/guard block` refusal for a catastrophic command, or None to go ahead.

    Every execution path asks this first. It used to sit inside `run_command` only,
    after the `admin` shortcut and with no counterpart in
    `start_background_command` — so elevation and the background start ran exactly
    the commands the gate exists to refuse.
    """
    from argent.config import get_command_guard
    if get_command_guard() != "block":
        return None
    level, reasons = assess_command_risk(command)
    if level != "block":
        return None
    why = "; ".join(reasons) or "catastrophic command"
    log.warning("Risk-gate BLOCKED: %s (%s)", command, why)
    return (
        f"Command BLOCKED by the risk-gate (/guard block): '{command}'.\n"
        f"Reason: {why}. It was refused without execution. Use a safer, scoped "
        f"command, or lower the gate with /guard warn to allow it with confirmation."
    )


def run_command(command: str, cwd: str | None = None, timeout: int | None = None,
                total_timeout: int | None = None, admin: bool = False) -> str:
    """Execute a console command and return its output. Requires user confirmation.

    `total_timeout` raises the absolute ceiling (300 s by default) for a command that is known to run
    long and keeps printing — a room's test step. It is not in the model's schema; only a room's
    `args` can set it.

    `admin` runs the command elevated (Windows UAC). It used to be a separate tool,
    `run_admin_command`, advertised next to this one — two nearly identical choices, and a
    model that picked the elevated one for an ordinary command got a UAC prompt for
    nothing. One tool with one flag cannot be picked wrong that way; the old name still
    works as an alias for saved traces.

    Streams output to the console while the command runs and returns the full
    result. `process_exit_time` and `pump_thread.is_alive()` let it detach from
    grandchildren that hold the stdout pipe (like `start /b`) and hand control
    back to the agent instead of freezing for 5 minutes.
    """
    # Before anything runs — elevation included.
    refusal = _guard_refusal(command)
    if refusal:
        return refusal

    if admin:
        return _run_elevated(command)

    if cwd:
        cwd_path = Path(cwd)
        if not cwd_path.exists() or not cwd_path.is_dir():
            return f"Error: Directory does not exist: {cwd}"

    """Run a shell command, showing live output and returning the full result.

    Uses `process_exit_time` and `pump_thread.is_alive()` to robustly detach 
    from grandchildren that hold the stdout pipe (like `start /b`), returning 
    control to the agent without freezing for 5 minutes.
    """
    approved = request_approval(
        f"{RUN_COMMAND_ACTION}{command}",
        destructive=is_destructive_command(command),
        grant_key=command_grant_key(command),
    )

    if not approved:
        return f"Execution aborted by user. The command '{command}' was NOT executed."

    try:
        import codecs
        class AutoDecoder:
            def __init__(self):
                self.switched = False
                self.utf8_decoder = codecs.getincrementaldecoder('utf-8')(errors='strict')
                # cmd.exe on Russian Windows usually uses cp866
                self.fallback_decoder = codecs.getincrementaldecoder('cp866' if os.name == 'nt' else 'utf-8')(errors='replace')

            def decode(self, b, final=False):
                if self.switched:
                    return self.fallback_decoder.decode(b, final)
                try:
                    return self.utf8_decoder.decode(b, final)
                except UnicodeDecodeError:
                    self.switched = True
                    return self.fallback_decoder.decode(b, final)

        decoder = AutoDecoder()
        from src.agent.secret_redactor import StreamRedactor, redact as _redact_secrets
        redactor = StreamRedactor()

        process = _spawn(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        # Close stdin immediately to prevent commands expecting EOF from hanging,
        # guarding against mock processes in tests that lack stdin.
        if getattr(process, "stdin", None):
            try:
                process.stdin.close()
            except Exception:
                pass
        
        # We need a queue to read lines without blocking the main thread,
        # which lets us enforce a silence timeout.
        COMMAND_TOTAL_LIMIT = 300
        if total_timeout:
            try:
                COMMAND_TOTAL_LIMIT = max(COMMAND_TOTAL_LIMIT, min(int(total_timeout), 4 * 3600))
            except (TypeError, ValueError):
                pass
        silence_limit = timeout if timeout is not None else COMMAND_SILENCE_LIMIT
        line_queue = queue.Queue()
        
        def _pump():
            fd = process.stdout.fileno()
            try:
                while True:
                    # Raw read avoids Python's line-buffering deadlock on Windows
                    # when the child process holds the pipe open but stops writing.
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    line_queue.put(chunk)
            except Exception:
                pass
            finally:
                line_queue.put(None)  # sentinel: stream closed

        pump_thread = threading.Thread(target=_pump, daemon=True)
        pump_thread.start()

        # Taken after the spawn: a stop asked for before this line belongs to the
        # previous turn, not to the command that just started.
        my_generation = cancel_generation()

        output_chunks = []
        interactive_timeout = False
        silent_timeout = False
        hard_timeout = False
        cancelled = False
        start_time = time.time()
        last_output_time = start_time
        process_exit_time = None
        
        # Max memory protection for massive logs (OOM cap)
        MAX_TOTAL_CHARS = 1_000_000  # Cap at ~1MB
        current_chars = 0
        truncated = False
        
        aborted = True
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= COMMAND_TOTAL_LIMIT:
                    hard_timeout = True
                    try:
                        _kill_process_tree(process)
                    except Exception:
                        pass
                    break
                
                try:
                    raw_chunk = line_queue.get(timeout=0.1)
                except queue.Empty:
                    # The user asked to stop: bring the tree down now instead of
                    # running a command they no longer want to completion.
                    if cancel_generation() != my_generation:
                        cancelled = True
                        try:
                            _kill_process_tree(process)
                        except Exception:
                            pass
                        break
                    # A prompt the secret filter was holding back ("Password:") is let through
                    # after a moment of silence, so it reaches the screen and the prompt detector.
                    released = redactor.release_stale()
                    if released and current_chars + len(released) <= MAX_TOTAL_CHARS:
                        current_chars += len(released)
                        output_chunks.append(released)
                        if current_chars < 50_000:
                            console.print(released, style="dim", end="", markup=False)
                    # Check if process exited while queue was draining
                    if process.poll() is not None:
                        if process_exit_time is None:
                            process_exit_time = time.time()
                        # Drain any leftover output in the pipe for up to 1 second
                        elif time.time() - process_exit_time > 1.0:
                            break
                    else:
                        silence_duration = time.time() - last_output_time

                        # Fast interactive prompt detection
                        if silence_duration >= FAST_INTERACTIVE_SILENCE_LIMIT and current_chars > 0:
                            current_text = "".join(output_chunks)
                            clean_text = _strip_escapes(current_text).rstrip()
                            last_line = clean_text.splitlines()[-1] if clean_text else ""
                            if any(rx.search(last_line) for rx in INTERACTIVE_PROMPT_PATTERNS):
                                interactive_timeout = True
                                try:
                                    _kill_process_tree(process)
                                except Exception:
                                    pass
                                break

                        if silence_duration >= silence_limit:
                            silent_timeout = True
                            try:
                                _kill_process_tree(process)
                            except Exception:
                                pass
                            break
                    continue
                    
                if raw_chunk is None:
                    break
                text = decoder.decode(raw_chunk, final=False)
                if text:
                    last_output_time = time.time()
                text = redactor.feed(text)
                if text:
                    last_output_time = time.time()
                    
                    if current_chars + len(text) > MAX_TOTAL_CHARS:
                        if not truncated:
                            output_chunks.append(f"\n\n[WARNING: Output exceeded {MAX_TOTAL_CHARS} chars. Truncating remaining output to prevent OOM...]\n")
                            truncated = True
                        continue
                        
                    current_chars += len(text)
                    output_chunks.append(text)
                    if current_chars < 50_000:
                        console.print(text, style="dim", end="", markup=False)
                        try:
                            console.file.flush()
                        except Exception:
                            pass

            trail = decoder.decode(b"", final=True)
            trail = redactor.feed(trail) + redactor.flush()
            if trail:
                if current_chars + len(trail) <= MAX_TOTAL_CHARS:
                    output_chunks.append(trail)
            aborted = False
        finally:
            if aborted or cancelled or interactive_timeout or silent_timeout or hard_timeout:
                try:
                    _kill_process_tree(process)
                except Exception:
                    pass
            else:
                try:
                    try:
                        process.wait(timeout=1.0)
                    except TypeError:
                        process.wait()
                except Exception:
                    try:
                        _kill_process_tree(process)
                    except Exception:
                        pass

            # Ensure pump thread has finished before closing pipes
            if pump_thread.is_alive():
                try:
                    pump_thread.join(timeout=1.0)
                except Exception:
                    pass

            _close_proc_pipes(process)

            try:
                try:
                    process.wait(timeout=1)
                except TypeError:
                    process.wait()
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        
        final_output = "".join(output_chunks).strip()
        clean_final_output = _strip_escapes(final_output)
        # Colour codes inside a key defeat the stream filter; one more pass over the clean text.
        clean_final_output = _redact_secrets(clean_final_output)
        clean_final_output = _collapse_carriage_returns(clean_final_output)
        output = f"Exit code: {process.returncode}\n"
        
        if cancelled:
            output += (
                "\n[STOPPED BY USER]: The user interrupted the turn (Ctrl+C) and the command was "
                "terminated. Do not retry it unless they ask for it.\n"
            )
        if interactive_timeout:
            trailing = clean_final_output.rstrip().splitlines()[-1] if clean_final_output else ""
            output += (
                f"\n[FAST INTERACTIVE TIMEOUT (3s)]: The command paused waiting for interactive user input: '{trailing}'.\n"
                f"`run_command` has a closed stdin and cannot accept interactive input.\n"
                f"Fix by providing non-interactive flags (e.g. -y, --yes, --batch, --non-interactive), "
                f"piping input (e.g. `echo y | ...`), or using `start_background_command(command, pty=true)` + "
                f"`read_background_command` + `send_background_command` (a real terminal, so it can be answered).\n"
            )
        elif silent_timeout:
            output += (
                f"\n[WARNING: Command was SILENT for over {silence_limit} seconds and was forcefully terminated!]\n"
                f"If you are running a long-running background task, you MUST use `start_background_command` instead.\n"
            )
        elif hard_timeout:
            output += (
                f"\n[WARNING: Command EXCEEDED the absolute execution limit of {COMMAND_TOTAL_LIMIT} seconds and was forcefully terminated!]\n"
                f"If you are running a server, watcher, or heavy task, you MUST use `start_background_command` instead.\n"
            )
        
        if clean_final_output:
            output += f"OUTPUT:\n{clean_final_output}\n"

        if interactive_timeout:
            pass  # Already diagnosed above
        elif silent_timeout:
            trailing = clean_final_output.rstrip().splitlines()[-1] if clean_final_output else ""
            if trailing:
                output += (
                    f"\n[DIAGNOSIS]: The command was silent for over {silence_limit}s and was terminated. "
                    f"It may have been waiting for an answer to: \"{trailing}\". "
                    f"Use `start_background_command` and `send_background_command` to interact "
                    f"with it, or provide the input non-interactively (e.g. `echo y | command`).\n"
                )
            else:
                output += (
                    f"\n[DIAGNOSIS]: The command was silent for over {silence_limit}s and was terminated. "
                    f"If this is a long compilation or background service, use `start_background_command`. "
                    f"Otherwise verify whether the process deadlocked or hung in an infinite loop.\n"
                )
        elif process.returncode != 0:
            hint = diagnose_command_error(command, clean_final_output, process.returncode)
            if hint:
                output += f"\n[DIAGNOSIS]: {hint}\n"
            elif clean_final_output and _runs_under_powershell(command):
                # The exit code is the weakest signal in PowerShell: warnings and
                # harmless "nothing matched" cases return it as readily as failures,
                # and the child's stderr is merged into this text, so "it printed
                # something and returned 1" says nothing on its own. Said only when
                # no more specific diagnosis exists, so the two never contradict.
                output += (
                    f"\n[NOTE]: exit code {process.returncode} does not by itself mean the "
                    "command failed — PowerShell and many tools it runs report warnings, "
                    "and \"no results\" this way. Read the text above before retrying or "
                    "concluding failure.\n")

        log.info("run_command: %s (exit=%s)", command, process.returncode)

        cmd_lower = command.strip().lower()
        is_fire_and_forget = any(cmd_lower.startswith(p) for p in ["explorer", "start ", "start.", 'start"'])

        exit_code = process.returncode if process.returncode is not None else -1
        # Phase 4: command exit code feeds the turn trajectory (telemetry only).
        try:
            from src.leeth import trajectory as leeth_trajectory
            if leeth_trajectory.current_trajectory_id():
                leeth_trajectory.record_event("command_exit", command=command, code=exit_code)
        except Exception:
            pass
        # Bookkeeping is not the command: a full disk or a read-only .argent
        # must never replace the output the user asked for with "Error running
        # command". MemoryManager already degrades; this is the second belt.
        try:
            if exit_code != 0 and not is_fire_and_forget:
                memory.add_error(f"Command '{command}' failed (exit={exit_code})")
            else:
                if hasattr(memory, "resolve_command_error"):
                    memory.resolve_command_error(command)
                memory.add_completed(f"Ran: {command}")
        except Exception as e:
            log.warning("memory update after command failed: %s", e)
        # The model's window may cut the middle of a long log: keep the whole of it in a file it can read.
        try:
            from argent.config import get_current_model
            from src.agent.output_store import keep_if_long
            return keep_if_long(output.strip(), command, get_current_model())
        except Exception:
            return output.strip()
    except Exception as e:
        log.error("run_command error: %s: %s", command, e)
        return f"Error running command '{command}': {e}"

def _encode_powershell_command(script: str) -> str:
    """Encode a PowerShell script for `powershell -EncodedCommand`: base64 of
    its UTF-16LE bytes. Unlike `-Command "..."`, this carries arbitrary quotes
    and special characters verbatim — no escaping, no injection surface."""
    import base64
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _wait_for_file(path, timeout: float, poll: float = 0.25) -> bool:
    """Wait until `path` appears. False when it never did."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(poll)
    return path.exists()


def _read_process_output_file(path):
    """Whatever an elevated child wrote, in whichever encoding it chose."""
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace").strip()
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace").strip()
    if b"\x00" in data:
        try:
            return data.decode("utf-16le").strip()
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace").strip()
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return data.decode("cp1251" if os.name == "nt" else "utf-8", errors="replace").strip()


def _elevated_wrapper(command: str, out_path, done_path) -> str:
    """The script the elevated child runs: all output captured, then a done marker.

    The command goes into a script block because `cmd1; cmd2 > file` redirects only
    the LAST statement — a two-part admin command reported half of its output. The
    marker is the child's final act, and its content is the exit code, so the caller
    can tell "finished" from "still running" instead of guessing from the first bytes.
    """
    return (
        f"& {{ {command} }} > '{out_path}' 2>&1; "
        f"$__code = if ($null -ne $LASTEXITCODE) {{ $LASTEXITCODE }} else {{ 0 }}; "
        f"Set-Content -LiteralPath '{done_path}' -Value $__code"
    )


def _collect_elevated_result(temp_out, done_path, timeout: float = 20.0) -> str:
    """Wait for the elevated child's own completion marker, then report.

    The child redirects its output to `temp_out` and, as its very last act, writes
    its exit code to `done_path`. The old loop returned as soon as the output file
    had *any* bytes in it — so a banner was reported as the finished result while
    the command was still running and the rest of its output was lost.
    """
    finished = _wait_for_file(done_path, timeout)

    output = ""
    if temp_out.exists():
        try:
            output = _read_process_output_file(temp_out)
        except Exception:
            output = ""      # still being written, or vanished

    if not finished:
        partial = f"\nOutput so far:\n{output}" if output else ""
        return (
            f"Error: the admin command did not finish within {int(timeout)} s and may still "
            "be running (an elevated process outlives this call). Check Task Manager before "
            "re-running; do not run project builds elevated — the files they create are owned "
            "by Administrator and later normal builds fail on them." + partial
        )

    code = ""
    try:
        code = done_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        code = ""
    return (
        f"Admin execution completed (exit code {code or 'unknown'}).\n"
        f"Output:\n{output}\n\n"
        "Note: files created by an elevated process are owned by Administrator; "
        "later non-elevated commands (a normal build, cargo, git) may be unable "
        "to modify them. Prefer running project commands without elevation."
    )


def _run_elevated(command: str) -> str:
    """Execute a PowerShell command with Administrator privileges (UAC prompt).

    Reached through `run_command(admin=True)`. It used to be a tool of its own,
    `run_admin_command`, advertised next to `run_command` — two nearly identical
    choices, and a model that picked the elevated one for an ordinary command got a
    UAC prompt for nothing. One tool with one flag cannot be picked wrong that way.
    """
    if os.name != "nt" or not hasattr(ctypes, "windll"):
        return "Error: elevation is only supported on Windows (UAC)."

    approved = request_approval(
        f"выполнить команду с правами АДМИНИСТРАТОРА (UAC): {command}",
        destructive=True,
    )

    if not approved:
        return f"Execution aborted by user. The admin command '{command}' was NOT run."
        
    import tempfile
    from pathlib import Path
    # Unique temp file in the user's temp dir — not a shared, hardcoded
    # C:/Windows/Temp path that would collide across concurrent/repeat admin
    # runs. Removed up front so its re-appearance signals the child finished.
    fd, temp_path = tempfile.mkstemp(prefix="argent_admin_", suffix=".txt")
    os.close(fd)
    temp_out = Path(temp_path)
    # Written by the child as its LAST act — that is what says "finished", which
    # the presence of output alone could never do.
    done_path = temp_out.with_name(temp_out.name + ".done")
    try:
        temp_out.unlink()  # elevated child re-creates it on completion

        # Pass the script via -EncodedCommand (base64 of UTF-16LE), not
        # -Command "…": inline quoting breaks the moment the user's command
        # contains a double quote (Write-Host "x"), which both fails valid
        # commands and is a quote-injection surface. EncodedCommand carries an
        # arbitrary script verbatim, no escaping needed.
        encoded = _encode_powershell_command(_elevated_wrapper(command, temp_out, done_path))
        result = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "powershell.exe",
            f"-NoProfile -EncodedCommand {encoded}", None, 0,
        )
        if result <= 32:
            return f"Error: UAC prompt was denied or execution failed. Error code: {result}"

        return _collect_elevated_result(temp_out, done_path)
    except Exception as e:
        return f"Error running admin command '{command}': {e}"
    finally:
        for leftover in (temp_out, done_path):
            try:
                leftover.unlink(missing_ok=True)
            except Exception:
                pass

def _close_proc_pipes_async(process):
    """Release a finished process's pipes without blocking the caller.

    `close()` on a Windows pipe waits while somebody is still inside `read()` on it,
    which happens exactly when a grandchild briefly outlives the process we killed.
    The descriptors are freed a moment later either way, so the agent thread — or,
    worse, a caller holding `ACTIVE_PROCESSES_LOCK` — must not be the one to wait.
    """
    threading.Thread(target=_close_proc_pipes, args=(process,), daemon=True).start()


def _close_proc_pipes(process):
    """Close all stdio pipes of a subprocess to avoid ResourceWarning leaks."""
    for attr in ("stdout", "stderr", "stdin"):
        try:
            pipe = getattr(process, attr, None)
            if pipe:
                pipe.close()
        except Exception:
            pass
    # a process on a pseudo-terminal (src/agent/pty_process.py) also owns the terminal itself
    try:
        from src.agent.pty_process import PtyPopen
        if isinstance(process, PtyPopen):
            process.close()
    except Exception:
        pass


def start_background_command(command: str, cwd: str | None = None, pty: bool = False) -> str:
    """Launch a command in the background and return its PID.

    pty=True gives the program a terminal of its own, for the ones that ask questions (`npm init`, `ssh`,
    `python -i`): read what it printed with read_background_command, answer with send_background_command.
    """
    if not command or not command.strip():
        return "Error: Command cannot be empty."
    command = command.strip()

    refusal = _guard_refusal(command)
    if refusal:
        return refusal

    if cwd:
        cwd_path = Path(cwd)
        if not cwd_path.exists() or not cwd_path.is_dir():
            return f"Error: Directory does not exist: {cwd}"

    with ACTIVE_PROCESSES_LOCK:
        # Auto-reap exited processes if reaching maximum capacity
        if len(ACTIVE_PROCESSES) >= MAX_BACKGROUND_PROCESSES:
            exited_pids = [
                p for p, info in ACTIVE_PROCESSES.items()
                if info.get("process") and info["process"].poll() is not None
            ]
            for p in exited_pids:
                info = ACTIVE_PROCESSES[p]
                proc = info.get("process")
                if proc:
                    # Never while holding the lock: a blocked close would freeze
                    # every other background call, not just this one.
                    _close_proc_pipes_async(proc)
                del ACTIVE_PROCESSES[p]
                _record_completed(p, info, proc.poll() if proc else -1)

        if len(ACTIVE_PROCESSES) >= MAX_BACKGROUND_PROCESSES:
            return f"Error: Maximum number of background processes ({MAX_BACKGROUND_PROCESSES}) reached. Stop an existing process first."
    
    approved = request_approval(
        f"{BACKGROUND_COMMAND_ACTION}{command}",
        destructive=is_destructive_command(command),
        grant_key=command_grant_key(command),
    )

    if not approved:
        return f"Execution aborted by user. The command '{command}' was NOT started."

    try:
        process = None
        on_terminal = False
        terminal_note = ""
        if pty:
            from src.agent import pty_process
            try:
                argv = build_command_argv(command, choose_shell(command)) if os.name == "nt" else command
                process = pty_process.spawn(argv, cwd=cwd)
                on_terminal = True
            except pty_process.PtyUnavailable as e:
                log.warning("no terminal for '%s': %s", command, e)
                terminal_note = (f" (a terminal could not be made: {e}. It runs without one, so it cannot be "
                                 f"answered if it asks a question.)")
        popen_kwargs = dict(
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if os.name == "nt":
            # Isolate background process from the interactive console's CTRL_C_EVENT group
            # so pressing Ctrl+C in Argent interrupts the wait/turn without killing background jobs.
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        if process is None:
            process = _spawn(
                command,
                **popen_kwargs,
            )

        # The handle people are given is the process's own OS PID, not a counter of
        # ours: it is the number `Get-Process`, Task Manager and psutil know, so a
        # background job can be looked at — and stopped — from outside Argent too.
        pid = str(process.pid)

        out_queue = queue.Queue()
        err_queue = queue.Queue()
        
        # The queue is drained by read_background_command; the tail is not, so a
        # finished process can still be asked what it printed.
        tail = _OutputTail()

        def reader(pipe, q):
            try:
                while True:
                    data = pipe.read(1024)
                    if not data:
                        break
                    q.put(data)
                    tail.add(data)
            except Exception:
                pass
                
        threading.Thread(target=reader, args=(process.stdout, out_queue), daemon=True).start()
        threading.Thread(target=reader, args=(process.stderr, err_queue), daemon=True).start()
        
        def process_waiter(proc, p_id):
            try:
                proc.wait()
                from src.agent.events import EVENT_QUEUE
                if EVENT_QUEUE is not None:
                    EVENT_QUEUE.put({"type": "bg_done", "pid": p_id, "exit_code": proc.returncode})
            except Exception:
                pass
            
        threading.Thread(target=process_waiter, args=(process, pid), daemon=True).start()
        
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[pid] = {
                "process": process,
                "out_queue": out_queue,
                "err_queue": err_queue,
                "command": command,
                "cwd": cwd,
                "started": time.time(),
                "pty": on_terminal,
                "tail": tail,
            }

        if on_terminal:
            return (f"Started background process with PID: {pid} on a terminal. If it asks something, read it "
                    f"with read_background_command and answer with send_background_command.")
        return f"Started background process with PID: {pid}{terminal_note}"
    except Exception as e:
        return f"Error starting background command '{command}': {e}"

def read_background_command(pid: Any) -> str:
    """Read the latest output from a background process."""
    pid = _normalize_pid(pid)
    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            if pid in COMPLETED_PROCESSES:
                comp = COMPLETED_PROCESSES[pid]
                if comp.get("stopped"):
                    return f"Error: Process {pid} has been stopped (terminated)."
                return f"--- Process {pid} has EXITED with code {comp['exit_code']} ---\nCommand: {comp['command']}\n(Process already completed and previous output was retrieved.)"
            return f"Error: No active process with PID {pid}."
        proc_info = ACTIVE_PROCESSES[pid]
        
    process = proc_info.get("process")
    
    def decode_q(q):
        data = bytearray()
        truncated = False
        while True:
            try:
                chunk = q.get_nowait()
                if len(data) + len(chunk) > 1_000_000:
                    if not truncated:
                        data.extend(b"\n\n[WARNING: Output exceeded 1MB. Truncating...]\n")
                        truncated = True
                    continue
                data.extend(chunk)
            except queue.Empty:
                break
        
        return _decode_output(bytes(data))
                
    stdout = decode_q(proc_info["out_queue"])
    stderr = decode_q(proc_info["err_queue"])
    
    retcode = process.poll() if process else -1
    if retcode is not None:
        time.sleep(0.05)
        trailing_out = decode_q(proc_info["out_queue"])
        trailing_err = decode_q(proc_info["err_queue"])
        if trailing_out:
            stdout = (stdout + trailing_out) if stdout else trailing_out
        if trailing_err:
            stderr = (stderr + trailing_err) if stderr else trailing_err

        status = f"Process {pid} has EXITED with code {retcode}."
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES.pop(pid, None)
            _record_completed(pid, proc_info, retcode)
        if process:
            # The process is gone; a grandchild may still hold the pipe, and a
            # synchronous close() would block this step of the agent on it.
            _close_proc_pipes_async(process)
    else:
        # Include elapsed runtime for human readability.
        started = proc_info.get("started")
        elapsed = int(time.time() - started) if started is not None else 0
        status = f"Process {pid} is RUNNING for {elapsed}s."
        
    stdout, stderr = _convert_stream(proc_info, stdout, stderr, retcode)

    res = f"--- {status} ---\n"
    if stdout:
        res += f"STDOUT:\n{stdout}\n"
    if stderr:
        res += f"STDERR:\n{stderr}\n"

    # A question whose answer is a secret is not the model's to answer: it would pass through its context.
    if stdout.strip():
        last_line = _strip_escapes(stdout).rstrip().splitlines()[-1] if _strip_escapes(stdout).strip() else ""
        proc_info["secret_prompt"] = bool(INTERACTIVE_PROMPT_PATTERNS[1].search(last_line)) and retcode is None
        if proc_info["secret_prompt"]:
            res += (f"\n[Hint]: this prompt asks for a secret (password, passphrase, token). Do not guess it and "
                    f"do not type one you were not given. Ask the user with ask_user_questions how to continue "
                    f"(a key instead of a password, a token from an environment variable), then stop the process "
                    f"with stop_background_command(pid=\"{pid}\") if it cannot go on.\n")

    has_output = bool(stdout or stderr)
    if has_output:
        # New data arrived — reset the idle counter.
        proc_info["idle_polls"] = 0
    else:
        res += "No new output.\n"
        # Track consecutive empty reads and nudge the model toward
        # wait_heartbeat instead of burning turns on busy-polling.
        idle = proc_info.get("idle_polls", 0) + 1
        proc_info["idle_polls"] = idle
        if idle >= IDLE_POLL_HINT_THRESHOLD and retcode is None:
            res += (
                f"\n[Hint]: No new output for {idle} consecutive reads. "
                f"Use wait_heartbeat(until='process_finished(\"{pid}\")') "
                f"to wait efficiently, or stop_background_command(pid=\"{pid}\") "
                f"if the task is done. Do NOT keep calling read_background_command in a loop."
            )
        
    return res

def _drain_queue(q) -> str:
    """Everything queued so far, decoded. Never blocks."""
    data = bytearray()
    while True:
        try:
            chunk = q.get_nowait()
        except queue.Empty:
            break
        data.extend(chunk)
        if len(data) > 1_000_000:
            data.extend(b"\n[WARNING: Output exceeded 1MB. Truncated.]\n")
            break
    return _decode_output(bytes(data)) if data else ""


def _convert_stream(proc_info: dict, stdout: str, stderr: str, retcode):
    """Raw captures -> what a person sees.

    Two stateful steps, both of which must happen exactly once per chunk and both of
    which live on `proc_info`: a PTY's escape sequences are resolved into plain text
    (otherwise the model reads `\x1b[?25l` instead of the prompt), and a streaming
    redactor keeps a secret split across two polls from leaking in halves.
    """
    if proc_info.get("pty"):
        from src.agent.pty_process import TerminalText
        terminal = proc_info.setdefault("terminal", TerminalText())
        stdout = terminal.feed(stdout)
        if retcode is not None:
            stdout += terminal.flush()

    from src.agent.secret_redactor import StreamRedactor
    r_out = proc_info.setdefault("redactor_out", StreamRedactor())
    r_err = proc_info.setdefault("redactor_err", StreamRedactor())
    stdout = r_out.feed(stdout)
    stderr = r_err.feed(stderr)
    if retcode is not None:
        stdout += r_out.flush()
        stderr += r_err.flush()
    else:
        stdout += r_out.release_stale(0.0)
        stderr += r_err.release_stale(0.0)
    return stdout, stderr


def _drain_briefly(pid: Any, proc_info: dict, process, wait_s: float = 0.25) -> str:
    """What the process said right after an answer.

    A wizard answers instantly, but "instantly" still means "in the next tick", so a
    short bounded wait turns send+read into one call. On timeout this returns "" and
    the caller says so honestly.
    """
    out_q = proc_info["out_queue"]
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if not out_q.empty():
            break
        if process is not None and process.poll() is not None:
            break
        time.sleep(0.02)

    stdout = _drain_queue(out_q)
    stderr = _drain_queue(proc_info["err_queue"])
    retcode = process.poll() if process else -1
    if retcode is not None and process is not None:
        # The same bookkeeping `read_background_command` does: a process that ended
        # while we were reading must leave the active table and be recorded, or the
        # next read reports "RUNNING" for something that is gone and has no output.
        with ACTIVE_PROCESSES_LOCK:
            info = ACTIVE_PROCESSES.pop(pid, None)
        if info:
            _record_completed(pid, info, retcode)
        _close_proc_pipes_async(process)
    if not stdout and not stderr and retcode is None:
        return ""
    stdout, stderr = _convert_stream(proc_info, stdout, stderr, retcode)

    started = proc_info.get("started")
    elapsed = int(time.time() - started) if started is not None else 0
    head = (f"--- Process {pid} has EXITED with code {retcode} ---" if retcode is not None
            else f"--- Process {pid} is RUNNING for {elapsed}s. ---")
    parts = [head]
    if stdout:
        parts.append("STDOUT:\n" + stdout.rstrip())
    if stderr:
        parts.append("STDERR:\n" + stderr.rstrip())
    return "\n".join(parts)


def send_background_command(pid: Any, input_string: str) -> str:
    """Send text to the standard input of a running background process."""
    pid = _normalize_pid(pid)
    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            return f"Error: No active process with PID {pid}."
        proc_info = ACTIVE_PROCESSES[pid]
        process = proc_info.get("process")
        secret_prompt = bool(proc_info.get("secret_prompt"))

    if not process or process.poll() is not None:
        with ACTIVE_PROCESSES_LOCK:
            proc_info = ACTIVE_PROCESSES.pop(pid, None)
            if proc_info:
                _record_completed(pid, proc_info, process.poll() if process else -1)
        if process:
            _close_proc_pipes_async(process)
        return f"Error: Process {pid} has already exited."
        
    if not getattr(process, "stdin", None):
        return f"Error: Process {pid} does not have an open stdin."

    try:
        shown = "•••••• (hidden: the program asked for a secret)" if secret_prompt else input_string.strip()
        console.print(f"\n[bold yellow]Agent sending input to PID {pid}:[/bold yellow] {shown}", markup=True,
                      highlight=False)
        if not input_string.endswith('\n'):
            input_string += '\n'
        process.stdin.write(input_string.encode('utf-8'))
        process.stdin.flush()
        # The answer and its reaction in one call.
        #
        # This used to return a constant "Sent input to PID X.", and answering a
        # wizard prompt by prompt cost two calls each: send, then read. Worse, the
        # constant result made a legitimate repeat look exactly like a stuck loop
        # (same call, same answer), so the model learned to vary its input — "y",
        # "Y", "yes" — to avoid the guard instead of answering the question.
        block = _drain_briefly(pid, proc_info, process)
        if block:
            return f"Sent input to PID {pid}.\n{block}"
        return (f"Sent input to PID {pid}.\n"
                "(no new output yet — call read_background_command if it is still working)")
    except Exception as e:
        return f"Error sending input to PID {pid}: {e}"

def stop_background_command(pid: Any) -> str:
    """Terminate a background process and its entire process tree."""
    pid = _normalize_pid(pid)
    if not pid:
        return "Error: PID cannot be empty."

    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            if pid in COMPLETED_PROCESSES:
                comp = COMPLETED_PROCESSES[pid]
                status_str = "stopped" if comp.get("stopped") else f"already exited (code {comp['exit_code']})"
                return f"Error: Process {pid} is not active ({status_str})."
            return f"Error: No active process with PID {pid}."
        proc_info = ACTIVE_PROCESSES[pid]
        process = proc_info.get("process")
        del ACTIVE_PROCESSES[pid]

        _record_completed(pid, proc_info, -15 if os.name != "nt" else 1, stopped=True)

    # Kill the whole process tree OUTSIDE the lock
    if process is not None:
        try:
            _kill_process_tree(process)
        except Exception as e:
            return f"Error terminating PID {pid}: {e}"

        _close_proc_pipes_async(process)

    return f"Terminated background process PID {pid}."

def list_background_commands() -> str:
    """List the background processes started this session, with PID, status and command."""
    with ACTIVE_PROCESSES_LOCK:
        items = list(ACTIVE_PROCESSES.items())

    if not items:
        return "No background processes are currently tracked."

    lines = []
    for pid, info in items:
        proc = info.get("process")
        if proc is None:
            status = "UNKNOWN"
        else:
            ret = proc.poll()
            if ret is None:
                started = info.get("started")
                elapsed = int(time.time() - started) if started is not None else 0
                status = f"RUNNING {elapsed}s"
            else:
                status = f"EXITED (code {ret})"
        lines.append(f"- PID {pid}: [{status}] {info.get('command', '')}")

    return (
        "Background processes:\n" + "\n".join(lines)
        + "\n\nUse read_background_command(pid) for output, "
          "send_background_command(pid, text) for input, "
          "stop_background_command(pid) to terminate."
    )


def _format_elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds}s"


def completed_background_commands(limit: int = 10) -> str:
    """The background commands that have finished, newest last, with exit codes."""
    with ACTIVE_PROCESSES_LOCK:
        items = list(COMPLETED_PROCESSES.items())
    if not items:
        return "Завершённых фоновых задач нет."
    rows = []
    for pid, info in items[-limit:]:
        code = info.get("exit_code")
        mark = "✓" if code == 0 else "✗"
        started, finished = info.get("started"), info.get("finished")
        took = f" · {_format_elapsed(finished - started)}" if started and finished else ""
        rows.append(f"  pid {pid}  {mark} {code}  {info.get('command', '')}{took}")
    more = f"\n  … ещё {len(items) - limit}" if len(items) > limit else ""
    return (f"Завершённые фоновые задачи ({len(items)}):\n" + "\n".join(rows) + more
            + "\nВывод: /jobs <pid>")


def background_output(pid: Any) -> str:
    """What a background process printed, from the tail kept for it."""
    pid = _normalize_pid(pid)
    if not pid:
        return "Error: PID cannot be empty."
    tail_obj = None
    info = None
    running = False
    with ACTIVE_PROCESSES_LOCK:
        if pid in ACTIVE_PROCESSES:
            info = dict(ACTIVE_PROCESSES[pid])
            tail_obj = ACTIVE_PROCESSES[pid].get("tail")
            running = True
        elif pid in COMPLETED_PROCESSES:
            info = dict(COMPLETED_PROCESSES[pid])
        else:
            return f"Error: No process with PID {pid}."

    text = tail_obj.text() if isinstance(tail_obj, _OutputTail) else (info.get("output") or "")
    clean = _ANSI_RE.sub("", text or "").strip()
    head = f"PID {pid} · {info.get('command', '')}"
    if not running:
        state = "остановлен" if info.get("stopped") else f"завершён, код {info.get('exit_code')}"
        head += f" · {state}"
    if not clean:
        return f"{head}\n(вывод не сохранён)"
    return f"{head}\n{'-' * 40}\n{clean}"


def background_view(pid: Any) -> dict:
    """A structured snapshot for a UI window: state, elapsed, exit code, output tail.

    Unlike `read_background_command` this never drains a queue: a window polls it a
    few times a second while the agent may be reading the same process, and taking
    the output away from the agent just to show it to the person would make the two
    fight over it. The tail is the same buffer `background_output` uses.
    """
    pid = _normalize_pid(pid)
    blank = {"pid": pid, "status": "unknown", "command": "", "elapsed": 0,
             "code": None, "output": "", "error": ""}
    if not pid:
        return {**blank, "error": "PID пуст"}
    info: dict = {}
    tail = None
    running = False
    with ACTIVE_PROCESSES_LOCK:
        if pid in ACTIVE_PROCESSES:
            info = dict(ACTIVE_PROCESSES[pid])
            tail = ACTIVE_PROCESSES[pid].get("tail")
            running = True
        elif pid in COMPLETED_PROCESSES:
            info = dict(COMPLETED_PROCESSES[pid])
        else:
            return {**blank, "error": f"задача {pid} не найдена (список обновился?)"}

    text = tail.text() if isinstance(tail, _OutputTail) else (info.get("output") or "")
    started = info.get("started")
    finished = info.get("finished")
    if running:
        status = "running"
        elapsed = int(time.time() - started) if started else 0
    elif info.get("stopped"):
        status = "stopped"
        elapsed = int(finished - started) if started and finished else 0
    else:
        status = "done"
        elapsed = int(finished - started) if started and finished else 0
    return {
        "pid": pid,
        "status": status,
        "command": str(info.get("command") or ""),
        "elapsed": elapsed,
        "code": None if running else info.get("exit_code"),
        "output": _ANSI_RE.sub("", text or ""),
        "error": "",
    }


def stop_all_background_processes() -> None:
    """Terminate all active background processes on shutdown/cleanup."""
    with ACTIVE_PROCESSES_LOCK:
        pids = list(ACTIVE_PROCESSES.keys())
    for pid in pids:
        try:
            stop_background_command(pid)
        except Exception:
            pass


atexit.register(stop_all_background_processes)



def read_git_diff() -> str:
    """Read the current unstaged and staged git diff of the project."""
    try:
        is_git = run_text("git rev-parse --is-inside-work-tree", shell=True, capture_output=True)
        if is_git.returncode != 0:
            return "Error: This directory is not a Git repository."

        unstaged = run_text("git diff", shell=True, capture_output=True).stdout
        staged = run_text("git diff --cached", shell=True, capture_output=True).stdout

        res = ""
        if staged:
            res += "=== STAGED CHANGES (READY TO COMMIT) ===\n" + staged + "\n"
        if unstaged:
            res += "=== UNSTAGED CHANGES ===\n" + unstaged + "\n"

        if not res:
            return "No changes detected in Git."

        # A full diff of a large refactor can be tens of thousands of tokens in a
        # single result. Cap it: show a per-file summary + a bounded slice, and
        # point the model at `git diff -- <file>` for the detail it actually needs.
        LIMIT = 12000
        if len(res) > LIMIT:
            stat = run_text("git diff --stat", shell=True, capture_output=True).stdout
            cached_stat = run_text("git diff --cached --stat", shell=True, capture_output=True).stdout
            summary = "\n".join(s for s in (cached_stat.strip(), stat.strip()) if s)
            return (
                f"[Diff is large ({len(res)} chars) — showing a file summary and the first "
                f"{LIMIT} characters. Inspect a specific file with `git diff -- <file>` via "
                f"run_command.]\n\n=== FILES CHANGED ===\n{summary}\n\n{res[:LIMIT]}\n…[diff truncated]"
            )
        return res
    except Exception as e:
        return f"Error reading git diff: {e}"
