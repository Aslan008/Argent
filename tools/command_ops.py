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
from approval import (
    request_approval, is_destructive_command, command_grant_key, assess_command_risk,
    RUN_COMMAND_ACTION, BACKGROUND_COMMAND_ACTION,
)
from src.agent.shell import (
    choose_shell, build_command_argv, run_text,
    kill_process_tree as _kill_process_tree,
)
from src.agent.command_diagnostics import diagnose_command_error
from memory_manager import memory
from ui import console
from logger import get_logger

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



def _spawn(command: str, cwd: str | None = None, **popen_kwargs):
    """Start a process, picking the right shell on Windows (PowerShell vs cmd)
    so PowerShell-only commands work and cmd-style && / || chains still run.
    On POSIX, fall back to the default shell."""
    if os.name == "nt":
        argv = build_command_argv(command, choose_shell(command))
        return subprocess.Popen(argv, cwd=cwd, **popen_kwargs)
    return subprocess.Popen(command, shell=True, cwd=cwd, **popen_kwargs)

ACTIVE_PROCESSES = {}
ACTIVE_PROCESSES_LOCK = threading.Lock()
COMPLETED_PROCESSES = {}
_pid_counter = 1
MAX_BACKGROUND_PROCESSES = 10

# After this many consecutive reads with no new output, inject a hint
# telling the model to switch to wait_heartbeat instead of busy-polling.
IDLE_POLL_HINT_THRESHOLD = 3

COMMAND_SILENCE_LIMIT = 180
FAST_INTERACTIVE_SILENCE_LIMIT = 3.0

from text_safety import strip_terminal_escapes as _strip_escapes

INTERACTIVE_PROMPT_PATTERNS = [
    re.compile(r"(?:\[y/n\]|\(y/n\)|\[yes/no\]|\(yes/no\))\s*$", re.IGNORECASE),
    re.compile(r"(?:password|passphrase|token)\s*:\s*$", re.IGNORECASE),
    re.compile(r"(?:select|choose)\s+(?:an?\s+)?(?:option|number|\d+)?\s*(?:\([^\)]+\)|\[[^\]]+\])?\s*[:?]?\s*$", re.IGNORECASE),
    re.compile(r"(?:press\s+any\s+key|hit\s+enter|press\s+enter)\s*(?:to\s+continue)?\s*(?:[\.!…]|\.\s*)+\s*$", re.IGNORECASE),
    re.compile(r"(?:are\s+you\s+sure\b.*|confirm(?:\s+action)?)\s*[:?]\s*$", re.IGNORECASE),
    re.compile(r"(?:do\s+you\s+want\s+to\s+continue|proceed)\s*[:?]?\s*$", re.IGNORECASE),
]

def run_command(command: str, cwd: str | None = None, timeout: int | None = None,
                total_timeout: int | None = None) -> str:
    """Execute a console command and return its output. Requires user confirmation.

    `total_timeout` raises the absolute ceiling (300 s by default) for a command that is known to run
    long and keeps printing — a room's test step. It is not in the model's schema; only a room's
    `args` can set it.

    Streams output to the console while the command runs and returns the full
    result. `process_exit_time` and `pump_thread.is_alive()` let it detach from
    grandchildren that hold the stdout pipe (like `start /b`) and hand control
    back to the agent instead of freezing for 5 minutes.
    """
    if cwd:
        cwd_path = Path(cwd)
        if not cwd_path.exists() or not cwd_path.is_dir():
            return f"Error: Directory does not exist: {cwd}"

    from config import get_command_guard
    guard = get_command_guard()
    level, reasons = assess_command_risk(command)
    if guard == "off":
        reasons = []

    # Hard refusal of catastrophic commands when the gate is set to block.
    if guard == "block" and level == "block":
        why = "; ".join(reasons) or "catastrophic command"
        log.warning("Risk-gate BLOCKED: %s (%s)", command, why)
        return (
            f"Command BLOCKED by the risk-gate (/guard block): '{command}'.\n"
            f"Reason: {why}. It was refused without execution. Use a safer, scoped "
            f"command, or lower the gate with /guard warn to allow it with confirmation."
        )
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

        output_chunks = []
        interactive_timeout = False
        silent_timeout = False
        hard_timeout = False
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
            if aborted or interactive_timeout or silent_timeout or hard_timeout:
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
        if "\r" in clean_final_output:
            lines = []
            for line in clean_final_output.split("\n"):
                if "\r" in line:
                    parts = [p.strip() for p in line.split("\r") if p.strip()]
                    lines.append(parts[-1] if parts else "")
                else:
                    lines.append(line)
            clean_final_output = "\n".join(lines)
        output = f"Exit code: {process.returncode}\n"
        
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
            trailing = clean_final_output.rstrip() if clean_final_output else ""
            output += (
                f"\n[DIAGNOSIS]: The command was waiting for an answer to: \"{trailing}\". "
                f"Use `start_background_command` and `send_background_command` to interact "
                f"with it, or provide the input non-interactively (e.g. `echo y | command`).\n"
            )
        elif process.returncode != 0:
            hint = diagnose_command_error(command, clean_final_output, process.returncode)
            if hint:
                output += f"\n[DIAGNOSIS]: {hint}\n"

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
        if exit_code != 0 and not is_fire_and_forget:
            memory.add_error(f"Command '{command}' failed (exit={exit_code})")
        else:
            if hasattr(memory, "resolve_command_error"):
                memory.resolve_command_error(command)
            memory.add_completed(f"Ran: {command}")
        # The model's window may cut the middle of a long log: keep the whole of it in a file it can read.
        try:
            from config import get_current_model
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


def run_admin_command(command: str) -> str:
    """Execute a PowerShell command with Administrator privileges (UAC prompt)."""
    if os.name != "nt" or not hasattr(ctypes, "windll"):
        return "Error: run_admin_command is only supported on Windows (UAC elevation)."

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
    try:
        temp_out.unlink()  # elevated child re-creates it on completion

        # Pass the script via -EncodedCommand (base64 of UTF-16LE), not
        # -Command "…": inline quoting breaks the moment the user's command
        # contains a double quote (Write-Host "x"), which both fails valid
        # commands and is a quote-injection surface. EncodedCommand carries an
        # arbitrary script verbatim, no escaping needed.
        wrapped_command = f"{command} > '{temp_out}' 2>&1"
        encoded = _encode_powershell_command(wrapped_command)
        result = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "powershell.exe",
            f"-NoProfile -EncodedCommand {encoded}", None, 0,
        )
        if result <= 32:
            return f"Error: UAC prompt was denied or execution failed. Error code: {result}"

        timeout = 20
        start_time = time.time()
        while time.time() - start_time < timeout:
            if temp_out.exists():
                try:
                    data = temp_out.read_bytes()
                    if data:
                        if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
                            out = data.decode("utf-16", errors="replace").strip()
                        elif data.startswith(b"\xef\xbb\xbf"):
                            out = data[3:].decode("utf-8", errors="replace").strip()
                        elif b"\x00" in data:
                            try:
                                out = data.decode("utf-16le").strip()
                            except UnicodeDecodeError:
                                out = data.decode("utf-8", errors="replace").strip()
                        else:
                            try:
                                out = data.decode("utf-8").strip()
                            except UnicodeDecodeError:
                                out = data.decode("cp1251" if os.name == "nt" else "utf-8", errors="replace").strip()
                        return f"Admin execution completed.\nOutput:\n{out}"
                except PermissionError:
                    pass  # still being written by the elevated process
            time.sleep(0.5)

        return ("Admin execution started, but timed out waiting for output. "
                "It may still be running in the background.")
    except Exception as e:
        return f"Error running admin command '{command}': {e}"
    finally:
        try:
            temp_out.unlink(missing_ok=True)
        except Exception:
            pass

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
                    _close_proc_pipes(proc)
                del ACTIVE_PROCESSES[p]
                if len(COMPLETED_PROCESSES) >= 50:
                    oldest_k = next(iter(COMPLETED_PROCESSES))
                    COMPLETED_PROCESSES.pop(oldest_k, None)
                COMPLETED_PROCESSES[p] = {
                    "exit_code": proc.poll() if proc else -1,
                    "command": info.get("command", ""),
                }

        if len(ACTIVE_PROCESSES) >= MAX_BACKGROUND_PROCESSES:
            return f"Error: Maximum number of background processes ({MAX_BACKGROUND_PROCESSES}) reached. Stop an existing process first."
    
    approved = request_approval(
        f"{BACKGROUND_COMMAND_ACTION}{command}",
        destructive=is_destructive_command(command),
        grant_key=command_grant_key(command),
    )

    if not approved:
        return f"Execution aborted by user. The command '{command}' was NOT started."
        
    global _pid_counter
    with ACTIVE_PROCESSES_LOCK:
        pid = str(_pid_counter)
        _pid_counter += 1
    
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
        
        out_queue = queue.Queue()
        err_queue = queue.Queue()
        
        def reader(pipe, q):
            try:
                while True:
                    data = pipe.read(1024)
                    if not data:
                        break
                    q.put(data)
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
        
        if not data:
            return ""
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return data.decode('cp866')
            except UnicodeDecodeError:
                return data.decode('cp1251', errors='replace')
                
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
            if pid in ACTIVE_PROCESSES:
                del ACTIVE_PROCESSES[pid]
            if len(COMPLETED_PROCESSES) >= 50:
                oldest_k = next(iter(COMPLETED_PROCESSES))
                COMPLETED_PROCESSES.pop(oldest_k, None)
            COMPLETED_PROCESSES[pid] = {
                "exit_code": retcode,
                "command": proc_info.get("command", ""),
            }
        if process:
            _close_proc_pipes(process)
    else:
        # Include elapsed runtime for human readability.
        started = proc_info.get("started")
        elapsed = int(time.time() - started) if started is not None else 0
        status = f"Process {pid} is RUNNING for {elapsed}s."
        
    if proc_info.get("pty"):
        from src.agent.pty_process import TerminalText
        terminal = proc_info.setdefault("terminal", TerminalText())
        stdout = terminal.feed(stdout)
        if retcode is not None:
            stdout += terminal.flush()

    # Output of a background command arrives in polls, not in one piece: one filter per process and
    # stream keeps a secret that straddles two polls from slipping through.
    from src.agent.secret_redactor import StreamRedactor
    _r_out = proc_info.setdefault("redactor_out", StreamRedactor())
    _r_err = proc_info.setdefault("redactor_err", StreamRedactor())
    stdout = _r_out.feed(stdout)
    stderr = _r_err.feed(stderr)
    if retcode is not None:
        stdout += _r_out.flush()
        stderr += _r_err.flush()
    else:
        stdout += _r_out.release_stale(0.0)
        stderr += _r_err.release_stale(0.0)

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

def send_background_command(pid: Any, input_string: str) -> str:
    """Send text to the standard input of a running background process."""
    pid = _normalize_pid(pid)
    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            return f"Error: No active process with PID {pid}."
        process = ACTIVE_PROCESSES[pid].get("process")
        secret_prompt = bool(ACTIVE_PROCESSES[pid].get("secret_prompt"))

    if not process or process.poll() is not None:
        with ACTIVE_PROCESSES_LOCK:
            proc_info = ACTIVE_PROCESSES.pop(pid, None)
            if proc_info:
                if len(COMPLETED_PROCESSES) >= 50:
                    oldest_k = next(iter(COMPLETED_PROCESSES))
                    COMPLETED_PROCESSES.pop(oldest_k, None)
                COMPLETED_PROCESSES[pid] = {
                    "exit_code": process.poll() if process else -1,
                    "command": proc_info.get("command", ""),
                }
        if process:
            _close_proc_pipes(process)
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
        return f"Sent input to PID {pid}."
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

        if len(COMPLETED_PROCESSES) >= 50:
            oldest_k = next(iter(COMPLETED_PROCESSES))
            COMPLETED_PROCESSES.pop(oldest_k, None)
        COMPLETED_PROCESSES[pid] = {
            "exit_code": -15 if os.name != "nt" else 1,
            "command": proc_info.get("command", ""),
            "stopped": True,
        }

    # Kill the whole process tree OUTSIDE the lock
    if process is not None:
        try:
            _kill_process_tree(process)
        except Exception as e:
            return f"Error terminating PID {pid}: {e}"

        # Close pipes in a daemon thread: if a reader thread is still blocked
        # on pipe.read() (possible on Windows when a grandchild briefly
        # survives the kill), close() would block the calling thread.
        threading.Thread(target=_close_proc_pipes, args=(process,), daemon=True).start()

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
