import os
import subprocess
import queue
import threading
import ctypes
import time
from approval import request_approval, is_destructive_command, command_grant_key, assess_command_risk
from src.agent.shell import choose_shell, build_command_argv, run_text
from src.agent.command_diagnostics import diagnose_command_error
from memory_manager import memory
from ui import console
from logger import get_logger

log = get_logger("tools")


def _spawn(command: str, **popen_kwargs):
    """Start a process, picking the right shell on Windows (PowerShell vs cmd)
    so PowerShell-only commands work and cmd-style && / || chains still run.
    On POSIX, fall back to the default shell."""
    if os.name == "nt":
        argv = build_command_argv(command, choose_shell(command))
        return subprocess.Popen(argv, **popen_kwargs)
    return subprocess.Popen(command, shell=True, **popen_kwargs)

ACTIVE_PROCESSES = {}
ACTIVE_PROCESSES_LOCK = threading.Lock()
_pid_counter = 1
MAX_BACKGROUND_PROCESSES = 10

# A foreground run_command that emits nothing for this long is almost certainly
# waiting on interactive input (a prompt) or hung — either way it would block the
# whole turn forever, so we terminate it and tell the model what to do instead.
COMMAND_SILENCE_LIMIT = 180

def run_command(command: str) -> str:
    """Execute a console command and return its output. Requires user confirmation. Streams output to console."""
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

    destructive = level != "safe"
    detail = f" [{'; '.join(reasons)}]" if reasons else ""
    if level == "block":
        label = f"выполнить ОПАСНУЮ команду{detail}"
    elif destructive:
        label = f"выполнить ДЕСТРУКТИВНУЮ команду{detail}"
    else:
        label = "выполнить команду"
    approved = request_approval(
        f"{label}: {command}",
        destructive=destructive,
        grant_key=command_grant_key(command),
    )

    if not approved:
        return f"Execution aborted by user. The command '{command}' was NOT run."
        
    try:
        def decode_output(b: bytes) -> str:
            if not b:
                return ""
            try:
                return b.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return b.decode('cp866')
                except UnicodeDecodeError:
                    return b.decode('cp1251', errors='replace')
                    
        process = _spawn(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Read on a helper thread so a silent, still-running command can't block
        # the turn forever: the main loop waits on the queue with a timeout and
        # terminates the process if it goes quiet for too long (likely an
        # interactive prompt).
        line_queue: "queue.Queue" = queue.Queue()

        def _pump():
            try:
                for raw in iter(process.stdout.readline, b""):
                    if not raw:
                        break
                    line_queue.put(raw)
            except Exception:
                pass
            finally:
                line_queue.put(None)  # sentinel: stream closed

        threading.Thread(target=_pump, daemon=True).start()

        output_lines = []
        silent_timeout = False
        while True:
            try:
                raw_line = line_queue.get(timeout=COMMAND_SILENCE_LIMIT)
            except queue.Empty:
                if process.poll() is None:      # alive but mute -> stuck/awaiting input
                    silent_timeout = True
                    try:
                        process.terminate()
                    except Exception:
                        pass
                break
            if raw_line is None:                # stream closed: command finished
                break
            decoded_line = decode_output(raw_line)
            output_lines.append(decoded_line)
            console.print(f"[dim]{decoded_line.rstrip()}[/dim]")

        try:
            process.stdout.close()
        except Exception:
            pass
        process.wait()

        final_output = "".join(output_lines).strip()
        output = f"Exit code: {process.returncode}\n"

        if final_output:
            output += f"OUTPUT:\n{final_output}\n"

        if silent_timeout:
            output += (
                f"\n[DIAGNOSIS]: No output for {COMMAND_SILENCE_LIMIT}s — the command "
                f"likely awaits interactive input (a prompt) or is hung, so it was "
                f"terminated. Re-run non-interactively (add flags like --yes/-y or pipe "
                f"the answers), or use start_background_command + send_background_command "
                f"for a program that must be driven interactively.\n"
            )
        # Append a concrete diagnostic for recognized command failures so the
        # model fixes the real cause instead of guessing.
        elif process.returncode != 0:
            hint = diagnose_command_error(command, final_output, process.returncode)
            if hint:
                output += f"\n[DIAGNOSIS]: {hint}\n"

        log.info("run_command: %s (exit=%d)", command, process.returncode)

        cmd_lower = command.strip().lower()
        is_fire_and_forget = any(cmd_lower.startswith(p) for p in ["explorer", "start ", "start.", 'start"'])

        if process.returncode != 0 and not is_fire_and_forget:
            memory.add_error(f"Command '{command}' failed (exit={process.returncode})")
        else:
            memory.add_completed(f"Ran: {command}")
        return output.strip()
    except Exception as e:
        log.error("run_command error: %s: %s", command, e)
        return f"Error running command '{command}': {e}"

def run_admin_command(command: str) -> str:
    """Execute a PowerShell command with Administrator privileges (UAC prompt)."""
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

        wrapped_command = f"{command} > '{temp_out}' 2>&1"
        result = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "powershell.exe",
            f"-Command \"{wrapped_command}\"", None, 0,
        )
        if result <= 32:
            return f"Error: UAC prompt was denied or execution failed. Error code: {result}"

        timeout = 20
        start_time = time.time()
        while time.time() - start_time < timeout:
            if temp_out.exists():
                try:
                    out = temp_out.read_text(encoding="utf-8", errors="replace").strip()
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

def start_background_command(command: str) -> str:
    """Launch a command in the background and return its PID."""
    with ACTIVE_PROCESSES_LOCK:
        if len(ACTIVE_PROCESSES) >= MAX_BACKGROUND_PROCESSES:
            return f"Error: Maximum number of background processes ({MAX_BACKGROUND_PROCESSES}) reached. Stop an existing process first."
    
    approved = request_approval(
        f"запустить фоновый процесс: {command}",
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
        process = _spawn(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
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
        
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[pid] = {
                "process": process,
                "out_queue": out_queue,
                "err_queue": err_queue,
                "command": command,
                "started": time.time(),
        }
        
        return f"Started background process with PID: {pid}"
    except Exception as e:
        return f"Error starting background command '{command}': {e}"

def read_background_command(pid: str) -> str:
    """Read the latest output from a background process."""
    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            return f"Error: No active process with PID {pid}."
        proc_info = ACTIVE_PROCESSES[pid]
        
    process = proc_info["process"]
    
    def decode_q(q):
        data = bytearray()
        while True:
            try:
                chunk = q.get_nowait()
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
    
    retcode = process.poll()
    if retcode is not None:
        status = f"Process {pid} has EXITED with code {retcode}."
        with ACTIVE_PROCESSES_LOCK:
            if pid in ACTIVE_PROCESSES:
                del ACTIVE_PROCESSES[pid]
    else:
        # Include elapsed runtime so repeated polls of a still-running process
        # produce DISTINCT results — otherwise the loop guard sees identical
        # "RUNNING / no new output" and force-stops a legitimate wait.
        started = proc_info.get("started")
        elapsed = int(time.time() - started) if started is not None else 0
        status = f"Process {pid} is RUNNING for {elapsed}s."
        
    res = f"--- {status} ---\n"
    if stdout:
        res += f"STDOUT:\n{stdout}\n"
    if stderr:
        res += f"STDERR:\n{stderr}\n"
        
    if not stdout and not stderr:
        res += "No new output.\n"
        
    return res

def send_background_command(pid: str, input_string: str) -> str:
    """Send text to the standard input of a running background process."""
    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            return f"Error: No active process with PID {pid}."
        process = ACTIVE_PROCESSES[pid]["process"]
        
    if process.poll() is not None:
        return f"Error: Process {pid} has already exited."
        
    try:
        console.print(f"\n[bold yellow]Agent sending input to PID {pid}:[/bold yellow] {input_string.strip()}")
        if not input_string.endswith('\n'):
            input_string += '\n'
        process.stdin.write(input_string.encode('utf-8'))
        process.stdin.flush()
        return f"Sent input to PID {pid}."
    except Exception as e:
        return f"Error sending input to PID {pid}: {e}"

def stop_background_command(pid: str) -> str:
    """Terminate a background process."""
    with ACTIVE_PROCESSES_LOCK:
        if pid not in ACTIVE_PROCESSES:
            return f"Error: No active process with PID {pid}."
        process = ACTIVE_PROCESSES[pid]["process"]
        try:
            process.terminate()
            del ACTIVE_PROCESSES[pid]
            return f"Terminated background process PID {pid}."
        except Exception as e:
            return f"Error terminating PID {pid}: {e}"

def list_background_commands() -> str:
    """List the background processes started this session, with PID, status and command."""
    with ACTIVE_PROCESSES_LOCK:
        items = list(ACTIVE_PROCESSES.items())

    if not items:
        return "No background processes are currently tracked."

    lines = []
    for pid, info in items:
        ret = info["process"].poll()
        if ret is None:
            started = info.get("started")
            elapsed = int(time.time() - started) if started is not None else 0
            status = f"RUNNING {elapsed}s"
        else:
            status = f"EXITED (code {ret})"
        lines.append(f"- PID {pid}: [{status}] {info['command']}")

    return (
        "Background processes:\n" + "\n".join(lines)
        + "\n\nUse read_background_command(pid) for output, "
          "send_background_command(pid, text) for input, "
          "stop_background_command(pid) to terminate."
    )


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
