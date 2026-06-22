import os
import subprocess
import queue
import threading
import ctypes
import time
from approval import request_approval, is_destructive_command, command_grant_key, assess_command_risk
from src.agent.shell import choose_shell, build_command_argv
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

        output_lines = []
        for raw_line in iter(process.stdout.readline, b""):
            if not raw_line:
                break
            decoded_line = decode_output(raw_line)
            output_lines.append(decoded_line)
            console.print(f"[dim]{decoded_line.rstrip()}[/dim]")

        process.stdout.close()
        process.wait()

        final_output = "".join(output_lines).strip()
        output = f"Exit code: {process.returncode}\n"

        if final_output:
            output += f"OUTPUT:\n{final_output}\n"

        # Append a concrete diagnostic for recognized command failures so the
        # model fixes the real cause instead of guessing.
        if process.returncode != 0:
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
        
    try:
        from pathlib import Path
        temp_out = Path("C:/Windows/Temp/argent_admin_out.txt")
        if temp_out.exists():
            temp_out.unlink()
            
        wrapped_command = f"{command} > '{temp_out}' 2>&1"
        
        result = ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas", 
            "powershell.exe", 
            f"-Command \"{wrapped_command}\"", 
            None, 
            0
        )
        
        if result <= 32:
            return f"Error: UAC prompt was denied or execution failed. Error code: {result}"
            
        timeout = 20
        start_time = time.time()
        while time.time() - start_time < timeout:
            if temp_out.exists():
                try:
                    with open(temp_out, "r", encoding="utf-8", errors="replace") as f:
                        out = f.read().strip()
                    temp_out.unlink()
                    return f"Admin execution completed.\nOutput:\n{out}"
                except PermissionError:
                    pass
            time.sleep(0.5)
            
        try:
            if temp_out.exists():
                temp_out.unlink()
        except Exception:
            pass
        return "Admin execution started, but timed out waiting for output file. It may still be running in the background."
        
    except Exception as e:
        return f"Error running admin command '{command}': {e}"

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
                "command": command
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
        status = f"Process {pid} is RUNNING."
        
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
        status = "RUNNING" if ret is None else f"EXITED (code {ret})"
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
        is_git = subprocess.run("git rev-parse --is-inside-work-tree", shell=True, capture_output=True, text=True)
        if is_git.returncode != 0:
            return "Error: This directory is not a Git repository."
            
        unstaged = subprocess.run("git diff", shell=True, capture_output=True, text=True).stdout
        staged = subprocess.run("git diff --cached", shell=True, capture_output=True, text=True).stdout
        
        res = ""
        if staged:
            res += "=== STAGED CHANGES (READY TO COMMIT) ===\n" + staged + "\n"
        if unstaged:
            res += "=== UNSTAGED CHANGES ===\n" + unstaged + "\n"
            
        return res if res else "No changes detected in Git."
    except Exception as e:
        return f"Error reading git diff: {e}"
