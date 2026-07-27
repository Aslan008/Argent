"""Read-only system inspection: event log, processes, registry, file search.

SECURITY: these tools take model-supplied strings (a registry path, a glob) and
must never let those strings become *code*. The original implementation built
PowerShell scripts with f-strings, so a path like `C:\\"; Invoke-WebRequest ...; #`
executed as a command. Two rules now hold everywhere in this module:

1. Prefer no shell at all — the registry and file search run on native Python
   (winreg / pathlib), which cannot interpret an argument as a command.
2. Where PowerShell is genuinely required (Get-WinEvent, Get-Process), values
   are passed through the child's ENVIRONMENT and referenced as $env:VAR inside
   the script. Environment values are data to the parser and are never parsed
   as code, so quoting tricks have nothing to break out of.

Everything here is read-only, so it is not approval-gated — same policy as
read_file / list_directory.
"""

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

from tools._helpers import log

# Bound the output of enumerating tools so a huge tree can't flood the context.
_MAX_FILE_RESULTS = 200
_PS_TIMEOUT = 30


def _run_powershell(script: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a FIXED PowerShell script; variable data arrives via the environment.

    script must be a literal — never an f-string carrying caller input.
    """
    env = dict(os.environ)
    for key, value in (extra_env or {}).items():
        env[key] = str(value)
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=_PS_TIMEOUT,
    )


def read_event_logs(log_name: str = "Application", entry_type: str = "Error,Warning", newest: int = 20) -> str:
    """Reads the Windows Event Log using PowerShell."""
    if not sys.platform.startswith("win"):
        return "Error: reading Windows event logs is only supported on Windows."
    try:
        # Validate rather than interpolate: the log name is an identifier and
        # the levels come from a fixed vocabulary.
        if not log_name or not all(ch.isalnum() or ch in "-_/ " for ch in log_name):
            return f"Error: invalid log name '{log_name}'."
        valid_levels = {"Critical", "Error", "Warning", "Information", "Verbose"}
        levels = [lv.strip().capitalize() for lv in (entry_type or "").split(",") if lv.strip()]
        bad = [lv for lv in levels if lv not in valid_levels]
        if bad:
            return f"Error: unknown entry type(s) {bad}. Valid: {sorted(valid_levels)}."
        try:
            newest = max(1, min(int(newest), 500))
        except (TypeError, ValueError):
            newest = 20

        script = (
            '$levels = $env:ARGENT_LEVELS -split ","; '
            'Get-WinEvent -LogName $env:ARGENT_LOG -MaxEvents ([int]$env:ARGENT_MAX) -ErrorAction Stop '
            '| Where-Object { $_.LevelDisplayName -in $levels } '
            '| Select-Object TimeCreated, Id, LevelDisplayName, Message '
            '| ConvertTo-Json -Compress'
        )
        result = _run_powershell(script, {
            "ARGENT_LOG": log_name,
            "ARGENT_LEVELS": ",".join(levels or ["Error", "Warning"]),
            "ARGENT_MAX": newest,
        })
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        if "No events were found" in (result.stderr or ""):
            return "No matching events found in the log."
        return f"Error or empty result reading event logs: {(result.stderr or '').strip()}"
    except subprocess.TimeoutExpired:
        return "Error: reading event logs timed out."
    except Exception as e:
        return f"Exception executing read_event_logs: {e}"


def get_process_info(process_name: str = "") -> str:
    """Gets information about running processes using PowerShell."""
    if not sys.platform.startswith("win"):
        return "Error: get_process_info is only supported on Windows."
    try:
        script = (
            '$pattern = $env:ARGENT_PROC; '
            '$procs = if ($pattern) { Get-Process -ErrorAction SilentlyContinue '
            '| Where-Object { $_.Name -like "*$pattern*" } } '
            'else { Get-Process -ErrorAction SilentlyContinue }; '
            '$procs | Select-Object Name, Id, CPU, WorkingSet | ConvertTo-Json -Compress'
        )
        result = _run_powershell(script, {"ARGENT_PROC": process_name or ""})
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "Process not found or error occurred."
    except subprocess.TimeoutExpired:
        return "Error: listing processes timed out."
    except Exception as e:
        return f"Exception executing get_process_info: {e}"


# Registry roots, accepted both as PowerShell drive names (HKLM:) and full hive
# names (HKEY_LOCAL_MACHINE) so existing model habits keep working.
_HIVES = {
    "HKLM": "HKEY_LOCAL_MACHINE", "HKCU": "HKEY_CURRENT_USER",
    "HKCR": "HKEY_CLASSES_ROOT", "HKU": "HKEY_USERS", "HKCC": "HKEY_CURRENT_CONFIG",
}


def query_registry(path: str, name: str = "") -> str:
    """Queries a registry key or value using the native Windows registry API."""
    if not sys.platform.startswith("win"):
        return "Error: the Windows registry is only available on Windows."
    try:
        import winreg
    except ImportError:
        return "Error: the Windows registry API is unavailable."

    if not path or not path.strip():
        return "Error: registry path is required."

    raw = path.strip().strip('"').strip("'").replace("/", "\\")
    root_part, _, sub_path = raw.partition("\\")
    root_key = root_part.rstrip(":").upper()
    hive_name = _HIVES.get(root_key, root_key)
    hive = getattr(winreg, hive_name, None)
    if hive is None or not isinstance(hive, int):
        return (f"Error: unknown registry root '{root_part}'. "
                f"Use one of: {', '.join(sorted(_HIVES))}.")

    try:
        with winreg.OpenKey(hive, sub_path) as key:
            if name:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    return f"Error: value '{name}' not found under '{path}'."
                return f"{name} = {value}"

            count = winreg.QueryInfoKey(key)[1]
            lines = []
            for i in range(count):
                try:
                    val_name, val_data, _ = winreg.EnumValue(key, i)
                except OSError:
                    continue
                lines.append(f"{val_name or '(Default)'} = {val_data}")
            if not lines:
                return f"Key '{path}' exists but has no values."
            return "\n".join(lines)
    except FileNotFoundError:
        return f"Error: registry key '{path}' not found."
    except PermissionError:
        return f"Error: access denied reading registry key '{path}'."
    except OSError as e:
        return f"Error reading registry: {e}"


def search_system_files(path: str, filter_pattern: str = "*") -> str:
    """Fast recursive search for files matching a glob pattern (native, no shell)."""
    if not path or not path.strip():
        return "Error: search path is required."
    root = Path(path).expanduser()
    if not root.exists():
        return f"Error: path '{path}' does not exist."
    if not root.is_dir():
        return f"Error: '{path}' is not a directory."

    pattern = (filter_pattern or "*").strip() or "*"
    matches = []
    truncated = False
    try:
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            for fname in filenames:
                if fnmatch.fnmatch(fname, pattern):
                    full = Path(dirpath) / fname
                    try:
                        size = full.stat().st_size
                    except OSError:
                        size = 0
                    matches.append(f"{full} ({size} bytes)")
                    if len(matches) >= _MAX_FILE_RESULTS:
                        truncated = True
                        break
            if truncated:
                break
    except Exception as e:
        return f"Exception executing search_system_files: {e}"

    if not matches:
        return "No files found or access denied."
    log.info("search_system_files: %s '%s' -> %d hits", root, pattern, len(matches))
    out = "\n".join(matches)
    if truncated:
        out += f"\n… (stopped at {_MAX_FILE_RESULTS} results — narrow the pattern)"
    return out
