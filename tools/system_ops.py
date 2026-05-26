import subprocess
import json
from tools._helpers import log

def read_event_logs(log_name: str = "Application", entry_type: str = "Error,Warning", newest: int = 20) -> str:
    """Reads the Windows Event Log using PowerShell."""
    try:
        cmd = f'Get-WinEvent -LogName {log_name} -MaxEvents {newest} -ErrorAction Stop | Where-Object {{ $_.LevelDisplayName -in "{entry_type.replace(",", "\",\"")}" }} | Select-Object TimeCreated, Id, LevelDisplayName, Message | ConvertTo-Json -Compress'
        result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            # Formatting JSON for better readability if needed, or return raw json
            return result.stdout.strip()
        else:
            if "No events were found" in result.stderr:
                return "No matching events found in the log."
            return f"Error or empty result reading event logs: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception executing read_event_logs: {e}"

def get_process_info(process_name: str = "") -> str:
    """Gets information about running processes using PowerShell."""
    try:
        name_filter = f"-Name *{process_name}*" if process_name else ""
        cmd = f'Get-Process {name_filter} -ErrorAction SilentlyContinue | Select-Object Name, Id, CPU, WorkingSet | ConvertTo-Json -Compress'
        result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "Process not found or error occurred."
    except Exception as e:
        return f"Exception executing get_process_info: {e}"

def query_registry(path: str, name: str = "") -> str:
    """Queries a registry key or value using PowerShell."""
    try:
        # Example path format expected by PS: "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion"
        if name:
            cmd = f'Get-ItemProperty -Path "{path}" -Name "{name}" -ErrorAction Stop | Select-Object "{name}" | ConvertTo-Json -Compress'
        else:
            cmd = f'Get-ItemProperty -Path "{path}" -ErrorAction Stop | ConvertTo-Json -Compress'
        result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return f"Error reading registry: {result.stderr.strip()}"
    except Exception as e:
         return f"Exception executing query_registry: {e}"

def search_system_files(path: str, filter_pattern: str = "*") -> str:
    """Fast search for files matching a pattern using PowerShell Get-ChildItem."""
    try:
        cmd = f'Get-ChildItem -Path "{path}" -Filter "{filter_pattern}" -Recurse -File -ErrorAction SilentlyContinue | Select-Object FullName, Length | ConvertTo-Json -Compress'
        result = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
             return result.stdout.strip()
        return "No files found or access denied."
    except Exception as e:
        return f"Exception executing search_system_files: {e}"
