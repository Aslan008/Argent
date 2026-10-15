"""Running due automations while Argent is closed, through the OS scheduler.

By design the in-app scheduler lives only while Argent is open. For the jobs that
must not wait for you to open it — the 03:00 backup, the morning digest — one
small entry in the operating system's own scheduler wakes up every few minutes,
runs whatever is due (`python src/automation/runner.py --tick`) and exits. It
makes the same decisions as the live scheduler, honours the pause, and shares the
per-automation lock with it, so having both running never doubles a job.

Per project, because automations are: each project that wants this gets its own
entry. Windows uses the Task Scheduler; elsewhere we print the cron line instead
of editing a crontab nobody asked us to touch.

The entry is registered from an XML definition rather than `schtasks /SC MINUTE`
flags: a task made with the flags refuses to start on battery power and is
stopped when the charger is unplugged, which on a laptop means "never runs when
you are away". The XML also lets a missed tick run as soon as the machine is up
(`StartWhenAvailable`) and forbids two ticks overlapping.
"""

import hashlib
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_MINUTES = 15
_ARGENT_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _ARGENT_ROOT / "src" / "automation" / "runner.py"

# A tick only decides and spawns; a job has its own timeout. This bounds a tick
# that hangs on something the jobs never touch.
EXECUTION_LIMIT = "PT2H"


def task_name(root: Path) -> str:
    root = Path(root)
    digest = hashlib.sha1(str(root.resolve()).lower().encode("utf-8")).hexdigest()[:6]
    return f"Argent Automations - {root.name[:30] or 'project'} ({digest})"


def _interpreter() -> str:
    """pythonw where it exists: a tick every 15 minutes must not flash a console."""
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return str(windowless if os.name == "nt" and windowless.exists() else exe)


def tick_arguments(root: Path) -> str:
    return f'"{_RUNNER}" --tick --project "{Path(root).resolve()}"'


def tick_command(root: Path) -> str:
    return f'"{_interpreter()}" {tick_arguments(root)}'


def cron_line(root: Path, minutes: int = DEFAULT_MINUTES) -> str:
    return f"*/{minutes} * * * * {tick_command(root)}"


def task_xml(root: Path, minutes: int = DEFAULT_MINUTES, now: datetime = None) -> str:
    """The Task Scheduler definition for one project's periodic tick."""
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    root = Path(root).resolve()
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{escape(f"Argent: runs the scheduled automations of {root} (src/automation/runner.py --tick)")}</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT{int(minutes)}M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{now:%Y-%m-%dT%H:%M:%S}</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <AllowHardTerminate>true</AllowHardTerminate>
    <ExecutionTimeLimit>{EXECUTION_LIMIT}</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(_interpreter())}</Command>
      <Arguments>{escape(tick_arguments(root))}</Arguments>
      <WorkingDirectory>{escape(str(root))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _schtasks(*args) -> list:
    return ["schtasks", *args]


def _text(output) -> str:
    """schtasks answers in the console's OEM code page (cp866 on a Russian
    Windows), which is not UTF-8: read bytes, decode them ourselves, never raise."""
    if isinstance(output, str):
        return output
    for encoding in ("utf-8", "oem"):
        try:
            return (output or b"").decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return (output or b"").decode("utf-8", errors="replace")


def install(root: Path, minutes: int = DEFAULT_MINUTES, run=subprocess.run):
    """Register the periodic tick. Returns (ok, message)."""
    minutes = max(5, min(int(minutes), 24 * 60 - 1))
    if os.name != "nt":
        return False, ("Автоустановка есть только для Windows. Добавьте в crontab (crontab -e):\n  "
                       + cron_line(root, minutes))
    name = task_name(root)
    handle, path = tempfile.mkstemp(prefix="argent-task-", suffix=".xml")
    os.close(handle)
    try:
        # schtasks insists on UTF-16 for an XML definition.
        Path(path).write_text(task_xml(root, minutes), encoding="utf-16")
        result = run(_schtasks("/Create", "/TN", name, "/XML", path, "/F"),
                     capture_output=True, timeout=60,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if result.returncode != 0:
        detail = " ".join((_text(result.stderr) or _text(result.stdout)).split())[:300]
        return False, f"Планировщик Windows отказал: {detail}"
    return True, (f"Готово: «{name}» проверяет расписание каждые {minutes} мин, даже когда Argent "
                  f"закрыт (пока вы вошли в Windows; пропущенное запускается, как только компьютер "
                  f"проснётся). Убрать: /tasks service remove")


def remove(root: Path, run=subprocess.run):
    """Remove the periodic tick. Returns (ok, message)."""
    if os.name != "nt":
        return False, "Автоустановки не было: уберите строку из crontab вручную."
    name = task_name(root)
    result = run(_schtasks("/Delete", "/TN", name, "/F"), capture_output=True,
                 timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        return False, f"«{name}» не найдена или не удалена."
    return True, f"«{name}» удалена: расписание снова работает только пока Argent открыт."


def status(root: Path, run=subprocess.run) -> dict:
    """{'installed': bool, 'name': str, 'supported': bool}"""
    info = {"installed": False, "name": task_name(root), "supported": os.name == "nt"}
    if os.name != "nt":
        return info
    try:
        result = run(_schtasks("/Query", "/TN", info["name"]), capture_output=True,
                     timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        info["installed"] = result.returncode == 0
    except Exception:
        pass
    return info
