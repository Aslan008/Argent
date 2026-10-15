"""Automation definitions and run history on disk.

Definitions live in <project>/.argent/automations.json, anchored to the project
root (not the CWD) like the rest of Argent's per-project state, so starting the
server from a subfolder doesn't silently produce a second, empty set.

Run history is a separate append-only JSONL: a definitions file that is
rewritten on every run would eventually lose definitions to a crash mid-write,
and the two have completely different write rates. Even so, the definitions ARE
rewritten after every run (the run stamps `last_run` on its definition), and the
run happens in another process — so every read-modify-write of that file takes a
short cross-process lock, or a `/tasks off` typed while a job finishes would be
silently overwritten by the job's stale copy.
"""

import contextlib
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from logger import get_logger

log = get_logger("automation")

_DEFS_NAME = "automations.json"
_RUNS_NAME = "automation_runs.jsonl"
_STATE_NAME = "automation_state.json"

# Keep the tail readable and bounded; this is a log, not an archive.
MAX_RUN_HISTORY = 200
_TRIM_AT_BYTES = 400_000
SUMMARY_LIMIT = 4000

MODES = ("agent", "command", "remind", "room")
NOTIFY_POLICIES = ("auto", "always", "failures", "changes", "never")
CHANNELS = ("telegram",)
MAX_NAME_LENGTH = 60


@dataclass
class Automation:
    """One scheduled task.

    ``mode`` says what a run IS:
      agent    the model works on ``task`` with a bounded toolset (the default);
      command  ``task`` is a shell command run as-is — no model, no tokens;
      remind   ``task`` is a message delivered as-is — no model, no process.
      room     ``task`` is ``<room> [key=value ...]``: that room runs unattended (a question to a
               person ends it as "needs a person"; a step that asks first is not run).
    The last two exist because most "automation" is not a thought: a backup, a
    health check or "call Anna at 15:00" should not cost an LLM turn or depend on
    a provider being up.

    ``allowed_tools`` is the capability boundary: an unattended task gets only
    the tools it needs, so a monitoring job cannot start editing files even if
    the model decides that would be helpful. Empty means "the default toolset",
    which is deliberately NOT the same as "everything is fine" — see runner.

    ``allowed_commands`` is the one place a human pre-approves something for the
    model: exact shell commands (``*`` matches within one command, never across
    ``; & |``). Anything else that needs approval is still refused.
    """
    name: str
    task: str
    schedule: str
    enabled: bool = True
    allowed_tools: list = field(default_factory=list)
    max_turns: int = 12
    last_run: str | None = None          # ISO timestamp
    last_status: str | None = None       # ok | error | denied | skipped
    mode: str = "agent"
    notify: str = "auto"                 # auto | always | failures | changes | never
    channels: list = field(default_factory=list)
    allowed_commands: list = field(default_factory=list)
    timeout_minutes: int = 0             # 0 = the runner's default
    report_file: str = ""
    retries: int = 1                     # extra tries after a failed daily/monthly/one-shot run
    catchup_minutes: int = 60            # how late a daily/monthly slot may still fire
    max_failures: int = 5                # consecutive failures before it pauses itself; 0 = never
    fail_streak: int = 0
    disabled_reason: str = ""
    created: str | None = None
    run_count: int = 0
    last_digest: str = ""                # fingerprint of the last report, for notify=changes

    def last_run_dt(self) -> datetime | None:
        if not self.last_run:
            return None
        try:
            return datetime.fromisoformat(self.last_run)
        except ValueError:
            return None


def _argent_dir() -> Path:
    from project_paths import project_root_or_cwd
    return project_root_or_cwd() / ".argent"


def defs_path() -> Path:
    return _argent_dir() / _DEFS_NAME


def runs_path() -> Path:
    return _argent_dir() / _RUNS_NAME


def state_path() -> Path:
    return _argent_dir() / _STATE_NAME


def project_root() -> Path:
    from project_paths import project_root_or_cwd
    return Path(project_root_or_cwd())


# --- validation --------------------------------------------------------------------

def validate_name(name: str) -> str | None:
    """A problem with the name, or None. Names are typed after `/tasks run`, used
    in file names and passed to a child process, so keep them plain."""
    if not name or not name.strip():
        return "имя не может быть пустым"
    if name != name.strip():
        return "имя не должно начинаться или заканчиваться пробелом"
    if len(name) > MAX_NAME_LENGTH:
        return f"имя длиннее {MAX_NAME_LENGTH} символов"
    if re.search(r"[|\x00-\x1f\"]", name):
        return "в имени нельзя использовать | \" и управляющие символы"
    if name.lower() in ("all", "все"):
        return f"'{name}' — зарезервированное слово"
    return None


def validate_automation(automation: Automation) -> list:
    """Every problem with a definition, as human-readable lines (empty = fine)."""
    from src.automation.report import ReportPathError, resolve_report_path
    from src.automation.schedule import ScheduleError, parse_schedule

    problems = []
    issue = validate_name(automation.name)
    if issue:
        problems.append(issue)
    if automation.mode not in MODES:
        problems.append(f"режим '{automation.mode}' неизвестен (допустимо: {', '.join(MODES)})")
    if not (automation.task or "").strip():
        problems.append({"command": "не указана команда", "remind": "нет текста напоминания",
                         "room": "не указана комната"}.get(automation.mode, "не указана задача"))
    elif automation.mode == "room":
        from src.rooms.headless import split_task
        room_name = split_task(automation.task)[0]
        if not room_name.replace("_", "").replace("-", "").replace(".", "").isalnum():
            problems.append(f"'{room_name}' не похоже на имя комнаты")
    try:
        parse_schedule(automation.schedule, now=datetime.now())
    except ScheduleError as e:
        problems.append(str(e))
    if automation.notify not in NOTIFY_POLICIES:
        problems.append(f"notify='{automation.notify}' неизвестен (допустимо: {', '.join(NOTIFY_POLICIES)})")
    bad = [c for c in automation.channels if c not in CHANNELS]
    if bad:
        problems.append(f"неизвестные каналы: {', '.join(bad)} (допустимо: {', '.join(CHANNELS)})")
    if not 0 <= int(automation.timeout_minutes) <= 24 * 60:
        problems.append("timeout должен быть от 0 до 1440 минут")
    if not 1 <= int(automation.max_turns) <= 200:
        problems.append("turns должен быть от 1 до 200")
    if not 0 <= int(automation.retries) <= 5:
        problems.append("retries должен быть от 0 до 5")
    if not 0 <= int(automation.max_failures) <= 100:
        problems.append("max_failures должен быть от 0 до 100")
    if any(not str(c).strip() for c in automation.allowed_commands):
        problems.append("в списке разрешённых команд есть пустая строка")
    if automation.report_file:
        try:
            resolve_report_path(automation.report_file, automation.name, datetime.now(),
                                project_root())
        except ReportPathError as e:
            problems.append(str(e))
    return problems


# --- cross-process lock for the definitions file ------------------------------------

_DEFS_LOCK_STALE_SECONDS = 30
_DEFS_LOCK_WAIT_SECONDS = 5.0


@contextlib.contextmanager
def _defs_lock():
    """Serialise read-modify-write of automations.json across processes.

    A file made with O_EXCL is the one thing two processes can agree on without a
    server. If the lock cannot be taken in time we proceed anyway: a lost update
    is bad, a scheduler that stops running because of a stuck lock is worse.
    """
    path = _argent_dir() / "automations.lock"
    held = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _DEFS_LOCK_WAIT_SECONDS
        while True:
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                held = True
                break
            except FileExistsError:
                try:
                    if time.time() - path.stat().st_mtime > _DEFS_LOCK_STALE_SECONDS:
                        path.unlink()
                        continue
                except OSError:
                    continue
                if time.monotonic() > deadline:
                    log.warning("automation definitions lock busy; continuing without it")
                    break
                time.sleep(0.02)
            except OSError:
                break
        yield
    finally:
        if held:
            try:
                path.unlink()
            except OSError:
                pass


# --- definitions -------------------------------------------------------------------

def load_automations() -> list:
    """All defined automations. A corrupt file yields [] rather than crashing
    the server it is loaded into."""
    path = defs_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("automations file unreadable (%s); treating as empty", e)
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        try:
            known = {k: v for k, v in item.items() if k in Automation.__dataclass_fields__}
            out.append(Automation(**known))
        except Exception as e:
            log.warning("skipping malformed automation %r: %s", item, e)
    return out


def save_automations(automations: list) -> None:
    from atomic_io import atomic_write_text
    payload = json.dumps([asdict(a) for a in automations], indent=2, ensure_ascii=False)
    atomic_write_text(defs_path(), payload)


def get_automation(name: str):
    for a in load_automations():
        if a.name == name:
            return a
    return None


def upsert_automation(automation: Automation) -> None:
    with _defs_lock():
        items = [a for a in load_automations() if a.name != automation.name]
        if not automation.created:
            automation.created = datetime.now().isoformat(timespec="seconds")
        items.append(automation)
        save_automations(items)


def update_automation(name: str, mutate) -> "Automation | None":
    """Apply ``mutate(automation)`` to the stored definition under the lock.

    Returns the updated definition, or None when there is no such automation.
    The mutation sees the CURRENT file contents, not the caller's older copy.
    """
    with _defs_lock():
        items = load_automations()
        for item in items:
            if item.name == name:
                mutate(item)
                save_automations(items)
                return item
    return None


def remove_automation(name: str) -> bool:
    with _defs_lock():
        items = load_automations()
        kept = [a for a in items if a.name != name]
        if len(kept) == len(items):
            return False
        save_automations(kept)
        return True


def find_automation(query: str):
    """An automation by exact name, else by a unique case-insensitive name or
    prefix. Returns (automation | None, [candidates]) — several candidates means
    the query was ambiguous and the caller should say so."""
    items = load_automations()
    for a in items:
        if a.name == query:
            return a, []
    folded = query.casefold()
    exact = [a for a in items if a.name.casefold() == folded]
    if len(exact) == 1:
        return exact[0], []
    starts = [a for a in items if a.name.casefold().startswith(folded)]
    if len(starts) == 1:
        return starts[0], []
    return None, [a.name for a in (exact or starts)]


# --- run history -------------------------------------------------------------------

def _trim_history(path: Path) -> None:
    """Keep the log bounded: MAX_RUN_HISTORY only ever limited what is *read*."""
    try:
        if path.stat().st_size < _TRIM_AT_BYTES:
            return
        from atomic_io import atomic_write_text
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        atomic_write_text(path, "\n".join(lines[-MAX_RUN_HISTORY:]) + "\n")
    except Exception as e:
        log.info("could not trim the automation run log: %s", e)


def record_run(name: str, status: str, summary: str, started: datetime,
               finished: datetime, denied_actions: list = None,
               new_items: int = 0, *, trigger: str = "schedule", alert: bool = None,
               silent: bool = False, digest: str = None, report: str = "",
               commands: list = None) -> None:
    """Append one run to the history and stamp the definition.

    The history is what you read in the morning to find out what the agent did
    while you weren't looking, so a denied action is recorded as loudly as a
    successful one.

    ``alert`` says whether this run earned your attention (it feeds the inbox).
    When the caller does not know, anything but a plain ok counts.
    """
    if alert is None:
        alert = status != "ok" or bool(new_items) or bool(denied_actions)
    entry = {
        "name": name,
        "status": status,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "seconds": round((finished - started).total_seconds(), 1),
        "summary": (summary or "").strip()[:SUMMARY_LIMIT],
        "denied_actions": denied_actions or [],
        "new_items": int(new_items or 0),
        "trigger": trigger,
        "alert": bool(alert),
        "silent": bool(silent),
    }
    if report:
        entry["report"] = report
    if commands:
        entry["commands"] = list(commands)[:20]
    try:
        path = runs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _trim_history(path)
    except Exception as e:
        log.warning("could not append automation run log: %s", e)

    def _stamp(a: Automation) -> None:
        from src.automation.schedule import is_one_shot
        a.last_run = finished.isoformat(timespec="seconds")
        a.last_status = status
        a.run_count = int(a.run_count or 0) + 1
        if digest is not None:
            a.last_digest = digest
        if status == "error":
            a.fail_streak = int(a.fail_streak or 0) + 1
            # A human asking for a run is watching it; only unattended failures
            # can switch a job off.
            if (trigger != "manual" and a.max_failures and a.fail_streak >= a.max_failures
                    and a.enabled):
                a.enabled = False
                a.disabled_reason = (f"{a.fail_streak} сбоев подряд — приостановлена "
                                     f"(исправьте причину и включите: /tasks on {a.name})")
                log.warning("automation %r paused after %d consecutive failures", a.name, a.fail_streak)
        else:
            a.fail_streak = 0
        if trigger != "manual" and is_one_shot(a.schedule) and a.enabled:
            # A one-shot that failed keeps its retries; every other outcome ends it.
            if status != "error" or int(a.fail_streak or 0) > int(a.retries or 0):
                a.enabled = False
                a.disabled_reason = "выполнена (разовая)"

    update_automation(name, _stamp)


# --- run locks ---------------------------------------------------------------
# Argent can be open twice at once — a terminal and the TUI — and each runs its
# own scheduler. "This automation is already running" lives in one process's
# memory, so both would fire the same due job and both would toast about it. A
# file created with O_EXCL is the one thing two processes can agree on without
# a server.

# A lock this old belongs to a run that cannot still be going (the runner kills
# a run long before this), so it is a crash's leftover, not a live run.
RUN_LOCK_STALE_SECONDS = 2 * 60 * 60


def _lock_path(name: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", name)[:60]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return _argent_dir() / "automation_locks" / f"{safe}-{digest}.lock"


def _pid_alive(pid: int) -> bool:
    """Whether a process exists. Unknown counts as alive: wrongly freeing a lock
    double-runs a job, wrongly keeping one only waits for the age limit.

    Not os.kill(pid, 0): on Windows signal 0 is CTRL_C_EVENT and would be
    delivered to the target's console."""
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True


def _lock_is_stale(path: Path, now: datetime) -> bool:
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        pid = int(info["pid"])
        since = datetime.fromisoformat(info["since"])
    except Exception:
        # Torn or foreign content: fall back on the file's own age.
        try:
            age = now.timestamp() - path.stat().st_mtime
        except OSError:
            return True
        return age > RUN_LOCK_STALE_SECONDS
    if not _pid_alive(pid):
        return True
    return (now - since).total_seconds() > RUN_LOCK_STALE_SECONDS


def claim_run(name: str, now: datetime = None) -> bool:
    """Take the right to run `name` right now. False: somebody else holds it."""
    now = now or datetime.now()
    path = _lock_path(name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("cannot create the automation lock directory: %s", e)
        return False
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _lock_is_stale(path, now):
                return False
            try:
                path.unlink()
            except OSError:
                return False
            continue
        except OSError as e:
            log.warning("cannot lock automation %r: %s", name, e)
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "since": now.isoformat(), "name": name}, f)
        return True
    return False


def release_run(name: str) -> None:
    """Give the lock back — only if it is still ours."""
    path = _lock_path(name)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if int(info.get("pid", -1)) != os.getpid():
            return
    except FileNotFoundError:
        return
    except Exception:
        pass            # unreadable: it was ours to create, so remove it
    try:
        path.unlink()
    except OSError:
        pass


def running_elsewhere() -> list:
    """Names of automations whose run lock is held right now (any process)."""
    folder = _argent_dir() / "automation_locks"
    if not folder.is_dir():
        return []
    now = datetime.now()
    names = []
    for lock in folder.glob("*.lock"):
        if _lock_is_stale(lock, now):
            continue
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            names.append((info.get("since", ""), info.get("name") or lock.stem))
        except Exception:
            continue
    return [stem for _, stem in sorted(names)]


def load_runs(limit: int = 20, name: str = None) -> list:
    """Most recent runs, newest last. Tolerates a torn final line."""
    path = runs_path()
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue                     # partially written tail
            if name is None or entry.get("name") == name:
                out.append(entry)
    except Exception as e:
        log.warning("could not read automation run log: %s", e)
        return []
    return out[-limit:] if limit else out


# --- global state: pause, and what you have already read -----------------------------

def load_state() -> dict:
    path = state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("automation state unreadable (%s); starting empty", e)
        return {}
    return data if isinstance(data, dict) else {}


def save_state(**changes) -> dict:
    """Merge ``changes`` into the state file (None removes a key)."""
    from atomic_io import atomic_write_text
    with _defs_lock():
        state = load_state()
        for key, value in changes.items():
            if value is None:
                state.pop(key, None)
            else:
                state[key] = value
        try:
            atomic_write_text(state_path(), json.dumps(state, indent=2, ensure_ascii=False))
        except Exception as e:
            log.warning("could not save automation state: %s", e)
    return state


def pause_until(now: datetime = None):
    """When the global pause ends: a datetime, datetime.max for "until resumed",
    or None when not paused. An expired pause reads as not paused."""
    state = load_state()
    if not state.get("paused"):
        return None
    until = state.get("paused_until")
    if not until:
        return datetime.max
    try:
        moment = datetime.fromisoformat(until)
    except ValueError:
        return datetime.max
    return moment if moment > (now or datetime.now()) else None


def set_paused(until: datetime = None) -> None:
    save_state(paused=True, paused_until=until.isoformat(timespec="seconds") if until else None)


def resume() -> None:
    save_state(paused=None, paused_until=None)


def last_seen() -> datetime | None:
    try:
        return datetime.fromisoformat(load_state().get("last_seen", ""))
    except (TypeError, ValueError):
        return None


def mark_seen(moment: datetime = None) -> None:
    save_state(last_seen=(moment or datetime.now()).isoformat(timespec="seconds"))


def unseen_runs(since: datetime = None) -> list:
    """Runs that finished after you last looked and earned your attention."""
    since = since or last_seen()
    if since is None:
        return []
    out = []
    for entry in load_runs(limit=0):
        try:
            finished = datetime.fromisoformat(entry.get("finished", ""))
        except ValueError:
            continue
        if finished > since and entry.get("alert") and entry.get("trigger") != "manual":
            out.append(entry)
    return out
