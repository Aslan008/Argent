"""Automation definitions and run history on disk.

Definitions live in <project>/.argent/automations.json, anchored to the project
root (not the CWD) like the rest of Argent's per-project state, so starting the
server from a subfolder doesn't silently produce a second, empty set.

Run history is a separate append-only JSONL: a definitions file that is
rewritten on every run would eventually lose definitions to a crash mid-write,
and the two have completely different write rates.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from logger import get_logger

log = get_logger("automation")

_DEFS_NAME = "automations.json"
_RUNS_NAME = "automation_runs.jsonl"

# Keep the tail readable and bounded; this is a log, not an archive.
MAX_RUN_HISTORY = 200


@dataclass
class Automation:
    """One scheduled task.

    ``allowed_tools`` is the capability boundary: an unattended task gets only
    the tools it needs, so a monitoring job cannot start editing files even if
    the model decides that would be helpful. Empty means "the default toolset",
    which is deliberately NOT the same as "everything is fine" — see runner.
    """
    name: str
    task: str
    schedule: str
    enabled: bool = True
    allowed_tools: list = field(default_factory=list)
    max_turns: int = 12
    last_run: str | None = None          # ISO timestamp
    last_status: str | None = None       # ok | error | denied | skipped

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
    items = [a for a in load_automations() if a.name != automation.name]
    items.append(automation)
    save_automations(items)


def remove_automation(name: str) -> bool:
    items = load_automations()
    kept = [a for a in items if a.name != name]
    if len(kept) == len(items):
        return False
    save_automations(kept)
    return True


def record_run(name: str, status: str, summary: str, started: datetime,
               finished: datetime, denied_actions: list = None) -> None:
    """Append one run to the history and stamp the definition.

    The history is what you read in the morning to find out what the agent did
    while you weren't looking, so a denied action is recorded as loudly as a
    successful one.
    """
    entry = {
        "name": name,
        "status": status,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "seconds": round((finished - started).total_seconds(), 1),
        "summary": (summary or "").strip()[:2000],
        "denied_actions": denied_actions or [],
    }
    try:
        path = runs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("could not append automation run log: %s", e)

    automation = get_automation(name)
    if automation is not None:
        automation.last_run = finished.isoformat(timespec="seconds")
        automation.last_status = status
        upsert_automation(automation)


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
    return out[-limit:]
