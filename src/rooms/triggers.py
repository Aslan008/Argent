"""What starts a room by itself: a schedule or a webhook, in one place.

A schedule is an ordinary automation of mode `room` (so the scheduler, the OS tick, retries, pause,
notifications and the report file all work as they do for any automation); a webhook is an entry of
webhooks.py. This module is the one door the editor and `/rooms triggers` use, so neither has to know
where each kind lives.
"""

import re
from datetime import datetime

from src.rooms import webhooks

_NAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _quote(value) -> str:
    text = str(value)
    if '"' in text or "\n" in text:
        raise ValueError(f"a parameter value cannot hold quotes or line breaks: {text[:40]!r}")
    return f'"{text}"' if (not text or any(c.isspace() for c in text)) else text


def task_text(room: str, task: str = "", params: dict = None) -> str:
    """The `task` of a room automation: `room key=value "key=two words" free task text`."""
    pieces = [room]
    for key, value in (params or {}).items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
            raise ValueError(f"'{key}' is not a usable parameter name")
        pieces.append(f"{key}={_quote(value)}")
    if task and task.strip():
        pieces.append(task.strip())
    return " ".join(pieces)


def _room_of(automation) -> str:
    from src.rooms.headless import split_task
    return split_task(automation.task)[0]


def list_triggers(room: str = None) -> list:
    """Every schedule and webhook, newest kinds together: dicts with `kind` ("schedule" | "webhook")."""
    from src.automation.store import load_automations
    out = []
    for a in load_automations():
        if a.mode != "room":
            continue
        name_room = _room_of(a)
        if room is not None and name_room != room:
            continue
        from src.rooms.headless import split_task
        _, text, params = split_task(a.task)
        out.append({"kind": "schedule", "name": a.name, "room": name_room, "schedule": a.schedule,
                    "enabled": bool(a.enabled), "task": text, "params": params,
                    "last_run": a.last_run, "last_status": a.last_status, "notify": a.notify,
                    "channels": list(a.channels or []), "paused_reason": a.disabled_reason})
    for h in webhooks.list_hooks(room):
        out.append({"kind": "webhook", "name": h["name"], "room": h["room"], "enabled": h["enabled"],
                    "url": webhooks.url_for(h["name"]), "params": h["params"], "task": h["task"],
                    "last_run": h["last_run"], "last_status": h["last_status"]})
    return out


def add_schedule(room: str, schedule: str, task: str = "", params: dict = None, name: str = None,
                 notify: str = "auto", channels=()) -> dict:
    """Make `room` run on `schedule` ("every day at 03:00", "каждый час", "on start", ...). Raises
    ValueError with the problems when the definition is not usable."""
    from src.automation.schedule import ScheduleError, normalize_schedule
    from src.automation.store import Automation, get_automation, upsert_automation, validate_automation
    try:
        canonical = normalize_schedule(schedule, datetime.now())
    except ScheduleError as exc:
        raise ValueError(str(exc))
    base = name or f"room:{_NAME_SAFE.sub('_', room)}"
    chosen, n = base, 1
    while get_automation(chosen) is not None:
        n += 1
        chosen = f"{base}:{n}"
    automation = Automation(name=chosen, task=task_text(room, task, params), schedule=canonical, mode="room",
                            notify=notify, channels=list(channels or []),
                            created=datetime.now().isoformat(timespec="seconds"))
    problems = validate_automation(automation)
    if problems:
        raise ValueError("; ".join(problems))
    upsert_automation(automation)
    return next(t for t in list_triggers(room) if t["kind"] == "schedule" and t["name"] == chosen)


def add_webhook(room: str, name: str = None, params=None, task: str = "") -> tuple:
    """Make `room` start on a POST to a URL. Returns (trigger, secret); the secret is not kept."""
    hook, secret = webhooks.add_hook(room, name or _NAME_SAFE.sub("_", room), params, task)
    trigger = next(t for t in list_triggers(room) if t["kind"] == "webhook" and t["name"] == hook["name"])
    return trigger, secret


def remove(kind: str, name: str) -> bool:
    if kind == "schedule":
        from src.automation.store import get_automation, remove_automation
        found = get_automation(name)
        if found is None or found.mode != "room":
            return False
        return bool(remove_automation(name))
    if kind == "webhook":
        return webhooks.remove_hook(name)
    raise ValueError("kind must be 'schedule' or 'webhook'")


def set_enabled(kind: str, name: str, enabled: bool) -> bool:
    if kind == "schedule":
        from src.automation.store import get_automation, update_automation

        def mutate(a):
            a.enabled = bool(enabled)
            if enabled:
                a.fail_streak = 0
                a.disabled_reason = ""
        found = get_automation(name)
        if found is None or found.mode != "room":
            return False
        return update_automation(name, mutate) is not None
    if kind == "webhook":
        return webhooks.set_enabled(name, enabled)
    raise ValueError("kind must be 'schedule' or 'webhook'")


def describe(trigger: dict) -> str:
    """One readable line for a list."""
    state = "" if trigger.get("enabled") else " [выключен]"
    if trigger["kind"] == "schedule":
        return f"расписание «{trigger['name']}»: {trigger['schedule']}{state}"
    return f"webhook «{trigger['name']}»: POST {trigger['url']}{state}"
