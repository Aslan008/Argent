"""What the host says in its own window about automations.

Both hosts (the TUI bridge and the classic REPL) need the same two things: a line
when a scheduled run finished and earned attention, and a "while you were away"
summary at start-up. Kept here, pure and testable, so the two cannot drift apart —
and so the rule stays the one notify.py states: speak only when there is
something to say.
"""

from collections import deque
from datetime import datetime

from src.automation import store
from src.automation.schedule import humanize_when

_NOTICE_LIMIT = 600


def _one_line(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text[:limit - 1] + "…" if len(text) > limit else text


def format_event_notice(event: dict):
    """The line to show for a scheduler event, or None to stay quiet.

    A manual run is always answered — you asked, and the result belongs in the
    window you asked from. A scheduled run speaks only if it earned an alert.
    """
    if event.get("event") != "finished":
        return None
    name = event.get("name", "?")
    status = event.get("status", "ok")
    summary = (event.get("summary") or "").strip()
    manual = event.get("trigger") == "manual"

    alert = event.get("alert")
    if alert is None:
        # An event from a runner that predates the flag.
        alert = status != "ok" or bool(event.get("denied")) or bool(event.get("new_items"))
    if not alert and not manual:
        return None

    if event.get("mode") == "remind" and status == "ok":
        return f"⏰ Напоминание «{name}»: {_one_line(summary, _NOTICE_LIMIT)}"

    icon = {"ok": "⏰", "error": "⚠", "denied": "🔒"}.get(status, "⏰")
    head = f"{icon} Автоматизация «{name}»"
    if manual:
        head += " выполнена" if status == "ok" else f": {status}"
    else:
        head += f": {status}"
    extra = []
    if event.get("new_items"):
        extra.append(f"нового: {event['new_items']}")
    if event.get("denied"):
        extra.append(f"отклонено действий: {event['denied']}")
    if extra:
        head += " — " + ", ".join(extra)
    body = _one_line(summary, _NOTICE_LIMIT)
    tail = "" if manual else " (подробности: /tasks inbox)"
    return f"{head}{(chr(10) + body) if body else ''}{tail}"


# The classic REPL cannot print from the scheduler's thread without scribbling
# over the line you are typing, so it parks the lines here and shows them before
# the next prompt is drawn.
_pending = deque(maxlen=50)


def enqueue_event(event: dict) -> None:
    text = format_event_notice(event)
    if text:
        _pending.append(text)


def drain_pending() -> list:
    out = []
    while _pending:
        out.append(_pending.popleft())
    return out


def startup_digest(now: datetime = None, limit_names: int = 4):
    """A short "while you were away" message, or None when there is nothing.

    Reads what runs recorded since you last opened `/tasks inbox` — runs by the
    scheduler in another window, or by the OS scheduler while Argent was closed.
    The first time ever there is no backlog to report: it starts the clock.
    """
    now = now or datetime.now()
    lines = []

    paused = store.pause_until(now)
    if paused is not None:
        until = "пока не включите: /tasks resume" if paused == datetime.max else (
            f"до {humanize_when(paused, now)}")
        lines.append(f"⏸ Автоматизации на паузе {until}")

    since = store.last_seen()
    if since is None:
        store.mark_seen(now)
    else:
        unseen = store.unseen_runs(since)
        if unseen:
            failed = sum(1 for r in unseen if r.get("status") == "error")
            blocked = sum(1 for r in unseen if r.get("status") == "denied")
            news = sum(int(r.get("new_items") or 0) for r in unseen)
            names = []
            for entry in unseen:
                if entry["name"] not in names:
                    names.append(entry["name"])
            parts = [f"прогонов: {len(unseen)}"]
            if news:
                parts.append(f"нового: {news}")
            if failed:
                parts.append(f"сбоев: {failed}")
            if blocked:
                parts.append(f"отказов без человека: {blocked}")
            shown = ", ".join(names[:limit_names]) + (" и др." if len(names) > limit_names else "")
            lines.append(f"⏰ Пока вас не было — {'; '.join(parts)} ({shown}). Смотреть: /tasks inbox")

    return "\n".join(lines) if lines else None
