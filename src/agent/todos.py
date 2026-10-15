"""The checklist of the task in hand.

A model working through a long job loses the thread: it forgets which steps are
done, repeats one, or declares victory with half the list untouched. A weak model
does it sooner. Writing the steps down — and having the *person* see them tick off
— fixes the model's memory and the person's blindness in one move.

The list is the model's own working notes, not authority: it grants nothing, and
rewriting it can never hurt anything. So it is stored per *scope* (the main agent,
or one sub-agent) in memory only. It dies with the conversation (`/clear`, a loaded
session) and is shown by the frontend next to the prompt.
"""

import contextvars
import json
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

PENDING, IN_PROGRESS, COMPLETED = "pending", "in_progress", "completed"
STATUSES = (PENDING, IN_PROGRESS, COMPLETED)

#: Bigger lists stop being a plan and become a transcript of the model's worries.
MAX_ITEMS = 30
MAX_ITEM_CHARS = 200

_MARKS = {PENDING: "[ ]", IN_PROGRESS: "[~]", COMPLETED: "[x]"}

# Which list a tool call belongs to. The main agent is "main"; a sub-agent that runs
# in-process takes its own scope for the duration of its work so it cannot overwrite
# the supervisor's list.
_scope = contextvars.ContextVar("argent_todo_scope", default="main")


@dataclass(frozen=True)
class Todo:
    content: str
    status: str = PENDING

    def as_dict(self) -> dict:
        return {"content": self.content, "status": self.status}


class TodoError(ValueError):
    """The list the model sent cannot be understood."""


_ALIASES = {
    "todo": PENDING, "open": PENDING, "not_started": PENDING, "not started": PENDING,
    "in progress": IN_PROGRESS, "inprogress": IN_PROGRESS, "doing": IN_PROGRESS,
    "active": IN_PROGRESS, "current": IN_PROGRESS, "started": IN_PROGRESS,
    "done": COMPLETED, "complete": COMPLETED, "finished": COMPLETED, "ok": COMPLETED,
    "x": COMPLETED, "[x]": COMPLETED, "[ ]": PENDING, "[~]": IN_PROGRESS,
}


def _status(value) -> str:
    text = str(value or PENDING).strip().lower()
    text = _ALIASES.get(text, text)
    if text not in STATUSES:
        raise TodoError(f"unknown status '{value}' (use pending, in_progress or completed)")
    return text


def parse_items(raw) -> List[Todo]:
    """A list from what a model sends: a JSON string, or a list of strings / objects."""
    if raw is None:
        raise TodoError("`todos` is required: a list of {content, status} items")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except ValueError:
            # A plain bulleted list is a reasonable thing for a small model to send.
            raw = [line.strip(" -*\t") for line in text.splitlines() if line.strip(" -*\t")]
    if isinstance(raw, dict):
        raw = raw.get("todos", [raw])
    if not isinstance(raw, (list, tuple)):
        raise TodoError("`todos` must be a list")
    if len(raw) > MAX_ITEMS:
        raise TodoError(f"at most {MAX_ITEMS} items — group the small steps")
    items = []
    for entry in raw:
        if isinstance(entry, str):
            content, status = entry, PENDING
        elif isinstance(entry, dict):
            content = entry.get("content") or entry.get("task") or entry.get("title") or entry.get("text") or ""
            status = entry.get("status", PENDING)
            if entry.get("done") is True or entry.get("completed") is True:
                status = COMPLETED
        else:
            raise TodoError(f"cannot read item {entry!r}")
        content = " ".join(str(content).split())
        if not content:
            raise TodoError("an item has no `content`")
        items.append(Todo(content[:MAX_ITEM_CHARS], _status(status)))
    return items


class TodoStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._lists = {}
        self._listeners: list = []

    def subscribe(self, listener) -> None:
        """`listener(scope, items)` is called whenever a list changes (a frontend that mirrors the checklist
        without polling: the Telegram panel). It must be quick and must not raise."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _changed(self, scope: str, items: List[Todo]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(scope, list(items))
            except Exception:
                pass

    @staticmethod
    def scope() -> str:
        return _scope.get()

    def get(self, scope: Optional[str] = None) -> List[Todo]:
        with self._lock:
            return list(self._lists.get(scope or self.scope(), ()))

    def set(self, items: Iterable[Todo], scope: Optional[str] = None) -> List[Todo]:
        items = list(items)
        scope = scope or self.scope()
        with self._lock:
            if items:
                self._lists[scope] = items
            else:
                self._lists.pop(scope, None)
        self._changed(scope, items)
        return items

    def clear(self, scope: Optional[str] = None) -> None:
        with self._lock:
            if scope is None:
                gone = list(self._lists)
                self._lists.clear()
            else:
                gone = [scope] if scope in self._lists else []
                self._lists.pop(scope, None)
        for name in gone:
            self._changed(name, [])

    def begin_turn(self, scope: Optional[str] = None) -> None:
        """A list that was finished last turn is history by the time a new request arrives."""
        items = self.get(scope)
        if items and all(t.status == COMPLETED for t in items):
            self.clear(scope or self.scope())

    def snapshot(self, scope: str = "main") -> List[dict]:
        return [t.as_dict() for t in self.get(scope)]


TODOS = TodoStore()


def restore(snapshot, scope: str = "main") -> None:
    """Put back a list saved with the session (`TODOS.snapshot`); anything unreadable leaves the list empty."""
    TODOS.clear(scope)
    if not snapshot:
        return
    try:
        TODOS.set(parse_items(snapshot), scope)
    except TodoError:
        pass


def use_scope(name: str):
    """Context manager: tool calls inside belong to `name`'s list."""
    class _Scope:
        def __enter__(self):
            self._token = _scope.set(name)
            return name

        def __exit__(self, *exc):
            _scope.reset(self._token)
            TODOS.clear(name)
            return False
    return _Scope()


def render(items: Iterable[Todo]) -> str:
    return "\n".join(f"{_MARKS[t.status]} {t.content}" for t in items)


def progress(items: Iterable[Todo]) -> str:
    items = list(items)
    done = sum(1 for t in items if t.status == COMPLETED)
    return f"{done}/{len(items)}"


def write(raw) -> str:
    """The `todo_write` tool body: replace the list, answer with what the model should do next."""
    try:
        items = parse_items(raw)
    except TodoError as e:
        return f"Error: {e}"
    TODOS.set(items)
    if not items:
        return "Todo list cleared."
    active = [t for t in items if t.status == IN_PROGRESS]
    lines = [f"Todo list updated ({progress(items)} done):", render(items)]
    if all(t.status == COMPLETED for t in items):
        lines.append("Every item is completed. Summarise the result for the user.")
    elif not active:
        lines.append("Nothing is marked in_progress — mark the step you are starting NOW as in_progress.")
    elif len(active) > 1:
        lines.append("More than one item is in_progress. Work on one at a time: "
                     "keep a single step in_progress and mark the rest pending.")
    else:
        lines.append(f"Current step: {active[0].content}. Mark it completed the moment it is done, "
                     "before starting the next one.")
    return "\n".join(lines)


def ephemeral_block(scope: str = "main") -> Optional[str]:
    """The list as the model should see it at the tail of its next request."""
    items = TODOS.get(scope)
    if not items:
        return None
    return ("## YOUR TODO LIST\n" + render(items) +
            "\n(Keep it current with todo_write: one item in_progress, mark each completed as soon as it is done.)")
