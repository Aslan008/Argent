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
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

PENDING, IN_PROGRESS, COMPLETED = "pending", "in_progress", "completed"
STATUSES = (PENDING, IN_PROGRESS, COMPLETED)

#: Bigger lists stop being a plan and become a transcript of the model's worries.
MAX_ITEMS = 30
MAX_ITEM_CHARS = 200

#: Turns a list may go untouched — with a step still in_progress — before the model
#: is reminded of it. A forgotten `in_progress` used to live until the end of the
#: session: the person kept reading "1/4" long after the work was finished, and the
#: model kept believing it was in the middle of something.
STALE_TURNS = 3

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
        #: When each in_progress item started, epoch seconds, per scope. The panel
        #: shows it, and a step that hangs is the one thing a person watching an
        #: unattended run needs to see.
        self._since = {}
        self._listeners: list = []
        # Turn counter and the turn each list was last written in: what makes
        # "nobody has moved this plan in a while" answerable at all. Counted per
        # scope, because a sub-agent's turns are not the supervisor standing still.
        self._turns: dict = {}
        self._changed_at_turn: dict = {}

    def turns_since_change(self, scope: Optional[str] = None) -> int:
        scope = scope or self.scope()
        with self._lock:
            turn = int(self._turns.get(scope, 0))
            return turn - int(self._changed_at_turn.get(scope, turn))

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
        now = time.time()
        with self._lock:
            self._changed_at_turn[scope] = self._turns.get(scope, 0)
            # A step still being worked on keeps its clock through both edits a model
            # makes to the list: the same wording keeps it wherever the item now sits
            # (a reorder), and a reworded item keeps the clock of its own slot. Without
            # the second rule, fixing a typo in the current step restarted the timer,
            # and a step twenty minutes old suddenly looked three seconds young.
            old = list(self._lists.get(scope, ()))
            old_since = dict(self._since.get(scope, {}))
            fresh: dict[int, float] = {}
            claimed: set = set()
            for i, t in enumerate(items):
                if t.status != IN_PROGRESS:
                    continue
                started = None
                for j, prev in enumerate(old):
                    if j not in claimed and prev.status == IN_PROGRESS and prev.content == t.content:
                        started = old_since.get(j)
                        claimed.add(j)
                        break
                if started is None and i < len(old) and i not in claimed:
                    if old[i].status == IN_PROGRESS:
                        started = old_since.get(i)
                        claimed.add(i)
                fresh[i] = now if started is None else started
            if items:
                self._lists[scope] = items
                self._since[scope] = fresh
            else:
                self._lists.pop(scope, None)
                self._since.pop(scope, None)
        self._changed(scope, items)
        return items

    def clear(self, scope: Optional[str] = None) -> None:
        with self._lock:
            if scope is None:
                gone = list(self._lists)
                self._lists.clear()
                self._since.clear()
                self._changed_at_turn.clear()
                self._turns.clear()
            else:
                gone = [scope] if scope in self._lists else []
                self._lists.pop(scope, None)
                self._since.pop(scope, None)
                self._changed_at_turn.pop(scope, None)
                self._turns.pop(scope, None)
        for name in gone:
            self._changed(name, [])

    def begin_turn(self, scope: Optional[str] = None) -> None:
        """A new turn starts: count it, and drop a list that was finished last time.

        The count is what makes a forgotten `in_progress` visible: without it a stale
        checklist simply sat there, and the person kept reading "1/4" long after the
        work was done.
        """
        with self._lock:
            scope = scope or self.scope()
            self._turns[scope] = self._turns.get(scope, 0) + 1
            self._changed_at_turn.setdefault(scope, self._turns[scope])
        items = self.get(scope)
        if items and all(t.status == COMPLETED for t in items):
            self.clear(scope or self.scope())

    def snapshot(self, scope: str = "main") -> List[dict]:
        items = self.get(scope)
        with self._lock:
            since = dict(self._since.get(scope, {}))
        out = []
        for i, t in enumerate(items):
            entry = t.as_dict()
            if t.status == IN_PROGRESS and i in since:
                entry["since"] = since[i]
            out.append(entry)
        return out

    def sub_scopes(self) -> List[str]:
        """Scopes other than the main list: the sub-agents working right now."""
        with self._lock:
            return [name for name in self._lists if name != "main"]

    def snapshot_subagents(self) -> dict:
        """The sub-agents' own checklists, by scope, for a frontend to show.

        Delegated work used to be invisible: the main list said "delegate" and what
        happened inside the sub-agent never reached the screen, which is exactly the
        part a person watching an unattended run wants to see.
        """
        return {name: self.snapshot(name) for name in self.sub_scopes()}

    def set_clocks(self, clocks: dict, scope: str = "main") -> None:
        """Give running steps back their start times (a restored session carries them)."""
        with self._lock:
            items = self._lists.get(scope, ())
            self._since[scope] = {
                int(i): float(t) for i, t in clocks.items()
                if 0 <= int(i) < len(items) and items[int(i)].status == IN_PROGRESS
            }


TODOS = TodoStore()


def restore(snapshot, scope: str = "main") -> None:
    """Put back a list saved with the session (`TODOS.snapshot`); anything unreadable leaves the list empty."""
    TODOS.clear(scope)
    if not snapshot:
        return
    try:
        items = parse_items(snapshot)
    except TodoError:
        return
    TODOS.set(items, scope)
    # A step that was already running keeps its clock: it is the same step, and
    # "3с" for a step that began before the restart would be a lie the timer tells.
    clocks = {}
    if isinstance(snapshot, (list, tuple)):
        for i, entry in enumerate(snapshot):
            if isinstance(entry, dict) and isinstance(entry.get("since"), (int, float)):
                clocks[i] = entry["since"]
    if clocks:
        TODOS.set_clocks(clocks, scope)


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


def render(items: Iterable[Todo], numbered: bool = False) -> str:
    """The list as text. Numbered, so a follow-up call can name one item."""
    lines = []
    for n, t in enumerate(items, 1):
        prefix = f"{n}. " if numbered else ""
        lines.append(f"{prefix}{_MARKS[t.status]} {t.content}")
    return "\n".join(lines)


def progress(items: Iterable[Todo]) -> str:
    items = list(items)
    done = sum(1 for t in items if t.status == COMPLETED)
    return f"{done}/{len(items)}"


def pending_note(items: Iterable[Todo]) -> str:
    """What a turn should not end silently on: steps the plan still calls undone.

    Left out, a half-finished list stays on screen and nobody — person or agent —
    can tell "almost done" from "forgotten". Empty string when nothing is left.
    """
    items = list(items)
    left = [t for t in items if t.status != COMPLETED]
    if not left:
        return ""
    lines = [f"Осталось {len(left)} из {len(items)}:"]
    for t in left[:6]:
        lines.append(f"  {_MARKS[t.status]} {t.content}")
    if len(left) > 6:
        lines.append(f"  … и ещё {len(left) - 6}")
    return "\n".join(lines)


def end_of_turn_note(items: Iterable[Todo], stopped: bool = False, changed: bool = True) -> str:
    """The line a turn ends with — or nothing when it should stay quiet.

    A stopped turn is another matter: there the interruption is the news, and
    listing what is left would blame the plan for the person's Ctrl+C.

    `changed` says whether the turn touched the list at all. A plan nobody moved is
    a plan, not a failure: repeating "Осталось 3 из 5" at the end of every turn — a
    question answered in passing, a look-up, a status check — turned the warning into
    wallpaper and taught the person to ignore it. A step left in_progress still
    speaks: the turn ended with work half-done, whether or not the list was rewritten.
    """
    if stopped:
        return ""
    items = list(items)
    if not changed and not any(t.status == IN_PROGRESS for t in items):
        return ""
    return pending_note(items)


def _report(items: List[Todo]) -> str:
    """What the model gets back: the list with indices, plus the one nudge it needs."""
    if not items:
        return "Todo list cleared."
    active = [t for t in items if t.status == IN_PROGRESS]
    lines = [f"Todo list updated ({progress(items)} done):", render(items, numbered=True)]
    if all(t.status == COMPLETED for t in items):
        lines.append("Every item is completed. Summarise the result for the user.")
    elif not active:
        lines.append("Nothing is marked in_progress — mark the step you are starting NOW as in_progress "
                     "(todo_write with index=<its number>, status='in_progress').")
    elif len(active) > 1:
        lines.append("More than one item is in_progress. Work on one at a time: "
                     "keep a single step in_progress and mark the rest pending.")
    else:
        lines.append(f"Current step: {active[0].content}. Mark it completed the moment it is done, "
                     "before starting the next one.")
    return "\n".join(lines)


def write(raw=None, index=None, status=None, content=None) -> str:
    """The `todo_write` tool body.

    Two shapes for one tool:

    * `raw` — the whole list, replacing whatever was there (the plan at the start);
    * `index` with `status` and/or `content` — one item changed in place.

    The second shape exists because the first one was used far more often than it
    should be: ticking step 5 of six meant resending all six, which costs tokens
    and can silently reword an item nobody meant to touch.
    """
    if raw is not None:
        try:
            items = parse_items(raw)
        except TodoError as e:
            return f"Error: {e}"
        TODOS.set(items)
        return _report(items)

    items = TODOS.get()
    if index is None:
        if not items:
            return "Error: `todos` is required: a list of {content, status} items"
        return ("Error: send `todos` (the whole list) or `index` with `status` and/or `content` "
                "to change one item in place.")
    if not items:
        return "Error: the list is empty — send `todos` to write the steps first."
    try:
        i = int(index)
    except (TypeError, ValueError):
        return f"Error: `index` must be a number from 1 to {len(items)}."
    if not 1 <= i <= len(items):
        return f"Error: `index` must be from 1 to {len(items)}.\n" + render(items, numbered=True)
    if status is None and content is None:
        return "Error: give `status` and/or `content` to change that item."

    cur = items[i - 1]
    new_status = cur.status
    if status is not None:
        try:
            new_status = _status(status)
        except TodoError as e:
            return f"Error: {e}"
    new_content = cur.content
    if content is not None:
        text = " ".join(str(content).split())[:MAX_ITEM_CHARS]
        if not text:
            return "Error: `content` cannot be empty."
        new_content = text

    items[i - 1] = Todo(new_content, new_status)
    TODOS.set(items)
    return _report(items)


def ephemeral_block(scope: str = "main") -> Optional[str]:
    """The list as the model should see it at the tail of its next request."""
    items = TODOS.get(scope)
    if not items:
        return None
    block = ("## YOUR TODO LIST\n" + render(items, numbered=True) +
             "\n(Keep it current with todo_write: one item in_progress, mark each completed as soon as it is done. "
             "To change one item, pass index=<its number> with status/content instead of resending the whole list.)")
    # A list nobody has moved in a while, with a step still "in progress", is the
    # shape a forgotten checklist takes. Said once, quietly, where the model looks:
    # a list that lies is worse than no list at all.
    stale = TODOS.turns_since_change(scope)
    if stale >= STALE_TURNS and any(t.status == IN_PROGRESS for t in items):
        block += (f"\n(Этот список не менялся {stale} ход(ов), а шаг всё ещё in_progress. "
                  f"Либо отметь сделанное, либо сними отметку — список, который врёт, хуже пустого.)")
    return block
