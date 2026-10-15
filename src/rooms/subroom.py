"""Sub-rooms and notifications: the two node types that reach outside one room.

A `room` node runs another room as a sub-routine — its `inputs` become the child's
state, its `outputs` are copied back, its exit name lands in a field the parent's
edges can branch on. That is what makes rooms modular: a verified "run the tests and
fix them" room is written once and used inside any recipe.

Bounds: a call depth ceiling, no room may (directly or through others) call itself
(`reference_errors` in library.py rejects it at load), and the tokens a child spends
count against the parent's room budget. A child is not journaled on its own — if the
process dies inside one, the parent's `room` node is the interrupted node and the
run goes to the person rather than re-running something half done.
"""

import threading

from src.rooms import template
from src.rooms.engine import Engine
from src.rooms.models import State
from src.rooms.supervisor import FAILURE_EXITS

MAX_DEPTH = 3


class SubRoomHandler:
    def __init__(self, library, max_depth: int = MAX_DEPTH):
        self.library = library
        self.max_depth = max_depth
        # per thread: parallel branches each call sub-rooms, and their calls must not add up as nesting
        self._local = threading.local()

    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @_depth.setter
    def _depth(self, value: int) -> None:
        self._local.depth = value

    def __call__(self, node, state, run_node, on_event=None) -> None:
        child = self.library.get(node.room)
        if child is None:
            raise RuntimeError(f"room node '{node.id}': there is no room '{node.room}'")
        if self._depth >= self.max_depth:
            raise RuntimeError(f"room node '{node.id}': sub-rooms are nested deeper than {self.max_depth}")

        data = dict(child.defaults or {})
        # `inputs` were rendered against the parent's state by the runner (see _materialize)
        data.update(node.inputs or {})
        data.setdefault("task", state.data.get("task", ""))
        child_state = State(data=data, rules=dict(state.rules))     # the caller's rules bind the called room

        self._depth += 1
        try:
            outcome = Engine().run(child, child_state, run_node, on_event=on_event)
        finally:
            self._depth -= 1

        for field in node.outputs:
            if field in child_state.data:
                state.data[field] = child_state.data[field]
        slot = node.into or f"{node.id}_exit"
        state.data[slot] = outcome.exit
        report = template.render(child.report, child_state.data, limit=1500).strip() if child.report else ""
        state.data[f"{node.id}.output"] = report
        state.usage["tokens"] = state.usage.get("tokens", 0) + child_state.usage.get("tokens", 0)
        try:
            self.library.record_run(child.room, success=outcome.exit not in FAILURE_EXITS)
        except Exception:
            pass


def default_notify(node, state) -> bool:
    """Production delivery for a `notify` node. Best-effort: a toast that cannot be shown must not
    fail a run. Returns whether it was delivered (the run records it as `<id>.delivered`)."""
    title = node.title or "Argent · комната"
    message = node.message or ""
    try:
        if node.channel == "desktop":
            from src.automation.notify import desktop_notify
            return bool(desktop_notify(title, message))
        if node.channel == "telegram":
            from src.automation.notify import telegram_notify
            return bool(telegram_notify(title, message))
        from ui import print_system
        print_system(f"[bold]{title}[/bold]: {message}")
        return True
    except Exception:
        return False
