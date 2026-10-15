"""The run on the person's screen.

An autonomous run that shows nothing for ten minutes gets interrupted, so the
engine's events are drawn two ways: a line per node in the console, and the
current room's nodes as the checklist the frontends already render above the
prompt (the TUI panel, the Telegram status). The checklist the person had before
is put back when the run ends.
"""

from typing import Callable, Optional

_KIND_MARK = {"tool": "⚙", "agent": "🤖", "condition": "◇", "human_pause": "✋", "spawn_room": "✚",
              "mcp": "🔌", "skill": "📘", "plugin": "🧩", "leeth": "⚡", "foreach": "↻", "wait": "⏱",
              "loop": "⟳", "break": "⤓", "continue": "⤒", "power": "⏻", "hint": "💡", "forget": "🧹", "assert": "❗", "parallel": "⑂", "join": "⑃"}


class RoomBoard:
    def __init__(self, library, printer: Optional[Callable[[str], None]] = None, todos=None, scope: str = "main"):
        self._library = library
        self._print = printer
        self._todos = todos
        self._scope = scope
        self._before = None
        self._room = None
        self._status = {}
        self._opened = False

    # -- plumbing ---------------------------------------------------------------------------------
    def _say(self, text: str) -> None:
        printer = self._print
        if printer is None:
            try:
                from ui import print_system
                printer = print_system
            except Exception:
                return
        try:
            printer(text)
        except Exception:
            pass

    def _store(self):
        if self._todos is None:
            try:
                from src.agent.todos import TODOS
                self._todos = TODOS
            except Exception:
                return None
        return self._todos

    def _open(self) -> None:
        store = self._store()
        if store is not None and not self._opened:
            self._opened = True
            try:
                self._before = store.get(self._scope)
            except Exception:
                self._before = None

    def _redraw(self) -> None:
        store = self._store()
        room = self._library.get(self._room) if self._room else None
        if store is None or room is None:
            return
        try:
            from src.agent.todos import COMPLETED, IN_PROGRESS, PENDING, Todo
            items = []
            for n in room.nodes:
                st = self._status.get(n.id, PENDING)
                items.append(Todo(f"{self._room} › {n.id}"[:200], {"done": COMPLETED, "run": IN_PROGRESS}.get(st, PENDING)))
            store.set(items, self._scope)
        except Exception:
            pass

    # -- engine events ----------------------------------------------------------------------------
    def on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "node_start":
            self._open()
            if event.get("room") != self._room:
                self._room = event.get("room")
                self._status = {}
            self._status[event["node"]] = "run"
            mark = _KIND_MARK.get(event.get("kind"), "·")
            self._say(f"[dim]{mark} {event.get('room')} › {event.get('node')}[/dim]")
            self._redraw()
        elif kind == "node_end":
            self._status[event["node"]] = "done"
            secs = event.get("seconds") or 0
            if secs >= 1:
                self._say(f"[dim]   ✓ {event.get('node')} — {secs:.0f} с[/dim]")
            self._redraw()
        elif kind == "room_end":
            why = f" ({event['reason']})" if event.get("reason") else ""
            self._say(f"[dim]■ {event.get('room')} → {event.get('exit')}{why}[/dim]")

    def close(self) -> None:
        """Put back the checklist the person had before the run."""
        store = self._store()
        if store is None or not self._opened:
            return
        try:
            if self._before:
                store.set(self._before, self._scope)
            else:
                store.clear(self._scope)
        except Exception:
            pass
