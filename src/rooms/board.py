"""The run on the person's screen.

An autonomous run that shows nothing for ten minutes gets interrupted, so the
engine's events are drawn two ways: a few lines per step in the console (the step, how
long it took and what it produced: a critic's remarks are read there, not found later in
a file), and the steps so far as the checklist the frontends already render above the
prompt (the TUI panel, the Telegram status). The checklist the person had before is put
back when the run ends.
"""

from typing import Callable, Optional

#: What a step is, in a word. Shown dimmed after its name.
_KIND_WORD = {
    "tool": "инструмент", "agent": "агент", "condition": "проверка", "human_pause": "вопрос вам",
    "spawn_room": "новая комната", "set": "запись", "notify": "уведомление", "room": "подкомната",
    "mcp": "MCP", "skill": "скилл", "plugin": "плагин", "leeth": "Leeth", "foreach": "цикл по списку",
    "wait": "пауза", "loop": "цикл", "break": "выход из цикла", "continue": "следующий круг",
    "power": "питание ПК", "hint": "подсказка", "forget": "забыть", "assert": "проверка",
    "parallel": "ветвление", "join": "слияние", "git": "git", "http": "запрос",
}

#: Steps whose result is worth reading on the screen (the others only move the run along).
_SHOWS_RESULT = {"agent", "tool", "mcp", "plugin", "leeth", "room", "human_pause", "http", "git", "skill"}

ACCENT = "#58a6ff"        # the room's name; the same blue as the interface's accent
MUTED = "#8b949e"
OK = "#3fb950"
BAD = "#f85149"

_LINE = 160               # a value longer than this is cut to this on its row
_ITEM = 420               # a list item (a finding is a sentence or two) may be longer
_ROWS = 10                # rows of one produced field
_FIELDS = 6               # produced fields shown for one step


def _esc(text) -> str:
    try:
        from rich.markup import escape
        return escape(str(text))
    except Exception:
        return str(text).replace("[", "\\[")


def _cut(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _value_lines(value) -> list:
    """A produced value as short rows: one row for a short value, a bullet per item for a list."""
    if isinstance(value, bool):
        return ["да" if value else "нет"]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        rows = []
        for item in value:
            if len(rows) >= _ROWS:
                rows.append(f"… ещё {len(value) - _ROWS}")
                break
            rows.append("• " + _cut(item if not isinstance(item, dict)
                                    else "; ".join(f"{k}: {v}" for k, v in item.items()), _ITEM))
        return rows
    if isinstance(value, dict):
        return [f"{k}: " + _cut(v, _LINE) for k, v in list(value.items())[:_ROWS]]
    text = str(value).strip()
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) <= 1:
        return [_cut(text, _ITEM)]
    shown = [_cut(r, _LINE) for r in rows[:_ROWS]]
    if len(rows) > _ROWS:
        shown.append(f"… ещё {len(rows) - _ROWS}")
    return shown


class RoomBoard:
    def __init__(self, library, printer: Optional[Callable[[str], None]] = None, todos=None, scope: str = "main"):
        self._library = library
        self._print = printer
        self._todos = todos
        self._scope = scope
        self._before = None
        self._opened = False
        self._main_room = None
        self._trail = []           # [[room, node, "run"|"done"]] in the order the steps began
        self._current = None

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

    def _entry(self, room, node):
        for item in self._trail:
            if item[0] == room and item[1] == node:
                return item
        return None

    def _redraw(self) -> None:
        """The checklist is the steps that have begun: a room is a graph, so "what is left" is not a
        list, but "what has been done and what is going on now" is, and it stays short."""
        store = self._store()
        if store is None or not self._trail:
            return
        try:
            from src.agent.todos import COMPLETED, IN_PROGRESS, Todo
            items = []
            for room, node, status in self._trail[-40:]:
                label = node if room == self._main_room else f"{room} › {node}"
                items.append(Todo(label[:200], COMPLETED if status == "done" else IN_PROGRESS))
            store.set(items, self._scope)
        except Exception:
            pass

    # -- engine events ----------------------------------------------------------------------------
    def on_event(self, event: dict) -> None:
        kind = event.get("type")
        room, node = event.get("room"), event.get("node")
        if kind == "node_start":
            self._open()
            if self._main_room is None:
                self._main_room = room
            entry = self._entry(room, node)
            if entry is None:
                self._trail.append([room, node, "run"])
            else:
                entry[2] = "run"
            word = _KIND_WORD.get(event.get("kind"), "")
            where = f"[bold {ACCENT}]{_esc(room)}[/bold {ACCENT}] [dim]›[/dim] [bold]{_esc(node)}[/bold]"
            if event.get("branch"):
                where += f" [dim]({_esc(event['branch'])})[/dim]"
            self._say(f"{where}  [dim]{word}[/dim]" if word else where)
            self._redraw()
        elif kind == "node_end":
            entry = self._entry(room, node)
            if entry is not None:
                entry[2] = "done"
            secs = event.get("seconds") or 0
            failed = bool(event.get("failed"))
            lines = []
            if failed or secs >= 1:
                mark = f"[{BAD}]не вышло[/{BAD}]" if failed else f"[{OK}]готово[/{OK}]"
                took = f"  [dim]{secs:.0f} с[/dim]" if secs >= 1 else ""
                lines.append(f"  {mark}{took}")
            if event.get("kind") in _SHOWS_RESULT:
                lines.extend(self._result_lines(event.get("produced") or {}))
            if lines:
                self._say(chr(10).join(lines))
            self._redraw()
        elif kind == "node_error":
            self._say(f"  [{BAD}]ошибка шага {_esc(node)}[/{BAD}] [dim]{_esc(_cut(event.get('error', ''), _LINE))}[/dim]")
        elif kind == "room_end":
            why = f" [dim]({_esc(event['reason'])})[/dim]" if event.get("reason") else ""
            self._say(f"[bold {ACCENT}]{_esc(room)}[/bold {ACCENT}] [dim]закончена:[/dim] "
                      f"[bold]{_esc(event.get('exit'))}[/bold]{why}")

    @staticmethod
    def _result_lines(produced: dict) -> list:
        out = []
        for name, value in list(produced.items())[:_FIELDS]:
            rows = _value_lines(value)
            if not rows:
                continue
            if len(rows) == 1 and not str(rows[0]).startswith("• "):
                out.append(f"  [{MUTED}]{_esc(name)}[/{MUTED}]  {_esc(rows[0])}")
                continue
            out.append(f"  [{MUTED}]{_esc(name)}[/{MUTED}]")
            out.extend(f"    {_esc(row)}" for row in rows)
        return out

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
