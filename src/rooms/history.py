"""Run records and the final report of a rooms run.

A run that leaves nothing behind but "done" cannot be trusted or learned from.
The recorder listens to the engine's events and, when the run ends, writes one
small JSON file per run (`.argent/rooms_runs/`): the task, the route, every node
with its time and the tokens it cost, the outcome and the key state fields.
`format_report` turns a run result into the text the person (or the model that
called the room) reads.
"""

import json
import re
import time
from pathlib import Path

from src.rooms import template

#: A state value longer than this is cut in the stored record.
MAX_STORED_CHARS = 1500
#: Fields never worth showing as "what came out" (engine plumbing).
_HIDDEN_SUFFIXES = (".output", ".declined")


def runs_dir() -> Path:
    return Path(".argent") / "rooms_runs"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40] or "run"


def _clip(value, limit=MAX_STORED_CHARS):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"… [+{len(value) - limit}]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:
        return _clip(json.dumps(value, ensure_ascii=False, default=str), limit)
    except Exception:
        return _clip(str(value), limit)


class RunRecorder:
    """Collects engine events; `finish` writes the record and returns its path."""

    def __init__(self, task: str, start_room: str, directory=None, clock=None, params=None):
        self.task = task
        self.start_room = start_room
        self.params = dict(params or {})
        self.directory = Path(directory) if directory else runs_dir()
        self._clock = clock or time.time
        self.started = self._clock()
        self.nodes = []

    def on_event(self, event: dict) -> None:
        if event.get("type") == "node_end":
            self.nodes.append({k: event.get(k) for k in ("room", "node", "kind", "seconds", "tokens")})

    def record(self, result, resumed: bool = False) -> dict:
        state = result.state
        data = {k: _clip(v) for k, v in (state.data if state else {}).items()}
        return {
            "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started)) + "-" + _slug(self.start_room),
            "started": self.started,
            "seconds": round(self._clock() - self.started, 1),
            "task": self.task,
            "start_room": self.start_room,
            "params": {k: _clip(v, 200) for k, v in self.params.items()},
            "outcome": result.outcome,
            "route": [[r, e] for r, e in result.history],
            "nodes": self.nodes,
            "usage": dict(state.usage) if state else {},
            "resumed": resumed,
            "state": data,
        }

    def finish(self, result, resumed: bool = False):
        rec = self.record(result, resumed)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{rec['id']}.json"
            n = 2
            while path.exists():
                path = self.directory / f"{rec['id']}-{n}.json"
                n += 1
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            self._prune()
            return path
        except OSError:
            return None

    def _prune(self, keep: int = 100) -> None:
        files = sorted(self.directory.glob("*.json"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass


def list_runs(directory=None, limit: int = 15) -> list:
    d = Path(directory) if directory else runs_dir()
    if not d.exists():
        return []
    out = []
    for path in sorted(d.glob("*.json"), reverse=True)[:limit]:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rec["_file"] = path.name
        out.append(rec)
    return out


def load_run(ref: str = "last", directory=None):
    runs = list_runs(directory, limit=200)
    if not runs:
        return None
    if ref in ("", "last"):
        return runs[0]
    for rec in runs:
        if rec.get("id", "").startswith(ref) or rec.get("_file", "").startswith(ref):
            return rec
    return None


_OUTCOME_TEXT = {
    "done": "выполнено",
    "escalated": "остановлено: нужен человек",
    "budget": "остановлено: исчерпан лимит комнат",
}


def _facts(state, library, rooms_run) -> list:
    """What came out of the run: the fields the run's rooms were allowed to write, plus the verdicts
    tool nodes set — never the raw tool outputs and never the person's own parameters."""
    wanted = []
    for name in rooms_run:
        room = library.get(name) if library is not None else None
        if room is None:
            continue
        for node in room.nodes:
            for f in list(node.writes) + list(node.sets) + ([node.into, f"{node.into}_approved"] if node.into else []):
                if f not in wanted:
                    wanted.append(f)
    lines = []
    for f in wanted:
        v = state.data.get(f)
        if v in (None, ""):
            continue
        text = str(v)
        if len(text) > 700:
            text = text[:700] + "…"
        lines.append(f"- {f}: {text}")
    return lines


def format_report(result, library=None, task: str = "", seconds: float = None) -> str:
    """The text a run ends with: outcome, route, what came out, what it cost."""
    state = result.state
    lines = [f"Итог: {_OUTCOME_TEXT.get(result.outcome, result.outcome)}"]
    if task:
        lines.append(f"Задача: {task}")
    if result.history:
        lines.append("Маршрут: " + " -> ".join(f"{r}:{e}" for r, e in result.history))
    rooms_run = [r for r, _ in result.history]

    custom = ""
    if library is not None and rooms_run and state is not None:
        last = library.get(rooms_run[-1])
        if last is not None and last.report:
            custom = template.render(last.report, state.data, limit=3000).strip()
    if custom:
        lines.append("")
        lines.append(custom)
    elif state is not None:
        facts = _facts(state, library, rooms_run)
        if facts:
            lines.append("")
            lines.append("Что получилось:")
            lines.extend(facts)

    if state is not None:
        declined = [k[: -len(".declined")] for k, v in state.data.items() if k.endswith(".declined") and v]
        if declined:
            lines.append("")
            lines.append("Не выполнено (не подтверждено): " + ", ".join(declined))

    spent = []
    if state is not None and state.usage.get("tokens"):
        spent.append(f"≈{int(state.usage['tokens'])} ток")
    if seconds is not None:
        spent.append(f"{seconds:.0f} с")
    if spent:
        lines.append("")
        lines.append("Расход: " + ", ".join(spent))
    return "\n".join(lines)
