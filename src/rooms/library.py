"""RoomLibrary: load, validate, index and track declarative rooms.

Rooms live as JSON files in a directory. On load each is parsed against the
schema and run through the RoomValidator against the tool registry; anything
that fails is kept out of the active set and its errors recorded (never silently
dropped). The library tracks per-room success/failure counts and auto-quarantines
a room whose success rate falls below a threshold over enough runs — the guard
against a growing library rotting with bad rooms.

spawn_room is a privileged node type: it is admitted only for rooms whose name is
in `spawn_rooms` (the triage room), enforcing "only triage may propose rooms".
"""

import json
from pathlib import Path

from src.rooms.models import Room
from src.rooms.validator import validate_room


def default_starter_dir() -> Path:
    return Path(__file__).parent / "starter"


class RoomLibrary:
    def __init__(self, spawn_rooms=("triage",), user_dir=None):
        self.rooms: dict = {}         # name -> Room
        self.stats: dict = {}         # name -> {successes, failures, quarantined}
        self.load_errors: dict = {}   # name/stem -> [error strings]
        self.sources: dict = {}       # name -> "starter" | "user"
        self._spawn_rooms = set(spawn_rooms)
        # Writable directory for user/AI-authored rooms (add_room persists here).
        self.user_dir = Path(user_dir) if user_dir else None

    def load_dir(self, directory, available_tools, origin: str = "starter") -> "RoomLibrary":
        for path in sorted(Path(directory).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                room = Room(**data)
            except Exception as e:
                self.load_errors[path.stem] = [f"parse/schema error: {e}"]
                continue
            errs = validate_room(
                room, set(available_tools),
                allow_spawn=(room.room in self._spawn_rooms),
            )
            if errs:
                self.load_errors[room.room] = errs
                continue
            self.rooms[room.room] = room
            self.sources[room.room] = origin
            self.stats.setdefault(room.room, {"successes": 0, "failures": 0, "quarantined": False})
        return self

    def add_room(self, room: Room, available_tools) -> list:
        """Validate and add a room, persisting it to the user directory. Returns
        the validation errors (empty list == added). This is the single gate for
        both user- and AI-authored rooms — nothing enters the graph unvalidated."""
        errors = validate_room(
            room, set(available_tools),
            allow_spawn=(room.room in self._spawn_rooms),
        )
        if errors:
            return errors
        self.rooms[room.room] = room
        self.sources[room.room] = "user"
        self.stats.setdefault(room.room, {"successes": 0, "failures": 0, "quarantined": False})
        if self.user_dir:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            (self.user_dir / f"{room.room}.json").write_text(
                room.model_dump_json(indent=2, by_alias=True, exclude_none=True),
                encoding="utf-8",
            )
        return []

    def remove_room(self, name: str) -> bool:
        """Remove a room from the library; delete its user-dir file if present.
        Returns False if there was no such room."""
        if name not in self.rooms:
            return False
        self.rooms.pop(name)
        self.stats.pop(name, None)
        self.sources.pop(name, None)
        if self.user_dir:
            f = self.user_dir / f"{name}.json"
            if f.exists():
                f.unlink()
        return True

    def get(self, name):
        return self.rooms.get(name)

    def names(self) -> list:
        return sorted(self.rooms)

    def find(self, query: str) -> list:
        """Rooms whose name or description contains `query` (substring, case-
        insensitive). A semantic retriever can replace this later."""
        q = (query or "").lower().strip()
        if not q:
            return []
        return [r for r in self.rooms.values()
                if q in r.room.lower() or q in r.description.lower()]

    def active_rooms(self) -> dict:
        """Rooms usable in the graph — quarantined ones excluded."""
        return {n: r for n, r in self.rooms.items() if not self.stats[n]["quarantined"]}

    def record_run(self, name: str, success: bool, min_runs: int = 3,
                   threshold: float = 0.5) -> dict:
        """Record an outcome; auto-quarantine a room that underperforms once it
        has enough runs to judge."""
        st = self.stats.setdefault(name, {"successes": 0, "failures": 0, "quarantined": False})
        st["successes" if success else "failures"] += 1
        runs = st["successes"] + st["failures"]
        if runs >= min_runs and (st["successes"] / runs) < threshold:
            st["quarantined"] = True
        return st


def describe_room(room: Room) -> str:
    """A readable summary of a room's graph — for `/rooms show`."""
    lines = [
        f"Room: {room.room}  ({room.description})",
        f"  budget: {room.budget.max_iterations} iters / {room.budget.max_tokens} tok   entry: {room.entry}",
        "  nodes:",
    ]
    for n in room.nodes:
        detail = ""
        if n.type == "tool":
            detail = f" tool={n.tool}" + (f" args={n.args}" if n.args else "")
        elif n.type == "agent":
            detail = f" writes={n.writes}" + (f" ctx={n.context}" if n.context else "")
        lines.append(f"    - {n.id} [{n.type}]{detail}")
    lines.append("  edges:")
    for e in room.edges:
        if e.kind == "if":
            label = f" if({e.if_})"
        elif e.kind == "while":
            label = f" while({e.while_}) x{e.max}"
        elif e.kind == "else":
            label = " else"
        else:
            label = ""
        lines.append(f"    {e.from_} ->{label} {e.to}")
    return "\n".join(lines)
