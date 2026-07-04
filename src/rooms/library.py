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
    def __init__(self, spawn_rooms=("triage",)):
        self.rooms: dict = {}         # name -> Room
        self.stats: dict = {}         # name -> {successes, failures, quarantined}
        self.load_errors: dict = {}   # name/stem -> [error strings]
        self._spawn_rooms = set(spawn_rooms)

    def load_dir(self, directory, available_tools) -> "RoomLibrary":
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
            self.stats.setdefault(room.room, {"successes": 0, "failures": 0, "quarantined": False})
        return self

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
