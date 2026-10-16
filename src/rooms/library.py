"""RoomLibrary: load, validate, index and track declarative rooms.

Rooms live as JSON files in a directory. On load each is parsed against the
schema and run through the RoomValidator against the tool registry; anything
that fails is kept out of the active set and its errors recorded (never silently
dropped). The library tracks per-room success/failure counts and auto-quarantines
a room whose success rate falls below a threshold over enough runs — the guard
against a growing library rotting with bad rooms. The counters are kept in a small
JSON file (``stats_path``) so a run today still counts tomorrow; without a path
the library is in-memory only (tests, one-off use).

spawn_room is a privileged node type: it is admitted only for rooms whose name is
in `spawn_rooms` (the triage room), enforcing "only triage may propose rooms".
"""

import json
from pathlib import Path

from src.rooms.models import Room
from src.rooms.validator import validate_room


HISTORY_KEEP = 5                       # earlier files kept per room


def _version_order(path) -> tuple:
    """Sort key of a saved version: (1, sequence) for "000007-<stamp>.json"; (0, name) for the older
    "<stamp>.json" files, which therefore count as older than any numbered one."""
    head = Path(path).stem.split("-", 1)[0]
    if len(head) == 6 and head.isdigit():
        return (1, int(head))
    return (0, Path(path).stem)


def default_starter_dir() -> Path:
    return Path(__file__).parent / "starter"


class RoomLibrary:
    def __init__(self, spawn_rooms=("triage",), user_dir=None, stats_path=None):
        self.rooms: dict = {}         # name -> Room
        self.stats: dict = {}         # name -> {successes, failures, quarantined}
        self.load_errors: dict = {}   # name/stem -> [error strings]
        self.sources: dict = {}       # name -> "starter" | "user"
        self._spawn_rooms = set(spawn_rooms)
        # Writable directory for user/AI-authored rooms (add_room persists here).
        self.user_dir = Path(user_dir) if user_dir else None
        # Further folders whose rooms are shown too (a project's own .argent/rooms). They are only read:
        # saving always goes to user_dir, and a name that user_dir has wins over the same name here.
        self.extra_dirs: list = []
        self.stats_path = Path(stats_path) if stats_path else None
        self._saved_stats: dict = self._read_stats()

    def _read_stats(self) -> dict:
        if not self.stats_path or not self.stats_path.exists():
            return {}
        try:
            data = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return {}          # a damaged stats file is no reason to lose the library
        return data if isinstance(data, dict) else {}

    def _fresh_stats(self, name: str) -> dict:
        saved = self._saved_stats.get(name)
        if isinstance(saved, dict):
            return {"successes": int(saved.get("successes", 0)), "failures": int(saved.get("failures", 0)),
                    "quarantined": bool(saved.get("quarantined", False))}
        return {"successes": 0, "failures": 0, "quarantined": False}

    def save_stats(self) -> None:
        if not self.stats_path:
            return
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            merged = dict(self._saved_stats)
            merged.update({n: st for n, st in self.stats.items() if n in self.rooms})
            self._saved_stats = merged
            tmp = self.stats_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.stats_path)
        except OSError:
            pass               # statistics are a convenience; never break a run over them

    def unquarantine(self, name: str) -> bool:
        """Put a quarantined room back into use (its counters start over)."""
        if name not in self.rooms:
            return False
        self.stats[name] = {"successes": 0, "failures": 0, "quarantined": False}
        self.save_stats()
        return True

    def load_dir(self, directory, available_tools, origin: str = "starter",
                 defer_to_user: bool = False) -> "RoomLibrary":
        """Load every room file of a folder. With `defer_to_user`, a room whose name is already a user's
        room (from the main folder) is left alone: one name, one room."""
        for path in sorted(Path(directory).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                room = Room(**data)
            except Exception as e:
                self.load_errors[path.stem] = [f"parse/schema error: {e}"]
                continue
            if defer_to_user and self.sources.get(room.room) == "user":
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
            self.stats.setdefault(room.room, self._fresh_stats(room.room))
        # Rooms may call each other (`room` nodes): a call to a room that is not there, or a circle of
        # calls, keeps the CALLER out of the library, as any other validation failure would.
        changed = True
        while changed:
            changed = False
            for name, errs in self.reference_errors().items():
                self.load_errors[name] = errs
                self.rooms.pop(name, None)
                self.sources.pop(name, None)
                self.stats.pop(name, None)
                changed = True
        return self

    @staticmethod
    def calls_of(room: Room) -> list:
        return [n.room for n in room.nodes if n.type == "room" and n.room]

    def reference_errors(self, rooms: dict = None) -> dict:
        """name -> problems with the rooms it calls (missing room, a room calling itself, a cycle)."""
        rooms = self.rooms if rooms is None else rooms
        errors = {}
        graph = {name: self.calls_of(r) for name, r in rooms.items()}
        for name, room in rooms.items():
            for n in room.nodes:
                if n.type == "room" and n.room not in rooms:
                    errors.setdefault(name, []).append(f"node '{n.id}': room '{n.room}' does not exist")
        # cycles: a room that can reach itself through calls
        for start in graph:
            seen, stack = set(), list(graph[start])
            while stack:
                cur = stack.pop()
                if cur == start:
                    errors.setdefault(start, []).append("rooms call each other in a circle (a room may not reach itself)")
                    break
                if cur in seen or cur not in graph:
                    continue
                seen.add(cur)
                stack.extend(graph[cur])
        return errors

    def referenced_by(self, name: str) -> list:
        """Rooms that call `name` from a `room` node."""
        return sorted(n for n, r in self.rooms.items() if name in self.calls_of(r) and n != name)

    def add_room(self, room: Room, available_tools) -> list:
        """Validate and add a room, persisting it to the user directory. Returns
        the validation errors (empty list == added). This is the single gate for
        both user- and AI-authored rooms — nothing enters the graph unvalidated."""
        errors = validate_room(
            room, set(available_tools),
            allow_spawn=(room.room in self._spawn_rooms),
        )
        if not errors:
            errors = self.reference_errors({**self.rooms, room.room: room}).get(room.room, [])
        if errors:
            return errors
        self.rooms[room.room] = room
        self.sources[room.room] = "user"
        self.stats.setdefault(room.room, self._fresh_stats(room.room))
        if self.user_dir:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot(room.room)
            (self.user_dir / f"{room.room}.json").write_text(
                room.model_dump_json(indent=2, by_alias=True, exclude_none=True),
                encoding="utf-8",
            )
        return []

    # -- versions --------------------------------------------------------------------------------------
    # Every write over a room (editor, /rooms build|edit, patch) and every delete first copies the file to
    # <user_dir>/.history/<name>/, so a bad edit or a deleted room can be brought back. The folder starts
    # with a dot and holds no top-level *.json, so load_dir never takes a copy for a room.

    def _history_dir(self, name: str) -> Path:
        return Path(self.user_dir) / ".history" / name

    def _snapshot(self, name: str) -> None:
        if not self.user_dir:
            return
        # the file that is about to be replaced or deleted: the shared one, else the project's own copy
        current = next((f for f in (Path(d) / f"{name}.json" for d in [self.user_dir, *self.extra_dirs])
                        if f.is_file()), None)
        try:
            if current is None:
                return
            text = current.read_text(encoding="utf-8")
            folder = self._history_dir(name)
            folder.mkdir(parents=True, exist_ok=True)
            kept = sorted(folder.glob("*.json"), key=_version_order)
            if kept and kept[-1].read_text(encoding="utf-8") == text:
                return                      # nothing changed since the last copy
            # Ordered by a sequence number, not by the clock: two saves in one millisecond used to get
            # "<stamp>.json" and "<stamp>-2.json", and "-" sorts before "." - the newer copy came out older.
            import time
            seq = max((_version_order(p)[1] for p in kept if _version_order(p)[0] == 1), default=0) + 1
            target = folder / f"{seq:06d}-{time.strftime('%Y%m%d-%H%M%S')}.json"
            target.write_text(text, encoding="utf-8")
            for old in sorted(folder.glob("*.json"), key=_version_order)[:-HISTORY_KEEP]:
                old.unlink()
        except OSError:
            pass                            # a copy that cannot be made must not block the save

    def versions(self, name: str) -> list:
        """Earlier files of a room, newest first (1 = the one just before the current)."""
        if not self.user_dir:
            return []
        try:
            return sorted(self._history_dir(name).glob("*.json"), key=_version_order, reverse=True)
        except OSError:
            return []

    def restore_version(self, name: str, which: int, available_tools) -> list:
        """Put an earlier file back (through the same gate as any save). Returns errors; [] == restored."""
        found = self.versions(name)
        if not 1 <= which <= len(found):
            return [f"нет версии {which} (есть: {len(found)})"]
        try:
            room = Room(**json.loads(found[which - 1].read_text(encoding="utf-8")))
        except Exception as e:
            return [f"версия не читается: {e}"]
        if room.room != name:
            return [f"в файле версии другая комната («{room.room}»)"]
        return self.add_room(room, available_tools)

    def remove_room(self, name: str) -> bool:
        """Remove a room from the library; delete its user-dir file if present.
        Returns False if there was no such room."""
        if name not in self.rooms:
            return False
        self.rooms.pop(name)
        self.stats.pop(name, None)
        self.sources.pop(name, None)
        self._snapshot(name)             # a deleted room can be brought back: /rooms restore
        # also from the project folders: a room deleted here must not come back from there
        for folder in ([self.user_dir] if self.user_dir else []) + list(self.extra_dirs):
            f = Path(folder) / f"{name}.json"
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
        has enough runs to judge. The packaged starter rooms are counted but never
        quarantined: they are curated and tested, and a run that failed because
        the PROJECT's tests are red says nothing about the room. The guard is for
        the rooms people and the model add."""
        st = self.stats.setdefault(name, self._fresh_stats(name))
        st["successes" if success else "failures"] += 1
        runs = st["successes"] + st["failures"]
        if (runs >= min_runs and (st["successes"] / runs) < threshold
                and self.sources.get(name) != "starter"):
            st["quarantined"] = True
        self.save_stats()
        return st


def short_description(room: Room, limit: int = 150) -> str:
    """The first sentence of a room's description, cut at a word boundary (for lists and hints)."""
    text = " ".join((room.description or "").split())
    first = text.split(". ")[0].rstrip(".")
    if len(first) <= limit:
        return first
    cut = first[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def describe_room(room: Room) -> str:
    """A readable summary of a room's graph — for `/rooms show`."""
    b = room.budget
    limit = f"{b.max_iterations} шагов / {b.max_tokens} ток" + (f" / {b.max_seconds} с" if b.max_seconds else "")
    lines = [
        f"Room: {room.room}  ({room.description})",
        f"  budget: {limit}   entry: {room.entry}",
    ]
    if room.tags:
        lines.append(f"  tags: {', '.join(room.tags)}")
    if room.defaults:
        lines.append("  params: " + ", ".join(f"{k}={v!r}" for k, v in room.defaults.items()))
    if room.next:
        lines.append("  next: " + ", ".join(f"{k} -> {v}" for k, v in room.next.items()))
    if room.rules is not None and room.rules.active():
        lines.append("  rules: " + ", ".join(f"{k}={v}" for k, v in room.rules.model_dump().items()
                                              if v and v != "isolated" and k != "memory_chars"))
    lines.append("  nodes:")
    for n in room.nodes:
        detail = ""
        if n.type == "tool":
            detail = f" tool={n.tool}" + (f" args={n.args}" if n.args else "")
            if n.sets:
                detail += f" sets={n.sets}"
            if n.confirm:
                detail += " [спрашивает]"
        elif n.type == "agent":
            detail = f" role={n.role or 'Coder'}"
            if n.tools is not None:
                detail += f" tools={n.tools}"
            detail += f" writes={n.writes}" + (f" ctx={n.context}" if n.context else "")
        elif n.type == "set":
            detail = f" values={n.values}"
        elif n.type == "notify":
            detail = f" channel={n.channel}"
        elif n.type == "room":
            detail = f" room={n.room}" + (f" inputs={n.inputs}" if n.inputs else "")
            detail += (f" outputs={n.outputs}" if n.outputs else "") + f" exit->{n.into or n.id + '_exit'}"
        elif n.type == "mcp":
            detail = f" server={n.server}" + (f" tool={n.tool}" if n.tool else "") + (f" args={n.args}" if n.args else "")
            if n.sets:
                detail += f" sets={n.sets}"
            if n.ignore:
                detail += f" ignore={n.ignore}"
            if n.confirm:
                detail += " [спрашивает]"
        elif n.type == "skill":
            detail = f" skill={n.skill}" + (f" ignore={n.ignore}" if n.ignore else "")
        elif n.type == "plugin":
            detail = f" command={n.command}" + (f" plugin={n.plugin}" if n.plugin else "")
        elif n.type == "leeth":
            detail = f" mode={n.mode or 'delegate'}" + (f" ask={n.instruction!r}" if n.instruction else "")
            if n.input_from:
                detail += f" list={n.input_from}"
            if n.confirm:
                detail += " [спрашивает]"
        elif n.type == "foreach":
            detail = f" over={n.input_from} item={n.item or 'item'}"
        elif n.type == "wait":
            detail = f" seconds={n.seconds}"
        elif n.type == "loop":
            detail = f" mode={n.mode or 'for'}" + (f" cond={n.condition}" if n.condition else "") +                 (f" start={n.start} stop={n.stop} step={n.step}" if (n.mode or "for") in ("for", "times") else "") +                 (f" over={n.input_from}" if n.input_from else "") + f" var={n.item or 'i'}"
        elif n.type in ("break", "continue"):
            detail = f" loop={n.loop}"
        elif n.type == "power":
            detail = f" mode={n.mode} seconds={n.seconds}" + (" [спрашивает]" if n.confirm else "")
        elif n.type == "hint":
            detail = f" mode={n.mode or 'next'} text={n.message[:60]!r}" + (" [пишет модель]" if n.compose else "")
        elif n.type == "forget":
            detail = f" mode={n.mode or 'all'}" + (f" clear={n.clear}" if n.clear else "")
        elif n.type == "assert":
            detail = f" cond={n.condition}"
        elif n.type == "http":
            detail = f" {(n.mode or 'get').upper()} {str((n.args or {}).get('url', ''))[:60]}" + (" [спрашивает]" if n.confirm else "")
        elif n.type == "git":
            detail = f" mode={n.mode or 'status'}" + (f" message={n.message[:50]!r}" if n.message else "") + (" [спрашивает]" if n.confirm else "")
        elif n.type == "parallel":
            detail = f" join={n.join} workers={n.workers or 'all'}"
        elif n.type == "join":
            detail = f" wait_for={n.wait_for or 'all'} merge={n.merge or 'error'} on_branch_fail={n.on_branch_fail or 'fail'}"
        if n.type == "agent" and (n.sampling or n.memory or n.hint):
            detail += (f" sampling={n.sampling}" if n.sampling else "") + (f" memory={n.memory}" if n.memory else "") +                 (" hint" if n.hint else "")
        if n.type == "agent":
            if n.skills:
                detail += f" skills={n.skills}"
            if n.mcp is not None:
                detail += f" mcp={n.mcp}"
            if n.ignore:
                detail += f" ignore={n.ignore}"
        if n.disabled:
            detail += " [отключён]"
        if n.retries:
            detail += f" retries={n.retries}"
        lines.append(f"    - {n.id} [{n.type}]{detail}")
    lines.append("  edges:")
    for e in room.edges:
        if e.kind == "if":
            label = f" if({e.if_})"
        elif e.kind == "while":
            label = f" while({e.while_}) x{e.max}"
        elif e.kind == "else":
            label = " else"
        elif e.kind == "error":
            label = " on_error"
        else:
            label = ""
        lines.append(f"    {e.from_} ->{label} {e.to}")
    return "\n".join(lines)
