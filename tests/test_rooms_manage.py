"""Room library management: persist/add/remove, round-trip, describe, user dir."""

import json

from src.rooms.library import RoomLibrary, describe_room
from src.rooms.models import Room


LINT_ROOM = {
    "room": "lint",
    "description": "run a linter",
    "budget": {"max_iterations": 3, "max_tokens": 100},
    "nodes": [{"id": "n", "type": "tool", "tool": "run_command", "args": {"command": "flake8"}}],
    "edges": [{"from": "n", "to": "EXIT:done"}],
}

COND_ROOM = {
    "room": "cond",
    "budget": {"max_iterations": 5, "max_tokens": 100},
    "nodes": [{"id": "a", "type": "tool", "tool": "run_command"},
              {"id": "b", "type": "tool", "tool": "run_command"}],
    "edges": [
        {"from": "a", "if": "state.ok", "to": "EXIT:yes"},
        {"from": "a", "else": True, "to": "b"},
        {"from": "b", "while": "state.go", "max": 2, "id": "c", "to": "a"},
        {"from": "b", "else": True, "to": "EXIT:no"},
    ],
}


class TestAddRemove:
    def test_add_persists_and_reloads(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        assert lib.add_room(Room(**LINT_ROOM), {"run_command"}) == []
        assert lib.get("lint") is not None and lib.sources["lint"] == "user"
        assert (tmp_path / "rooms" / "lint.json").exists()

        reloaded = RoomLibrary().load_dir(tmp_path / "rooms", {"run_command"}, origin="user")
        assert reloaded.get("lint") is not None

    def test_add_rejects_invalid_without_writing(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        bad = Room(room="bad", budget={"max_iterations": 2, "max_tokens": 10},
                   nodes=[{"id": "n", "type": "tool", "tool": "ghost"}],
                   edges=[{"from": "n", "to": "EXIT:x"}])
        errs = lib.add_room(bad, {"run_command"})   # ghost tool doesn't exist
        assert errs and lib.get("bad") is None
        assert not (tmp_path / "rooms" / "bad.json").exists()

    def test_remove(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        lib.add_room(Room(**LINT_ROOM), {"run_command"})
        assert lib.remove_room("lint") is True
        assert lib.get("lint") is None
        assert not (tmp_path / "rooms" / "lint.json").exists()
        assert lib.remove_room("nope") is False


class TestRoundTrip:
    def test_saved_room_roundtrips_edge_aliases(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        lib.add_room(Room(**COND_ROOM), {"run_command"})
        reloaded = RoomLibrary().load_dir(tmp_path / "rooms", {"run_command"}, origin="user")
        room = reloaded.get("cond")
        assert room is not None
        kinds = sorted(e.kind for e in room.edges)
        assert kinds == ["else", "else", "if", "while"]   # aliases survived the round-trip


class TestDescribe:
    def test_describe_room(self):
        s = describe_room(Room(**COND_ROOM))
        assert "Room: cond" in s
        assert "a [tool]" in s
        assert "EXIT:yes" in s and "if(state.ok)" in s and "while(state.go) x2" in s


class TestBuildLibraryUserDir:
    def test_build_library_loads_user_rooms(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        udir = tmp_path / ".argent" / "rooms"
        udir.mkdir(parents=True)
        (udir / "lint.json").write_text(json.dumps(LINT_ROOM), encoding="utf-8")

        from src.rooms.session import build_library
        lib = build_library()   # real tool registry
        assert lib.get("lint") is not None and lib.sources["lint"] == "user"
        assert "debug_loop" in lib.names()   # starter rooms still present
