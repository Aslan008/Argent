"""RoomLibrary: loading, validation gating, indexing, stats and quarantine."""

import json

from src.rooms.library import RoomLibrary, default_starter_dir


TOOLS = {"run_command", "search_web"}


def _lib():
    return RoomLibrary().load_dir(default_starter_dir(), TOOLS)


class TestStarterLoad:
    def test_all_starter_rooms_valid(self):
        lib = _lib()
        assert lib.load_errors == {}
        assert set(lib.names()) == {"analyze", "debug_loop", "triage"}

    def test_triage_spawn_room_admitted(self):
        # triage carries a spawn_room node; it loads only because triage is a
        # privileged spawn room. (A non-triage room with spawn_room is rejected
        # below.)
        lib = _lib()
        assert lib.get("triage") is not None

    def test_find_by_substring(self):
        lib = _lib()
        assert [r.room for r in lib.find("debug")] == ["debug_loop"]
        assert [r.room for r in lib.find("codebase")] == ["analyze"]


class TestValidationGating:
    def test_unknown_tool_room_rejected(self, tmp_path):
        (tmp_path / "bad.json").write_text(json.dumps({
            "room": "bad", "budget": {"max_iterations": 3, "max_tokens": 100},
            "nodes": [{"id": "n", "type": "tool", "tool": "does_not_exist"}],
            "edges": [{"from": "n", "to": "EXIT:done"}],
        }), encoding="utf-8")
        lib = RoomLibrary().load_dir(tmp_path, TOOLS)
        assert "bad" in lib.load_errors and lib.get("bad") is None

    def test_spawn_room_outside_triage_rejected(self, tmp_path):
        (tmp_path / "sneaky.json").write_text(json.dumps({
            "room": "sneaky", "budget": {"max_iterations": 3, "max_tokens": 100},
            "nodes": [{"id": "s", "type": "spawn_room"}],
            "edges": [{"from": "s", "to": "EXIT:done"}],
        }), encoding="utf-8")
        lib = RoomLibrary().load_dir(tmp_path, TOOLS)
        assert "sneaky" in lib.load_errors
        assert any("spawn_room" in e for e in lib.load_errors["sneaky"])

    def test_malformed_json_recorded(self, tmp_path):
        (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
        lib = RoomLibrary().load_dir(tmp_path, TOOLS)
        assert "broken" in lib.load_errors and not lib.rooms


class TestStatsAndQuarantine:
    def test_quarantine_after_repeated_failure(self):
        lib = _lib()
        for _ in range(3):
            lib.record_run("debug_loop", success=False)
        assert lib.stats["debug_loop"]["quarantined"] is True
        assert "debug_loop" not in lib.active_rooms()

    def test_good_room_not_quarantined(self):
        lib = _lib()
        lib.record_run("debug_loop", True)
        lib.record_run("debug_loop", True)
        lib.record_run("debug_loop", False)
        assert lib.stats["debug_loop"]["quarantined"] is False
        assert "debug_loop" in lib.active_rooms()

    def test_no_quarantine_before_min_runs(self):
        lib = _lib()
        lib.record_run("analyze", False)  # one failure, below min_runs
        assert lib.stats["analyze"]["quarantined"] is False
