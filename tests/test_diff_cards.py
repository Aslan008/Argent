"""Diff cards: file editors queue their unified diff, the agent forwards it as
a structured chunk, and the GUI can reject a card (undo) or rewind a turn."""

from fastapi.testclient import TestClient

import pytest

from src.server.app import create_app
from src.server.session import AgentSession
from tools import _helpers
from tools._helpers import _print_diff, drain_diff_events


@pytest.fixture(autouse=True)
def _clean_queue():
    drain_diff_events()
    yield
    drain_diff_events()


class TestDiffQueue:
    def test_print_diff_queues_plain_text(self):
        _print_diff("a\nb\n", "a\nc\n", "game/menu.py")
        events = drain_diff_events()
        assert len(events) == 1
        assert events[0]["file"] == "game/menu.py"
        assert "-b" in events[0]["diff"] and "+c" in events[0]["diff"]

    def test_drain_clears_the_queue(self):
        _print_diff("a\n", "b\n", "f.py")
        assert drain_diff_events()
        assert drain_diff_events() == []

    def test_no_change_queues_nothing(self):
        _print_diff("same\n", "same\n", "f.py")
        assert drain_diff_events() == []

    def test_queue_is_bounded(self):
        for i in range(_helpers._PENDING_DIFFS_MAX + 10):
            _print_diff("a\n", f"b{i}\n", "f.py")
        assert len(drain_diff_events()) == _helpers._PENDING_DIFFS_MAX

    def test_real_editor_queues_diff(self, tmp_path, monkeypatch):
        from tools import file_ops
        monkeypatch.setattr(file_ops, "snapshot", lambda *a, **k: True)
        f = tmp_path / "menu.py"
        f.write_text("def menu():\n    return 1\n", encoding="utf-8")

        result = file_ops.replace_in_file(str(f), "return 1", "return 2")
        assert result.startswith("Successfully")
        events = drain_diff_events()
        assert len(events) == 1
        assert "+    return 2" in events[0]["diff"]


class TestDiffEvent:
    def test_chunk_maps_to_structured_event(self):
        from src.server.events import to_event
        ev = to_event({"type": "diff", "file": "a.py", "diff": "-x\n+y\n"})
        assert ev == {"type": "diff", "file": "a.py", "diff": "-x\n+y\n"}


class _IdleAgent:
    def process_user_input(self, text, **kw):
        yield {"type": "content_stream", "content": "ok"}


class TestWsUndoFile:
    def test_undo_round_trip(self, monkeypatch):
        import file_tracker
        monkeypatch.setattr(file_tracker, "undo",
                            lambda fp: f"Restored '{fp}' to previous version.")
        app = create_app(session_factory=lambda: AgentSession(agent=_IdleAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "undo_file", "file": "menu.py"})
            res = ws.receive_json()
            assert res["type"] == "undo_result"
            assert res["ok"] is True and res["file"] == "menu.py"

    def test_undo_failure_reports_not_ok(self, monkeypatch):
        import file_tracker
        monkeypatch.setattr(file_tracker, "undo",
                            lambda fp: f"No snapshots found for '{fp}'. Cannot undo.")
        app = create_app(session_factory=lambda: AgentSession(agent=_IdleAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "undo_file", "file": "menu.py"})
            assert ws.receive_json()["ok"] is False

    def test_busy_session_refuses(self):
        session = AgentSession(agent=_IdleAgent())
        session._active = True                      # a turn is in flight
        app = create_app(session_factory=lambda: session)
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "undo_file", "file": "menu.py"})
            res = ws.receive_json()
            assert res["ok"] is False and "in flight" in res["text"]


class TestWsRewind:
    def test_rewind_round_trip(self, monkeypatch):
        from src.agent import checkpoints
        monkeypatch.setattr(checkpoints, "rewind_to",
                            lambda sha: f"Rewound to checkpoint {sha} — 'label'.")
        app = create_app(session_factory=lambda: AgentSession(agent=_IdleAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "rewind", "sha": "abc1234"})
            res = ws.receive_json()
            assert res["type"] == "rewind_result"
            assert res["ok"] is True and res["sha"] == "abc1234"

    def test_unsafe_rewind_reports_error(self, monkeypatch):
        from src.agent import checkpoints

        def boom(sha):
            raise checkpoints.CheckpointError("real commits would be discarded")
        monkeypatch.setattr(checkpoints, "rewind_to", boom)
        app = create_app(session_factory=lambda: AgentSession(agent=_IdleAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "rewind", "sha": "abc1234"})
            res = ws.receive_json()
            assert res["ok"] is False and "discarded" in res["text"]

    def test_busy_session_refuses(self):
        session = AgentSession(agent=_IdleAgent())
        session._active = True
        app = create_app(session_factory=lambda: session)
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "rewind", "sha": "abc1234"})
            assert ws.receive_json()["ok"] is False
