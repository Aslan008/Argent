"""GUI parity backend: state snapshot, vibe switch, cooperative turn stop."""

import threading

import pytest
from fastapi.testclient import TestClient

import approval
from src.server.app import create_app
from src.server.session import AgentSession


class FakeAgent:
    model_name = "test-model:7b"
    provider = "ollama"

    def get_context_usage(self):
        return {"tokens": 1200, "max": 8000, "percent": 15.0}

    def process_user_input(self, text, **kw):
        yield {"type": "content_stream", "content": "hi"}


@pytest.fixture
def restore_policy():
    yield
    approval.set_policy(approval.POLICY_ASK)


class TestState:
    def test_snapshot_shape(self, monkeypatch):
        from src.agent import checkpoints
        monkeypatch.setattr(checkpoints, "list_checkpoints",
                            lambda limit=10: [{"sha": "abc1234", "label": "before: fix", "age": "5m"}])
        s = AgentSession(agent=FakeAgent())
        st = s.state()
        assert st["model"] == "test-model:7b" and st["provider"] == "ollama"
        assert st["context"] == {"tokens": 1200, "max": 8000, "percent": 15}
        assert st["vibe"] is False
        assert st["checkpoints"] == [{"sha": "abc1234", "label": "before: fix"}]

    def test_bare_fake_agent_never_crashes(self):
        st = AgentSession(agent=object()).state()
        assert st["model"] == "?" and st["provider"] == "?"

    def test_ws_get_state(self, monkeypatch):
        from src.agent import checkpoints
        monkeypatch.setattr(checkpoints, "list_checkpoints", lambda limit=10: [])
        app = create_app(session_factory=lambda: AgentSession(agent=FakeAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "get_state"})
            st = ws.receive_json()
            assert st["type"] == "state" and st["model"] == "test-model:7b"

    def test_state_pushed_after_done(self, monkeypatch):
        from src.agent import checkpoints
        monkeypatch.setattr(checkpoints, "list_checkpoints", lambda limit=10: [])
        app = create_app(session_factory=lambda: AgentSession(agent=FakeAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "message", "text": "go"})
            types = [ws.receive_json()["type"] for _ in range(3)]
            assert types == ["content", "done", "state"]


class TestVibeSwitch:
    def test_set_vibe_toggles_policy(self, restore_policy):
        s = AgentSession(agent=FakeAgent())
        assert s.set_vibe(True) is True
        assert approval.get_policy() == approval.POLICY_AUTO
        assert s.state()["vibe"] is True
        assert s.set_vibe(False) is False
        assert approval.get_policy() == approval.POLICY_ASK

    def test_ws_vibe_round_trip(self, restore_policy):
        app = create_app(session_factory=lambda: AgentSession(agent=FakeAgent()))
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_json({"type": "vibe", "enabled": True})
            assert ws.receive_json() == {"type": "vibe_state", "enabled": True}
            assert approval.get_policy() == approval.POLICY_AUTO


class SlowAgent:
    """Yields one chunk, then waits for the gate before yielding 50 more —
    lets the test stop the turn at a deterministic point."""

    def __init__(self):
        self.gate = threading.Event()
        self.closed = False

    def process_user_input(self, text, **kw):
        try:
            yield {"type": "content_stream", "content": "first"}
            self.gate.wait(timeout=5)
            for i in range(50):
                yield {"type": "content_stream", "content": f"chunk{i}"}
        except GeneratorExit:
            self.closed = True
            raise


class TestCooperativeStop:
    def test_stop_aborts_between_chunks(self):
        agent = SlowAgent()
        s = AgentSession(agent=agent)
        s.start("go")
        assert s.get_event() == {"type": "content", "text": "first"}

        s.cancel()                # stop requested while the generator is gated
        agent.gate.set()          # release: the next chunk hits the stop check

        ev = s.get_event()
        assert ev["type"] == "notice" and "stopped" in ev["text"]
        assert s.get_event()["type"] == "done"
        assert agent.closed is True   # GeneratorExit reached the turn

    def test_next_turn_runs_after_a_stop(self):
        agent = SlowAgent()
        s = AgentSession(agent=agent)
        s.start("go")
        s.get_event()
        s.cancel()
        agent.gate.set()
        while s.get_event()["type"] != "done":
            pass

        fresh = AgentSession(agent=FakeAgent())
        fresh.start("again")
        assert fresh.get_event() == {"type": "content", "text": "hi"}
        assert fresh.get_event()["type"] == "done"
