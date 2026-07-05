"""FastAPI WebSocket endpoint — streamed over a real (test) transport."""

from fastapi.testclient import TestClient

from src.server.app import create_app
from src.server.session import AgentSession


class FakeAgent:
    def __init__(self, chunks):
        self.chunks = chunks

    def process_user_input(self, text, **kwargs):
        yield from self.chunks


def test_health():
    # Default factory, but /health never builds a session/agent.
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}


def test_ws_streams_turn():
    fake = FakeAgent([
        {"type": "content_stream", "content": "Hi "},
        {"type": "tool_start", "name": "read_file", "args": {"p": "x"}},
        {"type": "tool_end", "name": "read_file", "result": "data"},
        {"type": "content_stream", "content": "there"},
        {"type": "error", "content": "\n[System: nudging...]"},
    ])
    app = create_app(session_factory=lambda: AgentSession(agent=fake))
    client = TestClient(app)

    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "message", "text": "hello"})
        events = []
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] == "done":
                break

    types = [e["type"] for e in events]
    assert types == ["content", "tool_start", "tool_end", "content", "notice", "done"]
    assert events[0]["text"] == "Hi "
    assert events[2]["result"] == "data"


def test_ws_ignores_non_message():
    fake = FakeAgent([{"type": "content_stream", "content": "ok"}])
    app = create_app(session_factory=lambda: AgentSession(agent=fake))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})            # ignored
        ws.send_json({"type": "message", "text": "go"})
        first = ws.receive_json()
        assert first == {"type": "content", "text": "ok"}
        assert ws.receive_json()["type"] == "done"
