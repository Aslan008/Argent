"""CSWSH defense: the WebSocket rejects foreign browser origins."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.server.app import _origin_allowed, create_app
from src.server.session import AgentSession


class _IdleAgent:
    def process_user_input(self, text, **kw):
        yield {"type": "content_stream", "content": "ok"}


class TestOriginPredicate:
    @pytest.mark.parametrize("origin", [
        None, "",
        "http://localhost:1420", "http://127.0.0.1:8756", "https://localhost",
        "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
    ])
    def test_allowed(self, origin):
        assert _origin_allowed(origin) is True

    @pytest.mark.parametrize("origin", [
        "https://evil.com",
        "http://attacker.example",
        "https://localhost.evil.com",     # suffix trick — hostname is not localhost
        "http://169.254.1.1",
    ])
    def test_rejected(self, origin):
        assert _origin_allowed(origin) is False


class TestWsHandshake:
    def _client(self):
        return TestClient(create_app(session_factory=lambda: AgentSession(agent=_IdleAgent())))

    def test_local_origin_connects(self):
        with self._client().websocket_connect(
            "/ws", headers={"origin": "http://localhost:1420"}
        ) as ws:
            ws.send_json({"type": "message", "text": "hi"})
            assert ws.receive_json()["type"] == "content"

    def test_no_origin_connects(self):
        # Native clients / tests send no Origin — allowed.
        with self._client().websocket_connect("/ws") as ws:
            ws.send_json({"type": "message", "text": "hi"})
            assert ws.receive_json()["type"] == "content"

    def test_foreign_origin_is_closed(self):
        with pytest.raises(WebSocketDisconnect):
            with self._client().websocket_connect(
                "/ws", headers={"origin": "https://evil.com"}
            ) as ws:
                ws.receive_json()
