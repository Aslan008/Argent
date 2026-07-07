"""Approvals-as-events: a gated action surfaces as an event and is answered by
the transport, so a non-TTY client never deadlocks on questionary."""

from fastapi.testclient import TestClient

import approval
from src.server.app import create_app
from src.server.session import AgentSession


class ApprovalAgent:
    """A turn that gates on one approval and reports the decision as content."""

    def __init__(self, destructive=True, grant_key=None):
        self.destructive = destructive
        self.grant_key = grant_key

    def process_user_input(self, text, **kw):
        ok = approval.request_approval(
            "delete X", destructive=self.destructive, grant_key=self.grant_key
        )
        yield {"type": "content_stream", "content": "yes" if ok else "no"}


class TestSessionApproval:
    def test_once_proceeds(self):
        s = AgentSession(agent=ApprovalAgent())
        s.start("go")
        req = s.get_event()
        assert req["type"] == "approval_request"
        assert req["destructive"] is True and req["action"] == "delete X"
        s.reply_approval(req["id"], "once")
        assert s.get_event() == {"type": "content", "text": "yes"}
        assert s.get_event()["type"] == "done"

    def test_deny_blocks(self):
        s = AgentSession(agent=ApprovalAgent())
        s.start("go")
        req = s.get_event()
        s.reply_approval(req["id"], "deny")
        assert s.get_event() == {"type": "content", "text": "no"}
        assert s.get_event()["type"] == "done"

    def test_cancel_denies_pending(self):
        s = AgentSession(agent=ApprovalAgent())
        s.start("go")
        assert s.get_event()["type"] == "approval_request"
        s.cancel()                                     # e.g. client disconnected
        assert s.get_event() == {"type": "content", "text": "no"}
        assert s.get_event()["type"] == "done"

    def test_always_adds_session_grant(self):
        approval.clear_session_grants()
        s = AgentSession(agent=ApprovalAgent(destructive=False, grant_key="git"))
        s.start("go")
        req = s.get_event()
        assert req["grant_key"] == "git"
        s.reply_approval(req["id"], "always")
        assert s.get_event() == {"type": "content", "text": "yes"}
        assert s.get_event()["type"] == "done"
        assert "git" in approval.get_session_grants()
        approval.clear_session_grants()


def test_ws_approval_round_trip():
    app = create_app(session_factory=lambda: AgentSession(agent=ApprovalAgent()))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "message", "text": "go"})
        req = ws.receive_json()
        assert req["type"] == "approval_request"
        ws.send_json({"type": "approval_reply", "id": req["id"], "decision": "once"})
        assert ws.receive_json() == {"type": "content", "text": "yes"}
        assert ws.receive_json()["type"] == "done"


def test_ws_stop_denies():
    app = create_app(session_factory=lambda: AgentSession(agent=ApprovalAgent()))
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "message", "text": "go"})
        assert ws.receive_json()["type"] == "approval_request"
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "content", "text": "no"}
        assert ws.receive_json()["type"] == "done"
