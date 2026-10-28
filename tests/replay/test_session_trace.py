import json

from src.bridge.session import AgentSession


class _FakeAgent:
    session_id = None

    def process_user_input(self, text):
        yield {"type": "content_stream", "content": "hello"}
        yield {"type": "tool_start", "name": "read_file", "args": {"file_path": "a.py"}}
        yield {"type": "tool_end", "name": "read_file", "result": "ok"}


def test_a_session_records_a_trace_only_when_replay_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "1")
    monkeypatch.setenv("ARGENT_REPLAY_DIR", str(tmp_path))
    session = AgentSession(agent=_FakeAgent())

    events = list(session.handle("hi"))
    assert events[-1] == {"type": "done"}
    path = session.trace_path
    assert path is not None and path.exists()
    session.close_trace()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "session_start"
    assert "content" in [r["kind"] for r in rows]
    assert "tool_start" in [r["kind"] for r in rows]
    assert "tool_end" in [r["kind"] for r in rows]
    assert rows[-1]["kind"] == "done"


def test_no_trace_file_is_created_when_replay_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "0")
    monkeypatch.setenv("ARGENT_REPLAY_DIR", str(tmp_path))
    session = AgentSession(agent=_FakeAgent())

    list(session.handle("hi"))
    assert session.trace_path is None
    assert not list(tmp_path.glob("*.jsonl"))
