"""Event normalization + AgentSession — the transport-agnostic core, no server."""

from src.server.events import to_event
from src.server.session import AgentSession


class TestToEvent:
    def test_content(self):
        assert to_event({"type": "content_stream", "content": "hi"}) == {"type": "content", "text": "hi"}
        assert to_event({"type": "content", "content": "yo"}) == {"type": "content", "text": "yo"}

    def test_content_replace_and_thinking(self):
        assert to_event({"type": "content_replace", "content": "x"})["type"] == "content_replace"
        assert to_event({"type": "thinking_stream", "content": "hmm"}) == {"type": "thinking", "text": "hmm"}

    def test_tool_events(self):
        assert to_event({"type": "tool_start", "name": "read_file", "args": {"p": 1}}) == \
            {"type": "tool_start", "name": "read_file", "args": {"p": 1}}
        assert to_event({"type": "tool_end", "name": "read_file", "result": 42}) == \
            {"type": "tool_end", "name": "read_file", "result": "42"}

    def test_usage(self):
        assert to_event({"type": "usage", "data": {"tokens": 5}}) == {"type": "usage", "data": {"tokens": 5}}

    def test_error_vs_notice(self):
        assert to_event({"type": "error", "content": "Provider error: boom"})["type"] == "error"
        assert to_event({"type": "error", "content": "\n[System: nudging...]"})["type"] == "notice"
        assert to_event({"type": "error", "content": "\n[Loop Guard]: stop"})["type"] == "notice"

    def test_ignored_chunks_return_none(self):
        assert to_event({"type": "tool_generating", "delta": "..."}) is None
        assert to_event({"type": "something_new"}) is None


class FakeAgent:
    def __init__(self, chunks):
        self.chunks = chunks
        self.seen = None

    def process_user_input(self, text, **kwargs):
        self.seen = text
        yield from self.chunks


class TestAgentSession:
    def test_streams_events_then_done(self):
        agent = FakeAgent([
            {"type": "content_stream", "content": "Hello "},
            {"type": "tool_start", "name": "t", "args": {}},
            {"type": "tool_end", "name": "t", "result": "ok"},
            {"type": "content_stream", "content": "world"},
        ])
        events = list(AgentSession(agent=agent).handle("hi"))
        assert agent.seen == "hi"
        assert events[-1] == {"type": "done"}
        types = [e["type"] for e in events]
        assert types == ["content", "tool_start", "tool_end", "content", "done"]

    def test_filters_ignored_chunks(self):
        agent = FakeAgent([
            {"type": "tool_generating", "delta": "x"},
            {"type": "content_stream", "content": "hi"},
        ])
        events = list(AgentSession(agent=agent).handle("q"))
        assert [e["type"] for e in events] == ["content", "done"]
