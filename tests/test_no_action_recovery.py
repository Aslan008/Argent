"""A model that ends its turn with neither an answer nor a tool call (common
with reasoning models that 'think' then stop) must be nudged to continue —
bounded — instead of the turn silently dead-ending (agent.py `else: break`)."""

from itertools import islice
from unittest.mock import MagicMock

import pytest

import agent as agent_module
from agent import ArgentAgent, MAX_NO_ACTION_CONTINUES


class FakeProvider:
    """Scripted provider: yields one chunk list per stream_chat call."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


HANG_CAP = 500


@pytest.fixture
def make_agent(monkeypatch):
    def _make(provider):
        fake_memory = MagicMock()
        fake_memory.data = {}
        monkeypatch.setattr(agent_module, "memory", fake_memory)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        monkeypatch.setattr(agent_module, "estimate_tokens",
                            lambda text, model, prov: len(text) // 4)
        return ArgentAgent()
    return _make


def consume(gen):
    chunks = list(islice(gen, HANG_CAP))
    assert next(gen, None) is None, "generator did not terminate"
    return chunks


THINKING_ONLY = [{"thinking": "Let me think...", "content": "", "tool_call_deltas": []}]
ANSWER = [{"content": "Here is the final answer.", "thinking": "", "tool_call_deltas": []}]


class TestNoActionRecovery:
    def test_thinking_only_is_nudged_then_stops(self, make_agent):
        agent = make_agent(FakeProvider([THINKING_ONLY]))  # always only-thinking
        chunks = consume(agent.process_user_input("do something"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        nudges = [e for e in errors if "nudging the model" in e]
        assert len(nudges) == MAX_NO_ACTION_CONTINUES
        assert any("no answer and no action after" in e for e in errors)
        user_msgs = [m["content"] for m in agent.messages if m.get("role") == "user"]
        assert any("CALL A TOOL" in m for m in user_msgs)

    def test_thinking_then_answer_completes(self, make_agent):
        agent = make_agent(FakeProvider([THINKING_ONLY, ANSWER]))
        chunks = consume(agent.process_user_input("question"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        assert sum("nudging the model" in e for e in errors) == 1
        assert not any("no answer and no action after" in e for e in errors)
        content = "".join(c["content"] for c in chunks if c["type"] == "content_stream")
        assert "final answer" in content

    def test_direct_answer_is_not_nudged(self, make_agent):
        agent = make_agent(FakeProvider([ANSWER]))
        chunks = consume(agent.process_user_input("question"))
        errors = [c["content"] for c in chunks if c["type"] == "error"]
        assert not any("nudging" in e for e in errors)
