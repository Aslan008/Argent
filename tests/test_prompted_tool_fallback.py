"""A cloud model with no native tool-use endpoint (e.g. an OpenRouter free
model → 'No endpoints found that support tool use') must not crash the turn:
Argent switches to prompted tool-calling (in-context catalog + raw-JSON) so the
model can still USE tools, not just answer in prose."""

from itertools import islice
from unittest.mock import MagicMock

import pytest

import agent as agent_module
from agent import ArgentAgent
from providers import ProviderError
from src.agent.strategy import CloudStrategy

HANG_CAP = 500


class ToolRejectingProvider:
    """Raises a 'no tool use' error when native tools are sent; otherwise serves
    scripted chunk lists (one per tools=None call)."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.no_tool_calls = 0
        self.rejected = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        if kwargs.get("tools"):
            self.rejected += 1
            raise ProviderError("HTTP 404: No endpoints found that support tool use.")
        script = self.scripts[min(self.no_tool_calls, len(self.scripts) - 1)]
        self.no_tool_calls += 1
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


@pytest.fixture
def make_agent(monkeypatch):
    def _make(provider):
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        a = ArgentAgent()
        a.model_name = "gpt-4o"
        a.strategy = CloudStrategy()  # native tools, no constrained decoding, no anchor
        a._native_tools_unsupported = False
        return a
    return _make


def consume(gen):
    chunks = list(islice(gen, HANG_CAP))
    assert next(gen, None) is None, "generator did not terminate"
    return chunks


def test_switches_to_prompted_and_answers(make_agent):
    provider = ToolRejectingProvider([[{"content": "Hi, no tools needed."}]])
    agent = make_agent(provider)
    chunks = consume(agent.process_user_input("hello"))
    errors = [c["content"] for c in chunks if c["type"] == "error"]
    assert any("no native tool support" in e for e in errors)
    assert agent._native_tools_unsupported is True
    content = "".join(c["content"] for c in chunks if c["type"] == "content_stream")
    assert "no tools needed" in content
    assert provider.rejected == 1  # rejected once, then retried without native tools


def test_tools_work_via_prompted_calling(make_agent):
    provider = ToolRejectingProvider([
        [{"content": '{"name": "calculate", "arguments": {"expression": "40 + 2"}}'}],
        [{"content": "The answer is 42."}],
    ])
    agent = make_agent(provider)
    chunks = consume(agent.process_user_input("what is 40+2"))
    tool_ends = [c for c in chunks if c["type"] == "tool_end"]
    assert any("42" in str(c.get("result", "")) for c in tool_ends)
    content = "".join(c["content"] for c in chunks if c["type"] == "content_stream")
    assert "42" in content
