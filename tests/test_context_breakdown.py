from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent


@pytest.fixture
def make_agent(monkeypatch):
    def _make(model):
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        a = ArgentAgent()
        a.set_model(model)
        a.provider = "ollama"
        a.refresh_tier()
        return a
    return _make


class TestContextBreakdown:
    def test_components_sum_to_total(self, make_agent):
        a = make_agent("glm-4.7")  # cloud -> native tools
        b = a.get_context_breakdown()
        assert {"system", "tools", "history", "total", "max", "percent"} <= set(b)
        assert b["total"] == b["system"] + b["tools"] + b["history"]

    def test_cloud_carries_tool_schemas(self, make_agent):
        a = make_agent("glm-4.7")
        b = a.get_context_breakdown()
        assert b["tools"] > 0 and b["tool_count"] > 0

    def test_tiny_sends_no_native_schemas(self, make_agent):
        # Tiny uses an in-prompt catalog, not native schemas -> 0 schema tokens.
        a = make_agent("qwen2.5:1.5b")
        b = a.get_context_breakdown()
        assert b["tools"] == 0 and b["tool_count"] == 0

    def test_small_tool_count_is_reduced(self, make_agent):
        # Small (3-7B, here 3b) keeps native tools but slimmed to the core
        # profile; cloud keeps the full set.
        small = make_agent("qwen2.5:3b").get_context_breakdown()
        large = make_agent("glm-4.7").get_context_breakdown()
        assert 0 < small["tool_count"] < large["tool_count"]
