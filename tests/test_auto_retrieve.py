import sys
import types
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent


@pytest.fixture
def agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    a = ArgentAgent()
    a.messages = [{"role": "system", "content": "sys"}]
    return a


def fake_rag(enabled=True, result="Snippet 1 | Rigidbody.AddForce\n```code```"):
    m = types.ModuleType("rag_engine")
    m.is_rag_enabled = lambda: enabled
    m.semantic_search = lambda q, n_results=5: result
    return m


class TestAutoRetrieve:
    def test_off_no_injection(self, agent, monkeypatch):
        monkeypatch.setattr(config, "get_auto_retrieve", lambda: False)
        agent._maybe_auto_retrieve("how does AddForce work in Unity")
        assert len(agent.messages) == 1

    def test_on_injects_context(self, agent, monkeypatch):
        monkeypatch.setattr(config, "get_auto_retrieve", lambda: True)
        monkeypatch.setitem(sys.modules, "rag_engine", fake_rag(True))
        agent._maybe_auto_retrieve("how does AddForce work in Unity")
        injected = [m for m in agent.messages if m["role"] == "system" and "RETRIEVED CONTEXT" in m["content"]]
        assert len(injected) == 1
        assert "Rigidbody.AddForce" in injected[0]["content"]

    def test_rag_disabled_no_injection(self, agent, monkeypatch):
        monkeypatch.setattr(config, "get_auto_retrieve", lambda: True)
        monkeypatch.setitem(sys.modules, "rag_engine", fake_rag(enabled=False))
        agent._maybe_auto_retrieve("how does AddForce work in Unity")
        assert len(agent.messages) == 1

    def test_short_query_skipped(self, agent, monkeypatch):
        monkeypatch.setattr(config, "get_auto_retrieve", lambda: True)
        monkeypatch.setitem(sys.modules, "rag_engine", fake_rag(True))
        agent._maybe_auto_retrieve("hi")
        assert len(agent.messages) == 1

    def test_no_results_no_injection(self, agent, monkeypatch):
        monkeypatch.setattr(config, "get_auto_retrieve", lambda: True)
        monkeypatch.setitem(sys.modules, "rag_engine", fake_rag(True, "No relevant snippets found."))
        agent._maybe_auto_retrieve("how does AddForce work in Unity")
        assert len(agent.messages) == 1

    def test_failure_is_swallowed(self, agent, monkeypatch):
        monkeypatch.setattr(config, "get_auto_retrieve", lambda: True)
        broken = types.ModuleType("rag_engine")
        broken.is_rag_enabled = lambda: True
        def boom(*a, **k):
            raise RuntimeError("chroma down")
        broken.semantic_search = boom
        monkeypatch.setitem(sys.modules, "rag_engine", broken)
        agent._maybe_auto_retrieve("how does AddForce work in Unity")  # must not raise
        assert len(agent.messages) == 1


class TestSemanticSearchDescription:
    def test_description_lists_enabled_kbs(self, monkeypatch):
        fake = types.ModuleType("rag_engine")
        fake.is_rag_enabled = lambda: True
        monkeypatch.setitem(sys.modules, "rag_engine", fake)
        monkeypatch.setattr(config, "get_external_kbs",
                            lambda: [{"id": "unity", "name": "Unity 6.4", "enabled": True}])
        from tools.schemas import get_tool_schemas
        schemas = get_tool_schemas()
        ss = next((s for s in schemas if s["function"]["name"] == "semantic_search"), None)
        assert ss is not None
        desc = ss["function"]["description"]
        assert "Unity 6.4" in desc
        assert "documentation" in desc.lower()
