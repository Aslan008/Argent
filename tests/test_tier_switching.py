"""Switching model or provider mid-session must fully swap the tier profile.

Regression: set_model used to update the strategy but keep the previous
tier's history budget, and /provider assigned agent.provider without
recomputing the strategy at all — so small-model accommodations leaked
into cloud sessions and vice versa.
"""

import pytest

import config
import agent as agent_module
from agent import ArgentAgent
from src.agent.strategy import TinyLocalStrategy, StandardLocalStrategy, CloudStrategy


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    # Neutralize any manual category override in the user's real config.
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    a = ArgentAgent()
    a.provider = "ollama"  # deterministic baseline regardless of user config
    return a


class TestModelSwitching:
    def test_tiny_to_cloud_swaps_strategy_and_history_budget(self, agent):
        agent.set_model("qwen2.5:1.5b")
        assert isinstance(agent.strategy, TinyLocalStrategy)
        tiny_budget = agent.max_history_messages

        agent.set_model("glm-4.7")
        assert isinstance(agent.strategy, CloudStrategy)
        assert agent.max_history_messages == CloudStrategy().get_max_history_messages("cloud")
        assert agent.max_history_messages > tiny_budget

    def test_cloud_to_tiny_shrinks_history_budget(self, agent):
        agent.set_model("glm-4.7")
        cloud_budget = agent.max_history_messages

        agent.set_model("llama3.2:1b")
        assert isinstance(agent.strategy, TinyLocalStrategy)
        assert agent.max_history_messages == TinyLocalStrategy().get_max_history_messages("tiny")
        assert agent.max_history_messages < cloud_budget

    def test_constrained_unsupported_flag_resets_on_model_switch(self, agent):
        agent._constrained_unsupported = True
        agent.set_model("llama3.2:1b")
        assert agent._constrained_unsupported is False


class TestProviderSwitching:
    def test_switch_to_zai_forces_cloud_strategy(self, agent):
        agent.set_model("qwen2.5:7b")
        assert isinstance(agent.strategy, StandardLocalStrategy)

        agent.set_provider("zai")
        assert isinstance(agent.strategy, CloudStrategy)
        assert agent.max_history_messages == CloudStrategy().get_max_history_messages("medium")

    def test_switch_back_to_ollama_restores_local_strategy(self, agent):
        agent.set_model("qwen2.5:7b")
        agent.set_provider("zai")
        agent.set_provider("ollama")
        assert isinstance(agent.strategy, StandardLocalStrategy)

    def test_constrained_unsupported_flag_resets_on_provider_switch(self, agent):
        agent._constrained_unsupported = True
        agent.set_provider("koboldcpp")
        assert agent._constrained_unsupported is False
