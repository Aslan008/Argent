"""Changing model or provider in the middle of a conversation.

refresh_tier's docstring says nothing from the previous tier may leak into the
new one — and then /provider assigned agent.model_name directly, skipping it.
Measured before the fix: after switching to tinyllama the agent was still
running CloudStrategy with an 80-message budget and a 256k window, so history
was never trimmed and the request overflowed.
"""

import pytest

import command_handler
from agent import ArgentAgent


@pytest.fixture
def agent(monkeypatch):
    a = ArgentAgent.__new__(ArgentAgent)
    a.messages = [{"role": "system", "content": "s"}]
    a.model_name = "minimax-m3:cloud"
    a.provider = "ollama"
    a.refresh_tier()
    return a


@pytest.fixture
def said(monkeypatch):
    lines = []
    monkeypatch.setattr(command_handler, "print_system", lambda t, *a, **k: lines.append(str(t)))
    monkeypatch.setattr(command_handler, "set_current_model", lambda m: None)
    return lines


class TestSwitchModel:
    def test_the_strategy_follows_the_model(self, agent, said):
        """The whole point: a cloud profile driving a 1B model is broken in
        every direction at once."""
        before = agent.strategy.__class__.__name__
        command_handler.switch_model(agent, "tinyllama")
        assert agent.strategy.__class__.__name__ != before

    def test_the_history_budget_follows_too(self, agent, said):
        before = agent.max_history_messages
        command_handler.switch_model(agent, "tinyllama")
        assert agent.max_history_messages < before

    def test_runtime_capability_flags_are_cleared(self, agent, said):
        """A new model may support what the previous one rejected; carrying the
        flag over silently keeps the fallback path forever."""
        agent._native_tools_unsupported = True
        agent._constrained_unsupported = True
        command_handler.switch_model(agent, "qwen3.5:9b")
        assert agent._native_tools_unsupported is False
        assert agent._constrained_unsupported is False

    def test_the_conversation_is_kept(self, agent, said):
        """The reason to switch mid-chat at all."""
        agent.messages += [{"role": "user", "content": "привет"},
                           {"role": "assistant", "content": "здравствуйте"}]
        command_handler.switch_model(agent, "qwen3.5:9b")
        assert [m["content"] for m in agent.messages[1:]] == ["привет", "здравствуйте"]

    def test_a_tier_change_is_announced(self, agent, said):
        command_handler.switch_model(agent, "tinyllama")
        assert any("cloud" in line and "tiny" in line for line in said)

    def test_the_same_tier_is_not_announced(self, agent, said):
        command_handler.switch_model(agent, "kimi-k2.6:cloud")
        assert not any("→" in line and "cloud" in line.split("→")[0][-12:] for line in said)


class TestWarnings:
    def test_a_shrinking_history_budget_warns_with_numbers(self, agent, said):
        """The trim happens silently at the start of the next turn, and the
        model then answers as if the earlier half was never said."""
        agent.messages += [{"role": "user", "content": f"m{i}"} for i in range(60)]
        command_handler.switch_model(agent, "tinyllama")
        warning = " ".join(said)
        assert "Бюджет истории" in warning and "60" in warning

    def test_no_warning_when_the_history_still_fits(self, agent, said):
        agent.messages += [{"role": "user", "content": "одно сообщение"}]
        command_handler.switch_model(agent, "tinyllama")
        assert not any("Бюджет истории" in line for line in said)

    def test_a_shrinking_context_window_warns(self, agent, monkeypatch, said):
        # agent.py binds the getter at import time, so patch it there.
        import agent as agent_module
        monkeypatch.setattr(agent_module, "get_context_window", lambda: 4096)
        command_handler.switch_model(agent, "qwen3.5:9b")
        assert any("Окно контекста" in line for line in said)

    def test_growing_limits_are_not_a_warning(self, agent, said):
        agent.max_history_messages = 5
        agent.max_context_tokens = 2048
        command_handler.switch_model(agent, "minimax-m3:cloud")
        assert not any("⚠" in line for line in said)


class TestNeverAssignDirectly:
    def test_no_call_site_sets_model_name_by_hand(self):
        """/provider used to do exactly this, which is how the tier leaked."""
        import re
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for name in ("command_handler.py", "main.py"):
            src = (root / name).read_text(encoding="utf-8")
            assert not re.search(r"agent\.model_name\s*=", src), (
                f"{name} assigns agent.model_name directly; use set_model/switch_model "
                f"so refresh_tier runs")
