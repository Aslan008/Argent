"""KV/prefix-cache friendliness: a stable system prompt, volatile facts in an
ephemeral tail, and trim hysteresis (deep cut once instead of a shave per turn)."""

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
    monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
    return ArgentAgent()


class TestStableSystemPrompt:
    def test_refresh_is_noop_when_inputs_unchanged(self, agent):
        agent._refresh_system_prompt()
        before = agent.messages[0]["content"]
        assert agent._refresh_system_prompt() is False     # nothing changed
        assert agent.messages[0]["content"] is before      # not even re-assigned

    def test_refresh_replaces_on_real_change(self, agent, monkeypatch):
        agent._refresh_system_prompt()
        # build_system_prompt now takes the turn's toolset (see
        # test_prompt_toolset_sync.py) — accept and ignore it here.
        monkeypatch.setattr(agent, "build_system_prompt", lambda *_: "NEW PROMPT")
        assert agent._refresh_system_prompt() is True
        assert agent.messages[0]["content"] == "NEW PROMPT"

    def test_volatile_sections_not_in_system_prompt(self, agent):
        prompt = agent.build_system_prompt()
        assert "REPOSITORY MAP" not in prompt
        assert "BACKGROUND PROCESSES" not in prompt

    def test_volatile_sections_live_in_ephemeral_tail(self, agent):
        eph = agent._build_ephemeral_context()
        assert eph is not None
        assert "REPOSITORY MAP" in eph                     # cwd listing moved here

    def test_ephemeral_reports_background_processes(self, agent, monkeypatch):
        import tools
        monkeypatch.setitem(tools.ACTIVE_PROCESSES, "99", {"command": "x"})
        try:
            eph = agent._build_ephemeral_context()
            assert "BACKGROUND PROCESSES" in eph and "1 background process" in eph
        finally:
            tools.ACTIVE_PROCESSES.pop("99", None)


class TestTrimHysteresis:
    def _fill(self, agent, n):
        agent.messages = [{"role": "system", "content": "sys"}]
        for i in range(n):
            agent.messages.append({"role": "user", "content": f"q{i}"})
            agent.messages.append({"role": "assistant", "content": f"a{i}"})

    def test_under_limit_is_append_only(self, agent, monkeypatch):
        monkeypatch.setattr(agent, "_estimate_tokens", lambda text: 1)
        agent.max_history_messages = 20
        agent.max_context_tokens = 100000
        self._fill(agent, 5)                                # 10 msgs < 20
        before = list(agent.messages)
        agent._trim_history()
        assert agent.messages == before                     # untouched -> cache warm

    def test_over_limit_trims_deep_then_stays_stable(self, agent, monkeypatch):
        monkeypatch.setattr(agent, "_estimate_tokens", lambda text: 1)
        monkeypatch.setattr(agent_module, "get_context_window", lambda: 100000)
        agent.max_history_messages = 10
        self._fill(agent, 8)                                # 16 msgs > 10 -> trim fires

        calls = {}
        real_trim = agent.strategy.trim_history

        def spy(messages, model_name, max_msgs, budget):
            calls["max_msgs"] = max_msgs
            return real_trim(messages, model_name, max_msgs, budget)

        monkeypatch.setattr(agent.strategy, "trim_history", spy)
        agent._trim_history()
        assert calls["max_msgs"] == 6                       # 60% of 10: deep cut

        # Headroom earned: next few turns must be no-ops (append-only).
        after = list(agent.messages)
        agent.messages.append({"role": "user", "content": "next"})
        agent.messages.append({"role": "assistant", "content": "ok"})
        agent._trim_history()
        assert agent.messages[: len(after)] == after        # prefix untouched
