"""The system prompt is derived from the toolset, not written alongside it.

Prompt sections and tool schemas used to be assembled independently, so the
prompt kept instructing the model to call tools the request never carried
(`create_artifact`, `list_directory`, `create_plugin` when disabled in config).
The model then emitted names that did not exist and the recovery layer guessed.

Also pins that a sub-agent keeps its role prompt: the inherited refresh used to
overwrite messages[0] on the sub-agent's first turn, dissolving every role.
"""

from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent, ArgentSubAgent
from prompt_compressor import drop_unavailable_tool_lines


@pytest.fixture
def agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
    return ArgentAgent()


class TestLineFilter:
    def test_drops_a_line_naming_an_absent_tool(self):
        text = "## X\n- Use `create_artifact` for plans.\n- Use `read_file` to read.\n"
        out = drop_unavailable_tool_lines(text, {"read_file"})
        assert "create_artifact" not in out
        assert "read_file" in out

    def test_keeps_neighbouring_rules_intact(self):
        """Filtering by section would have thrown away the valid rules too."""
        text = "## X\n- a: `read_file`\n- b: `create_artifact`\n- c: `write_file`\n"
        out = drop_unavailable_tool_lines(text, {"read_file", "write_file"})
        assert "- a:" in out and "- c:" in out and "- b:" not in out

    def test_heading_with_nothing_left_under_it_is_dropped(self):
        text = "## ONLY PLUGINS\n- Use `create_plugin`.\n- Then `delete_plugin`.\n"
        assert drop_unavailable_tool_lines(text, {"read_file"}).strip() == ""

    def test_prose_without_tool_names_survives(self):
        text = "## RULES\n- Think before acting.\n- Be concise.\n"
        out = drop_unavailable_tool_lines(text, {"read_file"})
        assert "Think before acting." in out and "Be concise." in out

    def test_empty_available_set_is_a_noop(self):
        """None/empty means 'caller does not know' — never strip on a guess."""
        text = "- Use `create_artifact`.\n"
        assert drop_unavailable_tool_lines(text, set()) == text


class TestPromptFollowsToolset:
    def test_absent_tools_are_not_advertised(self, agent):
        available = {"read_file", "write_file", "run_command", "ask_user_questions"}
        prompt = agent.build_system_prompt(available)
        for gone in ("create_artifact", "request_user_approval", "create_plugin",
                     "create_svg_image", "list_directory"):
            assert gone not in prompt, f"prompt still instructs the model to use {gone}"

    def test_available_tools_are_still_advertised(self, agent):
        prompt = agent.build_system_prompt(agent.effective_tool_names(None))
        assert "read_file" in prompt

    def test_none_keeps_everything(self, agent):
        """Diagnostics ask for the full text; only an explicit set filters."""
        full = agent.build_system_prompt(None)
        assert "## 1. OPERATIONAL PROTOCOL" in full

    def test_mcp_section_skipped_without_call_mcp_tool(self, agent, monkeypatch):
        monkeypatch.setattr(
            agent_module, "get_mcp_servers",
            lambda: [{"name": "srv", "type": "stdio", "command": "x", "args": []}],
        )
        with_tool = agent.build_system_prompt({"read_file", "call_mcp_tool"})
        without = agent.build_system_prompt({"read_file"})
        assert "MCP SERVERS" in with_tool
        assert "MCP SERVERS" not in without

    def test_section_about_a_tool_goes_when_the_tool_does(self, agent):
        """Line filtering leaves the plugin section's hook paths and event names
        behind — prose for work the model can no longer do."""
        without = agent.build_system_prompt({"read_file", "run_command"})
        assert "PLUGIN DEVELOPMENT" not in without
        assert "HOOKS_DIR" not in without

    def test_that_section_stays_when_its_tool_is_present(self, agent):
        prompt = agent.build_system_prompt(
            {"read_file", "create_plugin", "delete_plugin"}
        )
        assert "PLUGIN DEVELOPMENT" in prompt

    def test_effective_names_respect_the_mode_allowlist(self, agent):
        names = agent.effective_tool_names(["read_file", "write_file"])
        assert names <= {"read_file", "write_file"}

    def test_prompt_is_stable_across_rebuilds(self, agent):
        """Deriving from the toolset must not churn the prefix cache."""
        allowed = ["read_file", "write_file", "run_command"]
        assert agent._refresh_system_prompt(allowed) is True     # first build differs
        assert agent._refresh_system_prompt(allowed) is False    # then stable
        assert agent._refresh_system_prompt(allowed) is False


class TestTierLadderPointsBothWays:
    """The ladder used to only shorten downward: tiny/small got a trimmed
    prompt and everyone else got the maximum, so the most capable model
    received the most guardrails — the opposite of what helps it."""

    def _core(self, monkeypatch, category):
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda n: category)
        monkeypatch.setattr(config, "get_model_category_override", lambda: None)
        a = ArgentAgent.__new__(ArgentAgent)
        a.model_name, a.provider = "m", "p"
        a.refresh_tier()
        return a.build_system_prompt(None).split("## GLOBAL INSTRUCTIONS")[0]

    def test_strong_model_gets_a_shorter_prompt_than_mid(self, monkeypatch):
        cloud = self._core(monkeypatch, "cloud")
        medium = self._core(monkeypatch, "medium")
        assert len(cloud) < len(medium)

    def test_honesty_requirement_survives_trimming(self, monkeypatch):
        """Everything else in the verify section describes what a capable model
        already does; this one holds it to not claiming what it did not see."""
        for cat in ("cloud", "large", "medium"):
            assert "Prove It" in self._core(monkeypatch, cat)

    def test_environment_facts_survive_trimming(self, monkeypatch):
        """Exit code 0 lying about success is not something a model can infer."""
        assert "False Success" in self._core(monkeypatch, "cloud")

    def test_language_rule_is_stated_once(self, monkeypatch):
        """It was stated at the top and then pointed at again from section 7 —
        a second mention that added nothing."""
        for cat in ("cloud", "medium", "small"):
            assert self._core(monkeypatch, cat).count("LANGUAGE RULE") == 1

    def test_response_length_is_requested_explicitly(self, monkeypatch):
        """Effort controls reasoning, not output length — it has to be asked for."""
        assert "concise" in self._core(monkeypatch, "cloud")


class TestSubAgentKeepsItsRole:
    def test_role_prompt_survives_a_refresh(self, monkeypatch):
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
        sub = ArgentSubAgent("Critic", "review this")
        before = sub.messages[0]["content"]
        assert sub._refresh_system_prompt([]) is False
        assert sub.messages[0]["content"] == before

    def test_critic_system_actually_reaches_the_model(self, monkeypatch):
        """test_critic.py checks the CRITIC_SYSTEM text; this checks it arrives."""
        from src.agent.critic import CRITIC_SYSTEM
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
        sub = ArgentSubAgent("Critic", "review this")
        sub._refresh_system_prompt([])
        assert CRITIC_SYSTEM[:60] in sub.messages[0]["content"]
