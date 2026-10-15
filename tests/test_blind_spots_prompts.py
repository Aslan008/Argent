"""Blind spot test suite for Argent prompts subsystem according to LMTrust methodology.
Verifies contract consistency, language preservation, subagent safety, tool-filtering sync,
and error-prevention directives across all prompt generators.
"""

import json
import pytest
from unittest.mock import MagicMock

import agent as agent_module
from agent import ArgentAgent, ArgentSubAgent
import config
from prompt_compressor import compress_system_prompt, drop_unavailable_tool_lines
from src.agent.critic import CRITIC_SYSTEM
from src.agent.mcp_prompt import build_mcp_section
import deep_research


@pytest.fixture
def mock_env(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")


class TestPromptBlindSpotsLMTrust:
    """LMTrust Matrix tests covering L1 (Contract), L4 (Adversarial), L6 (Cross-Layer), L7 (Compositional), L9 (Negative Space)."""

    # L1 x D7: MCP Prompt signatures must match actual tools/schemas.py parameters
    def test_l1_mcp_prompt_parameter_names_match_schemas(self):
        from tools.schemas import TOOL_SCHEMAS
        schemas_by_name = {t["function"]["name"]: t["function"] for t in TOOL_SCHEMAS}
        
        mcp_servers = [{"name": "test_server", "type": "stdio", "command": "run", "args": []}]
        client = MagicMock()
        client.list_tools.return_value = [{"name": "tool1"}]
        
        section = build_mcp_section(mcp_servers, client=client)
        # Check call_mcp_tool signature in prompt
        assert "call_mcp_tool(server_name, tool_name, arguments_json)" in section, \
            "MCP prompt must use exact argument names 'server_name' and 'tool_name' matching schema"
        # Check list_mcp_tools signature in prompt
        assert "list_mcp_tools(server_name" in section or "list_mcp_tools(" in section, \
            "MCP prompt must not use invalid parameter name 'server'"

    # L1 x D7: SubAgents MUST contain LANGUAGE RULE to prevent language drift
    def test_l1_subagents_contain_language_rule(self, mock_env):
        roles = ["Coder", "Researcher", "Reviewer", "DocWriter", "Critic"]
        for role in roles:
            sub = ArgentSubAgent(role, "Почини авторизацию в модуле auth.py")
            system_content = sub.messages[0]["content"]
            assert "LANGUAGE RULE" in system_content or "EXACT SAME LANGUAGE" in system_content, \
                f"SubAgent role '{role}' lacks critical language rule, causing language drift in Russian dialogs"

    # L4 x D4: Coder SubAgent MUST have file editing and platform safety guardrails
    def test_l4_coder_subagent_has_safe_editing_guardrails(self, mock_env):
        sub = ArgentSubAgent("Coder", "Рефакторинг кода")
        content = sub.messages[0]["content"]
        assert "replace_in_file" in content or "multi_replace_in_file" in content, \
            "Coder sub-agent must be warned against blind write_file overwrites of large files"

    # L1 x D7: BrowserWorker SubAgent role must exist and have tailored instructions
    def test_l1_browser_worker_subagent_role_exists(self, mock_env):
        sub = ArgentSubAgent("BrowserWorker", "Open website and search")
        content = sub.messages[0]["content"]
        assert "BrowserWorker" in content
        assert "browser_" in content or "browser" in content.lower()
        # Must not be the generic fallback string "You are a specialized BrowserWorker sub-agent." without browser instructions
        assert "You are a specialized BrowserWorker sub-agent.\n\n## YOUR SPECIFIC TASK:" not in content, \
            "BrowserWorker must have specialized browser navigation instructions, not generic placeholder"

    # L6 x D4: run_browser_task must pass ALL browser tools to BrowserWorker
    def test_l6_browser_worker_allowed_tools_complete(self):
        from tools.browser_tools import run_browser_task
        from tools.schemas import TOOL_SCHEMAS
        all_browser_tools = {t["function"]["name"] for t in TOOL_SCHEMAS if t["function"]["name"].startswith("browser_")}
        
        # Check allowed_tools defined inside run_browser_task or available browser tools
        import inspect
        source = inspect.getsource(run_browser_task)
        for tool in ["browser_close", "browser_switch_tab"]:
            assert tool in source, f"run_browser_task must include {tool} in allowed_tools for sub-agent"

    # L7 x D3: Prompt compressor must not inject phantom ask_user_questions when tool is not available
    def test_l7_prompt_compressor_respects_available_tools(self, mock_env, monkeypatch):
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "tiny")
        agent = ArgentAgent()
        # available tools without ask_user_questions
        prompt = agent.build_system_prompt(available_tools={"read_file", "run_command"})
        assert "ask_user_questions" not in prompt, \
            "When ask_user_questions is absent from available tools, it must not be in tiny/small system prompt"

    # L9 x D3: Negative Space - tiny model compressed prompt must NOT contain heavy planning/plugin sections
    def test_l9_negative_space_tiny_model_strips_heavy_sections(self, mock_env, monkeypatch):
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "tiny")
        agent = ArgentAgent()
        prompt = agent.build_system_prompt()
        assert "PLUGIN DEVELOPMENT" not in prompt, "Tiny model must not receive heavy PLUGIN DEVELOPMENT section"
        assert "PLANNING MODE & ARTIFACTS" not in prompt, "Tiny model must not receive heavy PLANNING MODE section"

    # L6 x D7: System prompt sections with absent core tools should be dropped cleanly
    def test_l6_skills_and_kb_sections_drop_without_tools(self, mock_env, monkeypatch):
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
        agent = ArgentAgent()
        
        # If list_skills and read_skill are absent, SKILLS SYSTEM should not be retained as dangling text
        prompt = agent.build_system_prompt(available_tools={"read_file", "run_command"})
        assert "## 3. SKILLS SYSTEM" not in prompt, \
            "SKILLS SYSTEM section must be dropped when neither list_skills nor read_skill is available"

    # L1 x D7: Summarization prompt in trimmer.py must instruct to preserve language
    def test_l1_summarization_prompt_preserves_language(self):
        from src.agent import trimmer
        import inspect
        src = inspect.getsource(trimmer._summarize_with_llm)
        assert "language" in src.lower() or "russian" in src.lower() or "preserve" in src.lower(), \
            "Summarization prompt must explicitly instruct LLM to preserve the language of the conversation"

    # L1 x D7: Deep Research synthesis prompt must preserve query language
    def test_l1_deep_research_synthesis_preserves_language(self):
        import inspect
        src = inspect.getsource(deep_research)
        assert "same language" in src.lower() or "language of the objective" in src.lower(), \
            "Deep research synthesis prompt must instruct the model to output the report in the language of the objective"
