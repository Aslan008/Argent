"""What the model is told must name tools it can call.

Two sentences in the system prompt said "use replace_in_file or
multi_replace_in_file" after multi_replace_in_file was folded into
replace_in_file: the most important editing guardrail — never overwrite a large
file with write_file — pointed the model at a tool that no longer exists, so it
emitted the call, got "not available", and the recovery layer had to guess. The
same renames left schema descriptions telling the model to "use
browser_wait_for", gone as well, exactly when it is waiting for a page; and
hints inside tool results sent it to browser_state / browser_click at the
moments it was already stuck.
"""

import re
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS

# Names that existed once and are not tools any more. The model must not be sent
# to any of them.
REMOVED_TOOLS = {
    "multi_replace_in_file", "check_code", "find_definition", "find_references",
    "find_implementations", "search_workspace_symbols", "get_call_hierarchy",
    "browser_state", "browser_click", "browser_input", "browser_select",
    "browser_screenshot", "browser_scroll", "browser_wait_for", "browser_get_content",
    "browser_switch_tab", "browser_press_key", "browser_fill_form",
    "browser_upload_file", "browser_back", "browser_reload", "browser_eval_js",
    "browser_act", "browser_extract", "browser_observe",
    "view_file", "view_symbol_definition", "add_work_task",
}
_NAME = re.compile(r"\b(" + "|".join(sorted(REMOVED_TOOLS, key=len, reverse=True)) + r")\b")


def _walk_text(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_text(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_text(value)


class TestSchemas:
    def test_no_description_sends_the_model_to_a_removed_tool(self):
        offenders = []
        for schema in TOOL_SCHEMAS:
            name = schema["function"]["name"]
            for text in _walk_text(schema["function"]):
                for removed in set(_NAME.findall(text)):
                    offenders.append(f"{name} mentions {removed}")
        assert not offenders, offenders

    def test_the_browser_wait_advice_names_a_real_action(self):
        """wait_heartbeat's description is where a model learns how to wait for a page."""
        heartbeat = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "wait_heartbeat")
        text = heartbeat["function"]["description"]
        assert 'browser_action(action="wait_for")' in text


@pytest.fixture
def make_agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)

    def _make(category):
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: category)
        return ArgentAgent()
    return _make


CATEGORIES = ["tiny", "small", "medium", "large", "cloud"]


class TestSystemPrompt:
    @pytest.mark.parametrize("category", CATEGORIES)
    def test_it_names_no_removed_tool(self, make_agent, category):
        prompt = make_agent(category).build_system_prompt()
        assert not set(_NAME.findall(prompt)), set(_NAME.findall(prompt))

    @pytest.mark.parametrize("category", CATEGORIES)
    def test_the_large_file_editing_rule_survives_with_the_real_toolset(self, make_agent, category):
        """With the toolset passed — as at run time — the rule must still be there."""
        agent = make_agent(category)
        prompt = agent.build_system_prompt(available_tools=set(AVAILABLE_TOOLS))
        assert "NEVER use write_file to overwrite existing large files" in prompt
        assert "replace_in_file" in prompt

    def test_the_sub_agent_coder_prompt_names_real_tools(self):
        source = open(agent_module.__file__, encoding="utf-8").read()
        assert not _NAME.findall(source.split('"Coder": (', 1)[1].split('"Researcher"', 1)[0])
