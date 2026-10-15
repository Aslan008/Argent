"""Tests for direct browser tools integration for the primary agent."""

import pytest
from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS, get_available_tools, get_tool_schemas
from main import chat_allowed_tools
from tool_profiles import slim_tools_for_category, CORE_CHAT_TOOLS

BROWSER_TOOLS = [
    "browser_open",
    "browser_action",
    "browser_inspect",
    "browser_close",
]


class TestBrowserToolsDirect:
    def test_all_browser_tools_registered_in_available_tools(self):
        for tool_name in BROWSER_TOOLS:
            assert tool_name in AVAILABLE_TOOLS, f"Tool '{tool_name}' missing from AVAILABLE_TOOLS"
            assert callable(AVAILABLE_TOOLS[tool_name]), f"Tool '{tool_name}' is not callable"

    def test_all_browser_tools_have_schemas(self):
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        for tool_name in BROWSER_TOOLS:
            assert tool_name in schema_names, f"Tool '{tool_name}' missing from TOOL_SCHEMAS"

    def test_run_browser_task_removed_from_advertised_tools(self):
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert "run_browser_task" not in AVAILABLE_TOOLS
        assert "run_browser_task" not in schema_names

    def test_chat_allowed_tools_includes_browser_tools(self):
        allowed = chat_allowed_tools()
        for tool_name in BROWSER_TOOLS:
            assert tool_name in allowed, f"Tool '{tool_name}' missing from chat_allowed_tools()"

    def test_slim_models_exclude_browser_tools(self):
        for cat in ("tiny", "small"):
            slim = slim_tools_for_category(BROWSER_TOOLS, cat)
            assert not any(t in slim for t in BROWSER_TOOLS), f"{cat} model should exclude browser tools"
        for t in BROWSER_TOOLS:
            assert t not in CORE_CHAT_TOOLS

    def test_standard_models_include_browser_tools(self):
        for cat in ("medium", "large", "cloud"):
            tools = slim_tools_for_category(BROWSER_TOOLS, cat)
            assert tools == BROWSER_TOOLS, f"{cat} model should keep all browser tools"

    def test_browser_tools_schemas_structure(self):
        browser_schemas = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] in BROWSER_TOOLS}
        assert len(browser_schemas) == 4

        # Verify browser_open requires url
        assert "url" in browser_schemas["browser_open"]["parameters"]["required"]

        # Verify browser_action requires action and has valid actions enum
        assert "action" in browser_schemas["browser_action"]["parameters"]["required"]
        action_enum = browser_schemas["browser_action"]["parameters"]["properties"]["action"]["enum"]
        for expected in ("click", "type", "select", "scroll", "press_key", "upload_file", "act"):
            assert expected in action_enum

        # Verify browser_inspect mode enum
        mode_enum = browser_schemas["browser_inspect"]["parameters"]["properties"]["mode"]["enum"]
        for expected in ("dom_interactive", "screenshot", "console_errors", "text_content", "html", "observe", "extract"):
            assert expected in mode_enum

        # Verify browser_close is registered
        assert "browser_close" in browser_schemas

