"""Deep mutation-killing tests for tool_profiles.

These tests complement tests/test_tool_profiles.py by focusing on the
conditional-core logic (core_tools_now, _CONDITIONAL_CORE) and by
asserting exact membership / structure so that subtle mutations — a
removed tool, a flipped lambda, a broken except branch — are caught.
"""

import pytest

import tool_profiles
from tool_profiles import (
    CORE_CHAT_TOOLS,
    _CONDITIONAL_CORE,
    core_tools_now,
    is_slim_category,
    slim_tools_for_category,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_conditional(monkeypatch):
    """Pin the conditional-core inputs so tests are deterministic.

    Without this, core_tools_now reads the developer's live config / LSP
    manager and the suite becomes machine-dependent.
    """
    monkeypatch.setattr("config.get_mcp_servers", lambda: [])
    # lsp_manager may not be importable on every machine; patch the module
    # attribute used by the lambda if the module exists.
    try:
        import src.lsp.manager as lsp_manager  # noqa: F401
        monkeypatch.setattr(lsp_manager.lsp_manager, "is_available", lambda: False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CORE_CHAT_TOOLS — exact membership
# ---------------------------------------------------------------------------

class TestCoreChatToolsMembership:
    def test_files_tools_present(self):
        for t in ("read_file", "write_file", "append_to_file", "delete_file",
                  "replace_in_file", "multi_replace_in_file", "get_file_outline",
                  "list_directory", "create_directory", "move_file"):
            assert t in CORE_CHAT_TOOLS, f"{t} should be in CORE_CHAT_TOOLS"

    def test_search_tools_present(self):
        for t in ("grep_search", "search_files", "search_web", "read_webpage",
                  "semantic_search"):
            assert t in CORE_CHAT_TOOLS

    def test_execution_tools_present(self):
        for t in ("run_command", "start_background_command",
                  "read_background_command", "stop_background_command",
                  "list_background_commands"):
            assert t in CORE_CHAT_TOOLS

    def test_interaction_utils_present(self):
        for t in ("ask_user_questions", "calculate", "set_goal",
                  "analyze_project", "filter_new_items"):
            assert t in CORE_CHAT_TOOLS

    def test_browser_tools_excluded(self):
        for t in ("browser_open", "browser_click", "browser_screenshot",
                  "run_browser_task", "view_image"):
            assert t not in CORE_CHAT_TOOLS

    def test_mcp_tools_excluded(self):
        for t in ("call_mcp_tool", "list_mcp_tools"):
            assert t not in CORE_CHAT_TOOLS

    def test_lsp_tools_excluded(self):
        for t in ("check_code", "find_implementations",
                  "search_workspace_symbols", "get_call_hierarchy"):
            assert t not in CORE_CHAT_TOOLS

    def test_system_ops_excluded(self):
        for t in ("query_registry", "read_event_logs", "get_process_info",
                  "run_admin_command", "search_system_files"):
            assert t not in CORE_CHAT_TOOLS

    def test_heavy_misc_excluded(self):
        for t in ("create_svg_image", "create_skill", "git_checkpoint",
                  "run_subagent", "run_swarm_workers", "write_project_spec",
                  "plan_work_changes"):
            assert t not in CORE_CHAT_TOOLS

    def test_is_a_set(self):
        assert isinstance(CORE_CHAT_TOOLS, set)


# ---------------------------------------------------------------------------
# _CONDITIONAL_CORE — structure
# ---------------------------------------------------------------------------

class TestConditionalCoreStructure:
    def test_is_a_dict(self):
        assert isinstance(_CONDITIONAL_CORE, dict)

    def test_keys_are_tuples_of_strings(self):
        for names in _CONDITIONAL_CORE:
            assert isinstance(names, tuple)
            assert len(names) >= 1
            for n in names:
                assert isinstance(n, str) and n

    def test_values_are_callables_returning_bool(self):
        for names, fn in _CONDITIONAL_CORE.items():
            assert callable(fn)
            result = fn()
            assert isinstance(result, bool)

    def test_contains_mcp_group(self):
        mcp_names = ("call_mcp_tool", "list_mcp_tools")
        assert mcp_names in _CONDITIONAL_CORE

    def test_contains_lsp_group(self):
        lsp_names = ("check_code", "find_implementations",
                     "search_workspace_symbols", "get_call_hierarchy")
        assert lsp_names in _CONDITIONAL_CORE

    def test_conditional_tools_not_in_core(self):
        """Conditional tools are deliberately outside the static core set."""
        for names in _CONDITIONAL_CORE:
            for n in names:
                assert n not in CORE_CHAT_TOOLS


# ---------------------------------------------------------------------------
# core_tools_now
# ---------------------------------------------------------------------------

class TestCoreToolsNow:
    def test_returns_a_set(self):
        assert isinstance(core_tools_now(), set)

    def test_contains_all_core_chat_tools(self):
        result = core_tools_now()
        assert result >= CORE_CHAT_TOOLS

    def test_specific_core_tools_present(self):
        result = core_tools_now()
        for t in ("read_file", "write_file", "run_command", "replace_in_file",
                  "grep_search", "ask_user_questions", "calculate",
                  "filter_new_items"):
            assert t in result

    def test_conditional_excluded_when_no_mcp_no_lsp(self):
        """With empty config and LSP off, conditional tools stay out."""
        result = core_tools_now()
        assert "call_mcp_tool" not in result
        assert "list_mcp_tools" not in result
        assert "check_code" not in result
        assert "find_implementations" not in result

    def test_includes_mcp_tools_when_servers_configured(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers",
                            lambda: ["unity", "github"])
        result = core_tools_now()
        assert "call_mcp_tool" in result
        assert "list_mcp_tools" in result

    def test_mcp_truthy_non_list_includes(self, monkeypatch):
        """Any truthy return from get_mcp_servers triggers inclusion."""
        monkeypatch.setattr("config.get_mcp_servers", lambda: True)
        result = core_tools_now()
        assert "call_mcp_tool" in result

    def test_mcp_empty_list_excludes(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers", lambda: [])
        result = core_tools_now()
        assert "call_mcp_tool" not in result

    def test_includes_lsp_tools_when_available(self, monkeypatch):
        import src.lsp.manager as lsp_manager
        monkeypatch.setattr(lsp_manager.lsp_manager, "is_available",
                            lambda: True)
        result = core_tools_now()
        assert "check_code" in result
        assert "find_implementations" in result
        assert "search_workspace_symbols" in result
        assert "get_call_hierarchy" in result

    def test_lsp_false_excludes(self, monkeypatch):
        import src.lsp.manager as lsp_manager
        monkeypatch.setattr(lsp_manager.lsp_manager, "is_available",
                            lambda: False)
        result = core_tools_now()
        assert "check_code" not in result

    def test_broken_config_does_not_shrink_core(self, monkeypatch):
        """If get_mcp_servers raises, core_tools_now must still return core."""
        def _boom():
            raise RuntimeError("config broken")
        monkeypatch.setattr("config.get_mcp_servers", _boom)
        result = core_tools_now()
        # Core set is intact.
        assert result >= CORE_CHAT_TOOLS
        # Conditional tools are not added (exception → pass).
        assert "call_mcp_tool" not in result

    def test_broken_lsp_does_not_shrink_core(self, monkeypatch):
        import src.lsp.manager as lsp_manager
        def _boom():
            raise RuntimeError("lsp broken")
        monkeypatch.setattr(lsp_manager.lsp_manager, "is_available", _boom)
        result = core_tools_now()
        assert result >= CORE_CHAT_TOOLS
        assert "check_code" not in result

    def test_both_conditional_on(self, monkeypatch):
        monkeypatch.setattr("config.get_mcp_servers", lambda: ["x"])
        import src.lsp.manager as lsp_manager
        monkeypatch.setattr(lsp_manager.lsp_manager, "is_available",
                            lambda: True)
        result = core_tools_now()
        assert "call_mcp_tool" in result
        assert "list_mcp_tools" in result
        assert "check_code" in result
        assert "get_call_hierarchy" in result
        # And core still fully present.
        assert result >= CORE_CHAT_TOOLS

    def test_returns_new_set_each_call(self):
        a = core_tools_now()
        a.add("__mutated__")
        b = core_tools_now()
        assert "__mutated__" not in b


# ---------------------------------------------------------------------------
# slim_tools_for_category
# ---------------------------------------------------------------------------

class TestSlimToolsForCategory:
    ALL = [
        "read_file", "write_file", "run_command", "grep_search", "calculate",
        "browser_open", "browser_click", "call_mcp_tool", "create_skill",
        "create_svg_image", "git_checkpoint", "run_subagent", "analyze_project",
        "filter_new_items", "semantic_search",
    ]

    def test_tiny_filters_to_core_only(self):
        slim = slim_tools_for_category(self.ALL, "tiny")
        core = core_tools_now()
        for n in slim:
            assert n in core

    def test_small_filters_to_core_only(self):
        slim = slim_tools_for_category(self.ALL, "small")
        core = core_tools_now()
        for n in slim:
            assert n in core

    def test_tiny_excludes_non_core(self):
        slim = slim_tools_for_category(self.ALL, "tiny")
        assert "browser_open" not in slim
        assert "call_mcp_tool" not in slim
        assert "create_svg_image" not in slim
        assert "run_subagent" not in slim

    def test_tiny_keeps_core_tools(self):
        slim = slim_tools_for_category(self.ALL, "tiny")
        assert "read_file" in slim
        assert "write_file" in slim
        assert "run_command" in slim
        assert "analyze_project" in slim

    def test_medium_passthrough(self):
        assert slim_tools_for_category(self.ALL, "medium") == self.ALL

    def test_large_passthrough(self):
        assert slim_tools_for_category(self.ALL, "large") == self.ALL

    def test_cloud_passthrough(self):
        assert slim_tools_for_category(self.ALL, "cloud") == self.ALL

    def test_medium_returns_new_list(self):
        out = slim_tools_for_category(self.ALL, "medium")
        assert out == self.ALL and out is not self.ALL

    def test_large_returns_new_list(self):
        out = slim_tools_for_category(self.ALL, "large")
        assert out == self.ALL and out is not self.ALL

    def test_preserves_order_tiny(self):
        slim = slim_tools_for_category(self.ALL, "tiny")
        core = core_tools_now()
        expected = [n for n in self.ALL if n in core]
        assert slim == expected

    def test_preserves_order_small(self):
        slim = slim_tools_for_category(self.ALL, "small")
        core = core_tools_now()
        expected = [n for n in self.ALL if n in core]
        assert slim == expected

    def test_empty_input_tiny(self):
        assert slim_tools_for_category([], "tiny") == []

    def test_empty_input_large(self):
        assert slim_tools_for_category([], "large") == []

    def test_all_non_core_tiny(self):
        non_core = ["browser_open", "create_svg_image", "run_subagent"]
        assert slim_tools_for_category(non_core, "tiny") == []

    def test_tiny_includes_conditional_when_on(self, monkeypatch):
        """slim must reflect conditional tools that are active right now."""
        monkeypatch.setattr("config.get_mcp_servers", lambda: ["unity"])
        slim = slim_tools_for_category(
            ["read_file", "call_mcp_tool", "list_mcp_tools", "browser_open"],
            "tiny")
        assert "read_file" in slim
        assert "call_mcp_tool" in slim
        assert "list_mcp_tools" in slim
        assert "browser_open" not in slim


# ---------------------------------------------------------------------------
# is_slim_category
# ---------------------------------------------------------------------------

class TestIsSlimCategory:
    def test_tiny_true(self):
        assert is_slim_category("tiny") is True

    def test_small_true(self):
        assert is_slim_category("small") is True

    def test_medium_false(self):
        assert is_slim_category("medium") is False

    def test_large_false(self):
        assert is_slim_category("large") is False

    def test_cloud_false(self):
        assert is_slim_category("cloud") is False

    def test_unknown_category_false(self):
        assert is_slim_category("enormous") is False
        assert is_slim_category("") is False

    def test_returns_bool_type(self):
        assert isinstance(is_slim_category("tiny"), bool)
        assert isinstance(is_slim_category("medium"), bool)