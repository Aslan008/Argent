import pytest

from tool_profiles import slim_tools_for_category, is_slim_category, CORE_CHAT_TOOLS


ALL = [
    "read_file", "write_file", "run_command", "grep_search", "calculate",
    "browser_open", "browser_click", "call_mcp_tool", "create_skill",
    "create_svg_image", "git_checkpoint", "run_subagent", "analyze_project",
]


@pytest.fixture(autouse=True)
def no_mcp_servers(monkeypatch):
    """Part of the core set is conditional on what is plugged in, so pin it.

    Without this the suite passes or fails depending on whose machine it runs
    on — these tests used to read the developer's live config by accident.
    Conditional membership is covered in test_mcp_prompt.py.
    """
    monkeypatch.setattr("config.get_mcp_servers", lambda: [])


class TestSlim:
    def test_tiny_and_small_get_core_only(self):
        for cat in ("tiny", "small"):
            slim = slim_tools_for_category(ALL, cat)
            assert "read_file" in slim and "run_command" in slim
            # Heavy/rare tools are dropped.
            assert "browser_open" not in slim
            assert "call_mcp_tool" not in slim
            assert "create_svg_image" not in slim

    def test_medium_large_cloud_keep_everything(self):
        for cat in ("medium", "large", "cloud"):
            assert slim_tools_for_category(ALL, cat) == ALL

    def test_preserves_order(self):
        slim = slim_tools_for_category(ALL, "small")
        assert slim == [n for n in ALL if n in CORE_CHAT_TOOLS]

    def test_empty(self):
        assert slim_tools_for_category([], "tiny") == []

    def test_returns_new_list_for_large(self):
        out = slim_tools_for_category(ALL, "cloud")
        assert out == ALL and out is not ALL


class TestIsSlim:
    def test_flags(self):
        assert is_slim_category("tiny") and is_slim_category("small")
        assert not is_slim_category("medium")
        assert not is_slim_category("cloud")


class TestCoreSet:
    def test_core_has_essentials(self):
        for t in ("read_file", "write_file", "replace_in_file", "run_command",
                  "grep_search", "ask_user_questions"):
            assert t in CORE_CHAT_TOOLS

    def test_core_excludes_heavy(self):
        for t in ("browser_open", "call_mcp_tool", "create_svg_image",
                  "query_registry", "run_subagent"):
            assert t not in CORE_CHAT_TOOLS
