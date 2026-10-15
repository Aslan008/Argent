import pytest

from tool_profiles import slim_tools_for_category, is_slim_category, CORE_CHAT_TOOLS


ALL = [
    "read_file", "write_file", "run_command", "grep_search", "calculate",
    "browser_open", "browser_click", "call_mcp_tool", "create_skill",
    "git_checkpoint", "run_subagent", "analyze_project",
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
            assert "create_skill" not in slim

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
                  "grep_search", "ask_user_questions", "wait_heartbeat"):
            assert t in CORE_CHAT_TOOLS

    def test_core_excludes_heavy(self):
        for t in ("browser_open", "call_mcp_tool", "create_skill",
                  "git_rollback", "run_subagent"):
            assert t not in CORE_CHAT_TOOLS


class TestModeTools:
    def test_wait_heartbeat_available_in_chat_tools_for_small(self):
        from tool_profiles import get_mode_tools
        from tools.schemas import AVAILABLE_TOOLS
        assert "end_auto_mode" not in AVAILABLE_TOOLS

        tools = get_mode_tools("chat", "small")
        assert "wait_heartbeat" in tools
        assert "read_file" in tools
        assert "end_auto_mode" not in tools

    def test_slim_tools_includes_wait_heartbeat(self):
        from tool_profiles import slim_tools_for_category
        all_candidates = ["read_file", "wait_heartbeat", "browser_open"]
        slim_chat = slim_tools_for_category(all_candidates, "small")
        assert "read_file" in slim_chat
        assert "wait_heartbeat" in slim_chat
        assert "browser_open" not in slim_chat


class TestSchemaMinification:
    def test_minify_tool_schema_preserves_structure(self):
        from tool_profiles import minify_tool_schema
        full_schema = {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": "Replaces a specific block of text in a file with new content. This is much safer than write_file for large files because it avoids wiping the rest of the file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to be edited. Must be an existing file."
                        },
                        "target_text": {
                            "type": "string",
                            "description": "The exact character-sequence to be replaced, including indentation."
                        },
                        "replacement_text": {
                            "type": "string",
                            "description": "The new content that will replace target_text."
                        }
                    },
                    "required": ["file_path", "target_text", "replacement_text"]
                }
            }
        }
        minified = minify_tool_schema(full_schema)
        assert minified["type"] == "function"
        assert minified["function"]["name"] == "replace_in_file"
        assert minified["function"]["parameters"]["required"] == ["file_path", "target_text", "replacement_text"]
        assert set(minified["function"]["parameters"]["properties"].keys()) == {"file_path", "target_text", "replacement_text"}
        # Description is shortened to first sentence
        assert len(minified["function"]["description"]) < len(full_schema["function"]["description"])
        assert minified["function"]["description"] == "Replaces a specific block of text in a file with new content."

    def test_minify_tool_schemas_list(self):
        from tool_profiles import minify_tool_schemas
        schemas = [
            {"type": "function", "function": {"name": "test1", "description": "Desc 1. Extra info.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "test2", "description": "Desc 2. Extra info.", "parameters": {"type": "object", "properties": {}}}}
        ]
        minified_list = minify_tool_schemas(schemas)
        assert len(minified_list) == 2
        assert minified_list[0]["function"]["name"] == "test1"
        assert minified_list[0]["function"]["description"] == "Desc 1."
        assert minified_list[1]["function"]["name"] == "test2"
        assert minified_list[1]["function"]["description"] == "Desc 2."

