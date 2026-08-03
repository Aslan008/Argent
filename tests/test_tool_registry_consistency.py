"""The tool registry, the chat toolset and the system prompt must agree.

These three drifted apart silently once already: `append_to_file` shipped a
schema with no registered function, so the model's call fell through to the
recovery layer and was fuzzy-matched to `read_file` — the file was read, the
content to append was dropped, and nothing reported an error. Meanwhile the
prompt told the model to use `list_directory`, `create_artifact` and
`multi_replace_in_file_chunk`, none of which were in the chat toolset.

Nothing about those failures is visible at runtime, so they are pinned here.
"""

import re

import pytest


@pytest.fixture(scope="module")
def registry():
    from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
    return {
        "schemas": {t["function"]["name"] for t in TOOL_SCHEMAS},
        "functions": set(AVAILABLE_TOOLS),
    }


class TestSchemaFunctionParity:
    def test_every_schema_has_a_function(self, registry):
        """A schema without an implementation is advertised to the model and
        then silently rewritten by the recovery layer when called."""
        orphans = registry["schemas"] - registry["functions"]
        assert not orphans, f"schema advertised but not callable: {sorted(orphans)}"

    def test_every_function_has_a_schema(self, registry):
        """A function with no schema can never be reached by the model."""
        unreachable = registry["functions"] - registry["schemas"]
        assert not unreachable, f"callable but never advertised: {sorted(unreachable)}"


class TestChatToolset:
    def test_no_dead_names(self, registry):
        """`semantic_search` is the one legitimate exception: it is appended at
        runtime by get_tool_schemas() only when RAG is enabled."""
        from main import CHAT_ALLOWED_TOOLS
        known = registry["schemas"] | {"semantic_search"}
        dead = [n for n in CHAT_ALLOWED_TOOLS if n not in known]
        assert not dead, f"CHAT_ALLOWED_TOOLS references non-existent tools: {dead}"

    def test_no_duplicates(self):
        from main import CHAT_ALLOWED_TOOLS
        dupes = {n for n in CHAT_ALLOWED_TOOLS if CHAT_ALLOWED_TOOLS.count(n) > 1}
        assert not dupes, f"duplicated entries: {sorted(dupes)}"


class TestPromptMatchesToolset:
    """Every tool the system prompt names must actually be sent in chat mode."""

    def test_prompt_only_references_available_tools(self, monkeypatch):
        import agent as agent_module
        from agent import ArgentAgent
        from main import CHAT_ALLOWED_TOOLS
        from tools.schemas import TOOL_SCHEMAS

        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")

        a = ArgentAgent.__new__(ArgentAgent)
        a.model_name, a.provider = "test-model", "test"
        text = a.build_system_prompt() + "\n" + (a._build_ephemeral_context() or "")

        known = {t["function"]["name"] for t in TOOL_SCHEMAS} | {"semantic_search"}
        mentioned = {w for w in re.findall(r"[a-z_]{4,}", text) if w in known}
        missing = sorted(mentioned - set(CHAT_ALLOWED_TOOLS))
        assert not missing, (
            f"system prompt instructs the model to use tools absent from the chat "
            f"toolset: {missing}"
        )


class TestRecoveryReportsSubstitutions:
    def test_rename_is_flagged(self):
        """A silent swap is indistinguishable from success — it must be marked."""
        from tool_recovery import recover_tool_call
        tools = {"read_file": lambda **k: None, "write_file": lambda **k: None}
        out = recover_tool_call(
            {"function": {"name": "read_fil", "arguments": {"file_path": "a.py"}}}, tools
        )
        assert out["function"]["name"] == "read_file"
        assert out["renamed_from"] == "read_fil"

    def test_exact_match_is_not_flagged(self):
        from tool_recovery import recover_tool_call
        tools = {"read_file": lambda **k: None}
        out = recover_tool_call(
            {"function": {"name": "read_file", "arguments": {}}}, tools
        )
        assert "renamed_from" not in out

    def test_a_write_call_never_resolves_to_a_reader(self):
        """The exact regression. A call carrying `content` is a write; resolving
        it to a tool with no `content` parameter drops the payload and returns
        a successful-looking read."""
        from tool_recovery import recover_tool_call
        from tools.schemas import AVAILABLE_TOOLS
        write_args = {"file_path": "a.py", "content": "x"}
        for foreign in ("append_to_file", "create_file", "edit_file", "str_replace"):
            out = recover_tool_call(
                {"function": {"name": foreign, "arguments": write_args}}, AVAILABLE_TOOLS
            )
            resolved = out["function"]["name"] if out else None
            assert resolved != "read_file", f"'{foreign}' was absorbed into read_file"

    def test_a_write_call_may_resolve_to_another_writer(self):
        """Signature compatibility permits the substitutions that preserve intent."""
        from tool_recovery import recover_tool_call
        from tools.schemas import AVAILABLE_TOOLS
        out = recover_tool_call(
            {"function": {"name": "write_to_file",
                          "arguments": {"file_path": "a.py", "content": "x"}}},
            AVAILABLE_TOOLS,
        )
        assert out["function"]["name"] == "write_file"
        assert out["renamed_from"] == "write_to_file"

    def test_genuine_typos_still_recover(self):
        from tool_recovery import recover_tool_call
        from tools.schemas import AVAILABLE_TOOLS
        cases = [
            ("read_fil", {"file_path": "a.py"}, "read_file"),
            ("write_files", {"file_path": "a.py", "content": "x"}, "write_file"),
            ("grep-search", {"pattern": "x"}, "grep_search"),
            ("runCommand", {"command": "ls"}, "run_command"),
            ("readfile", {"file_path": "a.py"}, "read_file"),
        ]
        for typo, args, want in cases:
            out = recover_tool_call(
                {"function": {"name": typo, "arguments": args}}, AVAILABLE_TOOLS
            )
            assert out and out["function"]["name"] == want, f"{typo} did not recover to {want}"

    def test_append_to_file_is_callable(self):
        """It shipped a schema with no function for long enough to matter."""
        from tools.schemas import AVAILABLE_TOOLS
        assert "append_to_file" in AVAILABLE_TOOLS
