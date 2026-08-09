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
    """The chat toolset is DERIVED from the registry, not hand-listed.

    It used to be an allowlist of names, and every tool added after it was
    written silently vanished from ordinary chat — measured, 27 of 66,
    including move_file, find_definition, list_mcp_tools and view_image. The
    model would call a name the prompt had given it, be told "not available",
    and be offered an unrelated substitute.
    """

    def test_no_dead_names(self, registry):
        from main import chat_allowed_tools
        known = registry["schemas"] | {"semantic_search"}
        dead = [n for n in chat_allowed_tools() if n not in known]
        assert not dead, f"chat toolset references non-existent tools: {dead}"

    def test_no_duplicates(self):
        from main import chat_allowed_tools
        names = chat_allowed_tools()
        dupes = {n for n in names if names.count(n) > 1}
        assert not dupes, f"duplicated entries: {sorted(dupes)}"

    def test_a_new_tool_is_available_without_touching_a_list(self):
        """The whole point of inverting it."""
        from main import chat_allowed_tools
        for recent in ("view_image", "list_mcp_tools", "filter_new_items",
                       "move_file", "find_definition"):
            assert recent in chat_allowed_tools(), f"{recent} is invisible in chat"

    def test_mode_tools_stay_out_of_chat(self):
        """The project brain is driven by the orchestrator's state machine;
        offering its steps in conversation invites a spec nobody asked for."""
        from main import CHAT_DENIED_TOOLS, chat_allowed_tools
        available = set(chat_allowed_tools())
        assert not (CHAT_DENIED_TOOLS & available)
        assert "write_project_spec" in CHAT_DENIED_TOOLS

    def test_tools_disabled_by_the_user_are_excluded(self, monkeypatch):
        """The old list still named six tools that no longer resolved."""
        from main import chat_allowed_tools
        monkeypatch.setattr("config.get_disabled_tools", lambda: ["read_file"])
        assert "read_file" not in chat_allowed_tools()


class TestPromptMatchesToolset:
    """Every tool the system prompt names must actually be sent in chat mode."""

    def test_prompt_only_references_available_tools(self, monkeypatch):
        import agent as agent_module
        from agent import ArgentAgent
        from main import chat_allowed_tools
        from tools.schemas import TOOL_SCHEMAS

        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")

        a = ArgentAgent.__new__(ArgentAgent)
        a.model_name, a.provider = "test-model", "test"
        # Built against the SAME toolset it is compared with — that is what
        # happens at runtime, and comparing an unfiltered prompt to a filtered
        # list reports lines the model would never have been shown.
        toolset = chat_allowed_tools()
        text = (a.build_system_prompt(set(toolset)) + "\n"
                + (a._build_ephemeral_context() or ""))

        known = {t["function"]["name"] for t in TOOL_SCHEMAS} | {"semantic_search"}
        mentioned = {w for w in re.findall(r"[a-z_]{4,}", text) if w in known}
        missing = sorted(mentioned - set(toolset))
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
