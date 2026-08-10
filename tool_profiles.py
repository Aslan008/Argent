"""
Tier-based tool profiles.

The full tool schema set (~52 tools) costs ~6.5k tokens in every request and
dilutes a weak model's tool selection. Small/tiny models are restricted to a
core toolset — the essentials a coding agent actually needs — which cuts the
schema budget roughly to a third and sharpens tool choice. Larger models keep
the full set. This only narrows what's advertised in the prompt; execution of
any tool the model still manages to call is unaffected.
"""

# Essentials for a coding agent: file editing, search, command execution,
# background processes, user interaction and a couple of utilities. Excludes
# browser automation, MCP, SVG, skill/plugin CRUD, project-brain
# tools and system_ops — all rarely usable by a 3-7B model and heavy on tokens.
CORE_CHAT_TOOLS = {
    # files
    "read_file", "write_file", "append_to_file", "delete_file",
    "replace_in_file", "multi_replace_in_file", "get_file_outline",
    "list_directory", "create_directory", "move_file",
    # search
    "grep_search", "search_files", "search_web", "read_webpage",
    "semantic_search",
    # execution
    "run_command", "start_background_command", "read_background_command",
    "stop_background_command", "list_background_commands",
    # interaction & utilities
    "ask_user_questions", "calculate", "set_goal", "analyze_project",
    # Memory across scheduled runs. Kept in the core set because an automation
    # that declared no toolset falls back to exactly this list, and without it a
    # weak model re-reports the same findings every single run.
    "filter_new_items",
}

# Categories that get the slim profile (they benefit most from a tight prompt).
_SLIM_CATEGORIES = {"tiny", "small"}

# Only worth their tokens once something is actually plugged in. A fixed core
# set cannot express "useful here, dead weight there", and a weak model that
# silently loses its only door to a configured Unity editor is worse off than
# one carrying two extra schemas.
_CONDITIONAL_CORE = {
    ("call_mcp_tool", "list_mcp_tools"): lambda: bool(__import__(
        "config").get_mcp_servers()),
}


def core_tools_now():
    """CORE_CHAT_TOOLS plus whatever this environment makes worth having."""
    tools = set(CORE_CHAT_TOOLS)
    for names, is_relevant in _CONDITIONAL_CORE.items():
        try:
            if is_relevant():
                tools.update(names)
        except Exception:
            pass          # a broken config must not shrink the toolset
    return tools


def slim_tools_for_category(tool_names, category: str):
    """Filter a list of tool names down to the core set for weak models.
    Returns the list unchanged for medium/large/cloud."""
    if category in _SLIM_CATEGORIES:
        core = core_tools_now()
        return [n for n in tool_names if n in core]
    return list(tool_names)


def is_slim_category(category: str) -> bool:
    return category in _SLIM_CATEGORIES
