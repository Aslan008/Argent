from argent.tools._helpers import (
    _resolve_path,
    _is_plugin_path_restricted,
    _validate_code_syntax,
    _print_diff,
    _build_match_hint,
    log,
)
from argent.tools.command_ops import (
    ACTIVE_PROCESSES,
    ACTIVE_PROCESSES_LOCK,
)
from argent.tools.schemas import (
    TOOL_SCHEMAS,
    AVAILABLE_TOOLS,
    get_tool_schemas,
    get_available_tools,
)


def delegate_local(*args, **kwargs):
    from argent.local_inference import delegate_local as _dl
    return _dl(*args, **kwargs)


def run_deep_research(*args, **kwargs):
    from argent.deep_research import run_deep_research as _rdr
    return _rdr(*args, **kwargs)


def __getattr__(name: str):
    if name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[name]
    raise AttributeError(f"module 'tools' has no attribute '{name}'")


#: The taxonomy `/help` and `docs/TOOLS.md` group by. A tool belongs to exactly one
#: group: the noun a person would look it up under, not the module it happens to live in.
#: Sixty flat names are a wall; ten groups with five names each are a map.
TOOL_GROUPS: dict[str, frozenset] = {
    "files": frozenset({
        "read_file", "write_file", "replace_in_file", "delete_file", "move_file",
        "copy_file", "create_directory", "list_directory", "get_file_outline",
        "run_deep_linter",
    }),
    "search": frozenset({
        "grep_search", "search_files", "analyze_project", "semantic_search",
    }),
    "processes": frozenset({
        "run_command", "start_background_command",
        "read_background_command", "send_background_command",
        "stop_background_command", "list_background_commands", "read_git_diff",
        "git_checkpoint", "git_rollback",
    }),
    "agents": frozenset({
        "run_subagent", "subagents", "ultracode", "rooms", "delegate_to_leeth",
        "leeth_map", "leeth_scan_page", "leeth_click_many", "leeth_fill_form",
    }),
    "planning": frozenset({
        "todo_write", "exit_plan_mode", "create_artifact",
        "request_user_approval", "ask_user_questions",
    }),
    "memory": frozenset({"memory", "list_skills", "read_skill"}),
    "web": frozenset({"search_web", "read_webpage", "run_deep_research"}),
    "browser": frozenset({
        "browser_open", "browser_action", "browser_inspect", "browser_close",
    }),
    "desktop": frozenset({
        "list_desktop_windows", "capture_desktop_window", "capture_desktop_screen",
        "get_clipboard_image", "view_image", "desktop_inspect", "desktop_act",
    }),
    "integrations": frozenset({
        "list_mcp_tools", "call_mcp_tool", "manage_automations", "filter_new_items",
        "wait_heartbeat", "calculate",
    }),
}


def group_of(tool_name: str) -> str:
    """Which group a tool belongs to; empty when the taxonomy forgot it (a test checks)."""
    for group, names in TOOL_GROUPS.items():
        if tool_name in names:
            return group
    return ""




#: Tools that exist only while a feature is installed (`semantic_search` joins when a
#: RAG database is configured). They are advertised on those machines and are not part
#: of the static surface, so budgets and "every tool has a test" checks skip them.
OPTIONAL_TOOLS: frozenset = frozenset({"semantic_search"})


__all__ = [
    "TOOL_SCHEMAS", "AVAILABLE_TOOLS", "get_tool_schemas", "get_available_tools",
    "ACTIVE_PROCESSES", "ACTIVE_PROCESSES_LOCK", "delegate_local", "run_deep_research", "log",
    "TOOL_GROUPS", "group_of", "OPTIONAL_TOOLS",
]


