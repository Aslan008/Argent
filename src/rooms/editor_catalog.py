"""What the room editor can offer to pick from: Argent's tools with their parameters, grouped.

The editor's forms are built from this, not from a hand-kept list: a tool's parameters,
which are required, and what each one is for come from the tool's own schema, so a new
tool shows up in the editor with a proper form the moment it is registered.
"""

from src.rooms.integrations import schema_params

#: group -> a person's name for it (the palette's categories)
GROUPS = {
    "files": "Файлы и папки",
    "shell": "Команды и фоновые процессы",
    "web": "Веб и браузер",
    "leeth": "Leeth (быстрая локальная модель)",
    "extend": "MCP и скиллы",
    "agents": "Агенты, память, планы",
    "desktop": "Экран и окна",
    "other": "Прочее",
}

_BY_NAME = {
    "files": {"read_file", "write_file", "replace_in_file", "delete_file", "create_directory", "move_file",
              "copy_file", "list_directory", "search_files", "grep_search", "get_file_outline", "read_git_diff",
              "analyze_project", "git_checkpoint", "git_rollback", "view_image", "run_deep_linter"},
    "shell": {"run_command", "run_admin_command", "start_background_command", "list_background_commands",
              "read_background_command", "send_background_command", "stop_background_command", "wait_heartbeat",
              "calculate"},
    "web": {"search_web", "read_webpage", "browser_open", "browser_action", "browser_inspect", "browser_close",
            "run_deep_research", "filter_new_items"},
    "leeth": {"delegate_to_leeth", "leeth_map", "leeth_scan_page", "leeth_click_many", "leeth_fill_form"},
    "extend": {"call_mcp_tool", "list_mcp_tools", "list_skills", "read_skill"},
    "agents": {"run_subagent", "subagents", "ultracode", "rooms", "memory", "todo_write", "manage_automations",
               "ask_user_questions", "request_user_approval", "exit_plan_mode", "create_artifact"},
    "desktop": {"list_desktop_windows", "capture_desktop_window", "capture_desktop_screen", "get_clipboard_image",
                "desktop_inspect", "desktop_act"},
}
_GROUP_OF = {name: group for group, names in _BY_NAME.items() for name in names}

#: Tools that run a command or reach into the outside world without being in plan mode's "mutating" set.
_ALSO_ACTS = frozenset({"run_command", "browser_open", "search_web", "read_webpage"})


def group_of(tool: str) -> str:
    return _GROUP_OF.get(tool, "other")


def tool_catalog(mutating=None) -> list:
    """[{name, description, group, params:[...], mutating, not_in_rooms}] for every registered tool,
    sorted by group then name."""
    from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
    from src.rooms.validator import NOT_IN_ROOMS
    if mutating is None:
        from src.agent.plan_mode import _mutates as mutating
    out, seen = [], set()
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function") or {}
        name = fn.get("name")
        if not name or name not in AVAILABLE_TOOLS:
            continue
        seen.add(name)
        out.append(_entry(name, (fn.get("description") or "").strip(), fn.get("parameters") or {}, mutating))
    for name in sorted(set(AVAILABLE_TOOLS) - seen):            # registered without a schema
        out.append(_entry(name, "", {}, mutating))
    for e in out:
        e["not_in_rooms"] = e["name"] in NOT_IN_ROOMS
    out.sort(key=lambda e: (list(GROUPS).index(e["group"]), e["name"]))
    return out


def _entry(name: str, description: str, parameters: dict, mutating) -> dict:
    return {
        "name": name,
        "description": description[:500],
        "group": group_of(name),
        "params": schema_params(parameters),
        "mutating": bool(mutating(name)),
    }


def acts_outside(tool: str, mutating=None) -> bool:
    """True when a step with this tool changes something or talks to the outside world (so running it
    once from the editor asks first)."""
    if mutating is None:
        from src.agent.plan_mode import _mutates as mutating
    return bool(mutating(tool)) or tool in _ALSO_ACTS
