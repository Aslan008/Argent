"""
Tier-based tool profiles.

The full tool schema set (~50 tools) costs ~6.5k tokens in every request and
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
    "read_file", "write_file", "delete_file",
    "replace_in_file", "get_file_outline",
    "list_directory", "create_directory", "move_file",
    # search. semantic_search exists only while RAG is on, so listing it here
    # costs nothing otherwise — and when it IS on, the docs it searches are the
    # reason a small model can answer about an API without inventing signatures.
    # Leaving it out hid it from exactly the models that need it most.
    "grep_search", "search_files", "search_web", "read_webpage", "semantic_search",
    # execution & waiting
    "run_command", "start_background_command", "read_background_command",
    "stop_background_command", "list_background_commands", "wait_heartbeat",
    # interaction & utilities
    "ask_user_questions", "calculate", "analyze_project", "todo_write", "exit_plan_mode",
    # long-term notes: one compact schema; plan_mode.hides() drops it while /memory off
    "memory",
}

# Mode-specific tool additions for slim/small profiles (if any).
MODE_TOOLS = {}

# Categories that get the slim profile (they benefit most from a tight prompt).
_SLIM_CATEGORIES = {"tiny", "small"}

# Only worth their tokens once something is actually plugged in. A fixed core
# set cannot express "useful here, dead weight there", and a weak model that
# silently loses its only door to a configured Unity editor is worse off than
# one carrying two extra schemas.
_CONDITIONAL_CORE = {
    ("desktop_inspect", "desktop_act"): lambda: __import__(
        "src.automation.desktop_guard", fromlist=["enabled"]).enabled(),
    ("call_mcp_tool", "list_mcp_tools"): lambda: bool(__import__(
        "config").get_mcp_servers()),
}


def core_tools_now(mode: str = None, extra_tools: set = None):
    """CORE_CHAT_TOOLS plus whatever this environment makes worth having, plus mode tools."""
    tools = set(CORE_CHAT_TOOLS)
    if mode and mode in MODE_TOOLS:
        tools.update(MODE_TOOLS[mode])
    if extra_tools:
        tools.update(extra_tools)
    for names, is_relevant in _CONDITIONAL_CORE.items():
        try:
            if is_relevant():
                tools.update(names)
        except Exception:
            pass          # a broken config must not shrink the toolset
    return tools


def slim_tools_for_category(tool_names, category: str, mode: str = None, extra_tools: set = None):
    """Filter a list of tool names down to the core set for weak models.
    Returns the list unchanged for medium/large/cloud."""
    if category in _SLIM_CATEGORIES:
        core = core_tools_now(mode=mode, extra_tools=extra_tools)
        return [n for n in tool_names if n in core]
    return list(tool_names)


def is_slim_category(category: str) -> bool:
    return category in _SLIM_CATEGORIES


def get_mode_tools(mode: str, category: str = None) -> list[str]:
    """Get the active tools for a specific mode and category.
    If category is slim ('tiny' or 'small'), filters down to core + mode tools.
    Otherwise returns all available tools for that mode."""
    from tools.schemas import AVAILABLE_TOOLS
    all_names = list(AVAILABLE_TOOLS.keys())
    if category in _SLIM_CATEGORIES:
        core = core_tools_now(mode=mode)
        return [n for n in all_names if n in core]
    return all_names


def _compress_docstring(text: str, max_chars: int = 120) -> str:
    """Extract a concise first sentence or prefix from a docstring."""
    if not text or not isinstance(text, str):
        return ""
    first_chunk = text.strip().split("\n")[0].split(". ")[0].strip()
    if not first_chunk.endswith("."):
        first_chunk += "."
    if len(first_chunk) > max_chars:
        cut = first_chunk[:max_chars].rsplit(" ", 1)[0]
        return cut.rstrip(",;:-") + "..."
    return first_chunk


def minify_tool_schema(schema: dict) -> dict:
    """Create a token-compact copy of a tool schema for small models."""
    if not isinstance(schema, dict) or "function" not in schema:
        return schema

    fn = schema.get("function", {})
    minified_fn = {
        "name": fn.get("name", ""),
        "description": _compress_docstring(fn.get("description", ""), 120),
    }

    params = fn.get("parameters")
    if isinstance(params, dict):
        minified_params = {
            "type": params.get("type", "object"),
        }
        if "properties" in params and isinstance(params["properties"], dict):
            min_props = {}
            for prop_name, prop_val in params["properties"].items():
                if isinstance(prop_val, dict):
                    prop_copy = dict(prop_val)
                    if "description" in prop_copy:
                        prop_copy["description"] = _compress_docstring(prop_copy["description"], 80)
                    min_props[prop_name] = prop_copy
                else:
                    min_props[prop_name] = prop_val
            minified_params["properties"] = min_props

        if "required" in params:
            minified_params["required"] = params["required"]
        minified_fn["parameters"] = minified_params

    return {
        "type": schema.get("type", "function"),
        "function": minified_fn
    }


def minify_tool_schemas(schemas: list[dict]) -> list[dict]:
    """Minify a list of tool schemas for context efficiency."""
    if not schemas:
        return []
    return [minify_tool_schema(s) for s in schemas]
