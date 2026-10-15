from tools._helpers import (
    _resolve_path,
    _is_plugin_path_restricted,
    _validate_code_syntax,
    _print_diff,
    _build_match_hint,
    log,
)
from tools.command_ops import (
    ACTIVE_PROCESSES,
    ACTIVE_PROCESSES_LOCK,
)
from tools.schemas import (
    TOOL_SCHEMAS,
    AVAILABLE_TOOLS,
    get_tool_schemas,
    get_available_tools,
)


def delegate_local(*args, **kwargs):
    from local_inference import delegate_local as _dl
    return _dl(*args, **kwargs)


def run_deep_research(*args, **kwargs):
    from deep_research import run_deep_research as _rdr
    return _rdr(*args, **kwargs)


def __getattr__(name: str):
    if name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[name]
    raise AttributeError(f"module 'tools' has no attribute '{name}'")


__all__ = [
    "TOOL_SCHEMAS", "AVAILABLE_TOOLS", "get_tool_schemas", "get_available_tools",
    "ACTIVE_PROCESSES", "ACTIVE_PROCESSES_LOCK", "delegate_local", "run_deep_research", "log",
]


