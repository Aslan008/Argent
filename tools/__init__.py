from tools._helpers import (
    _resolve_path,
    _is_plugin_path_restricted,
    _validate_code_syntax,
    _print_diff,
    _build_match_hint,
)
from tools.project_tools import (
    add_project_task,
    complete_project_task,
    plan_work_changes,
    add_work_task,
    list_project_tasks,
    write_project_spec,
    write_project_architecture,
    write_file_spec,
)
from tools.file_ops import (
    read_file,
    delete_file,
    write_file,
    append_to_file,
    replace_python_function,
    replace_in_file,
    multi_replace_in_file,
    get_file_outline,
    list_directory,
    create_directory,
    move_file,
    copy_file,
)

from tools.search_ops import (
    search_files,
    grep_search,
)
from tools.command_ops import (
    run_command,
    run_admin_command,
    start_background_command,
    read_background_command,
    send_background_command,
    stop_background_command,
    read_git_diff,
    ACTIVE_PROCESSES,
    ACTIVE_PROCESSES_LOCK,
)
from tools.web_tools import (
    search_web,
    read_webpage,
)
from tools.plugin_tools import (
    create_plugin,
    delete_plugin,
)
from tools.skill_tools import (
    list_skills,
    read_skill,
    create_skill,
    delete_skill,
)
from tools.system_ops import (
    read_event_logs,
    get_process_info,
    query_registry,
    search_system_files,
)
from tools.misc_tools import (
    ask_user_questions,
    create_svg_image,
    find_definition,
    find_references,
    git_checkpoint,
    git_rollback,
    call_mcp_tool,
    run_subagent,
    wait_heartbeat,
    end_auto_mode,
)

from tools.browser_tools import (
    browser_open,
    browser_state,
    browser_click,
    browser_input,
    browser_screenshot,
    browser_scroll,
    browser_get_content,
    browser_close,
)
from tools.schemas import (
    TOOL_SCHEMAS,
    AVAILABLE_TOOLS,
    get_tool_schemas,
    get_available_tools,
)
from deep_research import run_deep_research
from tools._helpers import log

__all__ = [
    "TOOL_SCHEMAS", "AVAILABLE_TOOLS", "get_tool_schemas", "get_available_tools",
    "ACTIVE_PROCESSES", "ACTIVE_PROCESSES_LOCK", "log",
]
