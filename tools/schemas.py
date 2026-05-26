from deep_research import run_deep_research
from tools.project_tools import (
    add_project_task, complete_project_task, plan_work_changes,
    add_work_task, list_project_tasks, write_project_spec,
    write_project_architecture, write_file_spec,
)
from tools.file_ops import (
    read_file, write_file, delete_file, create_directory,
    move_file, copy_file, replace_in_file, replace_python_function,
    multi_replace_in_file, get_file_outline, list_directory,
)
from tools.obsidian_ops import (
    write_obsidian_note, search_obsidian_notes, update_obsidian_properties,
)
from tools.search_ops import search_files, grep_search
from tools.command_ops import (
    run_command, run_admin_command, start_background_command,
    read_background_command, send_background_command, stop_background_command,
    read_git_diff,
)
from tools.web_tools import search_web, read_webpage
from tools.plugin_tools import create_plugin, delete_plugin
from tools.system_ops import (
    read_event_logs, get_process_info, query_registry, search_system_files
)
from tools.skill_tools import list_skills, read_skill, create_skill, delete_skill
from tools.misc_tools import (
    find_definition, find_references, git_checkpoint, git_rollback,
    call_mcp_tool, run_subagent, create_svg_image, ask_user_questions,
    wait_heartbeat, end_auto_mode,
)
from tools.browser_tools import (
    browser_open, browser_state, browser_click, browser_input,
    browser_screenshot, browser_scroll, browser_get_content, browser_close,
)

AVAILABLE_TOOLS = {
    "add_project_task": add_project_task,
    "complete_project_task": complete_project_task,
    "list_project_tasks": list_project_tasks,
    "write_project_spec": write_project_spec,
    "write_project_architecture": write_project_architecture,
    "write_file_spec": write_file_spec,
    "plan_work_changes": plan_work_changes,
    "add_work_task": add_work_task,
    "read_file": read_file,
    "write_file": write_file,
    "delete_file": delete_file,
    "create_directory": create_directory,
    "move_file": move_file,
    "copy_file": copy_file,
    "write_obsidian_note": write_obsidian_note,
    "search_obsidian_notes": search_obsidian_notes,
    "update_obsidian_properties": update_obsidian_properties,
    "run_deep_research": run_deep_research,
    "replace_in_file": replace_in_file,
    "replace_python_function": replace_python_function,
    "list_directory": list_directory,
    "search_files": search_files,
    "grep_search": grep_search,
    "run_command": run_command,
    "run_admin_command": run_admin_command,
    "start_background_command": start_background_command,
    "read_background_command": read_background_command,
    "send_background_command": send_background_command,
    "stop_background_command": stop_background_command,
    "search_web": search_web,
    "read_webpage": read_webpage,
    "get_file_outline": get_file_outline,
    "multi_replace_in_file": multi_replace_in_file,
    "read_git_diff": read_git_diff,
    "create_plugin": create_plugin,
    "delete_plugin": delete_plugin,
    "read_event_logs": read_event_logs,
    "get_process_info": get_process_info,
    "query_registry": query_registry,
    "search_system_files": search_system_files,
    "list_skills": list_skills,
    "read_skill": read_skill,
    "create_skill": create_skill,
    "delete_skill": delete_skill,
    "wait_heartbeat": wait_heartbeat,
    "end_auto_mode": end_auto_mode,
    "find_definition": find_definition,
    "find_references": find_references,
    "git_checkpoint": git_checkpoint,
    "git_rollback": git_rollback,
    "call_mcp_tool": call_mcp_tool,
    "run_subagent": run_subagent,
    "create_svg_image": create_svg_image,
    "ask_user_questions": ask_user_questions,
    "browser_open": browser_open,
    "browser_state": browser_state,
    "browser_click": browser_click,
    "browser_input": browser_input,
    "browser_screenshot": browser_screenshot,
    "browser_scroll": browser_scroll,
    "browser_get_content": browser_get_content,
    "browser_close": browser_close,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_project_task",
            "description": "Add a task to the current project plan. Use this to break down the project into steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A concrete description of what this task should accomplish."
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_work_changes",
            "description": "Submit your investigation and proposed changes for an existing codebase. Call this ONCE during Phase 1 investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "High-level description of what you will change and how you will solve the problem."
                    },
                    "files_to_edit": {
                        "type": "string",
                        "description": "Comma-separated list of EXISTING files you need to modify."
                    },
                    "files_to_create": {
                        "type": "string",
                        "description": "Comma-separated list of NEW files you need to create."
                    }
                },
                "required": ["strategy", "files_to_edit", "files_to_create"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_work_task",
            "description": "Add a micro-task to the project plan during Phase 2. Make tasks small, like modifying a single method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task."
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_project_task",
            "description": "Mark a project task as completed. You MUST provide a summary of what you did.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to complete."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of what was accomplished. Mention files created or modified."
                    }
                },
                "required": ["task_id", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_project_tasks",
            "description": "View the current project status with all tasks and their completion summaries.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": "Call a standardized tool via the Model Context Protocol (MCP). Use this to interact with external services like GitHub, Slack, or Google Search via a standard interface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": { "type": "string", "description": "The name of the MCP server (e.g., 'github')." },
                    "tool_name": { "type": "string", "description": "The name of the tool to call." },
                    "arguments_json": { "type": "string", "description": "JSON string of arguments for the tool." }
                },
                "required": ["server_name", "tool_name", "arguments_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_subagent",
            "description": "Spawn a specialized sub-agent to handle a specific part of a complex task in isolation. This is great for research, code reviews, or implementing small isolated modules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": { "type": "string", "description": "The role of the sub-agent: 'Coder', 'Researcher', 'Reviewer', 'DocWriter'." },
                    "task": { "type": "string", "description": "The specific task instructions for the sub-agent." },
                    "tools_json": { "type": "string", "description": "Optional JSON list of tool names allowed for this sub-agent (e.g., '[\"read_file\", \"write_file\"]')." }
                },
                "required": ["role", "task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_project_spec",
            "description": "Write the detailed technical specification for the current project. Describe all files, classes, fields (with types), methods (with parameters), and relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "string",
                        "description": "The full technical specification text describing all files, classes, fields, methods, and their relationships."
                    }
                },
                "required": ["spec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_project_architecture",
            "description": "Write the high-level architecture map for the project. List ALL files needed, their purpose, and dependencies between them. Do NOT describe implementation details — just the structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "architecture": {
                        "type": "string",
                        "description": "The architecture map: list of all files, their purpose, and which files they depend on."
                    },
                    "files": {
                        "type": "string",
                        "description": "Comma-separated list of ALL project file paths to be created. Example: 'src/main.py, src/calculator.py, src/utils.py'. Use full paths including folders."
                    }
                },
                "required": ["architecture", "files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file_spec",
            "description": "Write a detailed specification for ONE specific file. Include: file path, all imports, class/function names, method signatures with parameter types and return types, field names with types, and a 1-2 sentence logic description for each method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The file path exactly as it appears in the architecture (e.g., 'src/converter.py')."
                    },
                    "spec": {
                        "type": "string",
                        "description": "The detailed specification: imports, classes, methods (name, params, return type, logic), fields (name, type, default)."
                    }
                },
                "required": ["filename", "spec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads file content. Files over 500 lines are auto-truncated; use start_line/end_line to read specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to read."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional. First line to read (1-indexed). Omit to start from the beginning."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional. Last line to read (1-indexed, inclusive). Omit to read to the end."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file, replacing its current contents. Creates intermediate directories if missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The string content to write into the file."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_obsidian_note",
            "description": "Creates or entirely overwrites a markdown note in the user's Obsidian vault. Automatically formats YAML frontmatter for tags and aliases. WARNING: Do not use this to append or modify just a small part of an existing note unless you intend to OVERWRITE the entire note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Relative path of the note inside the vault (e.g., 'Ideas/Game Concept.md'). Extension .md is added automatically if missing."
                    },
                    "content": {
                        "type": "string",
                        "description": "The main text content of the note (Markdown formatted)."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags WITHOUT the '#' symbol (e.g., ['npc', 'boss'])."
                    },
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of alternative titles (aliases) for the note."
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If true, overwrites the note if it already exists. Default is false to prevent accidental data loss."
                    }
                },
                "required": ["note_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_obsidian_notes",
            "description": "Searches for markdown notes in the Obsidian vault by text content or tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional text to search for within the note contents (case-insensitive)."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional tag to search for, either in the YAML frontmatter or inline as #tag (e.g., 'idea', 'boss')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_svg_image",
            "description": "Create an SVG image from code and automatically open it in the web browser for the user to see. Use this to explain complex concepts, show UI designs, or create architecture diagrams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "svg_code": {
                        "type": "string",
                        "description": "The complete SVG XML code."
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (e.g., 'architecture_diagram.svg')."
                    }
                },
                "required": ["svg_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_plugin",
            "description": "Creates a new Python plugin (hook/command) in the './plugins/' directory. Automatically handles imports, syntax validation, and reloads Argent to active the new command immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The filename of the plugin (e.g., 'weather_plugin' or 'weather_plugin.py')."
                    },
                    "code": {
                        "type": "string",
                        "description": "The full Python code for the plugin. Remember to use 'command_NAME' for slash-commands and 'from ui import console' for output."
                    }
                },
                "required": ["name", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_plugin",
            "description": "Deletes an existing plugin from the './plugins/' directory and reloads Argent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The filename of the plugin to delete (e.g., 'weather_plugin.py')."
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "Lists all available markdown-based skills. Use this to discover specialized instructions you or the user have created.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Reads the full instructions of a specific skill. Use this to follow complex workflows or expert guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to read (e.g., 'SQL_Expert')."
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Creates a new markdown-based skill or updates an existing one. Use this to persist complex workflows or expert personas for future use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill (e.g., 'Code_Reviewer')."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "The detailed instructions that Argent must follow when this skill is active."
                    },
                    "description": {
                        "type": "string",
                        "description": "A short one-sentence summary of what this skill does."
                    }
                },
                "required": ["name", "instructions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Deletes an existing skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to delete."
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_obsidian_properties",
            "description": "Safely updates tags, aliases, and custom properties in the YAML frontmatter of an existing Obsidian note. WARNING: This tool DOES NOT modify the main text body of the note. If the user asks to rewrite, expand, or add examples to an Obsidian note, DO NOT USE THIS TOOL. Use `replace_in_file` instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Relative path of the note inside the vault to update."
                    },
                    "add_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags to add WITHOUT the '#' symbol."
                    },
                    "remove_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags to remove WITHOUT the '#' symbol."
                    },
                    "add_aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of aliases to add."
                    },
                    "remove_aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of aliases to remove."
                    },
                    "properties": {
                        "type": "object",
                        "description": "Dictionary of any key-value pairs to set in YAML frontmatter. To delete a key, set its value to null."
                    }
                },
                "required": ["note_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replaces a specific text block in a file with new content. Use this to edit existing files without rewriting them entirely. The target text must be a unique, exact match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit."
                    },
                    "target_text": {
                        "type": "string",
                        "description": "The exact text block to be replaced. Must match exactly, including indentation and newlines."
                    },
                    "replacement_text": {
                        "type": "string",
                        "description": "The new text to insert in place of the target_text."
                    }
                },
                "required": ["file_path", "target_text", "replacement_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_python_function",
            "description": "Surgically replace an entire top-level function or class method in a Python file. Extremely reliable. Solves indentation and matching issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the python file."
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function. Use 'my_func' for top-level, or 'MyClass.my_method' for class methods."
                    },
                    "new_code": {
                        "type": "string",
                        "description": "The complete replacement code for the function, INCLUDING the 'def' line and full body. Indentation of the new code will be auto-corrected if it's a class method."
                    }
                },
                "required": ["file_path", "function_name", "new_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Deletes a file from the file system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to delete."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Creates a new directory and all parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "The path of the directory to create."
                    }
                },
                "required": ["dir_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Moves or renames a file from one path to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "The current path of the file."
                    },
                    "destination": {
                        "type": "string",
                        "description": "The new path for the file."
                    }
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copies a file from one path to another. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "The path of the file to copy."
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination path for the copy."
                    }
                },
                "required": ["source", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists all files and subdirectories within a given directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "The path to the directory to list."
                    }
                },
                "required": ["dir_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Recursively search for files matching specific criteria. Use this to find files by pattern (e.g., *.cs), by name, or by content without requiring user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The root directory to start the search from. Default is '.'."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for file matching (e.g., '*.py', '**/*.cs'). Default is '*'."
                    },
                    "name_contains": {
                        "type": "string",
                        "description": "Case-insensitive substring that must be in the filename."
                    },
                    "content_contains": {
                        "type": "string",
                        "description": "Case-insensitive substring that must be inside the file's content."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of files to return. Default is 50."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search file contents using a regex pattern. Returns matching lines with file paths and line numbers. Faster and more precise than search_files for finding specific code, function calls, or text patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The root directory to search in."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for (e.g., 'def my_func', 'import os', 'class.*Model')."
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., '*.py', '*.cs'). Default searches all files."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Default is 30."
                    }
                },
                "required": ["directory", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Runs a CLI shell command on the user's system and returns the stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in the shell."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_admin_command",
            "description": "Runs a PowerShell command with Administrator privileges. This triggers a Windows UAC prompt for the user. Use this only when you explicitly need elevated permissions (e.g., editing registry, setting global system variables, installing system-wide services).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command string to execute as Administrator."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web using DuckDuckGo to find up-to-date information, documentation, or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_event_logs",
            "description": "Reads recent errors and warnings from the Windows Event Log. Extremely useful for diagnosing application crashes and OS issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_name": {
                        "type": "string",
                        "description": "The name of the log to read (e.g., 'Application', 'System'). Default is 'Application'."
                    },
                    "entry_type": {
                        "type": "string",
                        "description": "Comma-separated list of entry types to filter by (e.g., 'Error,Warning'). Default is 'Error,Warning'."
                    },
                    "newest": {
                        "type": "integer",
                        "description": "Number of recent events to retrieve. Default is 20."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_process_info",
            "description": "Gets information about running processes on the system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "process_name": {
                        "type": "string",
                        "description": "Optional substring to filter process names (e.g., 'chrome')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_registry",
            "description": "Reads a key or specific value from the Windows Registry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The registry path (e.g., 'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion')."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name of the specific value to read within the key."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_system_files",
            "description": "Performs a fast recursive search for files on the disk. Useful for finding leftover files of deleted applications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The root path to search in (e.g., 'C:\\Users\\Username\\AppData')."
                    },
                    "filter_pattern": {
                        "type": "string",
                        "description": "The pattern to search for (e.g., '*discord*'). Default is '*'."
                    }
                },
                "required": ["path", "filter_pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Reads and extracts the main text content from a specific webpage URL. Use this to dive deeper into results found via search_web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to read."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds. Default is 15."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_background_command",
            "description": "Starts a command in the background (e.g. dev servers, infinite loops) and returns a PID. Use this instead of run_command for long-running processes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in the background."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_background_command",
            "description": "Reads newer output (stdout/stderr) from a currently running background process by PID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "The Process ID (PID) to read from."
                    }
                },
                "required": ["pid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_background_command",
            "description": "Sends string input to the stdin of a running background process. Use this to interact with REPLs or commands waiting for input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "The Process ID (PID)."
                    },
                    "input_string": {
                        "type": "string",
                        "description": "The text to send to the command. Must include newline if you want to submit it."
                    }
                },
                "required": ["pid", "input_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_background_command",
            "description": "Terminates a background process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "The Process ID (PID) to terminate."
                    }
                },
                "required": ["pid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_outline",
            "description": "Get the structural outline (classes and methods) of a Python file without reading the entire file body. Ideal for exploring large projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multi_replace_in_file",
            "description": "Perform multiple target/replacement edits across one or several files in a single tool call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "changes_json": {
                        "type": "string",
                        "description": "A serialized JSON array of objects. Example: '[{\"file_path\": \"app.py\", \"target_text\": \"old\", \"replacement_text\": \"new\"}]'"
                    }
                },
                "required": ["changes_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_git_diff",
            "description": "Read the current unstaged and staged Git differences in the project. Use this to understand what has changed compared to the last commit.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_deep_research",
            "description": "Starts an autonomous Deep Research Sub-Agent that deeply researches an objective using search engines, reads top web pages, extracts data, and returns a massive synthesized technical report. Use this instead of search_web for broad topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The specific research objective or question (e.g., 'Best Unity DOTS optimizations for CPU spikes')."
                    }
                },
                "required": ["objective"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": "Find the definition of a class, function, or variable at a specific line and column. Use this instead of grep for precise navigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "The path to the file." },
                    "line": { "type": "integer", "description": "Line number (1-indexed)." },
                    "column": { "type": "integer", "description": "Column number (0-indexed)." }
                },
                "required": ["file_path", "line", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Find all usages (references) of a symbol at a specific line and column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "The path to the file." },
                    "line": { "type": "integer", "description": "Line number (1-indexed)." },
                    "column": { "type": "integer", "description": "Column number (0-indexed)." }
                },
                "required": ["file_path", "line", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkpoint",
            "description": "Create a temporary git commit to save current progress. Use this before making risky changes or running experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Brief description of why you are checkpointing." }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_rollback",
            "description": "Revert all changes to the last 'Argent Checkpoint'. Use this if an experiment failed or logic is broken beyond simple repair.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_questions",
            "description": "Ask the user a series of structured questions. Use this when you need clarification, preferences, or decisions on multiple points before proceeding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "A list of question objects.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["text", "single_choice", "multi_choice"],
                                    "description": "The type of question."
                                },
                                "question": {
                                    "type": "string",
                                    "description": "The question text."
                                },
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Required for single_choice/multi_choice. A list of string options."
                                }
                            },
                            "required": ["type", "question"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_heartbeat",
            "description": "Used in Auto Mode to sleep for a specified number of seconds before waking up to check a condition. Use this to wait for compilation, server startup, or long-running background tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_seconds": {
                        "type": "integer",
                        "description": "The number of seconds to wait before waking up."
                    },
                    "condition_to_check": {
                        "type": "string",
                        "description": "What you want to check when you wake up. This will be sent back to you as context."
                    }
                },
                "required": ["delay_seconds", "condition_to_check"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_auto_mode",
            "description": "Ends the autonomous experimental mode and explicitly returns control to the user. MUST call this when the task is fully completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for ending auto mode (e.g., 'Task completed successfully', 'Encountered unrecoverable error')."
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": "Open a URL in the built-in browser. Creates a browser session if needed. Use this to navigate to websites for scraping, form filling, or interaction. After opening, call browser_state to see the interactive elements on the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to (e.g. 'https://example.com')."
                    },
                    "session": {
                        "type": "string",
                        "description": "Name of the browser session. Use different names for parallel browsing. Default: 'default'."
                    },
                    "headed": {
                        "type": "boolean",
                        "description": "If true, shows the browser window (for debugging). Default: false (headless)."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_state",
            "description": "Get the current state of the browser page: URL, title, and a numbered list of all interactive elements (buttons, links, inputs, etc). ALWAYS call this after browser_open or after any action to see the updated page. Use the element indices from this output for browser_click and browser_input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click on an interactive element by its index number. The index comes from browser_state output (e.g. [3] button \"Submit\"). IMPORTANT: After clicking, call browser_state again to see the updated page — old indices become invalid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The element index from browser_state (e.g. 3 for [3] button \"Submit\")."
                    },
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    }
                },
                "required": ["index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_input",
            "description": "Type text into an input field or textarea by its index from browser_state. Clears the existing text before typing. Use for filling forms, search boxes, login fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The element index from browser_state."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to type into the element."
                    },
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    }
                },
                "required": ["index", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take a screenshot of the current browser page and save it to disk. Returns the file path of the saved image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional file path to save the screenshot. If not provided, saves to the visuals directory with a timestamp."
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "If true, captures the entire scrollable page. Default: false (viewport only)."
                    },
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the browser page up or down. Use this to reveal content below the fold or to navigate long pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down"],
                        "description": "Scroll direction. Default: 'down'."
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Pixels to scroll. Default: 500."
                    },
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_content",
            "description": "Extract content from the current browser page. Can return plain text, markdown, or raw HTML. Use 'text' for body text, 'markdown' for structured readable content, 'html' for raw source. Use 'selector' to target a specific part of the page (e.g. '#comments' for YouTube comments, '.vacancy-list' for job listings, 'main' for main content).",
            "parameters": {
                "type": "object",
                "properties": {
                    "content_type": {
                        "type": "string",
                        "enum": ["text", "markdown", "html"],
                        "description": "Type of content to extract. Default: 'text'."
                    },
                    "index": {
                        "type": "integer",
                        "description": "Optional element index to get text from a specific element only."
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to target a specific section of the page. Examples: '#comments', '.job-list', 'main', 'article'. Much more precise than reading the entire page."
                    },
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_close",
            "description": "Close a browser session and free resources. Always close sessions when done with browser automation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Name of the session to close. Default: 'default'."
                    }
                }
            }
        }
    }
]

# Dynamically add the semantic search tool to the list of available tools ONLY if RAG is enabled.
try:
    from rag_engine import semantic_search, is_rag_enabled
    # We use a getter function that returns the tools, rather than a static list, 
    # to account for runtime changes (like enabling RAG mid-session).
except ImportError:
    pass

def get_available_tools() -> dict:
    """Returns the dictionary of Python functions the LLM can call."""
    from config import get_disabled_tools
    disabled = get_disabled_tools()
    
    tools = {k: v for k, v in AVAILABLE_TOOLS.items() if k not in disabled}
    try:
        from rag_engine import semantic_search, is_rag_enabled
        if is_rag_enabled() and "semantic_search" not in disabled:
            tools["semantic_search"] = semantic_search
    except ImportError:
        pass
    return tools

def get_tool_schemas() -> list[dict]:
    """Returns the JSON schemas for the available tools, dynamically adding RAG if enabled."""
    from config import get_disabled_tools
    disabled = get_disabled_tools()
    
    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] not in disabled]
    try:
        from rag_engine import is_rag_enabled
        if is_rag_enabled() and "semantic_search" not in disabled:
            schemas.append({
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Searches the project's codebase conceptually using AI embeddings. Returns relevant code snippets regardless of exact keywords. Use this when you need to understand where a feature is implemented.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A natural language query describing what code you want to find (e.g., 'where does the player take damage?')."
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of snippets to return (default is 5, recommend keeping under 10)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
    except ImportError:
        pass
    return schemas
