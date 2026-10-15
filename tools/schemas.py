from tools.lazy_registry import LazyToolRegistry, DEFAULT_TOOL_LOADERS

AVAILABLE_TOOLS = LazyToolRegistry(DEFAULT_TOOL_LOADERS)


# Tools that take over the terminal to ask the user something. The CLI must not
# run its "Выполнение…" spinner around these: Rich's status display owns the
# same lines questionary draws on, so the question is rendered and immediately
# overwritten — the user sees the preamble, no prompt, and a program that will
# not continue. request_user_approval sat outside this set and did exactly
# that; git_rollback was one report away from the same.
#
# Declared here, beside the registry, because the list lived in the CLI and
# fell behind every tool added since. test_interactive_tools derives the true
# set from the sources and fails if this one drifts again.
INTERACTIVE_TOOLS = frozenset({
    "ask_user_questions", "request_user_approval", "exit_plan_mode",
    "run_command", "run_admin_command", "start_background_command",
    "delete_file", "git_rollback", "desktop_act",
})

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_leeth",
            "description": "Delegates a routine subtask or chain of simple actions (reading files, directory listings, grep search, terminal command execution, calculations, checking background tasks) to the local ultra-fast Leeth coprocessor (10–25 ms, 0 API tokens). Returns an aggregated report of all executed steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language instruction for the routine action or chain (e.g. 'read config.py', 'grep for LeethModel in src/', 'check background tasks')."
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "leeth_map",
            "description": "Answer ONE question about MANY texts at once with the local Leeth model (milliseconds per item, 0 API tokens): filter ('yesno': is this a complaint? is this a recipe?) or categorise ('categories') a list of posts, bookmarks, file names, tickets. State the criterion once. Returns index lists plus the items Leeth is unsure about -- judge those yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "string", "description": "The texts: a JSON list or one per line."},
                    "criterion": {"type": "string", "description": "mode='yesno': a yes/no question about one item. mode='categories': an optional question ('What is this about?')."},
                    "mode": {"type": "string", "enum": ["yesno", "categories"], "description": "Default 'yesno'."},
                    "categories": {"type": "string", "description": "For mode='categories': 2+ allowed categories, a JSON list or one per line."},
                    "accept_above": {"type": "number", "description": "yesno: probability from which an item is 'yes' (default 0.7)."},
                    "reject_below": {"type": "number", "description": "yesno: probability up to which an item is 'no' (default 0.3); in between it is 'unsure'."}
                },
                "required": ["items", "criterion"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "leeth_scan_page",
            "description": "Read-only: classify the clickable elements (links / buttons) of the CURRENT browser page against a criterion in one batched pass with the local Leeth model. Returns the matching elements (index, label, link, probability) and the unsure ones; changes nothing. Follow with leeth_click_many to act. Use for 'find / remove all bookmarks about X', 'open every post that ...', 'which posts in this feed are about Y' (include='posts', scroll_pages for endless feeds).",
            "parameters": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string", "description": "A yes/no question about one element ('Is this a recipe?'), or for mode='categories' the question."},
                    "mode": {"type": "string", "enum": ["yesno", "categories"]},
                    "categories": {"type": "string", "description": "For mode='categories': the categories (JSON list or one per line)."},
                    "include": {"type": "string", "enum": ["links", "buttons", "all", "posts"], "description": "What to judge (default 'links'). 'posts' = the repeated text blocks of the page (posts, cards, rows, comments): their text is judged and each reported post lists its own buttons for leeth_click_many."},
                    "scroll_pages": {"type": "integer", "description": "Scroll the page down up to this many times and judge what appears too (endless feeds). Default 0, max 30."},
                    "max_items": {"type": "integer", "description": "How many items are judged at most (default 300)."},
                    "session": {"type": "string"}
                },
                "required": ["criterion"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "leeth_click_many",
            "description": "Click the elements that the LAST leeth_scan_page returned, one after another; each is re-found by label + link before its click, so a list that re-renders after a delete does not misdirect later clicks. Destructive-looking elements (delete, pay, confirm, purchase) are NOT clicked without confirm=True -- ask the user for approval first and tell them how many items there are.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {"type": "string", "description": "Element indices from the last leeth_scan_page: a JSON list or comma-separated."},
                    "confirm": {"type": "boolean", "description": "The user has approved clicking destructive elements."},
                    "session": {"type": "string"}
                },
                "required": ["indices"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "leeth_fill_form",
            "description": "Fill the form on the CURRENT browser page from ONE instruction in a single pass with the local Leeth model: text fields, dropdowns (matched by meaning: 'Гонконг' -> 'Hong Kong') and checkboxes, then re-read the page to verify. A field whose value is not in the instruction is left empty and reported, never invented. Payment / delete / confirm buttons are never pressed. Use instead of typing field by field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "What to enter, in the user's own words (e.g. 'логин Валера, рост 175 см, возраст 29, город Гонконг')."},
                    "dry_run": {"type": "boolean", "description": "Only show the plan; change nothing."},
                    "submit": {"type": "boolean", "description": "Also press the submit button when it is not a payment / delete / confirm one."},
                    "min_confidence": {"type": "number", "description": "A value below it is not entered (default 0.55)."},
                    "session": {"type": "string"}
                },
                "required": ["instruction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_desktop_windows",
            "description": "List all visible top-level desktop windows on the system (HWND, PID, title, geometry). Use this before capture_desktop_window to find target window title or HWND.",
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
            "name": "capture_desktop_window",
            "description": "Capture a screenshot of a specific desktop window by its title substring or numeric HWND. Safe targeted capture using Win32 API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Window title substring (e.g. 'Unity', 'Notepad', 'Argent') or numeric HWND."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional custom path to save PNG image. Defaults to .argent/artifacts/desktop/."
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_desktop_screen",
            "description": "Capture a screenshot of the entire desktop screen (primary monitor or all virtual monitors). Useful for fullscreen games, Unity Play Mode, or overall desktop view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monitor": {
                        "type": "string",
                        "description": "'primary' (default) for main display, or 'all' for the full multi-monitor virtual desktop.",
                        "default": "primary"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional custom path to save PNG image. Defaults to .argent/artifacts/desktop/."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard_image",
            "description": "Retrieve an image or screenshot currently stored in the system clipboard (e.g. copied via Win+Shift+S, PrintScreen, or image copy).",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Optional custom path to save PNG image. Defaults to .argent/artifacts/desktop/."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_artifact",
            "description": "Create a Markdown artifact in the .argent/artifacts/ directory. Useful for documenting plans, research, or long text. To inspect or resume an artifact, use read_file with path '.argent/artifacts/<filename>'.",

            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file (e.g. implementation_plan.md)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The markdown content of the artifact."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_user_approval",
            "description": "Pause execution and wait for the user to explicitly approve a plan or action. Use this before making major changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "A short summary of what you want the user to approve."
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": (
                "Call a tool on an external MCP server (Unity Editor, GitHub, Slack, a database…). "
                "The servers connected right now are listed in the MCP SERVERS section of your "
                "instructions. If you do not know the exact tool name or its arguments, call "
                "list_mcp_tools first — guessing a signature wastes a turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": { "type": "string", "description": "The name of the MCP server (e.g., 'unity')." },
                    "tool_name": { "type": "string", "description": "Exact tool name, as returned by list_mcp_tools." },
                    "arguments_json": { "type": "string", "description": "JSON string of arguments for the tool." }
                },
                "required": ["server_name", "tool_name", "arguments_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_mcp_tools",
            "description": (
                "Look up what an MCP server can do. A single server can expose well over a "
                "hundred tools, so their signatures are NOT in your instructions — you fetch "
                "the ones you need, when you need them.\n"
                "  list_mcp_tools() — which servers exist and how many tools each has\n"
                "  list_mcp_tools('unity', 'navmesh') — signatures of the matching tools\n"
                "Filter by what you are trying to do ('scene', 'animation', 'build'); an "
                "unfiltered list on a big server gives names only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": { "type": "string", "description": "Server to inspect. Omit to list the servers themselves." },
                    "filter": { "type": "string", "description": "Substring matched against tool names and descriptions." }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_subagent",
            "description": (
                "Spawn a specialized sub-agent to handle a specific part of a complex task in isolation. Great for "
                "research, code reviews, or implementing a separate module.\n"
                "TWO OPTIONS make it much more useful. background=true returns at once and the sub-agent works while "
                "you do something else — start several in ONE response to run them in parallel (up to 4), then "
                "collect the reports. isolation='worktree' gives it a private copy of the repository: it can edit "
                "freely without touching the user's files or its sibling agents, and its work comes back as a patch "
                "you apply (subagents action='apply') or drop. Use worktree for any sub-agent that WRITES code while "
                "others or you are working. A sub-agent cannot ask the user anything and can run only shell commands "
                "the user already allowed; give it a self-contained task, it does not see this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": { "type": "string", "description": "The role of the sub-agent: 'Coder', 'Researcher', 'Reviewer', 'DocWriter', 'Critic' (adversarial reviewer of plans/ideas)." },
                    "task": { "type": "string", "description": "The specific, self-contained task instructions for the sub-agent." },
                    "tools_json": { "type": "string", "description": "Optional JSON list of tool names allowed for this sub-agent (e.g., '[\"read_file\", \"write_file\"]')." },
                    "background": { "type": "boolean", "description": "Return immediately; the sub-agent keeps running. Default false (wait for its report)." },
                    "isolation": { "type": "string", "enum": ["none", "worktree"], "description": "'worktree' = work in a private git worktree copy; the result is a patch. Default 'none'." }
                },
                "required": ["role", "task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads file content. Every line is prefixed with its line number and a tab — that gutter is NOT part of the file: never include it in target_text or in content you write back. Long files are truncated; use start_line/end_line to read specific sections.",
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
            "description": "Writes content to a file. WARNING: DO NOT USE THIS TO EDIT EXISTING FILES >150 LINES! It will be blocked to prevent truncation. Use replace_in_file instead (it also takes a `changes` list to apply several edits at once). This tool is ONLY for creating NEW files or writing very small scripts.",
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
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Optional. Set to true ONLY if you want to completely DESTROY and replace a large existing file. Defaults to false to prevent accidental corruption."
                    }
                },
                "required": ["file_path", "content"]
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
            "description": "Reads the full instructions of a specific skill. Use this to follow complex workflows or expert guidelines. SKILL.md folder skills also list bundled scripts/references/assets and their folder path — read or run those with read_file / run_command when the instructions call for them.",
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
            "name": "replace_in_file",
            "description": "Replaces a specific text block in a file with new content, or applies a batch of replacements across one or more files with atomic rollback if any edit fails. For single replacement, provide 'file_path', 'target_text', 'replacement_text' (and optional 'start_line', 'end_line' 1-indexed). For batch/multi-diff replacements, provide 'changes' list.",
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
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional starting line number (1-indexed) to narrow the search range. Omit to search the entire file."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional ending line number (1-indexed, inclusive) to narrow the search range. Omit to search the entire file."
                    },
                    "changes": {
                        "type": "array",
                        "description": "Optional batch of edits to apply atomically. Each item can contain {'file_path', 'target_text', 'replacement_text', 'start_line', 'end_line'}.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "target_text": {"type": "string"},
                                "replacement_text": {"type": "string"},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"}
                            }
                        }
                    }
                },
                "required": ["file_path"]
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
            "description": "Universal search and semantic code navigation. By default (target='all'), performs fast fulltext ripgrep across any file type (.py, .ts, .cs, .env, .json, .md, logs). When target is 'definition', 'references', 'implementations', or 'call_hierarchy', uses LSP semantic intelligence without requiring manual coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search string, regular expression pattern, or symbol name."
                    },
                    "query": {
                        "type": "string",
                        "description": "Alias for pattern. Search string or symbol name."
                    },
                    "directory": {
                        "type": "string",
                        "description": "The directory or specific file path to search in. Default is current directory ('.')."
                    },
                    "path": {
                        "type": "string",
                        "description": "Alias for directory. The directory or specific file path to search in."
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., '*.py', '*.cs'). Default searches all files."
                    },
                    "target": {
                        "type": "string",
                        "enum": ["all", "definition", "references", "implementations", "call_hierarchy"],
                        "description": "Search mode: 'all' (default ripgrep fulltext), 'definition' (jump to definition via LSP/AST), 'references' (find all symbol usages), 'implementations' (find interface/trait implementations), 'call_hierarchy' (caller/callee tree)."
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["incoming", "outgoing"],
                        "description": "Direction for target='call_hierarchy': 'incoming' (who calls this function, default) or 'outgoing' (who this function calls)."
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Optional explicit symbol name when navigating code via LSP."
                    },
                    "line": {
                        "type": "integer",
                        "description": "Optional 1-indexed line number of the symbol for precise LSP navigation."
                    },
                    "column": {
                        "type": "integer",
                        "description": "Optional 0-indexed column number of the symbol for precise LSP navigation."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return. Default is 30."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Runs a CLI shell command on the user's system and returns the stdout and stderr. Always prefer non-interactive flags (e.g. `npx -y ...`, `npm init -y`, `pip install --no-input ...`, `--no-pager`) so tools do not hang awaiting interactive input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in the shell."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory in which to run the command. Defaults to the current working directory."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional silence timeout in seconds (defaults to 180). Terminate command early if no output is produced for this duration."
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
            "description": (
                "Searches the web across several engines at once (DuckDuckGo, Wikipedia, "
                "StackOverflow, GitHub issues, and Brave when configured) and merges the results.\n"
                "Search operators are supported and worth using — each engine is automatically "
                "given only the ones it understands:\n"
                "  \"exact phrase\"  — the words must appear together, verbatim. The single most "
                "effective technique for error messages: \"NullReferenceException in LoadAsync\".\n"
                "  site:docs.unity3d.com  — restrict to one domain (vendor docs, a specific forum).\n"
                "  filetype:pdf  — restrict to a file type (also ext:).\n"
                "  -word  — exclude results containing it, e.g. -tutorial to skip beginner pages.\n"
                "  intitle:word  — the word must be in the TITLE, not merely mentioned. Use it "
                "when a common term drowns in passing mentions; it is translated to each engine's "
                "native form (GitHub's in:title, StackOverflow's title filter).\n"
                "Combine them: 'site:forum.unity.com \"Addressables\" memory leak -tutorial'.\n"
                "Quoted phrases work everywhere; the rest are applied where the engine supports "
                "them and dropped where it does not, so writing them is never harmful.\n"
                "WARNING: this returns only short snippets and URLs. To read the actual content, "
                "you MUST call `read_webpage` with the returned URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, optionally with operators (see the description)."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return. Default is 5."
                    },
                    "engine": {
                        "type": "string",
                        "description": "Optional: specify a single search engine to query ('duckduckgo', 'brave', 'ollama', 'github', 'stackoverflow', 'wikipedia') or 'all' for federated meta-search across all active engines. Default is 'all'.",
                        "enum": ["all", "duckduckgo", "brave", "ollama", "github", "stackoverflow", "wikipedia"]
                    }
                },
                "required": ["query"]
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
                    },
                    "raw_mode": {
                        "type": "boolean",
                        "description": "If true, bypasses smart text extraction and returns all text from the body. Use this if the default extraction returns incomplete or chopped text (e.g. on forums)."
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
            "description": "Starts a command in the background (e.g. dev servers, infinite loops) and returns a PID. Use this instead of run_command for long-running processes. For a program that asks questions (npm init, ssh, python -i, an installer) set pty=true: it gets a real terminal; read its questions with read_background_command and answer with send_background_command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in the background."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Optional working directory in which to run the background command. Defaults to the current working directory."
                    },
                    "pty": {
                        "type": "boolean",
                        "description": "Run on a real terminal so the program can ask questions. Default false."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_project",
            "description": "Analyze the current project structure in one call: directory tree, detected languages/frameworks, entry points, build/test commands and file statistics. Use this FIRST when onboarding to a codebase or writing/updating AGENTS.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Project root to analyze. Defaults to the current directory."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_background_commands",
            "description": "List the background processes you started this session, with their PID, status (running/exited) and command. Use this to recover a PID you forgot before reading output or stopping a process.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_background_command",
            "description": "Reads newer output (stdout/stderr) from a currently running background process by PID. If there is no new output, do NOT keep calling this in a loop — use wait_heartbeat(until='process_finished(\"PID\")') to sleep until the process finishes, or proceed with other work.",
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
            "description": "Get the structural outline (classes and methods) of a Python (.py) or C# (.cs) file without reading the entire file body. Ideal for exploring large projects.",
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
            "description": "Starts an autonomous Deep Research Sub-Agent that deeply researches an objective using search engines, reads top web pages, extracts data, and returns a massive synthesized technical report. Use this instead of search_web for broad topics. You can guide the research with focus/context parameters and control depth with max_rounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The specific research objective or question (e.g., 'Best Unity DOTS optimizations for CPU spikes')."
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional: aspect to prioritise in the research. The report will lean toward this angle. E.g., 'memory usage' or 'CPU profiling'."
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional: already-known facts. The pipeline avoids re-searching these and the extractor focuses on NEW information beyond this context. Pass a summary of what you already know."
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Total rounds including the initial one. 1 = no gap-filling (fastest), 2 = one gap-filling pass (default), 3 = two gap-filling passes (most thorough). Default: 2.",
                        "default": 2
                    },
                    "max_sources": {
                        "type": "integer",
                        "description": "Maximum new sources to fetch per round. Default: 15. Increase for broader topics, decrease for faster runs.",
                        "default": 15
                    }
                },
                "required": ["objective"]
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
            "description": "Revert all changes to an 'Argent Checkpoint'. Use this if an experiment failed, logic is broken beyond simple repair, or the user asks to undo/return to an earlier state. Argent auto-creates a checkpoint before the first edit of each turn, so recent states are usually available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to_checkpoint": {
                        "type": "string",
                        "description": "Optional: pick an OLDER checkpoint by short sha or a substring of its message (e.g. 'before: fix menu'). Omit to roll back to the most recent checkpoint. On no match the tool lists available checkpoints."
                    }
                }
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates exact arithmetic or symbolic math expressions (e.g. arithmetic, trigonometry, calculus, algebra) and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '(15 * 4) + sqrt(144)', 'integrate(x**2, x)', 'solve(x**2 - 4, x)')."
                    }
                },
                "required": ["expression"]
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
                                    "description": (
                                        "Required for single_choice/multi_choice. An array of PLAIN "
                                        'STRINGS: ["Only critical paths", "Full coverage"]. NOT objects '
                                        '— {"label": ..., "description": ...} is wrong and the user will '
                                        "see an unusable menu. Put any explanation inside the string itself."
                                    )
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
            "description": (
                "Pause execution while something finishes: a compilation, a server "
                "starting, a long background task.\n"
                "IMPORTANT: Do NOT use wait_heartbeat for web pages or browser tasks (such as file/video upload or page loading) — it only monitors local filesystem files and background OS processes. For browser and web tasks, you MUST use browser_action(action=\"wait_for\") instead.\n"
                "PREFER `until`: waking up costs a full turn, so polling 'is it done yet?' ten "
                "times costs ten turns, while a condition is checked locally for free and wakes "
                "you once — when it actually happened. Fall back to `delay_seconds` only when "
                "nothing observable marks the end.\n"
                "Checks available to `until`: file_exists(path), file_missing(path), "
                "file_contains(path, text), process_finished(pid), process_running(pid), "
                "subagent_finished(id). "
                "Combine them with and / or / not.\n"
                "Examples:\n"
                "  until='file_contains(\"build.log\", \"BUILD SUCCESSFUL\")'\n"
                "  until='process_finished(\"3\")', timeout_seconds=1200\n"
                "  until='file_exists(\"dist/app.exe\") and file_missing(\".lock\")'\n"
                "You are told whether the condition was MET or the wait TIMED OUT — a timeout is "
                "not success, so re-check before assuming the event happened.\n"
                "If the result says the wait does NOT block (a watch was registered), do not sleep or poll: "
                "finish your turn with a short message to the user (what is running, what you wait for); "
                "you are woken with a '[Heartbeat ...]' message when it happens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "until": {
                        "type": "string",
                        "description": "Condition to wait for, e.g. 'file_contains(\"build.log\", \"Done\")'. Preferred over delay_seconds."
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Maximum wait for `until` before giving up and waking anyway. Default 600."
                    },
                    "delay_seconds": {
                        "type": "integer",
                        "description": "Fixed sleep in seconds. Use only when no condition can express what you are waiting for."
                    },
                    "condition_to_check": {
                        "type": "string",
                        "description": "Human-readable note on WHY you are waiting; sent back to you on wake-up."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": (
                "LOOK at an image file — a screenshot, a mockup, a diagram, a photo "
                "the user pointed at. Use it when the answer depends on what something LOOKS "
                "like: verifying a layout you just changed, reading a chart, checking a "
                "rendered page.\n"
                "The image arrives in the NEXT message, not in this tool's result, so "
                "continue after calling it and describe what you actually see. Do not claim "
                "to have seen anything before that.\n"
                "Text is still cheaper: for the CONTENT of a web page use read_webpage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the image (png, jpg, gif, webp, bmp). `path` is accepted as an alias."},
                    "path": {"type": "string", "description": "Alias for file_path."},
                    "question": {"type": "string", "description": "Optional: what you are trying to find out, echoed back to keep you on task."},
                    "crop": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Optional bounding box [ymin, xmin, ymax, xmax] normalized to [0..1000] to crop and focus on a specific region."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": "Open a URL in the browser. Creates a browser session if needed. Automatically returns initial page state (URL, title, and interactive elements).\n\nVISIBILITY: By default, the browser runs silently in the background (headed=false). If the user asks you to OPEN, SHOW, VIEW, or DISPLAY a website on their desktop screen, you MUST set headed=true so a real visible browser window appears.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to (e.g. 'https://example.com' or 'http://localhost:3000')."
                    },
                    "session": {
                        "type": "string",
                        "description": "Name of the browser session. Default: 'default'."
                    },
                    "headed": {
                        "type": "boolean",
                        "description": "If true, displays a visible browser window on desktop. If false (default), runs headless."
                    },
                    "return_state": {
                        "type": "boolean",
                        "description": "If true (default), returns updated page state."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "user", "isolated"],
                        "description": "Browser environment: 'user' (user's real browser profile with logins/cookies), 'isolated' (clean temporary Chromium), or 'auto' (default)."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": "Execute an action on the active browser page. PREFER action='act' with instruction for navigating, clicking, and typing (runs in 15ms via local Leeth coprocessor, 0 tokens, avoiding large DOM dumps). Use manual click/type only as fallback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["act", "click", "type", "select", "scroll", "press_key", "upload_file", "wait_for", "back", "reload", "switch_tab", "close_tab", "eval_js"],
                        "description": "Action type. Preferred: 'act' with instruction for fast semantic execution via Leeth. Deterministic: click, type, select, scroll, press_key, wait_for, etc."
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector or element text/label to target."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text an action works with: what to type, the element label to click, the value to set, the JavaScript source for eval_js, the absolute file path for upload_file, the text to wait for in wait_for, or the URL/title match for close_tab."
                    },
                    "value": {
                        "type": "string",
                        "description": "Alias for text: input value or dropdown option."
                    },
                    "key": {
                        "type": "string",
                        "description": "Key name for press_key action (e.g. 'Enter', 'Tab', 'Escape')."
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["down", "up"],
                        "description": "Scroll direction: 'down' (default) or 'up'."
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Scroll distance in pixels (scroll) or the timeout in seconds (wait_for). Default is 500."
                    },
                    "index": {
                        "type": "integer",
                        "description": "Optional numeric index of the interactive element, or tab index for switch_tab / close_tab."
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction for action='act' (e.g. 'click login', 'type python in search and press Enter', 'select filter Комплект'). Handled instantly by Leeth."
                    },
                    "session": {
                        "type": "string",
                        "description": "Browser session name. Default: 'default'."
                    },
                    "return_state": {
                        "type": "boolean",
                        "description": "If true (default), returns updated page state after the action."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_inspect",
            "description": "Inspect the active browser page state, list open tabs, take a screenshot, check console errors, or extract content. Prefer browser_action(action='act') over dumping dom_interactive repeatedly to manually click elements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["dom_interactive", "tabs", "screenshot", "console_errors", "text_content", "html", "observe", "extract"],
                        "description": "Inspection target: 'dom_interactive' (compact accessibility tree of buttons/inputs/links, max 15KB), 'tabs' (list all open browser tabs with indices, titles, and URLs), 'screenshot' (captures visual PNG), 'console_errors' (DevTools JS logs/errors), 'text_content' (readable text), 'html' (markup), 'observe' (semantic element discovery via Stagehand), 'extract' (semantic structured data extraction via Stagehand)."
                    },
                    "selector": {
                        "type": "string",
                        "description": "Optional CSS selector to scope text/html inspection."
                    },
                    "index": {
                        "type": "integer",
                        "description": "Optional numeric element index to inspect."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional custom file path to save screenshot PNG."
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Guidance instruction or data description for 'observe' or 'extract' modes."
                    },
                    "query": {
                        "type": "string",
                        "description": "Filter keyword for dom_interactive or instruction alias."
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
            "description": "Close an open browser session and release all associated processes, or close a specific tab if 'index' or 'text' is provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session": {
                        "type": "string",
                        "description": "Browser session name to close. Default: 'default'."
                    },
                    "index": {
                        "type": "integer",
                        "description": "Optional 1-based index of a specific tab to close. If provided, only this tab is closed instead of the whole browser session."
                    },
                    "text": {
                        "type": "string",
                        "description": "Optional title or URL substring to match and close a specific tab."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_deep_linter",
            "description": "Run a deep static analysis (Pylint) on a file or directory. This is a heavy operation (~5-10s) but it catches complex issues like shadowing, unused variables, and control flow errors that fast AST linters miss. Use this when you've finished major architectural changes or if you suspect hidden bugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to a file or directory to lint. Defaults to the current project directory ('.')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_new_items",
            "description": "Filter a list of items (URLs, IDs, or titles) to return only those not previously reported by this automated or monitoring task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of item identifiers to check against historical run memory."
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional category or description label for the reported batch."
                    }
                },
                "required": ["items"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_automations",
            "description": (
                "Set up, inspect or run work that happens by itself on a schedule while the user is away: "
                "monitoring a page or repository, a daily digest, a recurring check, a reminder. Use it when "
                "the user asks for something 'every day', 'every hour', 'by schedule', 'remind me', or 'keep an eye on'. "
                "Modes: 'agent' (a model works on `task` with limited tools — the default), 'command' (`task` is a shell "
                "command run as-is, no model, cheapest for tests/backups/health checks), 'remind' (`task` is just a message). "
                "Schedules: 'every 30m', 'every 2h on weekdays between 09:00 and 18:00', 'daily at 09:00', 'weekdays at 09:00', "
                "'mon,fri at 18:30', 'monthly on 1 at 09:00', 'in 2h', 'tomorrow at 08:30', 'on start' (Russian works too). "
                "An unattended run cannot ask questions or get approvals: give it only the `tools` it needs and list in "
                "`commands` the exact shell commands it may run (`*` matches arguments, never `;` `&` `|`). "
                "Every change asks the user for approval. Actions: list, show, create, update, remove, enable, disable, run, runs "
                "(history), templates (ready-made: morning, tests, command, page, releases, news, todo, deps)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["list", "show", "create", "update", "remove", "enable", "disable",
                                        "run", "runs", "templates"]},
                    "name": {"type": "string", "description": "Automation name (for create: a short readable one)."},
                    "schedule": {"type": "string", "description": "When it runs, e.g. 'weekdays at 09:00'."},
                    "task": {"type": "string",
                             "description": "What to do (agent), the command line (command), or the message (remind)."},
                    "mode": {"type": "string", "enum": ["agent", "command", "remind"]},
                    "notify": {"type": "string", "enum": ["auto", "failures", "changes", "always", "never"],
                               "description": "When to tell the user: auto = failure or something new; "
                                              "changes = report differs from last time; always = every run."},
                    "tools": {"type": "array", "items": {"type": "string"},
                              "description": "Tools an agent run may use (narrow is safer)."},
                    "commands": {"type": "array", "items": {"type": "string"},
                                 "description": "Shell commands the agent run may execute without approval."},
                    "channels": {"type": "array", "items": {"type": "string"},
                                 "description": "Extra delivery besides the desktop: ['telegram']."},
                    "report_file": {"type": "string",
                                    "description": "Append each report to this .md/.txt/.log/.jsonl file inside the project; "
                                                   "{date} {name} {datetime} are filled in."},
                    "timeout_minutes": {"type": "integer"},
                    "max_turns": {"type": "integer"},
                    "retries": {"type": "integer"},
                    "template": {"type": "string", "description": "For create: start from a ready-made template."},
                    "template_arg": {"type": "string", "description": "The template's argument (URL, repo, command, topic)."},
                    "limit": {"type": "integer", "description": "For runs: how many entries."},
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subagents",
            "description": (
                "Manage the sub-agents you started with run_subagent. action='list' shows them all; 'result' returns "
                "one's report (wait=true blocks until it finishes); 'stop' kills a running one; 'apply' puts the "
                "changes an isolated (worktree) sub-agent made into the real working tree; 'discard' throws them away. "
                "Read a patch (the path is in the report) before applying it, and run the tests afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "result", "stop", "apply", "discard"]},
                    "id": {"type": "integer", "description": "The sub-agent number (#N) from run_subagent."},
                    "wait": {"type": "boolean", "description": "With action='result': block until it is finished."},
                    "timeout_seconds": {"type": "integer", "description": "Longest wait for wait=true (default 600)."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ultracode",
            "description": (
                "Run a BIG job as a planned, checked workflow (Ultracode): a plan of steps, workers in isolated copies "
                "of the tree, the step's own checks, an independent critic reading every patch, an advisor reviewing "
                "the plan and the result. Use it for work that touches several files or modules and has checks; NOT "
                "for questions, explanations or small edits. It works only while the person has switched Ultracode "
                "on (/ultracode on). Blocks until the workflow ends and returns its report; the report tells which "
                "steps were done, what failed and what is left. Put the WHOLE job, every requirement and the checks "
                "the person named into `goal`. action='resume' continues a stopped workflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "resume", "status", "stop"]},
                    "goal": {"type": "string", "description": "For start: the complete job, with how to verify it."},
                    "run_id": {"type": "string", "description": "For resume: a workflow id (default: the last one)."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rooms",
            "description": (
                "Run a reusable, bounded RECIPE (a 'room') instead of improvising a procedure: fix_bug, "
                "debug_loop (red tests -> fix -> re-run until green), code_review (the uncommitted diff), "
                "implement_feature (plan -> the person approves -> code -> tests -> review), release_check, "
                "research (web question with sources), explain_repo, analyze. Rooms have hard step/token/time "
                "limits, verdicts come from processes (the test command's exit code), risky steps (a commit) ask "
                "the person, and a report comes back. action='list' shows the rooms with their parameters; "
                "'show' draws one; 'run' executes it AUTONOMOUSLY with real tools and sub-agents (the person is "
                "asked first, every time) and blocks until the report; 'resume' continues an interrupted run. "
                "'build' writes a NEW room from the person's description and 'edit' changes one (a second model "
                "rewrites the whole room). For PRECISE work use 'patch' (change an existing room) and 'create' "
                "(a new room put together block by block): you give a list of point operations in `ops` and "
                "nothing else is touched; they apply all together or not at all. Whatever the way, the person "
                "sees the drawn flow and decides whether it is saved. "
                "ops (each an object with `op`): "
                "add_node {node:{id,type,...}} | insert_node {node:{...}, between:[from,to]} (a block ON an arrow) | "
                "remove_node {node, reconnect:true} | rename_node {node,new_id} | "
                "set_param {node, field, value} or {node, values:{field:value}} (any block parameter; value null "
                "resets it; dotted paths like args.file_path, sampling.temperature, budget.max_tokens, sets.ok) | "
                "connect {from,to, if:'state.ok' | else:true | while:'..',max,id | on_error:true} (to may be "
                "'EXIT:done') | disconnect {from,to, if|else|all} | set_edge {from,to, if|else (which arrow), "
                "set:{if,else,while,max,id,on_error,to}} | set_room {field,value} (description, budget, entry, "
                "defaults, next, tags, report, rules; dotted paths too). Use rooms(action='show') first to see "
                "the ids and parameters. "
                "Use a room when the task matches one; for anything else work as usual."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "show", "run", "resume", "build", "edit", "patch", "create", "last", "history"]},
                    "room": {"type": "string", "description": "Room name (for show / run / edit / patch; for build and create the name of the new room)."},
                    "task": {"type": "string", "description": "For run: the task in the person's words, complete. For build: what the new room must do and what to do in each case (on failure, on a 'no'). For edit: what to change."},
                    "params": {"type": "object", "description": "For run: parameters of the room, e.g. {\"test_command\": \"python -m pytest tests/unit -q\"}."},
                    "ops": {"type": "array", "items": {"type": "object"}, "description": "For patch / create: the point operations, applied in order to a copy of the room (at most 40). Example: [{\"op\":\"insert_node\",\"node\":{\"id\":\"lint\",\"type\":\"tool\",\"tool\":\"run_command\",\"args\":{\"command\":\"ruff check .\"},\"sets\":{\"lint_ok\":\"ok\"}},\"between\":[\"plan\",\"code\"]},{\"op\":\"set_param\",\"node\":\"critic\",\"field\":\"sampling.temperature\",\"value\":0.2}]"},
                    "save_as": {"type": "string", "description": "For patch: save the changed copy under this name instead of replacing the room (a packaged room is always saved as <name>_custom)."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_inspect",
            "description": (
                "See another program's window as text (Windows UI Automation): its buttons, fields and labels, each "
                "with an id like e3. action='windows' lists the windows, 'tree' shows one window's controls, "
                "'find' filters them by words, 'wait' waits for a window or control to appear. Needs desktop "
                "control switched on by the person; works only in programs they allowed. Always inspect before "
                "desktop_act: its ids come from here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["windows", "tree", "find", "wait"]},
                    "window": {"type": "string", "description": "Part of the window title."},
                    "query": {"type": "string", "description": "For find/wait: words in a control's name, id or value."},
                    "role": {"type": "string", "description": "Optional control type filter: Button, Edit, CheckBox..."},
                    "timeout": {"type": "integer", "description": "For wait: seconds, up to 60 (default 10)."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "desktop_act",
            "description": (
                "Act in another program's window: click, type, set a field's value, press keys, scroll, drag, bring "
                "a window to the front. Use element ids from the LATEST desktop_inspect of that window; the answer shows the "
                "window after the action, so check it before the next step. Every action is shown to the person for "
                "approval. Never for passwords, payments or security dialogs. Does nothing unless the person "
                "switched desktop control on and allowed that program."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["click", "double_click", "right_click", "type", "set_value", "hotkey", "focus",
                                        "scroll", "drag"]},
                    "window": {"type": "string", "description": "Part of the window title."},
                    "element": {"type": "string", "description": "Element id from desktop_inspect, e.g. e3."},
                    "text": {"type": "string", "description": "For type / set_value: the text."},
                    "keys": {"type": "string", "description": "For hotkey: e.g. ctrl+s, alt+f4, enter."},
                    "x": {"type": "integer", "description": "For a click without an element: screen x."},
                    "y": {"type": "integer", "description": "For a click without an element: screen y."},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"],
                                  "description": "For scroll (element = the list or pane; none = the window's middle)."},
                    "amount": {"type": "integer", "description": "For scroll: small steps, 1-20 (default 3)."},
                    "to_element": {"type": "string", "description": "For drag: the element to drop it on."},
                    "to_x": {"type": "integer", "description": "For drag: screen x to drop at."},
                    "to_y": {"type": "integer", "description": "For drag: screen y to drop at."}
                },
                "required": ["action", "window"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory",
            "description": (
                "Long-term memory: notes kept between sessions, for this project and for the user in general. "
                "action='list' shows the index, 'read' opens a note, 'search' finds notes by words, 'write' saves "
                "one (name, description = one line saying what it is for, body), 'forget' moves one to the trash. "
                "Save what will matter in a later session and cannot be read from the code: the user's preferences "
                "and corrections, a decision and its reason, a hidden constraint. Never progress notes or secrets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "read", "search", "write", "forget"]},
                    "name": {"type": "string", "description": "Note name (letters, digits, '-'); 'global:name' for a global one."},
                    "query": {"type": "string", "description": "For search: the words to look for."},
                    "description": {"type": "string", "description": "For write: one line, what the note is for."},
                    "body": {"type": "string", "description": "For write: the text; link other notes as [[name]]."},
                    "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"],
                             "description": "For write: user = who the person is, feedback = how to work with them, "
                                            "project = decisions/goals, reference = where things are."},
                    "scope": {"type": "string", "enum": ["project", "global"],
                              "description": "For write: 'global' only for what holds in EVERY project."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Keep a checklist of the task you are working on and show it to the user. USE IT for any job "
                "with three or more steps, and whenever the user gives you several things to do: write the steps "
                "FIRST, then work through them. Each call REPLACES the whole list, so send every item every time. "
                "Rules: exactly ONE item is in_progress at a time; mark an item completed the moment it is really "
                "done (tests passed, file written) — not before and not in a batch at the end; add steps you discover "
                "on the way. Do not use it for a one-step request or a plain question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete list, in the order the work will happen.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string",
                                            "description": "One concrete step, e.g. 'Add the retry to fetch_page()'."},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "exit_plan_mode",
            "description": (
                "ONLY while plan mode is on: hand the finished plan to the user for approval. Plan mode means you "
                "may look but not change anything. Explore the code, then call this with the COMPLETE plan — goal, "
                "files to change and how, order of steps, how you will verify, risks. If the user approves, plan mode "
                "ends and you can start the work; if not, revise the plan and call it again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "The full plan in Markdown."}
                },
                "required": ["plan"]
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
    try:
        from config import is_local_agent_enabled
        if is_local_agent_enabled() and "delegate_local" not in disabled:
            from local_inference import delegate_local
            tools["delegate_local"] = delegate_local
    except Exception:
        pass
    return tools

def get_tool_schemas(include_hidden: bool = False) -> list[dict]:
    """Returns the JSON schemas for the available tools, dynamically adding RAG if enabled."""
    from config import get_disabled_tools
    disabled = get_disabled_tools()
    
    from src.agent.plan_mode import hides
    schemas = [s for s in TOOL_SCHEMAS
               if s["function"]["name"] not in disabled and not hides(s["function"]["name"])]
    try:
        from rag_engine import is_rag_enabled
        if is_rag_enabled() and "semantic_search" not in disabled:
            # Tell the model which indexed documentation/KBs are actually
            # searchable, so it knows to consult them (the old description only
            # mentioned "the project's codebase").
            kb_note = ""
            kb_ids = []
            try:
                from config import get_external_kbs
                kb_list = [kb for kb in get_external_kbs() if kb.get("enabled", True)]
                kb_names = [kb.get("name", kb.get("id")) for kb in kb_list]
                kb_ids = [kb.get("id") for kb in kb_list]
                if kb_names:
                    kb_note = f" Indexed documentation available: {', '.join(kb_names)}. Target KB IDs: {', '.join(kb_ids)}."
            except Exception:
                pass
            schemas.append({
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": (
                        "Searches the project's codebase AND any indexed documentation / knowledge bases "
                        "(library, framework or API docs) using AI embeddings + keywords. "
                        "ALWAYS call this BEFORE answering a question about an external library, engine or API "
                        "that has indexed docs — it returns exact, sourced snippets instead of relying on memory, "
                        "which prevents hallucinated method names/signatures." + kb_note
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Use 2-4 keywords or a short natural query (e.g. 'Rigidbody.AddForce' or 'where does the player take damage?'). NEVER use a long list of 10+ words (keyword stuffing) as it ruins search relevance."
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of snippets to return (default is 5, recommend keeping under 10)."
                            },
                            "target_kb": {
                                "type": "string",
                                "description": f"Target specific knowledge base to search. 'all' searches everything. 'local' searches only the project code. To search ONLY external docs, use the exact ID (e.g. {kb_ids[0] if kb_ids else 'unity'}). Default is 'all'."
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
    except ImportError:
        pass

    try:
        from config import is_local_agent_enabled
        if is_local_agent_enabled() and "delegate_local" not in disabled:
            from local_inference import DELEGATE_LOCAL_SCHEMA
            schemas.append(DELEGATE_LOCAL_SCHEMA)
    except Exception:
        pass

    return schemas

