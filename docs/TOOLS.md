# Argent tools

<!-- GENERATED FILE — do not edit by hand.
     Regenerate: .venv\Scripts\python.exe scripts/gen_tools_doc.py
     Source: argent/tools/schemas.py + argent/tools/TOOL_GROUPS. -->

59 tools in 10 groups. The model is given one line per tool; the long form — `ops` for rooms, the checks `wait_heartbeat` accepts, the operators `search_web` understands — lives in `skills/argent-tools-reference/SKILL.md` and is read with `read_skill`.

Total description budget: **16810 characters** (paid in every request; `tests/test_tool_contracts.py` keeps it from growing).

## files (10)

| tool | what it does | required |
|---|---|---|
| `copy_file` | Copies a file from one path to another. | `source`, `destination` |
| `create_directory` | Creates a new directory and all parent directories if needed. | `path` |
| `delete_file` | Deletes a file from the file system. | `path` |
| `get_file_outline` | Get the structural outline (classes and methods) of a Python (.py) or C# (.cs) file without reading the entire file… | `path` |
| `list_directory` | Lists all files and subdirectories within a given directory. | `path` |
| `move_file` | Moves or renames a file from one path to another. | `source`, `destination` |
| `read_file` | Reads file content. | `path` |
| `replace_in_file` | Replaces a specific text block in a file with new content, or applies a batch of replacements across one or more… | `path` |
| `run_deep_linter` | Run a deep static analysis (Pylint) on a file or directory. | — |
| `write_file` | Writes content to a file. | `path`, `content` |

## search (3)

| tool | what it does | required |
|---|---|---|
| `analyze_project` | Analyze the current project structure in one call: directory tree, detected languages/frameworks, entry points… | — |
| `grep_search` | Universal search and semantic code navigation. | `pattern` |
| `search_files` | Recursively find files BY NAME or glob (e.g., '*.cs') under a directory, no confirmation needed. | — |

## processes (9)

| tool | what it does | required |
|---|---|---|
| `git_checkpoint` | Save a snapshot of the current files to return to later (git_rollback). | `message` |
| `git_rollback` | Revert all changes to an 'Argent Checkpoint'. | — |
| `list_background_commands` | List the background processes you started this session, with their PID, status (running/exited) and command. | — |
| `read_background_command` | Reads newer output (stdout/stderr) from a currently running background process by PID. | `pid` |
| `read_git_diff` | Read the current unstaged and staged Git differences in the project. | — |
| `run_command` | Runs a CLI shell command on the user's system and returns the stdout and stderr. | `command` |
| `send_background_command` | Sends string input to the stdin of a running background process. | `pid`, `input_string` |
| `start_background_command` | Starts a command in the background (e.g. | `command` |
| `stop_background_command` | Terminates a background process. | `pid` |

## agents (9)

| tool | what it does | required |
|---|---|---|
| `delegate_to_leeth` | Delegates a routine subtask or chain of simple actions (reading files, directory listings, grep search, terminal… | `task` |
| `leeth_click_many` | Click the elements that the LAST leeth_scan_page returned, one after another; each is re-found by label + link before… | `indices` |
| `leeth_fill_form` | Fill the form on the CURRENT browser page from ONE instruction in a single pass with the local Leeth model: text… | `instruction` |
| `leeth_map` | Answer ONE question about MANY texts at once with the local Leeth model (milliseconds per item, 0 API tokens): filter… | `items`, `criterion` |
| `leeth_scan_page` | Read-only: classify the clickable elements (links / buttons) of the CURRENT browser page against a criterion in one… | `criterion` |
| `rooms` | Run a reusable, bounded RECIPE (a 'room') instead of improvising: fix_bug, debug_loop, code_review… | `action` |
| `run_subagent` | Spawn a specialized sub-agent for one self-contained part of the job (research, review, a separate module). | `role`, `task` |
| `subagents` | Manage the sub-agents started with run_subagent: 'list' (all), 'models' (available provider:model targets and… | `action` |
| `ultracode` | Run a BIG job as a planned, checked workflow: a plan of steps, workers in isolated copies of the tree, each step's… | `action` |

## planning (5)

| tool | what it does | required |
|---|---|---|
| `ask_user_questions` | Ask the user a series of structured questions. | `questions` |
| `create_artifact` | Create a Markdown artifact in the .argent/artifacts/ directory. | `filename`, `content` |
| `exit_plan_mode` | ONLY while plan mode is on: hand the finished plan to the user for approval. | `plan` |
| `request_user_approval` | Pause until the person answers. | `message` |
| `todo_write` | Keep a checklist and show it to the person. | — |

## memory (3)

| tool | what it does | required |
|---|---|---|
| `list_skills` | Lists all available markdown-based skills. | — |
| `memory` | Long-term memory: notes kept between sessions, for this project and for the user in general. | `action` |
| `read_skill` | Reads the full instructions of a specific skill. | `name` |

## web (3)

| tool | what it does | required |
|---|---|---|
| `read_webpage` | Reads and extracts the main text content from a specific webpage URL. | `url` |
| `run_deep_research` | Starts an autonomous Deep Research Sub-Agent that deeply researches an objective using search engines, reads top web… | `objective` |
| `search_web` | Searches several engines at once (DuckDuckGo, Wikipedia, StackOverflow, GitHub issues, and Brave when configured) and… | `query` |

## browser (4)

| tool | what it does | required |
|---|---|---|
| `browser_action` | Execute an action on the active browser page. | `action` |
| `browser_close` | Close an open browser session and release all associated processes, or close a specific tab if 'index' or 'text' is… | — |
| `browser_inspect` | Inspect the active browser page state, list open tabs, take a screenshot, check console errors, or extract content. | — |
| `browser_open` | Open a URL in the browser. | `url` |

## desktop (7)

| tool | what it does | required |
|---|---|---|
| `capture_desktop_screen` | Screenshot of the whole screen (primary or all monitors). | — |
| `capture_desktop_window` | Screenshot of ONE window by title substring or HWND. | `target` |
| `desktop_act` | Act in another program's window: click, type, set a field's value, press keys, scroll, drag, bring a window to the… | `action`, `window` |
| `desktop_inspect` | See another program's window as text (Windows UI Automation): its buttons, fields and labels, each with an id like e3. | `action` |
| `get_clipboard_image` | Retrieve an image or screenshot currently stored in the system clipboard (e.g. | — |
| `list_desktop_windows` | List visible top-level windows (HWND, PID, title, geometry); use it to find the target for capture_desktop_window. | — |
| `view_image` | LOOK at an image file — a screenshot, a mockup, a diagram, a photo the user pointed at. | `path` |

## integrations (6)

| tool | what it does | required |
|---|---|---|
| `calculate` | Evaluates exact arithmetic or symbolic math expressions (e.g. | `expression` |
| `call_mcp_tool` | Call a tool on an external MCP server (Unity, GitHub, Slack…), listed in MCP SERVERS below. | `server_name`, `tool_name` |
| `filter_new_items` | Filter a list of items (URLs, IDs, or titles) to return only those not previously reported by this automated or… | `items` |
| `list_mcp_tools` | Look up what an MCP server can do. | — |
| `manage_automations` | Work that happens on a schedule while the person is away: monitoring, a digest, a recurring check, a reminder. | `action` |
| `wait_heartbeat` | Pause while something finishes (a build, a server starting, a background job). | — |

---

See also: `docs/TOOLS.md` is generated; behaviour rules live in `ARGENT_BACKLOG.md`, the plan in `.argent/artifacts/tools-plan.md`.
