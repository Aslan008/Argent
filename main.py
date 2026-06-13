import sys
import os
import signal
import atexit
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import prompt as ptk_prompt # Renamed to avoid conflict with our own prompt
from src.cli.cli_prompt import build_prompt_session
from rich.markdown import Markdown
from rich.console import Console
import questionary

import approval
from agent import ArgentAgent
from config import (
    get_current_model, set_current_model, get_obsidian_vault, set_obsidian_vault,
    get_hooks_dir, set_hooks_dir, get_autonomous_plugins_enabled, set_autonomous_plugins_enabled,
    get_disabled_tools, set_disabled_tools,
    get_provider, set_provider, get_zai_api_key, set_zai_api_key,
    get_zai_endpoint, set_zai_endpoint, ZAI_ENDPOINT_GENERAL, ZAI_ENDPOINT_CODING,
    get_verbose_status, set_verbose_status
)
from ui import (
    console, print_markdown, print_system, print_error,
    print_tool_start, print_tool_end, select_model, 
    print_context_usage, s, get_code_blocks, clear_code_blocks
)
from rich.markup import escape
from command_handler import export_chat_history, handle_slash_command
from hook_manager import hook_manager
from rag_engine import enable_rag_for_project, disable_rag
import subprocess
import json
from pathlib import Path
from project_manager import ProjectManager
from session import save_session, load_session, list_sessions, delete_session, get_last_session
from file_tracker import snapshot, get_diff, undo, get_pending_changes, undo_all
from prompts import (
    build_spec_prompt, build_work_investigation_prompt,
    build_work_planning_prompt, build_planning_prompt, build_architecture_prompt
)
from src.cli.cli_ui import render_response_stream
from src.project.orchestrator import ProjectOrchestrator

# Default tools allowed in regular chat (excludes Project Brain tools and bloat OS tools)
CHAT_ALLOWED_TOOLS = [
    "read_file", "write_file", "append_to_file", "delete_file", "replace_in_file", "replace_python_function",
    "grep_search", "search_files", "run_command", "run_admin_command",
    "start_background_command", "read_background_command", "send_background_command",
    "stop_background_command", "search_web", "read_webpage", "get_file_outline", 
    "multi_replace_in_file", "write_obsidian_note", "search_obsidian_notes", 
    "update_obsidian_properties", "semantic_search", "create_plugin", "delete_plugin",
    "create_skill", "read_skill", "list_skills", "delete_skill",
    "ask_user_questions", "wait_heartbeat", "end_auto_mode",
    "browser_open", "browser_state", "browser_click", "browser_input",
    "browser_screenshot", "browser_scroll", "browser_get_content", "browser_close",
    "call_mcp_tool", "calculate"
]



def offer_safety_checkpoint(task: str) -> None:
    """Before unattended /auto work: offer one git checkpoint if the tree is
    dirty. A clean tree needs no insurance; outside git there is nothing to do.
    Rollback path: git_rollback (only ever touches 'Argent Checkpoint' commits)."""
    try:
        is_git = subprocess.run("git rev-parse --is-inside-work-tree",
                                shell=True, capture_output=True, text=True)
        if is_git.returncode != 0:
            return
        dirty = subprocess.run("git status --porcelain",
                               shell=True, capture_output=True, text=True).stdout.strip()
        if not dirty:
            return
        approved = questionary.confirm(
            "Рабочее дерево содержит незакоммиченные изменения. "
            "Создать страховочный git-чекпоинт перед автономной работой?",
            default=True
        ).ask()
        if approved:
            from tools.misc_tools import git_checkpoint
            from ui import print_system as _ps
            _ps(git_checkpoint(f"перед /auto: {task[:60]}"))
    except Exception:
        pass


def main():
    agent = None

    def cleanup():
        nonlocal agent
        # Shut down browser engine
        try:
            from browser_engine import browser_engine
            browser_engine.run(browser_engine.shutdown(), timeout=5)
        except Exception:
            pass
        # Stop background command processes
        try:
            from tools import ACTIVE_PROCESSES
            for pid, proc_info in list(ACTIVE_PROCESSES.items()):
                try:
                    proc_info["process"].terminate()
                except Exception:
                    pass
        except Exception:
            pass
        # Stop MCP servers
        try:
            from mcp_client import mcp_client
            mcp_client.stop_all()
        except Exception:
            pass
        # Stop background agents or hooks if any
        if agent and agent.messages:
            try:
                save_session(agent.messages, {
                    "model": agent.model_name,
                    "provider": get_provider(),
                })
            except Exception:
                pass

    atexit.register(cleanup)

    os.system('cls' if os.name == 'nt' else 'clear')
    console.rule("[bold cyan]Argent Coder[/bold cyan]")
    print_system("Argent Coder. Autonomous Development Environment.")
    print_system("Type /help for commands.")
    
    agent = ArgentAgent()
    orchestrator = ProjectOrchestrator(agent)
    
    builtin_cmds = [
        # Base commands
        '/help', '/provider', '/model', '/clear', '/research', '/rag_toggle',
        '/hooks', '/tools', '/save', '/project', '/work', '/commit',
        '/sessions', '/load', '/copy', '/logs', '/skills', '/auto', '/verbose', '/debug', '/browser', '/exit', '/quit',
        '/mcp', '/thinking', '/temp', '/temperature',
        '/cd', '/undo', '/diff', '/changes', '/stats',
        
        # Subcommands and parameter variations
        '/mcp list', '/mcp add', '/mcp remove', '/mcp test', '/mcp start', '/mcp stop',
        '/browser user', '/browser isolated', '/browser chrome', '/browser yandex', '/browser edge', '/browser brave', '/browser auto',
        '/hooks auto',
        '/temp 0.2', '/temp 0.7', '/temp 1.0',
        '/temperature 0.2', '/temperature 0.7', '/temperature 1.0',
        '/work --auto',
        '/logs clear', '/logs error'
    ]
    
    def get_all_commands():
        custom_names = [f"/{c}" for c in hook_manager.get_custom_commands().keys()]
        return builtin_cmds + custom_names

    # Shared mutable state the bottom toolbar reads each keystroke.
    ui_state = {"mode": "CHAT"}
    prompt_session = build_prompt_session(
        get_all_commands, agent, ui_state,
        history_path=Path(".argent") / "input_history",
    )
    
    print_system(f"Active Provider: {get_provider().upper()}")
    print_system(f"Active Model: {get_current_model()}")
    print_system(f"Working Directory: {os.getcwd()}")
    vault = get_obsidian_vault()
    if vault:
        print_system(f"Obsidian Vault: {vault}")

    from config import get_mcp_servers
    from mcp_client import mcp_client
    mcp_servers = get_mcp_servers()
    if mcp_servers:
        for srv in mcp_servers:
            name = srv["name"]
            cfg = {k: v for k, v in srv.items() if k != "name"}
            try:
                result = mcp_client.register_and_start(name, cfg)
                print_system(result)
            except Exception as e:
                print_system(f"MCP server '{name}': failed to start — {e}")

    # Offer to restore last session
    last = get_last_session()
    if last and last.get("preview"):
        restore = questionary.confirm(
            f"Last session found ({last['saved_at'][:16]}): \"{last['preview']}\". Restore?"
        ).ask()
        if restore:
            data = load_session(last["id"])
            if data and data.get("messages"):
                agent.messages = data["messages"]
                print_system(f"Restored {len(data['messages'])} messages from last session.")

    # Trigger Startup Hook
    hook_manager.call_hook("on_startup")

    is_project_mode = False
    is_auto_mode = False
    auto_sleep_time = 0
    auto_wake_context = ""
    auto_continue_input = None
    last_task_id = None
    task_retries = 0
    project_iterations = 0
    MAX_TASK_RETRIES = 3
    turn_counter = 0
    
    from config import get_auto_rag
    if get_auto_rag():
        import threading
        cwd = os.getcwd()
        print_system(f"Auto-RAG is enabled. Indexing {cwd} in background...")
        def _bg_auto_rag():
            try:
                from rag_engine import enable_rag_for_project
                result = enable_rag_for_project(cwd)
                print(f"\n[RAG Status] {result}")
            except ImportError:
                print("\n[RAG Status] ChromaDB not installed. Auto-RAG failed.")
        threading.Thread(target=_bg_auto_rag, daemon=True).start()

    while True:
        try:
            print() # Visual spacing
            
            if is_auto_mode:
                if auto_sleep_time > 0:
                    print_system(f"*[Heartbeat] Переход в сон на {auto_sleep_time} сек. (Ctrl+C для прерывания)*")
                    import time
                    try:
                        time.sleep(auto_sleep_time)
                    except KeyboardInterrupt:
                        print_system("Состояние Heartbeat прервано. Выход из автоматического режима.")
                        is_auto_mode = False
                        continue
                    auto_sleep_time = 0
                
                user_input = auto_wake_context if auto_wake_context else "[Режим Автоматизма] Продолжай автономную работу. Анализируй результат предыдущего шага. Если нужно подождать — используй `wait_heartbeat`. Если глобальная задача завершена — вызови `end_auto_mode`."
                auto_wake_context = ""
                print_system("\n❯ [Автономный импульс]")
            elif auto_continue_input:
                user_input = auto_continue_input
                auto_continue_input = None
                print_system("Продолжение рабочего процесса...")
            else:
                try:
                    user_input = prompt_session.prompt("❯ ")
                except EOFError:
                    break
                    
                if not user_input.strip():
                    continue
                    
                user_input = hook_manager.call_modifier_hook("pre_prompt", user_input)
                if not user_input.strip():
                    continue
                    
                # Reset project mode on new manual input
                is_project_mode = False
                project_iterations = 0
            
            active_tools = None

            if user_input.startswith("/research"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Please specify a topic. Example: /research Unity DOTS")
                    continue
                topic = parts[1].strip()
                active_tools = ["run_deep_research"]
                user_input = (
                    f"Please act as an autonomous Research Agent for the topic: '{topic}'.\n\n"
                    f"Your STRICT workflow is:\n"
                    f"1. MUST CALL `run_deep_research(objective='{topic}')` right now to let the sub-agent gather massive information.\n"
                    f"2. Read the final synthesized report returned by the sub-agent.\n"
                    f"3. Present the findings directly to me in a highly structured, readable, and detailed format right here in the chat.\n"
                    f"Do NOT write any notes to Obsidian unless I explicitly ask you to do so. Just give me the info.\n"
                    f"Do not ask for permission, start by calling `run_deep_research` right away." 
                )
                print_system(f"Starting auto-research on: {topic}...")
            elif user_input.startswith("/auto"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Укажите задачу. Пример: /auto Написать проект на python")
                    continue
                auto_task = parts[1].strip()
                offer_safety_checkpoint(auto_task)
                is_auto_mode = True
                auto_sleep_time = 0
                auto_wake_context = ""
                user_input = (
                    f"Активирован режим полного автоматизма 'Экспериментатор'.\n\n"
                    f"Ваша общая задача: {auto_task}\n\n"
                    f"ПРАВИЛА:\n"
                    f"1. Вы полностью самостоятельны. Используйте все доступные инструменты для разведки, планирования, написания кода или тестирования.\n"
                    f"2. Вам не нужно ждать ответа или одобрения от пользователя, просто продолжайте работать.\n"
                    f"3. Вы можете развивать идею задачи. Если считаете нужным добавить функционал — добавляйте.\n"
                    f"4. Если вам нужно 'уснуть' и дождаться события (окончание установки, ответ сервера и т.д.), используйте инструмент `wait_heartbeat`.\n"
                    f"5. Когда вы ПОЛНОСТЬЮ закончите работу над задачей, обязательно вызовите инструмент `end_auto_mode`."
                )
                print_system(f"🚀 Запущен автоматический режим для: {auto_task}")
                is_project_mode = False
            elif user_input.startswith("/browser"):
                parts = user_input.strip().split()
                if len(parts) == 1:
                    # Show current status and detected browsers
                    from config import get_browser_mode, get_browser_name
                    from browser_detect import detect_browsers
                    mode = get_browser_mode()
                    name = get_browser_name()
                    browsers = detect_browsers()
                    print_system(f"Browser Mode: [bold cyan]{mode}[/bold cyan]")
                    print_system(f"Selected Browser: [bold cyan]{name}[/bold cyan]")
                    if browsers:
                        print_system("Detected browsers:")
                        for b in browsers:
                            print_system(f"  - [bold yellow]{b.key}[/bold yellow]: {b.exe_path}")
                    else:
                        print_system("[dim]No Chromium-based browsers detected.[/dim]")
                    print_system("\nUsage: /browser user | isolated | chrome | yandex | edge | brave | auto")
                elif parts[1] in ("user", "isolated"):
                    from config import set_browser_mode
                    set_browser_mode(parts[1])
                    if parts[1] == "user":
                        print_system("Browser mode: [bold green]USER[/bold green] — AI will use your real browser via CDP.")
                    else:
                        print_system("Browser mode: [bold yellow]ISOLATED[/bold yellow] — AI will use Playwright Chromium.")
                elif parts[1] in ("chrome", "yandex", "edge", "brave", "auto"):
                    from config import set_browser_name, set_browser_mode
                    set_browser_name(parts[1])
                    set_browser_mode("user")
                    print_system(f"Browser set to: [bold green]{parts[1]}[/bold green] (user mode enabled).")
                else:
                    print_error(f"Unknown browser option: {parts[1]}. Valid: user, isolated, chrome, yandex, edge, brave, auto")
                continue

            elif user_input.strip() == "/rag_toggle":
                from config import get_auto_rag, set_auto_rag
                current = get_auto_rag()
                new_state = not current
                set_auto_rag(new_state)
                
                status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
                print_system(f"Auto-RAG (Semantic Search indexing on startup) is now: {status}")
                
                if new_state:
                    print_system("RAG will be enabled the next time you start Argent.")
                    print_system("If you want to start it now without restarting, type `/work` and the agent can use semantic search if it's already active.")
                else:
                    try:
                        from rag_engine import disable_rag
                        disable_rag()
                        print_system("Semantic Search (RAG) has been disabled for the current session.")
                    except ImportError:
                        pass
                continue

            elif user_input.startswith("/skills"):
                from skill_manager import skill_manager
                skills = skill_manager.list_skills()
                if not skills:
                    print_system("No skills found. You can create one via `create_skill` tool.")
                else:
                    print_system("[bold cyan]Available Skills:[/bold cyan]")
                    for skill in skills:
                        print_system(f"- [bold yellow]{skill['name']}[/bold yellow]: {skill['description']}")
                continue
            elif user_input.startswith("/hooks"):
                parts = user_input.split(" ")
                if len(parts) == 1:
                    status = "ENABLED" if get_autonomous_plugins_enabled() else "DISABLED"
                    print_system(f"Current Hooks Directory: [bold cyan]{get_hooks_dir()}[/bold cyan]")
                    print_system(f"Autonomous Plugin Creation: [bold yellow]{status}[/bold yellow]")
                    
                    # Add interactive plugin toggle
                    from config import get_disabled_plugins, set_disabled_plugins
                    
                    hooks_dir = Path(get_hooks_dir()).expanduser().resolve()
                    if hooks_dir.exists():
                        all_plugins = [item.stem for item in hooks_dir.iterdir() 
                                     if item.is_file() and item.suffix == ".py" and not item.name.startswith("_")]
                        
                        if all_plugins:
                            disabled = set(get_disabled_plugins())
                            choices = [
                                questionary.Choice(p, checked=(p not in disabled))
                                for p in all_plugins
                            ]
                            
                            selected_plugins = questionary.checkbox(
                                "Select the plugins you want to ENABLE:",
                                choices=choices
                            ).ask()
                            
                            if selected_plugins is not None:
                                new_disabled = [p for p in all_plugins if p not in selected_plugins]
                                set_disabled_plugins(new_disabled)
                                hook_manager.reload_plugins()
                                print_system(f"Plugins updated. Disabled: {', '.join(new_disabled) if new_disabled else 'None'}")
                        else:
                            print_system("No plugins found in the directory.")
                    else:
                        print_error(f"Hooks directory not found: {hooks_dir}")
                elif parts[1].lower() == "auto":
                    if len(parts) > 2:
                        val = parts[2].lower()
                        if val in ("on", "true", "yes", "1"):
                            set_autonomous_plugins_enabled(True)
                            print_system("Autonomous Plugin Creation [bold green]ENABLED[/bold green]. AI can now create tools on its own.")
                        else:
                            set_autonomous_plugins_enabled(False)
                            print_system("Autonomous Plugin Creation [bold red]DISABLED[/bold red]. AI will only create plugins when asked.")
                    else:
                        status = "ENABLED" if get_autonomous_plugins_enabled() else "DISABLED"
                        print_system(f"Autonomous Plugin Creation is currently: [bold yellow]{status}[/bold yellow]")
                else:
                    new_path = user_input.split(" ", 1)[1].strip()
                    set_hooks_dir(new_path)
                    hook_manager.reload_plugins(new_path)
                    print_system(f"Hooks Directory changed to: [bold green]{new_path}[/bold green]")
                continue

            elif user_input.strip() == "/tools":
                from tools import AVAILABLE_TOOLS
                all_tools = list(AVAILABLE_TOOLS.keys())
                
                # Check if RAG is available to show it in the list
                try:
                    from rag_engine import is_rag_enabled
                    if is_rag_enabled() and "semantic_search" not in all_tools:
                        all_tools.append("semantic_search")
                except ImportError:
                    pass
                
                disabled = set(get_disabled_tools())
                
                choices = [
                    questionary.Choice(t, checked=(t not in disabled))
                    for t in all_tools
                ]
                
                selected_tools = questionary.checkbox(
                    "Select the tools you want Argent to have access to:",
                    choices=choices
                ).ask()
                
                if selected_tools is not None:
                    new_disabled = [t for t in all_tools if t not in selected_tools]
                    set_disabled_tools(new_disabled)
                    print_system(f"Tools updated. Disabled tools: {', '.join(new_disabled) if new_disabled else 'None'}")
                continue
            elif user_input.startswith("/save"):
                parts = user_input.split(" ", 1)
                filename = parts[1].strip() if len(parts) > 1 else None
                export_chat_history(agent, filename=filename, auto=False)
                # Also save as a restorable session
                try:
                    save_session(agent.messages, {
                        "model": agent.model_name,
                        "provider": get_provider(),
                    })
                except Exception:
                    pass
                continue
            elif user_input.strip() == "/sessions":
                sessions = list_sessions()
                if not sessions:
                    print_system("No saved sessions found.")
                else:
                    console.print("[bold cyan]Saved Sessions:[/bold cyan]")
                    for i, s in enumerate(sessions[:20]):
                        date = s.get("saved_at", "")[:16]
                        model = s.get("model", "?")
                        preview = s.get("preview", "")
                        count = s.get("message_count", 0)
                        console.print(f"  [dim][{i+1}][/dim] {date} [dim]|[/dim] [cyan]{model}[/cyan] [dim]|[/dim] {count} msgs [dim]|[/dim] [italic]\"{preview}\"[/italic]")
                    print_system("Use /load <number> to restore a session.")
                continue
            elif user_input.startswith("/load"):
                parts = user_input.strip().split()
                if len(parts) < 2:
                    print_error("Usage: /load <session-number>")
                    continue
                sessions = list_sessions()
                try:
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(sessions):
                        data = load_session(sessions[idx]["id"])
                        if data and data.get("messages"):
                            agent.messages = data["messages"]
                            print_system(f"Restored {len(data['messages'])} messages from {sessions[idx].get('saved_at', '')[:16]}.")
                        else:
                            print_error("Failed to load session data.")
                    else:
                        print_error("Invalid session number.")
                except ValueError:
                    print_error("Please enter a valid number.")
                continue

            elif user_input.startswith("/copy"):
                import pyperclip
                blocks = get_code_blocks()
                parts = user_input.strip().split()
                if not blocks:
                    print_system("No code blocks in current response.")
                elif len(parts) < 2:
                    print_system("Usage: /copy <number>")
                    print_system(f"Available blocks: {', '.join(f'[{b['index']}] {b['lang']}' for b in blocks)}")
                else:
                    try:
                        idx = int(parts[1])
                        block = next((b for b in blocks if b['index'] == idx), None)
                        if block:
                            pyperclip.copy(block['code'])
                            print_system(f"Copied block [{idx}] ({block['lang']}, {len(block['code'].splitlines())} lines) to clipboard.")
                        else:
                            print_error(f"Block {idx} not found. Available: 1-{len(blocks)}")
                    except ValueError:
                        print_error("Usage: /copy <number>")
                continue

            elif user_input.startswith("/logs"):
                log_dir = Path.home() / ".argent" / "logs"
                parts = user_input.strip().split()
                
                if len(parts) > 1 and parts[1] == "clear":
                    for f in log_dir.glob("*.log"):
                        f.write_text("", encoding="utf-8")
                    print_system("All logs cleared.")
                    continue
                
                errors_only = "error" in parts
                module_filter = None
                count = 30
                for p in parts[1:]:
                    if p == "error":
                        continue
                    elif p.isdigit():
                        count = int(p)
                    else:
                        module_filter = p
                
                log_files = sorted(log_dir.glob("*.log"))
                if not log_files:
                    print_system("No log files found at ~/.argent/logs/")
                    continue
                
                if module_filter:
                    log_files = [f for f in log_files if f.stem == module_filter]
                    if not log_files:
                        available = ", ".join(f.stem for f in sorted(log_dir.glob("*.log")))
                        print_error(f"No log '{module_filter}'. Available: {available}")
                        continue
                
                output_lines = []
                for lf in log_files:
                    try:
                        lines = lf.read_text(encoding="utf-8").splitlines()
                    except Exception:
                        continue
                    if errors_only:
                        lines = [l for l in lines if "[ERROR]" in l or "[WARNING]" in l]
                    recent = lines[-count:]
                    if recent:
                        output_lines.append(f"\n[bold cyan]--- {lf.stem}.log ---[/bold cyan]")
                        for line in recent:
                            if "[ERROR]" in line:
                                output_lines.append(f"[red]{escape(line)}[/red]")
                            elif "[WARNING]" in line:
                                output_lines.append(f"[yellow]{escape(line)}[/yellow]")
                            else:
                                output_lines.append(f"[dim]{escape(line)}[/dim]")
                
                if output_lines:
                    for line in output_lines:
                        console.print(line)
                else:
                    print_system("No log entries found.")
                continue
            
            elif user_input.strip() == "/stats":
                from config import (
                    get_context_window, get_model_size_category,
                    get_model_category_override
                )
                
                model = get_current_model()
                provider = get_provider()
                ctx = get_context_window()
                msg_count = len(agent.messages)
                
                category = get_model_size_category(model)
                override = get_model_category_override()
                cat_source = "manual" if override else "auto"
                
                mcp_servers = mcp_client.get_servers()
                active_mcp = [s['name'] for s in mcp_servers if s['running']]
                
                plugins = list(hook_manager.plugins.keys())
                
                stats_msg = (
                    f"[bold cyan]Argent Diagnostics:[/bold cyan]\n"
                    f"  [dim]Directory:[/dim] {os.getcwd()}\n"
                    f"  [dim]Model:[/dim] {model} ({provider})\n"
                    f"  [dim]Classification:[/dim] {category} ({cat_source})\n"
                    f"  [dim]Context Window:[/dim] {ctx} tokens\n"
                    f"  [dim]History:[/dim] {msg_count} messages\n"
                )
                
                if active_mcp:
                    stats_msg += f"  [dim]MCP Servers:[/dim] {', '.join(active_mcp)}\n"
                if plugins:
                    stats_msg += f"  [dim]Plugins:[/dim] {', '.join(plugins)}\n"
                    
                print_system(stats_msg)
                continue

            elif user_input.startswith("/cd"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_system(f"Current Working Directory: [bold cyan]{os.getcwd()}[/bold cyan]")
                    print_system("Usage: /cd <path>")
                    continue
                target_dir = parts[1].strip()
                try:
                    resolved = Path(target_dir).expanduser().resolve()
                    if not resolved.exists():
                        print_error(f"Directory does not exist: {resolved}")
                        continue
                    if not resolved.is_dir():
                        print_error(f"Not a directory: {resolved}")
                        continue
                    os.chdir(resolved)
                    print_system(f"Working Directory changed to: [bold green]{os.getcwd()}[/bold green]")
                except Exception as e:
                    print_error(f"Failed to change directory: {e}")
                continue

            elif user_input.startswith("/undo"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Usage: /undo <file_path>")
                    continue
                result = undo(parts[1].strip())
                print_system(result)
                continue

            elif user_input.startswith("/diff"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Usage: /diff <file_path>")
                    continue
                result = get_diff(parts[1].strip())
                if result.startswith("Error") or result.startswith("No") or result.startswith("File"):
                    print_system(result)
                else:
                    from rich.syntax import Syntax
                    syntax = Syntax(result, "diff", theme="monokai", word_wrap=True)
                    console.print(syntax)
                continue

            elif user_input.strip() == "/changes":
                changes = get_pending_changes()
                if not changes:
                    print_system("No tracked file changes in this session.")
                else:
                    print_system("[bold cyan]Tracked File Changes:[/bold cyan]")
                    for ch in changes:
                        print_system(f"  - {ch['key']} ({ch['snapshot_count']} snapshots)")
                    print_system("\nUse /diff <path> to see changes, /undo <path> to restore.")
                continue

            elif user_input.startswith("/project"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Please specify a project prompt. Example: /project Build a Snake game in Python")
                    continue
                proj_prompt = parts[1].strip()
                is_project_mode, user_input = orchestrator.start_project(proj_prompt)
                

                
            elif user_input.startswith("/work"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Please specify a work task. Example: /work Fix the null reference in player.py")
                    continue
                work_prompt = parts[1].strip()
                
                auto_mode = False
                if work_prompt.startswith("--auto"):
                    auto_mode = True
                    work_prompt = work_prompt.replace("--auto", "", 1).strip()
                    if not work_prompt:
                        print_error("Please specify a work task after --auto.")
                        continue
                
                is_project_mode, user_input = orchestrator.start_work(work_prompt, auto_mode=auto_mode)

            elif user_input.startswith("/commit"):
                try:
                    git_check = subprocess.run("git rev-parse --git-dir", capture_output=True, text=True)
                    if git_check.returncode != 0:
                        print_error("Not a git repository. Navigate to a git project first.")
                        continue
                    
                    staged_diff = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True).stdout
                    if not staged_diff.strip():
                        print_error("No staged changes found. Use 'git add' first.")
                        continue
                        
                    print_system("Generating commit message based on staged changes...")
                    
                    commit_prompt = (
                        "You are a Senior Developer. Generate a concise, professional Git commit message following Conventional Commits "
                        "specification based on the following diff. Only output the commit message, nothing else.\n\n"
                        f"{staged_diff}"
                    )
                    
                    gen_message = ""
                    try:
                        from providers import create_provider
                        provider = create_provider()
                        gen_message = provider.sync_chat(
                            model=agent.model_name,
                            messages=[{"role": "user", "content": commit_prompt}]
                        ).strip().strip('"').strip("'")
                    except Exception as e:
                        print_error(f"Failed to generate commit message: {e}")
                        continue
                    
                    if not gen_message:
                        print_error("Failed to generate commit message.")
                        continue
                        
                    print_system(f"Suggested commit message:\n[bold cyan]{gen_message}[/bold cyan]")
                    
                    approved = questionary.confirm("Do you want to commit with this message?").ask()
                    if approved:
                        subprocess.run(["git", "commit", "-m", gen_message], check=True)
                        print_system("Commited successfully!")
                    else:
                        custom_msg = questionary.text("Enter custom commit message (leave empty to prevent commit):").ask()
                        if custom_msg:
                            subprocess.run(["git", "commit", "-m", custom_msg], check=True)
                            print_system("Commited successfully with custom message!")
                        else:
                            print_system("Commit aborted.")
                            
                except Exception as e:
                    print_error(f"Error during commit: {e}")
                continue

            elif user_input.startswith("/"): #
                should_exit = handle_slash_command(user_input, agent)
                if should_exit:
                    break
                continue
                
            if user_input.startswith("!"):#
                cmd = user_input[1:].strip()
                if not cmd:
                    continue
                print_system(f"Running local command: {cmd}")
                
                try:
                    import shlex
                    try:
                        cmd_parts = shlex.split(cmd, posix=(os.name != 'nt'))
                    except ValueError:
                        cmd_parts = cmd.split()
                    result = subprocess.run(
                        cmd_parts,
                        shell=False,
                        capture_output=True,
                        text=True
                    )
                    out = result.stdout.strip()
                    err = result.stderr.strip()
                    
                    if out:
                        print(f"{out}")
                    if err:
                        print_error(f"{err}")
                        
                    if result.returncode != 0:
                        print_error(f"Command failed with exit code {result.returncode}.")
                        ask_fix = questionary.confirm("Would you like Argent to help fix this error?").ask()
                        if ask_fix:
                            user_input = (
                                f"I ran the command `{cmd}` and it failed with exit code {result.returncode}.\n"
                                f"STDOUT:\n{out}\n"
                                f"STDERR:\n{err}\n"
                                f"Please analyze the error and help me fix it."
                            )
                        else:
                            continue
                    else:
                        continue
                except Exception as e:
                    print_error(f"Failed to execute command: {e}")
                    continue
                
            if is_project_mode:
                active_tools = orchestrator.get_active_tools() or CHAT_ALLOWED_TOOLS
            else:
                # Regular chat mode
                active_tools = CHAT_ALLOWED_TOOLS

            # Sync the approval policy with the current mode: in autonomous mode
            # safe actions are auto-approved, destructive ones still prompt.
            approval.set_policy(approval.POLICY_AUTO if is_auto_mode else approval.POLICY_ASK)

            # Reflect the active mode in the status bar.
            ui_state["mode"] = "AUTO" if is_auto_mode else ("PROJECT" if is_project_mode else "CHAT")

            response_chunks = agent.process_user_input(user_input, allowed_tools=active_tools)
            streamed_text, is_auto_mode, auto_sleep_time, auto_wake_context = render_response_stream(
                agent, response_chunks, is_auto_mode=is_auto_mode
            )
            
            # Show context usage after response
            usage = agent.get_context_usage()
            print_context_usage(usage["tokens"], usage["max"], usage["percent"])
            
            # Auto-save session every 5 turns
            turn_counter += 1
            if turn_counter % 5 == 0:
                try:
                    save_session(agent.messages, {
                        "model": agent.model_name,
                        "provider": get_provider(),
                    })
                except Exception:
                    pass
            
            # Trigger Post Response Hook
            if agent.messages and agent.messages[-1].get("role") in ("assistant", "model"):
                hook_manager.call_hook("post_response", agent.messages[-1].get("content", ""))
                    
            # === Project Brain: State Machine ===
            if is_project_mode:
                is_project_mode, auto_continue_input = orchestrator.step()
            
        except KeyboardInterrupt:
            # Heal the history: an interrupted turn may have left tool_calls
            # without matching tool results, which would poison the next request.
            try:
                agent.repair_history()
            except Exception:
                pass
            if is_auto_mode:
                is_auto_mode = False
                print_system("\n[bold yellow]Выполнение прервано пользователем (Ctrl+C). Выход из автоматического режима.[/bold yellow]")
            continue
        except EOFError:
            export_chat_history(agent, auto=True)
            break
        except Exception as e:
            print_error(f"Main loop error: {e}")
            continue
    
    print_system("Goodbye!")

if __name__ == "__main__":
    main()