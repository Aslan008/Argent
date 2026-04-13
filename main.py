import sys
import os
import signal
import atexit
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.shortcuts import prompt as ptk_prompt # Renamed to avoid conflict with our own prompt
from rich.markdown import Markdown
from rich.console import Console
import questionary

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
from hook_manager import hook_manager
from rag_engine import enable_rag_for_project, disable_rag
import subprocess
import json
from pathlib import Path
from project_manager import ProjectManager
import ollama
from session import save_session, load_session, list_sessions, delete_session, get_last_session
from file_tracker import snapshot, get_diff, undo, get_pending_changes, undo_all
from pipeline import Pipeline

# Default tools allowed in regular chat (excludes Project Brain tools)
CHAT_ALLOWED_TOOLS = [
    "read_file", "write_file", "delete_file", "replace_in_file", "replace_python_function",
    "list_directory", "grep_search", "search_files", "run_command", "run_admin_command",
    "start_background_command", "read_background_command", "send_background_command",
    "stop_background_command", "search_web", "read_webpage", "get_file_outline", 
    "multi_replace_in_file", "write_obsidian_note", "search_obsidian_notes", 
    "update_obsidian_properties", "semantic_search", "create_plugin", "delete_plugin",
    "create_skill", "read_skill", "list_skills", "delete_skill", "create_svg_image",
    "ask_user_questions", "create_directory", "move_file", "copy_file"
]


def _build_spec_prompt(objective: str, architecture: str, filename: str) -> str:
    """Build the Phase 1b prompt for detailing one file's specification."""
    return (
        f"You are writing a DETAILED specification for ONE file in the project: '{objective}'.\n\n"
        f"=== PROJECT ARCHITECTURE ===\n"
        f"{architecture}\n\n"
        f"Your task: write a DETAILED specification for the file: {filename}\n\n"
        f"You MUST call `write_file_spec(filename='{filename}', spec=...)` with a TEXT DESCRIPTION (NOT code!).\n\n"
        f"CRITICAL: Write a TEXT DESCRIPTION, NOT Python/C#/JS code! The spec must describe WHAT to implement, "
        f"not be the implementation itself.\n\n"
        f"Your spec MUST include:\n"
        f"1. File path\n"
        f"2. ALL imports/dependencies (exact module names)\n"
        f"3. ALL classes: name, inheritance\n"
        f"4. ALL methods/functions: name, ALL parameters with types, return type, and WHAT THE METHOD DOES (1-2 sentences of logic)\n"
        f"5. ALL fields/variables: name, type, default value\n"
        f"6. Which other project files this file imports and what names it uses from them\n\n"
        f"EXAMPLE OF A GOOD SPEC (this is what you should write):\n"
        f"  File: converter.py\n"
        f"  Imports: json (standard library)\n"
        f"  Dependencies: uses get_rate() from api_client.py\n"
        f"  Class: CurrencyConverter\n"
        f"    Fields:\n"
        f"      - rates_cache: dict, default empty dict\n"
        f"    Methods:\n"
        f"      - __init__(self): initializes empty rates_cache\n"
        f"      - convert(self, amount: float, from_cur: str, to_cur: str) -> float: calls get_rate(), multiplies amount by rate, returns result\n"
        f"      - supported_currencies(self) -> list[str]: returns hardcoded list of supported currency codes\n\n"
        f"EXAMPLE OF A BAD SPEC (DO NOT write code like this):\n"
        f"  def convert(self, amount, from_cur, to_cur):\n"
        f"      rate = get_rate(from_cur, to_cur)\n"
        f"      return amount * rate\n\n"
        f"BE EXTREMELY SPECIFIC with names and types. The AI that implements this will have NO OTHER CONTEXT.\n\n"
        f"Call `write_file_spec` NOW!"
    )

def _build_work_investigation_prompt(objective: str, research_context: str = "") -> str:
    """Build the Phase 1 prompt for investigating an existing codebase."""
    res = (
        f"You are the Lead Investigator for an existing codebase. Your objective is: '{objective}'.\n\n"
    )
    if research_context:
        res += f"=== LATEST RESEARCH CONTEXT ===\n{research_context}\n===============================\n\n"
    res += (
        f"Your task is to INVESTIGATE the current codebase and devise a plan to achieve the objective.\n"
        f"RULES:\n"
        f"1. You MUST use tools like `list_directory`, `grep_search`, and CRITICALLY `read_file` to examine the ACTUAL CODE.\n"
        f"2. You MUST NOT guess or hallucinate class names, method names, or file paths. READ THE FILES FIRST.\n"
        f"3. When you have a complete plan, call `plan_work_changes` with:\n"
        f"   - strategy: Detailed explanation of how you will solve the objective.\n"
        f"   - files_to_edit: Comma-separated list of EXISTING files to modify.\n"
        f"   - files_to_create: Comma-separated list of NEW files to create (leave empty if none).\n"
        f"4. ONLY call `plan_work_changes` when you are absolutely sure about the exact files to touch.\n\n"
        f"Start by listing the directory or searching for relevant files now!"
    )
    return res

def _build_work_planning_prompt(objective: str, strategy: str, files_to_edit: list, files_to_create: list) -> str:
    """Build the Phase 2 prompt for breaking a work strategy into micro-tasks."""
    return (
        f"The objective is: '{objective}'.\n"
        f"The agreed strategy is:\n{strategy}\n\n"
        f"Files to modify: {', '.join(files_to_edit) if files_to_edit else 'None'}\n"
        f"Files to create: {', '.join(files_to_create) if files_to_create else 'None'}\n\n"
        f"Now create implementation MICRO-TASKS using `add_work_task`. RULES:\n"
        f"1. Break the work down into a sequence of extremely small tasks.\n"
        f"2. For EXISTING files, create a task for EACH specific method or logic block you need to change.\n"
        f"3. For NEW files, Task 1 must be creating the skeleton (imports, empty classes/methods), followed by tasks for each method.\n"
        f"4. ONLY call `add_work_task`. Do NOT write code or complete tasks yet!\n"
        f"5. IMPORTANT: When you have added ALL tasks, you MUST stop calling tools and reply exactly with 'DONE'.\n\n"
        f"Call `add_work_task` for EVERY step NOW, then stop."
    )

def _build_planning_prompt(objective: str, specs_summary: str) -> str:
    """Build the Phase 2 prompt for breaking raw specs into micro-tasks."""
    return (
        f"The project '{objective}' has detailed specifications for these files:\n{specs_summary}\n\n"
        f"Now create implementation MICRO-TASKS. RULES:\n"
        f"1. Break EACH FILE down into a sequence of small tasks.\n"
        f"2. Task 1 for a file MUST be creating the 'skeleton' (imports, empty classes, methods with 'pass').\n"
        f"3. Task 2, Task 3, etc. for that file MUST implement EXACTLY ONE method each.\n"
        f"4. CRITICAL: ENTRY POINT FILES (like main.py) MUST have implementation tasks! Do NOT just create a skeleton for main.py. You MUST create a task to implement the main logic/function.\n"
        f"5. Create tasks in DEPENDENCY ORDER: standalone files first, then files that depend on them.\n"
        f"6. ONLY call `add_project_task`. Do NOT call `complete_project_task` or `write_file`!\n\n"
        f"EXAMPLE of Micro-Tasks for 'database.py':\n"
        f"  add_project_task('Create skeleton for database.py (imports and empty DB class)')\n"
        f"  add_project_task('Implement DB.__init__ method in database.py')\n"
        f"  add_project_task('Implement DB.save method in database.py')\n\n"
        f"Call `add_project_task` for EVERY file's micro-tasks NOW. When finished adding all tasks, stop calling tools and reply exactly with 'DONE'."
    )

def export_chat_history(agent: ArgentAgent, filename: str = None, auto: bool = False):
    """Exports the current chat history to a Markdown file."""
    from datetime import datetime
    from ui import s, print_system, print_error
    
    if auto and not s.get("auto_save_chat", True):
        return
        
    # Check if there's actual user interaction
    if not any(m.get("role") == "user" for m in agent.messages):
        if not auto:
            print_system("Chat history is empty. Nothing to save.")
        return
        
    cwd = os.getcwd()
    chats_dir = os.path.join(cwd, "exports")
    os.makedirs(chats_dir, exist_ok=True)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"chat_{timestamp}.md"
    elif not filename.endswith(".md"):
        filename += ".md"
        
    filepath = os.path.join(chats_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 🗓 Argent Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            for msg in agent.messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system" or not content:
                    continue
                    
                if role == "user":
                    f.write(f"### 👤 Пользователь\n{content}\n\n")
                elif role in ("assistant", "model"):
                    f.write(f"### 🤖 Argent\n{content}\n\n")
                    
        if auto:
            print_system(f"Chat autosaved to: {filepath}")
        else:
            print_system(f"Chat saved successfully to: {filepath}")
            
        hook_manager.call_hook("on_chat_saved", filepath)
            
    except Exception as e:
        print_error(f"Failed to save chat: {e}")

def handle_slash_command(command: str, agent: ArgentAgent) -> bool:
    """Handle slash commands. Returns True if REPL should exit."""
    cmd = command.lower().strip()
    if cmd in ("/exit", "/quit"):
        export_chat_history(agent, auto=True)
        return True
    elif cmd == "/clear":
        agent.clear_history()
        print_system("Conversation history cleared.")
    elif cmd == "/model":
        current = get_current_model()
        new_model = select_model(current)
        if new_model and new_model != current:
            set_current_model(new_model)
            agent.set_model(new_model)
            print_system(f"Model updated to: {new_model}")
        else:
            print_system("Model unchanged.")
    elif cmd == "/provider":
        current_prov = get_provider()
        choices = ["ollama", "zai"]
        new_prov = questionary.select(
            "Select API Provider:",
            choices=choices,
            default=current_prov
        ).ask()
        
        if new_prov:
            set_provider(new_prov)
            agent.provider = new_prov
            options_text = ""
            if new_prov == "zai":
                current_key = get_zai_api_key()
                if not current_key:
                    new_key = questionary.password("Enter Z.AI API Key:").ask()
                    if new_key:
                        set_zai_api_key(new_key)
                        options_text = " (API Key saved)"
                else:
                    change_key = questionary.confirm("Z.AI API Key is already set. Do you want to change it?").ask()
                    if change_key:
                        new_key = questionary.password("Enter New Z.AI API Key:").ask()
                        if new_key:
                            set_zai_api_key(new_key)
                            options_text = " (API Key updated)"

                endpoint_choice = questionary.select(
                    "Select Z.AI Endpoint:",
                    choices=[
                        "Coding Plan (api.z.ai/api/coding/paas/v4) - for GLM Coding Plan subscribers",
                        "General API (api.z.ai/api/paas/v4) - standard pay-per-token",
                    ],
                    default="Coding Plan (api.z.ai/api/coding/paas/v4) - for GLM Coding Plan subscribers"
                ).ask()
                if endpoint_choice and "Coding Plan" in endpoint_choice:
                    set_zai_endpoint(ZAI_ENDPOINT_CODING)
                else:
                    set_zai_endpoint(ZAI_ENDPOINT_GENERAL)

            print_system(f"API Provider updated to: {new_prov}{options_text}")
            if new_prov == "zai":
                print_system("Select a Z.AI model to use:")
                new_model = select_model(get_current_model())
                if new_model != get_current_model():
                    set_current_model(new_model)
                    agent.model_name = new_model
                    print_system(f"Model updated to: {new_model}")
    elif cmd.startswith("/obsidian"):
        parts = command.split(" ", 1)
        if len(parts) > 1:
            vault_path = parts[1].strip()
        else:
            vault_path = questionary.path("Enter the path to your Obsidian vault:").ask()
            
        if vault_path:
            # basic clean up of path
            vault_path = vault_path.strip('\'"')
            set_obsidian_vault(vault_path)
            print_system(f"Obsidian vault path set to: {vault_path}")
        else:
            print_system("Obsidian vault path unchanged.")
    elif cmd == "/help":
        help_text = (
            "**Argent Coder Commands:**\n"
            "- `/provider` - Select API Provider (Ollama / Z.ai) and endpoint\n"
            "- `/model` - Select active LLM model\n"
            "- `/obsidian [path]` - Set the path to your Obsidian vault\n"
            "- `/research [topic]` - Enter Auto-Research mode to search the web and generate notes\n"
            "- `/enable_rag` - Index the current project codebase for Semantic AI Search\n"
            "- `/disable_rag` - Turn off Semantic AI Search\n"
            "- `/hooks [path]` - View or change the global plugins (hooks) directory\n"
            "- `/sandbox` - Enter an isolated Code Playground to execute and test code safely\n"
            "- `/tools` - Open interactive menu to enable/disable tools\n"
            "- `/save [name]` - Export the current conversation to a Markdown file\n"
            "- `/sessions` - List saved sessions\n"
            "- `/load <n>` - Restore a saved session by number\n"
            "- `/diff [file]` - Show changes made to files\n"
            "- `/undo [file]` - Restore a file to its previous version\n"
            "- `/undo_all` - Restore all modified files\n"
            "- `/copy <n>` - Copy code block #n to clipboard\n"
            "- `/pipeline [task]` - Execute a multi-step task automatically\n"
            "- `/logs [module] [n]` - View logs (e.g. /logs tools 20, /logs error)\n"
            "- `/skills` - List available AI skills\n"
            "- `/setup_terminal` - Make the terminal look incredibly professional (Fonts & Colors)\n"
            "- `/project [prompt]` - Force the AI to build a massive multi-step project from scratch\n"
            "- `/work [--auto] [task]` - Modify or fix an EXISTING codebase safely\n"
            "- `/commit` - Generate AI commit message and commit changes\n"
            "- `/verbose` - Toggle live status indicators (spinners)\n"
            "- `/clear` - Clear conversation history\n"
            "- `/help` - Show this help message\n"
            "- `/exit` - Exit the application\n"
        )
        
        # Add custom commands to help
        custom_cmds = hook_manager.get_custom_commands()
        if custom_cmds:
            help_text += "\n**Plugin Commands:**\n"
            for c in custom_cmds:
                help_text += f"- `/{c}`\n"
                
        print_markdown(help_text)
    elif cmd == "/verbose":
        current = get_verbose_status()
        new_val = not current
        set_verbose_status(new_val)
        state = "[bold green]ON[/bold green]" if new_val else "[bold red]OFF[/bold red]"
        print_system(f"Live status indicators: {state}")
    else:
        # Check for custom commands in hooks
        custom_cmds = hook_manager.get_custom_commands()
        # strip slash and get base command
        base_cmd = command.strip().split(" ")[0].lstrip("/")
        if base_cmd in custom_cmds:
            try:
                # Pass any arguments after the command
                args = command.strip().split(" ")[1:]
                custom_cmds[base_cmd](*args)
            except Exception as e:
                print_error(f"Custom command '/{base_cmd}' failed: {e}")
        else:
            print_error(f"Unknown command: {command}. Type /help for available commands.")
    return False

def main():
    agent = None

    def cleanup():
        nonlocal agent
        try:
            from tools import ACTIVE_PROCESSES
            for pid, proc_info in list(ACTIVE_PROCESSES.items()):
                try:
                    proc_info["process"].terminate()
                except Exception:
                    pass
        except Exception:
            pass
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
    
    builtin_cmds = [
        '/help', '/provider', '/model', '/obsidian', '/clear', '/research', '/enable_rag', '/disable_rag',
        '/hooks', '/sandbox', '/tools', '/save', '/setup_terminal', '/project', '/work', '/commit',
        '/sessions', '/load', '/diff', '/undo', '/undo_all', '/copy', '/pipeline', '/logs', '/skills', '/verbose', '/exit', '/quit'
    ]
    
    def get_all_commands():
        custom_names = [f"/{c}" for c in hook_manager.get_custom_commands().keys()]
        return builtin_cmds + custom_names
    
    command_completer = WordCompleter(get_all_commands, ignore_case=True, match_middle=False, sentence=True)
    session_history = InMemoryHistory()
    
    print_system(f"Active Provider: {get_provider().upper()}")
    print_system(f"Active Model: {get_current_model()}")
    print_system(f"Working Directory: {os.getcwd()}")
    vault = get_obsidian_vault()
    if vault:
        print_system(f"Obsidian Vault: {vault}")

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
    auto_continue_input = None
    last_task_id = None
    task_retries = 0
    project_iterations = 0
    MAX_TASK_RETRIES = 3
    turn_counter = 0
    turn_counter = 0
    
    while True:
        try:
            print() # Visual spacing
            
            if auto_continue_input:
                user_input = auto_continue_input
                auto_continue_input = None
                print_system("Продолжение рабочего процесса...")
            else:
                try:
                    user_input = prompt("❯ ", completer=command_completer, history=session_history)
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
            elif user_input.strip() == "/enable_rag":
                cwd = os.getcwd()
                print_system(f"Enabling RAG for project at {cwd}...")
                result = enable_rag_for_project(cwd)
                if "ERROR: 'chromadb' is not installed" in result:
                    print_error("ChromaDB is not installed.")
                    install = questionary.confirm("Would you like Argent to install it now? (pip install chromadb)").ask()
                    if install:
                        os.system("pip install chromadb sentence-transformers")
                        print_system("Libraries installed! Attempting to enable RAG again...")
                        result = enable_rag_for_project(cwd)
                
                print_system(result)
                continue
            elif user_input.strip() == "/disable_rag":
                disable_rag()
                print_system("Semantic Search (RAG) has been disabled.")
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
                    from pathlib import Path
                    
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
            elif user_input.strip() == "/setup_terminal":
                font_guide = """
# 🎨 Установка Серьезного UI (Шрифты Терминала)

К сожалению, сам Python не имеет прав менять шрифт твоего системного терминала. Но чтобы Argent выглядел *по-настоящему стильно* и профессионально, тебе нужен шрифт программиста.

**Рекомендуемый Шрифт:** `Fira Code Nerd Font` или `JetBrains Mono`.

## Как установить (Займет 1 минуту):
1. **Скачай шрифт**: Перейди на страницу [Nerd Fonts](https://www.nerdfonts.com/font-downloads) и скачай *FiraCode*.
2. **Установи**: Распакуй архив, выдели все файлы `.ttf`, нажми ПКМ -> `Установить`.
3. **Настрой терминал**:
   - Если ты используешь **Windows Terminal** (настоятельно рекомендуем): Нажми шестеренку (Настройки) -> Профили -> Оформление -> Шрифт -> Выбери `FiraCode NF`.
   - Если стандартный PowerShell: ПКМ по рамке окна -> Свойства -> Шрифт -> `Fira Code`.

## Настройка цветов:
В папке с Argent автоматически создался файл `theme.yaml`. Ты можешь открыть его в любом редакторе и настроить любые цвета (например, заменить синие рамки ИИ на хакерские зеленые `green_yellow`).
                """
                print_markdown(font_guide)
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
            elif user_input.startswith("/diff"):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2:
                    pending = get_pending_changes()
                    if not pending:
                        print_system("No pending file changes.")
                    else:
                        print_system("[bold cyan]Modified files:[/bold cyan]")
                        for ch in pending:
                            print_system(f"  - {ch['key']} ({ch['snapshot_count']} snapshots)")
                        print_system("Use /diff <filepath> to see changes.")
                else:
                    diff_output = get_diff(parts[1])
                    print_markdown(f"```diff\n{diff_output}\n```")
                continue
            elif user_input.strip() == "/undo":
                pending = get_pending_changes()
                if not pending:
                    print_system("No files to undo.")
                elif len(pending) == 1:
                    result = undo(pending[0]["key"])
                    print_system(result)
                else:
                    print_system("Multiple modified files. Use /undo <filepath> or /undo_all.")
                continue
            elif user_input.startswith("/undo "):
                filepath = user_input.strip()[6:]
                result = undo(filepath)
                print_system(result)
                continue
            elif user_input.strip() == "/undo_all":
                result = undo_all()
                print_system(result)
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
            elif user_input.startswith("/pipeline"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Usage: /pipeline <task description>")
                    print_system("Example: /pipeline Find all TODO comments in the codebase and create a report")
                    continue
                pipeline_task = parts[1].strip()
                print_system(f"Planning pipeline for: {pipeline_task}")
                try:
                    pipe = Pipeline(agent)
                    steps = pipe.plan(pipeline_task)
                    if not steps:
                        print_error("Failed to plan pipeline steps.")
                        continue
                    print_system(f"[bold cyan]Pipeline planned: {len(steps)} steps[/bold cyan]")
                    for s in steps:
                        print_system(f"  {s.get('step', '?')}. {s.get('action', s.get('task', ''))}")
                    approved = questionary.confirm("Execute this pipeline?").ask()
                    if approved:
                        result = pipe.execute()
                        print_system("[bold green]Pipeline complete.[/bold green]")
                        print_markdown(result)
                    else:
                        print_system("Pipeline cancelled.")
                except Exception as e:
                    print_error(f"Pipeline error: {e}")
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
            elif user_input.startswith("/project"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Please specify a project prompt. Example: /project Build a Snake game in Python")
                    continue
                proj_prompt = parts[1].strip()
                
                # Phase 0: Ask for Deep Research
                run_research = questionary.confirm("Run Deep Research (Phase 0) to gather up-to-date context before planning?").ask()
                
                # TDD Mode
                tdd_mode = questionary.confirm("Enable TDD Mode? (AI will write tests BEFORE code)").ask()
                
                # Obsidian Mode
                use_obsidian = False
                if get_obsidian_vault():
                    use_obsidian = questionary.confirm("Enable Obsidian integration for this project? (Create notes instead of normal files)").ask()
                
                # Initialize Project Brain
                pm = ProjectManager()
                pm.destroy()
                
                if run_research:
                    pm.create(proj_prompt, status="researching", tdd_mode=tdd_mode, use_obsidian=use_obsidian)
                    user_input = (
                        f"You are the Phase 0 Research Agent for the new project: '{proj_prompt}'.\n\n"
                        f"Your task is to gather the MAXIMUM amount of up-to-date information, best practices, and API references required to build this project.\n"
                        f"You MUST call `run_deep_research(objective='...')` right now to search the web.\n"
                        f"When the research is complete, output a detailed markdown report of everything you found.\n"
                        f"DO NOT write code or architecture yet. Just gather information."
                    )
                    print_system(f"[Brain] Phase 0: Starting Deep Research...")
                else:
                    pm.create(proj_prompt, status="specifying_architecture", tdd_mode=tdd_mode, use_obsidian=use_obsidian)
                    user_input = (
                        f"You are the architect for this project: '{proj_prompt}'.\n\n"
                        f"Your task is to design the HIGH-LEVEL ARCHITECTURE.\n"
                        f"You MUST call `write_project_architecture` with TWO parameters:\n"
                        f"  1. architecture = text description of all files and their dependencies\n"
                        f"  2. files = comma-separated list of ALL file paths to create\n\n"
                        f"EXAMPLE call:\n"
                        f"  write_project_architecture(\n"
                        f"    architecture='Files:\\n1. MyApp/main.py — Entry point. Uses: calculator.py\\n2. MyApp/calculator.py — Math logic. Standalone.',\n"
                        f"    files='MyApp/main.py, MyApp/calculator.py'\n"
                        f"  )\n\n"
                        f"RULES:\n"
                        f"- ALL file paths MUST include the project folder (e.g. MyProject/main.py, NOT just main.py)\n"
                        f"- The 'files' parameter must list ONLY the files to CREATE, not referenced libraries\n"
                        f"- Do NOT describe implementation details (no method names, no types)\n"
                        f"- Do NOT create any files yet\n"
                        f"- Do NOT call any other tools\n"
                        f"- Call `write_project_architecture` NOW!"
                    )
                    print_system(f"[Brain] Phase 1a: Designing architecture...")
                
                is_project_mode = True
                last_task_id = None
                task_retries = 0
                
            elif user_input.strip() == "/sandbox":
                from sandbox import SandboxManager
                sm = SandboxManager()
                
                print_system("\n[bold cyan]=== ENTERING ARGENT SANDBOX ===[/bold cyan]")
                print_system("Welcome to the isolated Code Playground.")
                print_system(f"Sandbox Directory: {sm.sandbox_dir}")
                print_system(f"Available Commands:\n"
                             f"  /run python - Execute sandbox_main.py\n"
                             f"  /run csharp - Compile and execute SandboxMain.cs\n"
                             f"  /run js     - Execute sandbox_main.js via Node.js\n"
                             f"  /run web    - Start Local HTTP server and open browser\n"
                             f"  /stop web   - Stop Local HTTP server\n"
                             f"  /export [dir]- Export working code to main project\n"
                             f"  /clean      - Wipe the sandbox clean\n"
                             f"  /exit       - Leave Sandbox and return to main chat")
                
                # Sandbox tool restrictions
                SANDBOX_ALLOWED_TOOLS = [
                    "read_file", "write_file", "delete_file", "replace_in_file", "replace_python_function",
                    "list_directory", "grep_search", "search_files", "run_command", "run_admin_command",
                    "start_background_command", "read_background_command", "send_background_command",
                    "stop_background_command", "search_web", "read_webpage", "get_file_outline", "multi_replace_in_file"
                ]

                # Switch AI context forcefully
                agent.messages.append({
                    "role": "system", 
                    "content": (
                        f"You are now in SANDBOX MODE. You MUST write all code ONLY to {sm.sandbox_dir}.\n"
                        "DO NOT modify the main project. DO NOT use Project Brain tools like `add_project_task` or `complete_project_task`.\n"
                        "Your goal is to experiment and build small prototypes or test isolated logic.\n"
                        "Wait for the user to ask you to write code before doing anything."
                    )
                })
                
                # Nested sandbox loop
                while True:
                    try:
                        print()
                        sb_input = prompt("Sandbox ❯ ", history=session_history).strip()
                        if not sb_input:
                            continue
                            
                        # Sandbox Commands
                        if sb_input in ("/exit", "/quit"):
                            sm.stop_web()
                            print_system("Exiting Sandbox. Returning to main chat.")
                            agent.messages.append({"role": "system", "content": "Sandbox mode disabled. You are back in the main project directory."})
                            break
                        elif sb_input.startswith("/run"):
                            args = sb_input.split(" ")
                            lang = args[1].lower() if len(args) > 1 else "python"
                            
                            res = ""
                            if lang == "python":
                                res = sm.run_python()
                            elif lang == "csharp":
                                res = sm.run_csharp()
                            elif lang == "js":
                                res = sm.run_js()
                            elif lang == "web":
                                res = sm.run_web()
                            else:
                                print_error(f"Unknown language: {lang}")
                                continue
                                
                            print_system(f"=== EXECUTION RESULT ===\n{res}")
                            
                            # Feed the result back to AI if it crashed
                            if "STDERR:" in res or "Compilation failed" in res:
                                auto_fix = questionary.confirm("Code generated an error. Upload error to AI for automatic fixing?").ask()
                                if auto_fix:
                                    sb_input = f"I ran the code but it failed with this output:\n```\n{res}\n```\nPlease fix the error in the sandbox code."
                                else:
                                    continue
                            else:
                                continue
                                
                        elif sb_input.startswith("/export"):
                            parts = sb_input.split(" ", 1)
                            if len(parts) > 1:
                                sm.export_files(parts[1].strip())
                            else:
                                print_error("Please specify destination directory. Example: /export src/new_feature")
                            continue
                        elif sb_input == "/clean":
                            sm.clean_sandbox()
                            continue
                        elif sb_input == "/stop web":
                            sm.stop_web()
                            continue
                            
                        # If normal chat input inside Sandbox, send to AI directly via inner spinner
                        response_chunks = agent.process_user_input(sb_input, allowed_tools=SANDBOX_ALLOWED_TOOLS)
                        
                        chunk_iterator = iter(response_chunks)
                        streamed_text = ""
                        is_tool_executing = False
                        
                        from ui import s
                        
                        while True:
                            try:
                                done = False
                                if not is_tool_executing:
                                    spinner_text = "[dim]AI is thinking...[/dim]" if s.get("use_spinners", True) else ""
                                    with console.status(spinner_text, spinner="dots", speed=1.5):
                                        while True:
                                            try:
                                                chunk = next(chunk_iterator)
                                            except StopIteration:
                                                if streamed_text:
                                                    print_markdown(streamed_text)
                                                done = True
                                                break
                                                
                                            type_ = chunk.get("type")
                                            
                                            if type_ in ("content_stream", "content"):
                                                streamed_text += chunk["content"]
                                            elif type_ == "content_replace":
                                                streamed_text = chunk["content"]
                                            else:
                                                if streamed_text:
                                                    print_markdown(streamed_text)
                                                    streamed_text = ""
                                                    
                                                if type_ == "tool_start":
                                                    print_tool_start(chunk["name"], chunk.get("args", {}))
                                                    is_tool_executing = True
                                                elif type_ == "tool_end":
                                                    print_tool_end(chunk["name"], chunk.get("result", ""))
                                                elif type_ == "error":
                                                    print_error(chunk["content"])
                                                
                                                break
                                else:
                                    try:
                                        chunk = next(chunk_iterator)
                                    except StopIteration:
                                        done = True
                                         
                                    if not done:
                                        type_ = chunk.get("type")
                                        if type_ == "tool_end":
                                            print_tool_end(chunk["name"], chunk.get("result", ""))
                                            is_tool_executing = False
                                        elif type_ == "tool_start":
                                            print_tool_start(chunk["name"], chunk.get("args", {}))
                                        elif type_ == "error":
                                            print_error(chunk["content"])
                                    
                                if done:
                                    break
                                             
                            except StopIteration:
                                break
                            
                    except KeyboardInterrupt:
                        continue
                    except EOFError:
                        break
                        
                continue # loop back to main loop AFTER exiting sandbox
                
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
                
                # Phase 0: Ask for Deep Research
                run_research = questionary.confirm("Run Deep Research (Phase 0) to gather up-to-date context before investigation?").ask()
                
                # TDD Mode
                tdd_mode = questionary.confirm("Enable TDD Mode? (AI will write tests BEFORE code)").ask()

                # Obsidian Mode
                use_obsidian = False
                if get_obsidian_vault():
                    use_obsidian = questionary.confirm("Enable Obsidian integration for this work task?").ask()

                # Initialize Project Brain in Work mode
                pm = ProjectManager()
                pm.destroy()
                
                if run_research:
                    pm.create(work_prompt, status="work_researching", mode="work", auto_mode=auto_mode, tdd_mode=tdd_mode, use_obsidian=use_obsidian)
                    user_input = (
                        f"You are the Phase 0 Research Agent for the codebase modification task: '{work_prompt}'.\n\n"
                        f"Your task is to gather the MAXIMUM amount of up-to-date information, best practices, and API references required for this task.\n"
                        f"You MUST call `run_deep_research(objective='...')` right now to search the web.\n"
                        f"When the research is complete, output a detailed markdown report of everything you found.\n"
                        f"DO NOT write code or investigate files yet. Just gather information."
                    )
                    print_system(f"[Brain] Phase 0: Starting Deep Research...")
                else:
                    pm.create(work_prompt, status="work_investigating", mode="work", auto_mode=auto_mode, tdd_mode=tdd_mode, use_obsidian=use_obsidian)
                    user_input = _build_work_investigation_prompt(work_prompt, "")
                    print_system(f"[Brain] Phase 1: Investigating codebase...")

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
                    result = subprocess.run(
                        cmd,
                        shell=True,
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
                pm_temp = ProjectManager()
                if pm_temp.active:
                    status = pm_temp.data.get("status", "")
                    if status in ["researching", "work_researching"]:
                        active_tools = ["run_deep_research"]
                    elif status == "specifying_architecture":
                        active_tools = ["write_project_architecture"]
                    elif status == "specifying_details":
                        active_tools = ["write_file_spec"]
                    elif status == "work_investigating":
                        active_tools = ["list_directory", "grep_search", "read_file", "plan_work_changes"]
                    elif status == "planning":
                        active_tools = ["add_project_task"]
                    elif status == "work_planning":
                        active_tools = ["add_work_task"]
                    elif status in ["executing", "work_executing"]:
                        active_tools = [
                            "read_file", "write_file", "delete_file", "replace_in_file",
                            "list_directory", "grep_search", "run_command", "run_admin_command",
                            "start_background_command", "read_background_command", "send_background_command",
                            "stop_background_command", "search_web", "read_webpage",
                            "complete_project_task", "create_svg_image"
                        ]
                        if pm_temp.data.get("use_obsidian", False):
                            active_tools.extend(["write_obsidian_note", "search_obsidian_notes", "update_obsidian_properties"])
                else:
                    # In normal chat, use CHAT_ALLOWED_TOOLS
                    active_tools = CHAT_ALLOWED_TOOLS
            else:
                # Regular chat mode
                active_tools = CHAT_ALLOWED_TOOLS

            response_chunks = agent.process_user_input(user_input, allowed_tools=active_tools)
            chunk_iterator = iter(response_chunks)
            streamed_text = ""
            streamed_thinking = ""
            is_tool_executing = False
            current_tool_name = ""
            
            # Using the rich Live display for real-time response rendering
            from rich.live import Live
            from rich.console import Group
            from rich.text import Text
            from rich.markdown import Markdown
            from rich.spinner import Spinner as RichSpinner
            from ui import s, c, create_response_panel, create_content_panel, print_reasoning, print_reasoning_header
            
            verbose = get_verbose_status()
            
            while True:
                try:
                    done = False
                    if not is_tool_executing:
                        # Phase 1: Thinking/Reasoning Stream (NATIVE STREAMING)
                        # We don't use Live here because Live prevents natural terminal scrolling 
                        # for content larger than the viewport.
                        has_started_reasoning = False
                        is_waiting_ttft = True
                        while True:
                            try:
                                if is_waiting_ttft and verbose:
                                    with console.status("[bold cyan]Обработка...[/bold cyan]", spinner="dots"):
                                        chunk = next(chunk_iterator)
                                    is_waiting_ttft = False
                                else:
                                    chunk = next(chunk_iterator)
                                    is_waiting_ttft = False
                            except StopIteration:
                                done = True
                                break
                            
                            type_ = chunk.get("type")
                            if type_ == "thinking_stream":
                                if not has_started_reasoning:
                                    print_reasoning_header()
                                    has_started_reasoning = True
                                
                                content = chunk["content"]
                                streamed_thinking += content
                                # Native print without Live keeps scrolling lock-free
                                console.print(content, end="", style=c.get("reasoning_text", "dim white"))
                            elif type_ == "tool_generating":
                                if has_started_reasoning:
                                    console.print("\n")
                                    has_started_reasoning = False
                                # Show a spinner while the LLM generates tool arguments
                                tool_gen_name = chunk.get("name", "?")
                                tool_gen_bytes = chunk.get("bytes", 0)
                                if verbose:
                                    with console.status(f"[dim cyan]Генерация {tool_gen_name}... ({tool_gen_bytes} B)[/dim cyan]", spinner="dots") as status:
                                        while True:
                                            try:
                                                chunk = next(chunk_iterator)
                                            except StopIteration:
                                                done = True
                                                break
                                            type_ = chunk.get("type")
                                            if type_ == "tool_generating":
                                                tool_gen_name = chunk.get("name", tool_gen_name)
                                                tool_gen_bytes = chunk.get("bytes", 0)
                                                status.update(f"[dim cyan]Генерация {tool_gen_name}... ({tool_gen_bytes} B)[/dim cyan]")
                                            else:
                                                break
                                else:
                                    while True:
                                        try:
                                            chunk = next(chunk_iterator)
                                        except StopIteration:
                                            done = True
                                            break
                                        type_ = chunk.get("type")
                                        if type_ != "tool_generating":
                                            break
                                # After exiting spinner, re-check what chunk type we got
                                if done:
                                    break
                                if type_ == "tool_start":
                                    print_tool_start(chunk["name"], chunk.get("args", {}))
                                    is_tool_executing = True
                                    current_tool_name = chunk["name"]
                                    break
                                elif type_ in ("content_stream", "content"):
                                    # Shouldn't happen often — but handle gracefully
                                    break
                                else:
                                    break
                            else:
                                if has_started_reasoning:
                                    console.print("\n") # Newline after reasoning
                                break
                            
                        # Phase 2: Main Content/Tool Stream (LIVE PANEL)
                        if not done and not is_tool_executing:
                            # Live with transient=True: panel disappears after streaming ends,
                            # replaced by clean final render without borders.
                            with Live(create_content_panel(""), console=console, refresh_per_second=10, transient=True) as live:
                                while True:
                                    # If type_ is already from the previous next() call, process it first
                                    if type_ in ("content_stream", "content"):
                                        streamed_text += chunk["content"]
                                        live.update(create_content_panel(streamed_text))
                                    elif type_ == "content_replace":
                                        streamed_text = chunk["content"]
                                        live.update(create_content_panel(streamed_text))
                                    elif type_ == "tool_generating":
                                        # Tool generation started mid-content — exit Live and let next iteration handle it
                                        break
                                    elif type_ not in ("thinking_stream"):
                                        # Tool or error, exit live content phase
                                        break
                                    
                                    # Show animated thinking indicator while waiting for next chunk.
                                    # Rich Live refreshes via a background thread, so the spinner
                                    # keeps animating even while next() blocks the main thread.
                                    if streamed_text:
                                        live.update(Group(
                                            create_content_panel(streamed_text),
                                            RichSpinner("dots", text=Text(" Генерация...", style="dim cyan"))
                                        ))
                                    
                                    try:
                                        chunk = next(chunk_iterator)
                                        type_ = chunk.get("type")
                                        # Restore clean content panel (remove spinner)
                                        if type_ in ("content_stream", "content", "content_replace"):
                                            pass  # Will be updated at the top of the loop
                                        elif streamed_text:
                                            live.update(create_content_panel(streamed_text))
                                    except StopIteration:
                                        done = True
                                        break
                        
                        # Re-process the last chunk that broke the Live content loop (tool or error)
                        if not done and not is_tool_executing:
                            if type_ == "tool_start":
                                print_tool_start(chunk["name"], chunk.get("args", {}))
                                is_tool_executing = True
                                current_tool_name = chunk["name"]
                            elif type_ == "tool_generating":
                                # Will be caught by the next outer loop iteration
                                pass
                            elif type_ == "tool_end":
                                print_tool_end(chunk["name"], chunk.get("result", ""))
                            elif type_ == "error":
                                print_error(chunk["content"])
                    
                    else:
                        # Tool is executing — wait for tool_end
                        # Skip spinner for interactive tools that need clean terminal
                        # (any tool that calls questionary.confirm or similar prompts)
                        interactive_tools = {
                            "ask_user_questions",
                            "run_command",
                            "run_admin_command",
                            "start_background_command",
                            "delete_file",
                            "plan_work_changes",
                        }
                        use_spinner = verbose and current_tool_name not in interactive_tools
                        
                        if use_spinner:
                            with console.status("[dim cyan]Выполнение...[/dim cyan]", spinner="dots"):
                                try:
                                    chunk = next(chunk_iterator)
                                except StopIteration:
                                    done = True
                        else:
                            try:
                                chunk = next(chunk_iterator)
                            except StopIteration:
                                done = True
                             
                        if not done:
                            type_ = chunk.get("type")
                            if type_ == "tool_end":
                                print_tool_end(chunk["name"], chunk.get("result", ""))
                                is_tool_executing = False
                                current_tool_name = ""
                            elif type_ == "tool_start":
                                print_tool_start(chunk["name"], chunk.get("args", {}))
                                current_tool_name = chunk["name"]
                            elif type_ == "error":
                                print_error(chunk["content"])
                        
                    if done:
                        break
                                 
                except StopIteration:
                    break
                except Exception as e:
                    print_error(f"Streaming error: {e}")
                    break
            
            # Render final response without borders for easy copy-paste
            if streamed_text:
                from ui import create_final_panel, safe_print as _safe_print
                for el in create_final_panel(streamed_text):
                    _safe_print(el)
                    _safe_print("")
            
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
                pm = ProjectManager()
                
                if not pm.active:
                    is_project_mode = False
                    print_system("[Brain] No active project found.")
                    continue
                
                status = pm.data.get("status", "")
                
                # Phase 0 → 1a: Research done, save data and generate architecture
                if status in ["researching", "work_researching"]:
                    if not pm.data.get("research_data"):
                        # Save the last AI response as the research synthesis
                        pm.save_research_data(agent.messages[-1]["content"])
                    
                    if pm.data.get("mode") == "work":
                        pm.set_status("work_investigating")
                        agent.inject_context()
                        auto_continue_input = _build_work_investigation_prompt(pm.data["objective"], pm.data.get("research_data", ""))
                        print_system(f"[Brain] Phase 1: Investigating codebase based on research...")
                    else:
                        pm.set_status("specifying_architecture")
                        agent.inject_context()
                        
                        proj_prompt = pm.data["objective"]
                        research_text = pm.data.get("research_data", "")
                        
                        auto_continue_input = (
                            f"You are the architect for this project: '{proj_prompt}'.\n\n"
                        )
                        
                        if research_text:
                            auto_continue_input += (
                                f"=== LATEST RESEARCH CONTEXT ===\n"
                                f"{research_text}\n\n"
                            )
                            
                        auto_continue_input += (
                            f"Your task is to design the HIGH-LEVEL ARCHITECTURE.\n"
                            f"You MUST call `write_project_architecture` with TWO parameters:\n"
                            f"  1. architecture = text description of all files and their dependencies\n"
                            f"  2. files = comma-separated list of ALL file paths to create\n\n"
                            f"EXAMPLE call:\n"
                            f"  write_project_architecture(\n"
                            f"    architecture='Files:\\n1. MyApp/main.py — Entry point. Uses: calculator.py\\n2. MyApp/calculator.py — Math logic. Standalone.',\n"
                            f"    files='MyApp/main.py, MyApp/calculator.py'\n"
                            f"  )\n\n"
                            f"RULES:\n"
                            f"- ALL file paths MUST include the project folder (e.g. MyProject/main.py, NOT just main.py)\n"
                            f"- The 'files' parameter must list ONLY the files to CREATE, not referenced libraries\n"
                            f"- Do NOT describe implementation details (no method names, no types)\n"
                            f"- Do NOT create any files yet\n"
                            f"- Do NOT call any other tools\n"
                            f"- Call `write_project_architecture` NOW!"
                        )
                        
                        print_system(f"[Brain] Phase 1a: Designing architecture based on research...")

                # --- WORK MODE ---
                # Phase 1: Investigation in progress ( waiting for plan_work_changes )
                elif status == "work_investigating":
                    agent.inject_context()
                    # We just re-prompt briefly to remind it to call plan_work_changes if it got distracted
                    auto_continue_input = "Investigation phase. Continue reading files, searching, or when ready, call `plan_work_changes`."
                    print_system("[Brain] Phase 1: Investigation in progress...")

                # Phase 1 Investigation done (plan_work_changes called) -> Phase 2 Micro-Tasking
                elif status == "work_planning" and not pm.has_pending():
                    # Tools.py changes status to work_planning when plan is accepted
                    agent.inject_context()
                    print_system("[Brain] Phase 2: Generating micro-tasks from investigation plan...")
                    auto_continue_input = _build_work_planning_prompt(
                        pm.data["objective"],
                        pm.data.get("work_strategy", ""),
                        pm.data.get("files_to_edit", []),
                        pm.data.get("files_to_create", [])
                    )
                
                # Phase 2 -> Phase 3: Start Execution
                elif status == "work_planning" and pm.has_pending():
                    pm.set_status("work_executing")
                    next_task = pm.get_next_pending()
                    last_task_id = next_task["id"]
                    task_retries = 0
                    agent.inject_context()
                    auto_continue_input = pm.build_execution_context(next_task)
                    progress = pm.get_progress_display()
                    print_system(f"[Brain] Phase 3: {progress} >> Work Task {next_task['id']}: {next_task['description']}")
                
                # Phase 3: Continue executing
                elif status == "work_executing" and pm.has_pending():
                    next_task = pm.get_next_pending()
                    if next_task["id"] == last_task_id:
                        task_retries += 1
                        if task_retries >= MAX_TASK_RETRIES:
                            print_system(f"[Brain] Task {next_task['id']} failed {MAX_TASK_RETRIES} times.")
                            action = questionary.select(
                                "What should we do with this stalled task?",
                                choices=["1. Continue trying (reset counter)", "2. Provide a hint to AI", "3. Skip task"]
                            ).ask()
                            
                            if action and action.startswith("1"):
                                task_retries = 0
                                print_system("Retrying task...")
                            elif action and action.startswith("2"):
                                hint = questionary.text("Enter your hint for the AI:").ask()
                                if hint:
                                    next_task['description'] += f"\n\n[USER HINT AFTER FAILURE]: {hint}"
                                    pm._save()
                                task_retries = 0
                                print_system("Hint added. Retrying task...")
                            else:
                                pm.complete_task(next_task["id"], "SKIPPED: failed to complete after multiple attempts")
                                last_task_id = None
                                task_retries = 0
                                continue
                    else:
                        last_task_id = next_task["id"]
                        task_retries = 0
                    
                    agent.inject_context()
                    auto_continue_input = pm.build_execution_context(next_task)
                    progress = pm.get_progress_display()
                    print_system(f"[Brain] {progress} >> Work Task {next_task['id']}: {next_task['description']}")
                
                # Phase 3 done
                elif status == "work_executing" and not pm.has_pending():
                    is_project_mode = False
                    pm.set_status("completed")
                    print_system(f"[Brain] {pm.get_progress_display()} Work complete! Returning to chat.")

                # --- PROJECT MODE ---
                # Phase 1a → 1b: Architecture written, now detail each file
                elif (status == "specifying" or status == "specifying_architecture") and pm.has_architecture():
                    pending_files = pm.get_pending_spec_files()
                    
                    if pending_files:
                        # Detail the next file
                        next_file = pending_files[0]
                        pm.set_status("specifying_details")
                        agent.inject_context()
                        
                        print_system(f"[Brain] Phase 1b: Detailing spec for {next_file} ({len(pending_files)} files remaining)...")
                        
                        auto_continue_input = _build_spec_prompt(
                            pm.data['objective'], pm.get_architecture(), next_file
                        )
                    else:
                        # All files detailed — move to task creation
                        pm.set_status("planning")
                        agent.inject_context()
                        
                        all_specs = pm.get_all_file_specs()
                        specs_summary = "\
".join([f"  - {fname}" for fname in all_specs.keys()])
                        
                        print_system(f"[Brain] Phase 2: Creating tasks from {len(all_specs)} file specs...")
                        
                        auto_continue_input = _build_planning_prompt(pm.data['objective'], specs_summary)
                # Phase 1b: Continue detailing files
                elif status == "specifying_details":
                    pending_files = pm.get_pending_spec_files()
                    
                    if pending_files:
                        next_file = pending_files[0]
                        agent.inject_context()
                        
                        print_system(f"[Brain] Phase 1b: Detailing spec for {next_file} ({len(pending_files)} files remaining)...")
                        
                        auto_continue_input = _build_spec_prompt(
                            pm.data['objective'], pm.get_architecture(), next_file
                        )
                    else:
                        # All files detailed — move to task creation
                        pm.set_status("planning")
                        agent.inject_context()
                        
                        all_specs = pm.get_all_file_specs()
                        specs_summary = "\n".join([f"  - {fname}" for fname in all_specs.keys()])
                        
                        print_system(f"[Brain] Phase 2: Creating tasks from {len(all_specs)} file specs...")
                        
                        auto_continue_input = _build_planning_prompt(pm.data['objective'], specs_summary)
                # Phase 2 → 3: Tasks created, start execution
                elif status == "planning" and pm.has_pending():
                    pm.set_status("executing")
                    next_task = pm.get_next_pending()
                    last_task_id = next_task["id"]
                    task_retries = 0
                    agent.inject_context()
                    auto_continue_input = pm.build_execution_context(next_task)
                    progress = pm.get_progress_display()
                    print_system(f"[Brain] Phase 3: {progress} >> Task {next_task['id']}: {next_task['description']}")
                # Phase 3: Continue executing next pending task
                elif status == "executing" and pm.has_pending():
                    next_task = pm.get_next_pending()
                    
                    # Per-task retry protection
                    if next_task["id"] == last_task_id:
                        task_retries += 1
                        if task_retries >= MAX_TASK_RETRIES:
                            print_system(f"[Brain] Task {next_task['id']} failed {MAX_TASK_RETRIES} times.")
                            action = questionary.select(
                                "What should we do with this stalled task?",
                                choices=["1. Continue trying (reset counter)", "2. Provide a hint to AI", "3. Skip task"]
                            ).ask()
                            
                            if action and action.startswith("1"):
                                task_retries = 0
                                print_system("Retrying task...")
                            elif action and action.startswith("2"):
                                hint = questionary.text("Enter your hint for the AI:").ask()
                                if hint:
                                    next_task['description'] += f"\n\n[USER HINT AFTER FAILURE]: {hint}"
                                    pm._save()
                                task_retries = 0
                                print_system("Hint added. Retrying task...")
                            else:
                                pm.complete_task(next_task["id"], "SKIPPED: failed to complete after multiple attempts")
                                last_task_id = None
                                task_retries = 0
                                continue # re-enter state machine to pick next task
                    else:
                        last_task_id = next_task["id"]
                        task_retries = 0
                    
                    agent.inject_context()
                    auto_continue_input = pm.build_execution_context(next_task)
                    progress = pm.get_progress_display()
                    print_system(f"[Brain] {progress} >> Task {next_task['id']}: {next_task['description']}")
                # Execution done → Finish project
                elif status == "executing" and not pm.has_pending():
                    is_project_mode = False
                    pm.set_status("completed")
                    print_system(f"[Brain] {pm.get_progress_display()} Project complete! Returning to chat.")
                # Fallback: no progress
                else:
                    is_project_mode = False
                    print_system("[Brain] Could not advance project. Check the model output.")
            
        except KeyboardInterrupt:
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