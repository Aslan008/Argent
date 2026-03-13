import sys
import os
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
    get_disabled_tools, set_disabled_tools
)
from ui import (
    console, print_markdown, print_system, print_error,
    print_tool_start, print_tool_end, select_model, s
)
from hook_manager import hook_manager
from rag_engine import enable_rag_for_project, disable_rag
import subprocess
import json
from pathlib import Path
from project_manager import ProjectManager
import ollama

# Default tools allowed in regular chat (excludes Project Brain tools)
CHAT_ALLOWED_TOOLS = [
    "read_file", "write_file", "delete_file", "replace_in_file", "replace_python_function",
    "list_directory", "grep_search", "run_command", "run_admin_command",
    "start_background_command", "read_background_command", "send_background_command",
    "stop_background_command", "search_web", "read_webpage", "get_file_outline", 
    "multi_replace_in_file", "write_obsidian_note", "search_obsidian_notes", 
    "get_obsidian_vault", "semantic_search"
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
    chats_dir = os.path.join(cwd, ".argent", "chats")
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
            "- `/model` - Select active Ollama model\n"
            "- `/obsidian [path]` - Set the path to your Obsidian vault\n"
            "- `/research [topic]` - Enter Auto-Research mode to search the web and generate notes\n"
            "- `/enable_rag` - Index the current project codebase for Semantic AI Search\n"
            "- `/disable_rag` - Turn off Semantic AI Search\n"
            "- `/hooks [path]` - View or change the global plugins (hooks) directory\n"
            "- `/sandbox` - Enter an isolated Code Playground to execute and test code safely\n"
            "- `/tools` - Open interactive menu to enable/disable tools\n"
            "- `/save [name]` - Export the current conversation to a Markdown file\n"
            "- `/setup_terminal` - Make the terminal look incredibly professional (Fonts & Colors)\n"
            "- `/project [prompt]` - Force the AI to build a massive multi-step project from scratch\n"
            "- `/work [--auto] [task]` - Modify or fix an EXISTING codebase safely\n"
            "- `/commit` - Generate AI commit message and commit changes\n"
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
    os.system('cls' if os.name == 'nt' else 'clear')
    console.rule("[bold cyan]Argent Coder[/bold cyan]")
    print_system("Welcome to Argent Coder. The elite AI pair programmer.")
    print_system("Type /help for commands. Type your prompt below.")
    
    agent = ArgentAgent()
    
    builtin_cmds = [
        '/help', '/model', '/obsidian', '/clear', '/research', '/enable_rag', '/disable_rag',
        '/hooks', '/sandbox', '/tools', '/save', '/setup_terminal', '/project', '/work', '/commit', '/exit', '/quit'
    ]
    
    def get_all_commands():
        custom_names = [f"/{c}" for c in hook_manager.get_custom_commands().keys()]
        return builtin_cmds + custom_names
    
    command_completer = WordCompleter(get_all_commands, ignore_case=True, match_middle=False, sentence=True)
    session_history = InMemoryHistory()
    
    print_system(f"Active Model: {get_current_model()}")
    print_system(f"Working Directory: {os.getcwd()}")
    vault = get_obsidian_vault()
    if vault:
        print_system(f"Obsidian Vault: {vault}")

    # Trigger Startup Hook
    hook_manager.call_hook("on_startup")

    is_project_mode = False
    auto_continue_input = None
    last_task_id = None
    task_retries = 0
    MAX_TASK_RETRIES = 3
    
    while True:
        try:
            print() # Visual spacing
            
            if auto_continue_input:
                user_input = auto_continue_input
                auto_continue_input = None
                print_system("🔄 Auto-continuing project workflow...")
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
                    # Refresh autocomplete
                    builtin_cmds = [
                        '/help', '/model', '/obsidian', '/clear', '/research', '/enable_rag', '/disable_rag',
                        '/hooks', '/sandbox', '/tools', '/save', '/setup_terminal', '/project', '/work', '/commit', '/exit', '/quit'
                    ]
                    custom_cmds = hook_manager.get_custom_commands()
                    all_available_cmds = builtin_cmds + [f"/{c}" for c in custom_cmds.keys()]
                    # Note: session (prompt_toolkit) doesn't easily allow updating completer at runtime 
                    # without full restart, but local plugin lookup still works.
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
                
                # Initialize Project Brain
                pm = ProjectManager()
                pm.destroy()
                
                if run_research:
                    pm.create(proj_prompt, status="researching", tdd_mode=tdd_mode)
                    user_input = (
                        f"You are the Phase 0 Research Agent for the new project: '{proj_prompt}'.\n\n"
                        f"Your task is to gather the MAXIMUM amount of up-to-date information, best practices, and API references required to build this project.\n"
                        f"You MUST call `run_deep_research(objective='...')` right now to search the web.\n"
                        f"When the research is complete, output a detailed markdown report of everything you found.\n"
                        f"DO NOT write code or architecture yet. Just gather information."
                    )
                    print_system(f"[Project Brain] Phase 0: Starting Deep Research...")
                else:
                    pm.create(proj_prompt, status="specifying_architecture", tdd_mode=tdd_mode)
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
                    print_system(f"[Project Brain] Phase 1a: Designing architecture...")
                
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
                    "list_directory", "grep_search", "run_command", "run_admin_command",
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

                # Initialize Project Brain in Work mode
                pm = ProjectManager()
                pm.destroy()
                
                if run_research:
                    pm.create(work_prompt, status="work_researching", mode="work", auto_mode=auto_mode, tdd_mode=tdd_mode)
                    user_input = (
                        f"You are the Phase 0 Research Agent for the codebase modification task: '{work_prompt}'.\n\n"
                        f"Your task is to gather the MAXIMUM amount of up-to-date information, best practices, and API references required for this task.\n"
                        f"You MUST call `run_deep_research(objective='...')` right now to search the web.\n"
                        f"When the research is complete, output a detailed markdown report of everything you found.\n"
                        f"DO NOT write code or investigate files yet. Just gather information."
                    )
                    print_system(f"[Work Brain] Phase 0: Starting Deep Research...")
                else:
                    pm.create(work_prompt, status="work_investigating", mode="work", auto_mode=auto_mode, tdd_mode=tdd_mode)
                    user_input = _build_work_investigation_prompt(work_prompt, "")
                    print_system(f"[Work Brain] Phase 1: Investigating codebase...")

            elif user_input.startswith("/commit"):
                try:
                    # 1. Get staged changes
                    staged_diff = subprocess.run("git diff --cached", shell=True, capture_output=True, text=True).stdout
                    if not staged_diff.strip():
                        print_error("No staged changes found. Use 'git add' first.")
                        continue
                        
                    print_system("Generating commit message based on staged changes...")
                    
                    commit_prompt = (
                        "You are a Senior Developer. Generate a concise, professional Git commit message following Conventional Commits "
                        "specification based on the following diff. Only output the commit message, nothing else.\n\n"
                        f"{staged_diff}"
                    )
                    
                    # Call LLM synchronously for quick message generation
                    response = ollama.chat(
                        model=agent.model_name,
                        messages=[{"role": "user", "content": commit_prompt}]
                    )
                    gen_message = response.get("message", {}).get("content", "").strip().strip('"').strip("'")
                    
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
                
                is_project_mode = True
                last_task_id = None
                task_retries = 0

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
                            "complete_project_task", "write_obsidian_note", "get_obsidian_vault"
                        ]
                else:
                    # In normal chat, use CHAT_ALLOWED_TOOLS
                    active_tools = CHAT_ALLOWED_TOOLS
            else:
                # Regular chat mode
                active_tools = CHAT_ALLOWED_TOOLS

            response_chunks = agent.process_user_input(user_input, allowed_tools=active_tools)
            chunk_iterator = iter(response_chunks)
            streamed_text = ""
            is_tool_executing = False
            
            # Using the rich status spinner instead of Live Markdown for a cleaner look
            # We import it here or at the top of the file
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
                except Exception as e:
                    print_error(f"Streaming error: {e}")
                    break
            
            # Trigger Post Response Hook
            if agent.messages and agent.messages[-1].get("role") in ("assistant", "model"):
                hook_manager.call_hook("post_response", agent.messages[-1].get("content", ""))
                    
            # === Project Brain: State Machine ===
            if is_project_mode:
                pm = ProjectManager()
                
                if not pm.active:
                    is_project_mode = False
                    print_system("[Project Brain] No active project found.")
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
                        print_system(f"[Work Brain] Phase 1: Investigating codebase based on research...")
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
                        
                        print_system(f"[Project Brain] Phase 1a: Designing architecture based on research...")

                # --- WORK MODE ---
                # Phase 1: Investigation in progress ( waiting for plan_work_changes )
                elif status == "work_investigating":
                    agent.inject_context()
                    # We just re-prompt briefly to remind it to call plan_work_changes if it got distracted
                    auto_continue_input = "Investigation phase. Continue reading files, searching, or when ready, call `plan_work_changes`."
                    print_system("[Work Brain] Phase 1: Investigation in progress...")

                # Phase 1 Investigation done (plan_work_changes called) -> Phase 2 Micro-Tasking
                elif status == "work_planning" and not pm.has_pending():
                    # Tools.py changes status to work_planning when plan is accepted
                    agent.inject_context()
                    print_system("[Work Brain] Phase 2: Generating micro-tasks from investigation plan...")
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
                    print_system(f"[Work Brain] Phase 3: {progress} >> Work Task {next_task['id']}: {next_task['description']}")
                
                # Phase 3: Continue executing
                elif status == "work_executing" and pm.has_pending():
                    next_task = pm.get_next_pending()
                    if next_task["id"] == last_task_id:
                        task_retries += 1
                        if task_retries >= MAX_TASK_RETRIES:
                            print_system(f"[Work Brain] Task {next_task['id']} failed {MAX_TASK_RETRIES} times, skipping...")
                            pm.complete_task(next_task["id"], "SKIPPED: failed to complete")
                            last_task_id = None
                            task_retries = 0
                            continue
                    else:
                        last_task_id = next_task["id"]
                        task_retries = 0
                    
                    agent.inject_context()
                    auto_continue_input = pm.build_execution_context(next_task)
                    progress = pm.get_progress_display()
                    print_system(f"[Work Brain] {progress} >> Work Task {next_task['id']}: {next_task['description']}")
                
                # Phase 3 done
                elif status == "work_executing" and not pm.has_pending():
                    is_project_mode = False
                    pm.set_status("completed")
                    print_system(f"[Work Brain] {pm.get_progress_display()} Work complete! Returning to chat.")

                # --- PROJECT MODE ---
                # Phase 1a → 1b: Architecture written, now detail each file
                elif (status == "specifying" or status == "specifying_architecture") and pm.has_architecture():
                    pending_files = pm.get_pending_spec_files()
                    
                    if pending_files:
                        # Detail the next file
                        next_file = pending_files[0]
                        pm.set_status("specifying_details")
                        agent.inject_context()
                        
                        print_system(f"[Project Brain] Phase 1b: Detailing spec for {next_file} ({len(pending_files)} files remaining)...")
                        
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
                        
                        print_system(f"[Project Brain] Phase 2: Creating tasks from {len(all_specs)} file specs...")
                        
                        auto_continue_input = _build_planning_prompt(pm.data['objective'], specs_summary)
                # Phase 1b: Continue detailing files
                elif status == "specifying_details":
                    pending_files = pm.get_pending_spec_files()
                    
                    if pending_files:
                        next_file = pending_files[0]
                        agent.inject_context()
                        
                        print_system(f"[Project Brain] Phase 1b: Detailing spec for {next_file} ({len(pending_files)} files remaining)...")
                        
                        auto_continue_input = _build_spec_prompt(
                            pm.data['objective'], pm.get_architecture(), next_file
                        )
                    else:
                        # All files detailed — move to task creation
                        pm.set_status("planning")
                        agent.inject_context()
                        
                        all_specs = pm.get_all_file_specs()
                        specs_summary = "\n".join([f"  - {fname}" for fname in all_specs.keys()])
                        
                        print_system(f"[Project Brain] Phase 2: Creating tasks from {len(all_specs)} file specs...")
                        
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
                    print_system(f"[Project Brain] Phase 3: {progress} >> Task {next_task['id']}: {next_task['description']}")
                # Phase 3: Continue executing next pending task
                elif status == "executing" and pm.has_pending():
                    next_task = pm.get_next_pending()
                    
                    # Per-task retry protection
                    if next_task["id"] == last_task_id:
                        task_retries += 1
                        if task_retries >= MAX_TASK_RETRIES:
                            print_system(f"[Project Brain] Task {next_task['id']} failed {MAX_TASK_RETRIES} times, skipping...")
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
                    print_system(f"[Project Brain] {progress} >> Task {next_task['id']}: {next_task['description']}")
                # Execution done → Finish project
                elif status == "executing" and not pm.has_pending():
                    is_project_mode = False
                    pm.set_status("completed")
                    print_system(f"[Project Brain] {pm.get_progress_display()} Project complete! Returning to chat.")
                # Fallback: no progress
                else:
                    is_project_mode = False
                    print_system("[Project Brain] Could not advance project. Check the model output.")
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            export_chat_history(agent, auto=True)
            break
        except Exception as e:
            print_error(f"Main loop error: {e}")
            break
    
    print_system("Goodbye!")

if __name__ == "__main__":
    main()