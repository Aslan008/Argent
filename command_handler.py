import os
import questionary

from agent import ArgentAgent
from config import (
    get_current_model, set_current_model, get_obsidian_vault, set_obsidian_vault,
    get_hooks_dir, set_hooks_dir, get_autonomous_plugins_enabled, set_autonomous_plugins_enabled,
    get_disabled_tools, set_disabled_tools,
    get_provider, set_provider, get_zai_api_key, set_zai_api_key,
    get_zai_endpoint, set_zai_endpoint, ZAI_ENDPOINT_GENERAL, ZAI_ENDPOINT_CODING,
    get_koboldcpp_url, set_koboldcpp_url,
    get_openrouter_api_key, set_openrouter_api_key,
    get_auxiliary_model, set_auxiliary_model,
    get_auxiliary_provider, set_auxiliary_provider,
    get_verbose_status, set_verbose_status,
    get_debug_mode, set_debug_mode,
    add_mcp_server, remove_mcp_server, get_mcp_servers,
    get_strip_reasoning, set_strip_reasoning,
    get_temperature, set_temperature,
    get_external_kbs, add_external_kb, remove_external_kb, toggle_external_kb
)
from ui import (
    console, print_markdown, print_system, print_error,
    select_model
)
from hook_manager import hook_manager


def switch_model(agent: ArgentAgent, new_model: str) -> None:
    """Point the agent at another model and say what that costs the history.

    Everything tier-dependent — strategy, history budget, context window,
    constrained decoding, native tool support — is swapped by set_model. The
    part nobody can see is that a conversation built on a cloud model does not
    fit a small one: the trim happens silently at the start of the next turn,
    and the model then answers as if the earlier half was never said.
    """
    from config import get_model_size_category

    before_tier = get_model_size_category(agent.model_name)
    before_history = getattr(agent, "max_history_messages", None)
    before_ctx = getattr(agent, "max_context_tokens", None)

    set_current_model(new_model)
    agent.set_model(new_model)          # never assign model_name directly

    after_tier = get_model_size_category(new_model)
    print_system(f"Model updated to: {new_model}"
                 + (f" [{before_tier} → {after_tier}]" if before_tier != after_tier else ""))

    kept = len([m for m in agent.messages if m.get("role") != "system"])
    if before_history and agent.max_history_messages < before_history and kept > agent.max_history_messages:
        print_system(f"[yellow]⚠ Бюджет истории: {before_history} → "
                     f"{agent.max_history_messages} сообщений.[/yellow] Сейчас в диалоге "
                     f"{kept} — лишние будут отброшены на следующем ходу.")
    if before_ctx and agent.max_context_tokens < before_ctx:
        print_system(f"[yellow]⚠ Окно контекста: {before_ctx} → "
                     f"{agent.max_context_tokens} токенов.[/yellow] Начало разговора "
                     f"может не поместиться.")


def export_chat_history(agent: ArgentAgent, filename: str = None, auto: bool = False):
    """Exports the current chat history to a Markdown file."""
    from datetime import datetime
    from ui import s, print_system, print_error
    
    if auto and not s.get("auto_save_chat", True):
        return
        
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
            switch_model(agent, new_model)
        else:
            print_system("Model unchanged.")
    elif cmd == "/provider":
        current_prov = get_provider()
        choices = ["ollama", "zai", "openrouter", "koboldcpp"]
        new_prov = questionary.select(
            "Select API Provider:",
            choices=choices,
            default=current_prov
        ).ask()
        
        if new_prov:
            set_provider(new_prov)
            agent.set_provider(new_prov)
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
            elif new_prov == "openrouter":
                current_key = get_openrouter_api_key()
                if not current_key:
                    new_key = questionary.password("Enter OpenRouter API Key (https://openrouter.ai/keys):").ask()
                    if new_key:
                        set_openrouter_api_key(new_key)
                        options_text = " (API Key saved)"
                else:
                    change_key = questionary.confirm("OpenRouter API Key is already set. Do you want to change it?").ask()
                    if change_key:
                        new_key = questionary.password("Enter New OpenRouter API Key:").ask()
                        if new_key:
                            set_openrouter_api_key(new_key)
                            options_text = " (API Key updated)"
            elif new_prov == "koboldcpp":
                current_url = get_koboldcpp_url()
                new_url = questionary.text("Enter KoboldCPP API URL:", default=current_url).ask()
                if new_url:
                    set_koboldcpp_url(new_url)
                    options_text = f" (URL: {new_url})"

            print_system(f"API Provider updated to: {new_prov}{options_text}")
            if new_prov != current_prov:
                # Every provider, ollama included: a model id almost never
                # exists on two of them, so keeping the old name means every
                # request 404s until the user works out they must run /model.
                print_system(f"Select a {new_prov.upper()} model to use:")
                new_model = select_model(get_current_model())
                if new_model and new_model != get_current_model():
                    switch_model(agent, new_model)
    elif cmd.startswith("/mcp"):
        _handle_mcp_command(command)
    elif cmd.startswith("/kb"):
        _handle_kb_command(command)
    elif cmd == "/jobs":
        from tools.command_ops import list_background_commands
        print_system(list_background_commands())
    elif cmd.startswith("/stop"):
        from tools.command_ops import stop_background_command, list_background_commands
        parts = command.strip().split()
        if len(parts) < 2:
            print_system(list_background_commands())
            print_system("Usage: /stop <pid>")
        else:
            print_system(stop_background_command(parts[1]))
    elif cmd == "/aux":
        _handle_aux_command()
    elif cmd == "/search":
        _handle_search_command()
    elif cmd == "/doctor":
        from doctor import print_diagnostics
        print_diagnostics()
    elif cmd == "/help":
        help_text = (
            "**Argent Coder — Commands**\n"
            "\n**Model & provider**\n"
            "- `/provider` - Select API provider (Ollama / Z.ai / OpenRouter / KoboldCPP) and key/endpoint\n"
            "- `/model` - Select the active LLM model (context window + tier auto-detected)\n"
            "- `/aux` - Pick a cheap/local auxiliary model for service tasks (summarization, /commit)\n"
            "- `/temp [value]` - Set or view the generation temperature (0.0 - 2.0)\n"
            "- `/thinking` - Toggle stripping reasoning blocks from history\n"
            "\n**Build & edit code**\n"
            "- `/work [--auto] [task]` - Modify or fix an EXISTING codebase safely\n"
            "- `/vibe` - Vibe mode: auto-approve safe actions, checkpoint every turn (undo via /rewind)\n"
            "- `/tasks` - Scheduled automations that run unattended while Argent is open\n"
            "- `/project [prompt]` - Build a large multi-step project from scratch\n"
            "- `/rooms <task>` - Experimental \"rooms and rails\" engine (declarative graph + triage)\n"
            "- `/init` - Analyze the project and generate .argent/AGENTS.md (project memory)\n"
            "- `/commit` - Generate an AI commit message and commit staged changes\n"
            "\n**Files & changes**\n"
            "- `/changes` - List files the AI modified this session\n"
            "- `/diff <file>` - Diff a file against its pre-edit snapshot\n"
            "- `/undo <file>` - Restore a file to its pre-edit snapshot\n"
            "- `/rewind` - Time machine: roll the whole tree back to any auto-checkpoint (one per turn)\n"
            "- `/cd [path]` - Change (or show) the working directory\n"
            "\n**Knowledge & search**\n"
            "- `/rag_toggle` - Enable/disable semantic-search indexing\n"
            "- `/auto_retrieve` - Toggle auto-injecting semantic-search results each query\n"
            "- `/research [topic]` - Autonomous web research → notes\n"
            "- `/search` - Web search settings: Brave API key, query languages, reranker model\n"
            "- `/skills` - List available skills (flat .md and SKILL.md folders)\n"
            "- `/skill import <source>` - Install a skill from a GitHub repo (owner/repo or URL) or local path\n"
            "\n**Safety & quality**\n"
            "- `/guard [off|warn|block]` - Command risk-gate: confirm risky / refuse catastrophic commands\n"
            "- `/critic [text]` - Red-team a plan/idea (or the AI's last plan) with an independent critic\n"
            "- `/critic on|off|model <name>` - Auto-critique the diff before /commit; choose the critic model\n"
            "- `/goal [text|clear]` - Show the goal & progress, set a new objective, or reset it\n"
            "\n**Sessions**\n"
            "- `/save [name]` - Export the conversation to Markdown\n"
            "- `/sessions` / `/load <n>` - List / restore saved sessions\n"
            "- `/copy <n>` - Copy code block #n to the clipboard\n"
            "- `/clear` - Clear the conversation history\n"
            "\n**Tools & extensions**\n"
            "- `/tools` - Enable/disable tools interactively\n"
            "- `/hooks [path]` - View/change the plugins (hooks) directory\n"
            "- `/mcp` - Manage MCP servers (add/remove/start/stop/test)\n"
            "\n**Background & diagnostics**\n"
            "- `/jobs` / `/stop <pid>` - List / terminate background processes\n"
            "- `/doctor` - Environment self-diagnostics (provider, tier, deps, browser)\n"
            "- `/stats` - Session diagnostics (model, context budget, plugins, MCP)\n"
            "- `/logs [module] [n]` - View logs (e.g. /logs tools 20, /logs error)\n"
            "- `/verbose` / `/debug` - Toggle status spinners / detailed tool logs\n"
            "\n**Other**\n"
            "- `/help` - Show this message   ·   `/exit` - Quit\n"
        )
        
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
    elif cmd == "/debug":
        current = get_debug_mode()
        new_val = not current
        set_debug_mode(new_val)
        state = "[bold green]ON[/bold green]" if new_val else "[bold red]OFF[/bold red]"
        print_system(f"Detailed logs in chat (Debug mode): {state}")
    elif cmd == "/thinking":
        current = get_strip_reasoning()
        new_val = not current
        set_strip_reasoning(new_val)
        state = "[bold green]ON[/bold green]" if new_val else "[bold red]OFF[/bold red]"
        print_system(f"Forced removal of reasoning blocks from history: {state}")
    elif cmd == "/temp" or cmd.startswith("/temp ") or cmd == "/temperature" or cmd.startswith("/temperature "):
        parts = command.strip().split(" ", 1)
        if len(parts) > 1:
            val_str = parts[1].strip()
            try:
                val = float(val_str)
                if 0.0 <= val <= 2.0:
                    set_temperature(val)
                    print_system(f"Model temperature updated to: {val}")
                else:
                    print_error("Temperature must be between 0.0 and 2.0.")
            except ValueError:
                print_error("Please provide a valid numeric value for temperature.")
        else:
            current_temp = get_temperature()
            choices = [
                "0.2 (Deterministic - Coding)",
                "0.7 (Balanced - Default)",
                "1.0 (Creative)",
                f"Keep Current ({current_temp})",
                "Custom Value..."
            ]
            
            choice = questionary.select(
                f"Set Model Temperature (Current: {current_temp}):",
                choices=choices,
                default=f"Keep Current ({current_temp})"
            ).ask()
            
            if choice:
                if choice.startswith("0.2"):
                    set_temperature(0.2)
                    print_system("Model temperature updated to: 0.2")
                elif choice.startswith("0.7"):
                    set_temperature(0.7)
                    print_system("Model temperature updated to: 0.7")
                elif choice.startswith("1.0"):
                    set_temperature(1.0)
                    print_system("Model temperature updated to: 1.0")
                elif choice == "Custom Value...":
                    custom_val = questionary.text("Enter custom temperature (0.0 to 2.0):").ask()
                    if custom_val:
                        try:
                            val = float(custom_val)
                            if 0.0 <= val <= 2.0:
                                set_temperature(val)
                                print_system(f"Model temperature updated to: {val}")
                            else:
                                print_error("Temperature must be between 0.0 and 2.0.")
                        except ValueError:
                            print_error("Please enter a valid number.")
    else:
        custom_cmds = hook_manager.get_custom_commands()
        base_cmd = command.strip().split(" ")[0].lstrip("/")
        if base_cmd in custom_cmds:
            try:
                args = command.strip().split(" ")[1:]
                custom_cmds[base_cmd](*args)
            except Exception as e:
                print_error(f"Custom command '/{base_cmd}' failed: {e}")
        else:
            print_error(f"Unknown command: {command}. Type /help for available commands.")
    return False


def _handle_search_command():
    """Configure web research: the optional Brave index, the query languages and
    the reranker.

    These three are one setting in practice. Adding a language without a
    multilingual reranker retrieves pages that are then scored so low they never
    reach the answer, and a multilingual reranker with English-only queries
    never sees anything to rerank — so the menu shows the mismatch instead of
    letting you configure half of it.
    """
    from config import (
        get_brave_api_key, set_brave_api_key,
        get_reranker_model, set_reranker_model,
        get_search_languages, set_search_languages,
    )
    from src.research.rerank import MULTILINGUAL_MODEL, _DEFAULT_MODEL
    from src.research.search import engine_labels

    key = get_brave_api_key()
    langs = get_search_languages()
    model = get_reranker_model() or _DEFAULT_MODEL
    multilingual = model != _DEFAULT_MODEL

    print_system(f"Движки: [bold cyan]{', '.join(engine_labels())}[/bold cyan]")
    print_system(f"Brave API-ключ: {'задан' if key else '[dim]не задан[/dim]'}")
    print_system(f"Языки запросов: [bold cyan]{', '.join(langs)}[/bold cyan]")
    print_system(f"Reranker: [bold cyan]{model.split('/')[-1]}[/bold cyan]"
                 f" ({'мультиязычный' if multilingual else 'только английский'})")
    if langs != ["en"] and not multilingual:
        print_system("[yellow]Внимание:[/yellow] запросы не только на английском, но reranker "
                     "англоязычный — найденные неанглийские страницы будут отброшены при ранжировании.")

    BRAVE = "Brave API-ключ (второй независимый индекс поиска)"
    LANGS = "Языки поисковых запросов"
    MODEL = "Модель reranker'а"
    choice = questionary.select("Что настроить?", choices=[BRAVE, LANGS, MODEL, "Отмена"]).ask()

    if choice == BRAVE:
        new_key = questionary.password(
            "Brave Search API key (пусто — отключить; ключ берётся на brave.com/search/api):"
        ).ask()
        if new_key is None:
            return
        set_brave_api_key(new_key)
        print_system("Brave включён — второй индекс добавлен к поиску."
                     if new_key.strip() else "Brave отключён.")
        return

    if choice == LANGS:
        picked = questionary.checkbox(
            "На каких языках писать поисковые запросы (English почти всегда нужен — "
            "техническая документация и ответы англоязычны):",
            choices=[
                questionary.Choice("English", value="en", checked="en" in langs),
                questionary.Choice("Русский", value="ru", checked="ru" in langs),
                questionary.Choice("Deutsch", value="de", checked="de" in langs),
                questionary.Choice("Français", value="fr", checked="fr" in langs),
                questionary.Choice("Español", value="es", checked="es" in langs),
                questionary.Choice("中文", value="zh", checked="zh" in langs),
            ],
        ).ask()
        if picked is None:
            return
        if not picked:
            print_error("Нужен хотя бы один язык — оставляю как было.")
            return
        set_search_languages(picked)
        print_system(f"Языки запросов: {', '.join(picked)}")
        if picked != ["en"] and not multilingual:
            print_system("[yellow]Теперь стоит переключить reranker на мультиязычный[/yellow] — "
                         "иначе неанглийские результаты не дойдут до ответа (/search → Модель reranker'а).")
        return

    if choice == MODEL:
        ENGLISH = f"Английский, 92 МБ — {_DEFAULT_MODEL.split('/')[-1]}"
        MULTI = f"Мультиязычный, ~490 МБ — {MULTILINGUAL_MODEL.split('/')[-1]}"
        OTHER = "Другая модель (ввести имя с HuggingFace)"
        picked = questionary.select(
            "Reranker ранжирует найденные фрагменты. Мультиязычный нужен, если ищете "
            "не только на английском (замерено: англоязычный ставит нерелевантный русский "
            "текст выше релевантного).",
            choices=[ENGLISH, MULTI, OTHER, "Отмена"],
        ).ask()
        if picked == ENGLISH:
            set_reranker_model("")
            print_system("Reranker: англоязычный (по умолчанию).")
        elif picked == MULTI:
            set_reranker_model(MULTILINGUAL_MODEL)
            print_system("Reranker: мультиязычный. Модель скачается при первом поиске "
                         "(~490 МБ), дальше берётся из кэша.")
        elif picked == OTHER:
            name = questionary.text("Имя модели на HuggingFace (cross-encoder):").ask()
            if name and name.strip():
                set_reranker_model(name)
                print_system(f"Reranker: {name.strip()}")
        return


def _handle_aux_command():
    """Configure the auxiliary model used for service tasks (summarization,
    /commit). Lets you point service work at a cheap/local model while the main
    model stays a powerful (possibly paid) one."""
    from providers import create_provider
    from ui import _select_from_list

    current_p = get_auxiliary_provider()
    current_m = get_auxiliary_model()
    if current_m:
        print_system(f"Текущая вспомогательная модель: [bold cyan]{current_p or get_provider()}:{current_m}[/bold cyan]")
    else:
        print_system("Вспомогательная модель не задана — сервисные задачи идут на основной модели.")

    DISABLE = "Отключить (использовать основную модель)"
    prov = questionary.select(
        "Провайдер для вспомогательных задач (суммаризация контекста, /commit):",
        choices=["ollama", "openrouter", "zai", "koboldcpp", DISABLE],
    ).ask()
    if not prov:
        return
    if prov == DISABLE:
        set_auxiliary_model(None)
        set_auxiliary_provider(None)
        print_system("Вспомогательная модель отключена.")
        return

    models = []
    try:
        models = create_provider(prov).list_models()
    except Exception:
        models = []

    if models:
        model = _select_from_list(f"Модель {prov} для сервисных задач:", models, models[0])
    else:
        model = questionary.text(f"Имя модели {prov} (список недоступен, введите вручную):").ask()

    if model:
        set_auxiliary_provider(prov)
        set_auxiliary_model(model)
        print_system(f"Вспомогательная модель: [bold cyan]{prov}:{model}[/bold cyan]")


def _handle_mcp_command(command: str):
    """Handle /mcp subcommands: list, add, remove, test."""
    from mcp_client import mcp_client, MCPTransportType

    parts = command.strip().split()
    
    if len(parts) == 1 or (len(parts) == 2 and parts[1] == "list"):
        servers = mcp_client.get_servers()
        configured = get_mcp_servers()
        if not configured:
            print_system("No MCP servers configured.")
            print_system("Usage:")
            print_system("  /mcp add <name> --stdio <command> [args...]")
            print_system("  /mcp add <name> --sse <url>")
            print_system("  /mcp add <name> --rest <url> [standard|unity_bridge]")
            return
        
        print_system("[bold cyan]MCP Servers:[/bold cyan]")
        for srv_info in configured:
            name = srv_info["name"]
            stype = srv_info.get("type", "stdio")
            running = name in mcp_client.servers and mcp_client.servers[name].is_running
            endpoint = srv_info.get("url") or f"{srv_info.get('command', '')} {' '.join(srv_info.get('args', []))}".strip()
            status = "[green]RUNNING[/green]" if running else "[red]STOPPED[/red]"
            tool_count = len(mcp_client.servers[name].list_tools()) if name in mcp_client.servers else 0
            print_system(f"  - [yellow]{name}[/yellow] ({stype}) {status} — {endpoint} [{tool_count} tools]")
        # RUNNING says nothing about whether the model can reach it — those are
        # two separate switches, and both read as ON.
        from src.agent.mcp_prompt import model_access_warning
        warning = model_access_warning()
        if warning:
            print_system(f"[yellow]⚠ {warning}[/yellow]")
        return

    subcmd = parts[1].lower() if len(parts) > 1 else ""

    if subcmd == "add":
        _mcp_add(parts)
    elif subcmd == "remove":
        if len(parts) < 3:
            print_error("Usage: /mcp remove <name>")
            return
        name = parts[2]
        remove_mcp_server(name)
        result = mcp_client.unregister_server(name)
        print_system(result)
    elif subcmd == "test":
        if len(parts) < 3:
            print_error("Usage: /mcp test <name>")
            return
        name = parts[2]
        result = mcp_client.test_connection(name)
        print_system(result)
    elif subcmd == "start":
        if len(parts) < 3:
            print_error("Usage: /mcp start <name>")
            return
        name = parts[2]
        configured = get_mcp_servers()
        entry = next((s for s in configured if s["name"] == name), None)
        if not entry:
            print_error(f"Server '{name}' not found in config. Use /mcp add first.")
            return
        cfg = {k: v for k, v in entry.items() if k != "name"}
        result = mcp_client.register_and_start(name, cfg)
        print_system(result)
    elif subcmd == "stop":
        if len(parts) < 3:
            print_error("Usage: /mcp stop <name>")
            return
        name = parts[2]
        result = mcp_client.unregister_server(name)
        print_system(result)
    else:
        print_error(f"Unknown /mcp subcommand: '{subcmd}'")
        print_system("Usage: /mcp [list|add|remove|start|stop|test]")


def _mcp_add(parts: list):
    """Parse /mcp add arguments and register server."""
    from mcp_client import mcp_client

    if len(parts) < 4:
        print_error("Usage:")
        print_system("  /mcp add <name> --stdio <command> [args...]")
        print_system("  /mcp add <name> --sse <url>")
        print_system("  /mcp add <name> --rest <url> [standard|unity_bridge]")
        print_system("")
        print_system("Examples:")
        print_system('  /mcp add files --stdio npx -y @anthropic/mcp-server-filesystem "C:/Projects"')
        print_system("  /mcp add github --stdio npx -y @anthropic/mcp-server-github")
        print_system("  /mcp add unity --stdio unity mcp")
        return

    name = parts[2]
    flag = parts[3].lower()

    if flag == "--stdio":
        if len(parts) < 5:
            print_error("Stdio requires a command. Example: /mcp add myserver --stdio npx -y @some/mcp-server")
            return
        command = parts[4]
        args = parts[5:] if len(parts) > 5 else []
        add_mcp_server(name, server_type="stdio", command=command, args=args)
        config = {"type": "stdio", "command": command, "args": args}
        result = mcp_client.register_and_start(name, config)
        print_system(result)

    elif flag == "--sse":
        if len(parts) < 5:
            print_error("SSE requires a URL. Example: /mcp add remote --sse http://localhost:3000")
            return
        url = parts[4]
        add_mcp_server(name, server_type="sse", url=url)
        config = {"type": "sse", "url": url}
        result = mcp_client.register_and_start(name, config)
        print_system(result)

    elif flag == "--rest":
        if len(parts) < 5:
            print_error("REST requires a URL. Example: /mcp add custom --rest http://localhost:7860 standard")
            return
        url = parts[4]
        rest_type = parts[5] if len(parts) > 5 else "standard"
        if rest_type not in ("standard", "unity_bridge"):
            print_error(f"Unknown REST type '{rest_type}'. Use: standard, unity_bridge")
            return
        add_mcp_server(name, server_type=rest_type, url=url)
        config = {"type": rest_type, "url": url}
        result = mcp_client.register_and_start(name, config)
        print_system(result)

    else:
        print_error(f"Unknown flag '{flag}'. Use: --stdio, --sse, --rest")

def _handle_kb_command(command: str):
    """Handle /kb subcommands: list, add, remove, toggle, index."""
    parts = command.strip().split()
    
    if len(parts) == 1 or parts[1] == "list":
        kbs = get_external_kbs()
        if not kbs:
            print_system("No External Knowledge Bases configured.")
            print_system("Usage:")
            print_system('  /kb add <id> "<Name>" "<Path>"')
            return
            
        print_system("[bold cyan]External Knowledge Bases:[/bold cyan]")
        for kb in kbs:
            status = "[green]ENABLED[/green]" if kb.get("enabled", True) else "[red]DISABLED[/red]"
            print_system(f"  - [yellow]{kb['id']}[/yellow] — {kb['name']} ({kb['path']}) {status}")
        return

    subcmd = parts[1].lower()

    if subcmd == "add":
        if len(parts) < 5:
            print_error("Usage: /kb add <id> <name> <path>")
            print_system('Example: /kb add unity64 "Unity 6.4" "D:/UnityDocs"')
            return
        kb_id = parts[2]
        
        # Need to re-parse considering quotes
        import shlex
        try:
            parsed = shlex.split(command)
        except ValueError as e:
            print_error(f"Error parsing arguments: {e}")
            return
            
        if len(parsed) < 5:
            print_error('Usage: /kb add <id> "Name" "Path"')
            return
            
        kb_id = parsed[2]
        name = parsed[3]
        path = parsed[4]
        
        add_external_kb(kb_id, name, path)
        print_system(f"Added Knowledge Base '{name}' ({kb_id}) at {path}.")
        print_system(f"Don't forget to index it: /kb index {kb_id}")
        
    elif subcmd == "remove":
        if len(parts) < 3:
            print_error("Usage: /kb remove <id>")
            return
        kb_id = parts[2]
        remove_external_kb(kb_id)
        print_system(f"Removed Knowledge Base '{kb_id}'.")
        
    elif subcmd == "toggle":
        if len(parts) < 3:
            print_error("Usage: /kb toggle <id>")
            return
        kb_id = parts[2]
        new_status = toggle_external_kb(kb_id)
        status_str = "ENABLED" if new_status else "DISABLED"
        print_system(f"Knowledge Base '{kb_id}' is now {status_str}.")
        
    elif subcmd == "index":
        if len(parts) < 3:
            print_error("Usage: /kb index <id>")
            return
        kb_id = parts[2]
        kbs = get_external_kbs()
        target_kb = next((kb for kb in kbs if kb["id"] == kb_id), None)
        if not target_kb:
            print_error(f"Knowledge Base '{kb_id}' not found.")
            return
            
        from rag_engine import index_external_kb
        print_system(f"[yellow]Starting indexing for {target_kb['name']}... This may take a while.[/yellow]")
        result = index_external_kb(target_kb)
        print_system(result)
    else:
        print_error(f"Unknown /kb subcommand: '{subcmd}'")
        print_system("Usage: /kb [list|add|remove|toggle|index]")
