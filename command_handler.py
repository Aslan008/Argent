import os
import questionary

from agent import ArgentAgent
from config import (
    get_current_model, set_current_model,
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
    from providers import create_provider

    # Spell the name the way the provider lists it ("Claude Sonnet 4.5" -> claude-sonnet-4-5), and
    # refuse a name the provider certainly does not have rather than save one that 404s.
    try:
        provider = create_provider(getattr(agent, "provider", None))
        resolved = provider.resolve_model(new_model)
        similar = [] if resolved else provider.suggest_models(new_model)
    except Exception:
        resolved, similar = new_model, []
    if not resolved:
        print_system(f"[red]Модель '{new_model}' не найдена у провайдера {getattr(agent, 'provider', '?')}.[/red]"
                     + (f" Похожие: {', '.join(similar)}." if similar else "")
                     + " Список: /model")
        return
    if resolved != new_model.strip():
        print_system(f"Имя модели исправлено: {new_model.strip()} → [bold]{resolved}[/bold]")
    new_model = resolved

    before_tier = get_model_size_category(agent.model_name)
    before_history = getattr(agent, "max_history_messages", None)
    before_ctx = getattr(agent, "max_context_tokens", None)

    set_current_model(new_model, getattr(agent, "provider", None))
    agent.set_model(new_model)          # never assign model_name directly

    after_tier = get_model_size_category(new_model)
    from claude_catalog import label_for
    shown = label_for(getattr(agent, "provider", None), new_model)
    print_system(f"Модель изменена: {shown}" + (f" ({new_model})" if shown != new_model else "")
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
            print_system("История пуста — сохранять нечего.")
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
                    
                if isinstance(content, list):
                    text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    content = "\n".join(text_parts) + "\n*[Вложение: медиа]*"
                    
                if role == "user":
                    f.write(f"### 👤 Пользователь\n{content}\n\n")
                elif role in ("assistant", "model"):
                    f.write(f"### 🤖 Argent\n{content}\n\n")
                    
        if auto:
            print_system(f"Диалог автосохранён: {filepath}")
        else:
            print_system(f"Диалог сохранён: {filepath}")
            
        hook_manager.call_hook("on_chat_saved", filepath)
            
    except Exception as e:
        print_error(f"Не удалось сохранить диалог: {e}")


def handle_slash_command(command: str, agent: ArgentAgent) -> bool:
    """Handle slash commands via CommandRegistry. Returns True if REPL should exit."""
    if not isinstance(command, str):
        return False

    from src.cli.commands import dispatch_command, CommandContext
    ctx = CommandContext(agent=agent, user_input=command)
    res = dispatch_command(command, ctx)
    if res.handled:
        return res.should_exit

    # Custom plugin commands via hook_manager
    from hook_manager import hook_manager
    custom_cmds = getattr(hook_manager, "custom_commands", {}) or {}
    cmd_clean = command.strip()
    if cmd_clean.startswith("/"):
        base_cmd = cmd_clean.split(" ")[0].lstrip("/")
        if base_cmd in custom_cmds:
            try:
                args = cmd_clean.split(" ")[1:]
                custom_cmds[base_cmd](*args)
                return False
            except Exception as e:
                print_error(f"Пользовательская команда '/{base_cmd}' упала: {e}")
                return False
        print_error(unknown_command_message(command))
    return False


def known_commands() -> list:
    """Every slash command the REPL dispatches: the registry's names and aliases,
    plus the commands plugins add.

    This used to be reconstructed by running regular expressions over main.py
    and command_handler.py for `cmd == "/x"` idioms. Once dispatch moved into
    the registry those idioms vanished and the list quietly shrank to whatever
    the registry held, while the code that still looked healthy kept guarding
    nothing. The registry is what dispatch consults, so it is the only honest
    source; a "did you mean" built from anything else can suggest a command
    that no longer exists.
    """
    from src.cli.commands import get_registry

    found = {c for c in get_registry().get_all_commands() if c.startswith("/")}
    try:
        from hook_manager import hook_manager
        found |= {f"/{name}" for name in hook_manager.get_custom_commands()}
    except Exception:
        pass
    return sorted(found)


def unknown_command_message(command: str) -> str:
    """Say what was typed, and the nearest real command if there is one.

    Argent already fuzzy-matches tool names for the MODEL — `read_fil` becomes
    `read_file` — while answering a human's typo with a flat "Unknown command".
    The person is the one who cannot see the list.
    """
    import difflib

    typed = (command or "").strip().split()[0] if (command or "").strip() else ""
    close = difflib.get_close_matches(typed, known_commands(), n=3, cutoff=0.6)
    if close:
        return (f"Неизвестная команда: {typed}. "
                f"Возможно, вы имели в виду: {', '.join(close)}")
    return f"Неизвестная команда: {typed}. Список команд: /help"


# Menu labels live here, not inline: the tests select by label, and a
# duplicated string drifts the moment either side is reworded.
SEARCH_MENU_ENGINE = "Основной поисковый движок"
SEARCH_MENU_BRAVE = "Brave API-ключ (независимый индекс поиска)"
SEARCH_MENU_OLLAMA = "Ollama API-ключ (независимый индекс поиска)"
SEARCH_MENU_LANGS = "Языки поисковых запросов"
SEARCH_MENU_MODEL = "Модель reranker'а"

SEARCH_ENGINE_NAMES = {
    "all": "Все доступные (мета-поиск)",
    "duckduckgo": "DuckDuckGo (общий веб)",
    "brave": "Brave Search",
    "ollama": "Ollama Web Search",
    "github": "GitHub Issues/PRs",
    "stackoverflow": "StackOverflow / StackExchange",
    "wikipedia": "Wikipedia",
}


def _handle_search_command(command: str = ""):
    """Configure web research: the default engine, optional Brave index, query languages and
    the reranker. Supports fast CLI switches: /search ddg, /search brave, /search all, /search status.
    """
    from config import (
        get_brave_api_key, set_brave_api_key,
        get_ollama_api_key, set_ollama_api_key,
        get_reranker_model, set_reranker_model,
        get_search_languages, set_search_languages,
        get_search_engine, set_search_engine,
    )
    from src.research.rerank import MULTILINGUAL_MODEL, _DEFAULT_MODEL
    from src.research.search import engine_labels

    parts = (command or "").strip().split()
    if len(parts) > 1:
        sub = parts[1].lower().strip()
        if sub in ("status", "info", "check"):
            curr_engine = get_search_engine()
            label = SEARCH_ENGINE_NAMES.get(curr_engine, curr_engine)
            print_system(f"Основной движок: [bold cyan]{label}[/bold cyan]")
            print_system(f"Активные движки: [bold cyan]{', '.join(engine_labels())}[/bold cyan]")
            print_system(f"Brave API-ключ: {'задан' if get_brave_api_key() else '[dim]не задан[/dim]'}")
            print_system(f"Ollama API-ключ: {'задан' if get_ollama_api_key() else '[dim]не задан[/dim]'}")
            return

        eng_alias_map = {
            "all": "all",
            "default": "all",
            "auto": "all",
            "meta": "all",
            "duckduckgo": "duckduckgo",
            "ddg": "duckduckgo",
            "brave": "brave",
            "ollama": "ollama",
            "github": "github",
            "gh": "github",
            "stackoverflow": "stackoverflow",
            "stackexchange": "stackoverflow",
            "so": "stackoverflow",
            "wikipedia": "wikipedia",
            "wiki": "wikipedia",
        }
        if sub in eng_alias_map:
            target = eng_alias_map[sub]
            set_search_engine(target)
            target_label = SEARCH_ENGINE_NAMES.get(target, target)
            print_system(f"Поисковая система переключена на: [bold cyan]{target_label}[/bold cyan]")
            if target == "brave" and not get_brave_api_key():
                print_system("[yellow]Внимание:[/yellow] Brave API-ключ не задан. Настройте его через /search или на brave.com/search/api.")
            elif target == "ollama" and not get_ollama_api_key():
                print_system("[yellow]Внимание:[/yellow] Ollama API-ключ не задан. Настройте его через /search или на ollama.com/settings/keys.")
            return
        else:
            print_system(f"[yellow]Неизвестный поисковый движок '{parts[1]}'.[/yellow] Доступные: all, ddg, brave, ollama, github, stackoverflow, wikipedia")
            return

    key = get_brave_api_key()
    ollama_key = get_ollama_api_key()
    langs = get_search_languages()
    model = get_reranker_model() or _DEFAULT_MODEL
    multilingual = model != _DEFAULT_MODEL
    curr_engine = get_search_engine()
    curr_engine_label = SEARCH_ENGINE_NAMES.get(curr_engine, curr_engine)

    print_system(f"Основной движок: [bold cyan]{curr_engine_label}[/bold cyan]")
    print_system(f"Движки: [bold cyan]{', '.join(engine_labels())}[/bold cyan]")
    print_system(f"Brave API-ключ: {'задан' if key else '[dim]не задан[/dim]'}")
    print_system(f"Ollama API-ключ: {'задан' if ollama_key else '[dim]не задан[/dim]'}")
    print_system(f"Языки запросов: [bold cyan]{', '.join(langs)}[/bold cyan]")
    print_system(f"Reranker: [bold cyan]{model.split('/')[-1]}[/bold cyan]"
                 f" ({'мультиязычный' if multilingual else 'только английский'})")
    if langs != ["en"] and not multilingual:
        print_system("[yellow]Внимание:[/yellow] запросы не только на английском, но reranker "
                     "англоязычный — найденные неанглийские страницы будут отброшены при ранжировании.")

    choice = questionary.select(
        "Что настроить?",
        choices=[SEARCH_MENU_ENGINE, SEARCH_MENU_BRAVE, SEARCH_MENU_OLLAMA, SEARCH_MENU_LANGS,
                 SEARCH_MENU_MODEL, "Отмена"]).ask()

    if choice == SEARCH_MENU_ENGINE:
        picked = questionary.select(
            "Какую поисковую систему использовать по умолчанию?",
            choices=[
                questionary.Choice("Все доступные (мета-поиск по всем источникам)", value="all"),
                questionary.Choice("DuckDuckGo (общий веб, без ключей)", value="duckduckgo"),
                questionary.Choice("Brave Search (независимый индекс)", value="brave"),
                questionary.Choice("Ollama Web Search (облачный поиск)", value="ollama"),
                questionary.Choice("GitHub (issues и PR)", value="github"),
                questionary.Choice("StackOverflow / StackExchange", value="stackoverflow"),
                questionary.Choice("Wikipedia", value="wikipedia"),
            ],
            default="all" if curr_engine not in SEARCH_ENGINE_NAMES else curr_engine,
        ).ask()
        if picked is None:
            return
        set_search_engine(picked)
        print_system(f"Поисковая система переключена на: [bold cyan]{SEARCH_ENGINE_NAMES.get(picked, picked)}[/bold cyan]")
        if picked == "brave" and not key:
            print_system("[yellow]Внимание:[/yellow] Brave API-ключ не задан. Настройте его через /search.")
        elif picked == "ollama" and not ollama_key:
            print_system("[yellow]Внимание:[/yellow] Ollama API-ключ не задан. Настройте его через /search.")
        return

    if choice == SEARCH_MENU_BRAVE:
        new_key = questionary.password(
            "Brave Search API key (пусто — отключить; ключ берётся на brave.com/search/api):"
        ).ask()
        if new_key is None:
            return
        set_brave_api_key(new_key)
        print_system("Brave включён — второй индекс добавлен к поиску."
                     if new_key.strip() else "Brave отключён.")
        return

    if choice == SEARCH_MENU_OLLAMA:
        new_key = questionary.password(
            "Ollama API key (пусто — отключить; ключ берётся на ollama.com/settings/keys):"
        ).ask()
        if new_key is None:
            return
        set_ollama_api_key(new_key)
        if new_key.strip():
            print_system("Ollama-поиск включён — ещё один независимый индекс.")
            # Said once, at the moment of enabling: this key also authenticates
            # Ollama's cloud models, and the search itself is a hosted service
            # even when the model answering is local.
            print_system("[dim]Запросы уходят на ollama.com даже для локальных моделей, "
                         "а ключ хранится в конфиге открытым текстом и даёт доступ "
                         "к вашим облачным моделям.[/dim]")
        else:
            print_system("Ollama-поиск отключён.")
        return

    if choice == SEARCH_MENU_LANGS:
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

    if choice == SEARCH_MENU_MODEL:
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


def _configure_compress_fallback(primary: str):
    """After an OpenRouter model is picked for context compression: what happens when it is
    out of quota or down, and which free models to move to."""
    from config import (
        get_compress_fallback_models, set_compress_fallback_models,
        get_compress_on_failure, set_compress_on_failure,
    )
    from providers import create_provider
    from ui import _select_from_list

    AUTO = "Автоматически перейти на другую бесплатную модель"
    ASK = "Спросить меня, какую выбрать"
    mode = _select_from_list(
        "Если модель сжатия исчерпала лимит или не отвечает:",
        [AUTO, ASK],
        ASK if get_compress_on_failure() == "ask" else AUTO,
    )
    if mode:
        set_compress_on_failure("ask" if mode == ASK else "auto")

    try:
        free = [m for m in create_provider("openrouter").list_free_models() if m != primary]
    except Exception:
        free = []
    if not free:
        return
    current = set(get_compress_fallback_models())
    picked = questionary.checkbox(
        "Резервные бесплатные модели (в порядке списка; ничего не выбрано — любые свободные):",
        choices=[questionary.Choice(m, checked=m in current) for m in free[:60]],
    ).ask()
    if picked is not None:
        set_compress_fallback_models(picked)
    shown = ", ".join(get_compress_fallback_models()) or "любые свободные бесплатные"
    print_system(
        f"Сжатие контекста: при сбое — {'спросить' if get_compress_on_failure() == 'ask' else 'перейти автоматически'}; "
        f"резерв: {shown}."
    )


def _handle_aux_command(preset_provider: str | None = None):
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
    prov = preset_provider or questionary.select(
        "Провайдер для вспомогательных задач (суммаризация контекста, /commit):",
        choices=["ollama", "openrouter", "zai", "claude", "deepseek", "xkiro", "wormsoft", "koboldcpp", DISABLE],
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

    if prov == "openrouter" and models:
        # Hundreds of paid and free models are mixed together; the free ones are what
        # compression is normally pointed at, and what it falls back among.
        free = sorted(m for m in models if m.endswith(":free"))
        if free and len(models) > len(free):
            only = f"Только бесплатные ({len(free)})"
            scope = _select_from_list("Какие модели OpenRouter показать?", [only, f"Все ({len(models)})"], only)
            if scope == only:
                models = free
        models = sorted(models, key=lambda m: (not m.endswith(":free"), m))

    if models:
        model = _select_from_list(f"Модель {prov} для сервисных задач:", models, models[0])
    else:
        model = questionary.text(f"Имя модели {prov} (список недоступен, введите вручную):").ask()

    if model:
        set_auxiliary_provider(prov)
        set_auxiliary_model(model)
        print_system(f"Вспомогательная модель: [bold cyan]{prov}:{model}[/bold cyan]")
        if prov == "openrouter":
            _configure_compress_fallback(model)


def _handle_mcp_command(command: str):
    """Handle /mcp subcommands: list, add, remove, test."""
    from mcp_client import mcp_client, MCPTransportType

    parts = command.strip().split()
    
    if len(parts) == 1 or (len(parts) == 2 and parts[1] == "list"):
        servers = mcp_client.get_servers()
        configured = get_mcp_servers()
        if not configured:
            print_system("MCP-серверы не настроены.")
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
            print_error("Использование: /mcp remove <имя>")
            return
        name = parts[2]
        remove_mcp_server(name)
        result = mcp_client.unregister_server(name)
        print_system(result)
    elif subcmd == "test":
        if len(parts) < 3:
            print_error("Использование: /mcp test <имя>")
            return
        name = parts[2]
        result = mcp_client.test_connection(name)
        print_system(result)
    elif subcmd == "start":
        if len(parts) < 3:
            print_error("Использование: /mcp start <имя>")
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
            print_error("Использование: /mcp stop <имя>")
            return
        name = parts[2]
        result = mcp_client.unregister_server(name)
        print_system(result)
    else:
        print_error(f"Unknown /mcp subcommand: '{subcmd}'")
        print_system("Использование: /mcp [list|add|remove|start|stop|test]")


def _mcp_add(parts: list):
    """Parse /mcp add arguments and register server."""
    from mcp_client import mcp_client

    _PRESETS = {
        "blender": {"server_type": "stdio", "command": "uvx", "args": ["blender-mcp"]},
        "unity": {"server_type": "stdio", "command": "unity", "args": ["mcp"]},
    }

    if len(parts) == 3:
        target_name = parts[2].lower()
        if target_name in _PRESETS:
            preset = _PRESETS[target_name]
            add_mcp_server(target_name, **preset)
            config = {"type": preset["server_type"], "command": preset["command"], "args": preset["args"]}
            print_system(f"Используется пресет для '{target_name}': {preset['command']} {' '.join(preset['args'])}")
            result = mcp_client.register_and_start(target_name, config)
            print_system(result)
            return

    if len(parts) < 4:
        print_error("Usage:")
        print_system("  /mcp add blender  (preset: uvx blender-mcp)")
        print_system("  /mcp add unity    (preset: unity mcp)")
        print_system("  /mcp add <name> --stdio <command> [args...]")
        print_system("  /mcp add <name> --sse <url>")
        print_system("  /mcp add <name> --rest <url> [standard|unity_bridge]")
        print_system("")
        print_system("Examples:")
        print_system("  /mcp add blender")
        print_system('  /mcp add files --stdio npx -y @anthropic/mcp-server-filesystem "C:/Projects"')
        print_system("  /mcp add github --stdio npx -y @anthropic/mcp-server-github")
        print_system("  /mcp add unity")
        return

    name = parts[2]
    import re
    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        print_error("Имя сервера может содержать только буквы, цифры, дефис и подчеркивание.")
        return
        
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
            print_system("Внешние базы знаний не настроены.")
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
            print_error("Использование: /kb add <id> <имя> <путь>")
            print_system('Example: /kb add unity64 "Unity 6.4" "D:/UnityDocs"')
            return
        kb_id = parts[2]
        
        # Need to re-parse considering quotes
        import shlex
        try:
            parsed = shlex.split(command)
        except ValueError as e:
            print_error(f"Не удалось разобрать аргументы: {e}")
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
            print_error("Использование: /kb remove <id>")
            return
        kb_id = parts[2]
        remove_external_kb(kb_id)
        print_system(f"Removed Knowledge Base '{kb_id}'.")
        
    elif subcmd == "toggle":
        if len(parts) < 3:
            print_error("Использование: /kb toggle <id>")
            return
        kb_id = parts[2]
        new_status = toggle_external_kb(kb_id)
        status_str = "ENABLED" if new_status else "DISABLED"
        print_system(f"Knowledge Base '{kb_id}' is now {status_str}.")
        
    elif subcmd == "index":
        if len(parts) < 3:
            print_error("Использование: /kb index <id>")
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
        print_system("Использование: /kb [list|add|remove|toggle|index]")


def _handle_lmtrust_command(command: str) -> None:
    """Handle /lmtrust slash command for local zero-API blind-spot testing bridge."""
    from src.bridge.lmtrust_bridge import (
        create_test_request, list_pending_requests,
        get_test_response, apply_test_response
    )
    parts = command.strip().split()
    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() in ("help", "-h", "--help")):
        print_system(
            "**LMTrust Local Bridge (Zero-API Blind Spot Testing)**\n"
            "- `/lmtrust <файл>` — Создать запрос на Deep Blind Spot тесты для файла\n"
            "- `/lmtrust status` — Список ожидающих запросов в очереди `.argent/lmtrust_bridge/`\n"
            "- `/lmtrust apply <request_id> [целевой_файл]` — Применить готовые тесты в проект\n"
        )
        return

    sub = parts[1].lower()
    if sub == "status":
        pending = list_pending_requests()
        if not pending:
            print_system("В очереди `.argent/lmtrust_bridge/inbox/` нет необработанных запросов.")
        else:
            print_system(f"**Ожидающие запросы LMTrust ({len(pending)}):**")
            for req in pending:
                print_system(f"- `{req['id']}`: {req['source_file']} (Слои: {', '.join(req['focus_layers'])})")
    elif sub == "apply" and len(parts) > 2:
        req_id = parts[2]
        target = parts[3] if len(parts) > 3 else None
        res = apply_test_response(req_id, target_file=target)
        print_system(res)
    else:
        source_file = parts[1]
        layers = parts[2] if len(parts) > 2 else "L1,L2,L4,L4b,L7,L9"
        try:
            req = create_test_request(source_file, focus_layers=layers)
            print_system(
                f"✅ **Запрос на LMTrust тесты создан в шлюзе:**\n"
                f"- **ID:** `{req['id']}`\n"
                f"- **Файл:** `{req['source_file']}`\n"
                f"- **Слои:** `{', '.join(req['focus_layers'])}`\n"
                f"- **Статус:** Ожидает обработки в текущем чате Antigravity."
            )
        except Exception as e:
            print_error(f"Не удалось создать запрос LMTrust: {e}")


def _handle_telegram_command(command: str) -> None:
    """Handle /telegram slash command for setup and control of the Telegram bridge."""
    import time
    from config import (
        get_telegram_token, set_telegram_token,
        get_telegram_chat_id, set_telegram_chat_id,
        get_telegram_enabled, set_telegram_enabled,
        get_telegram_mode, set_telegram_mode
    )
    from src.telegram.bot import TelegramClient, telegram_bridge

    parts = command.strip().split()
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub in ("status", ""):
        token = get_telegram_token()
        chat_id = get_telegram_chat_id()
        mode = get_telegram_mode()
        mode_label = "Компаньон (shared) — единая сессия с ПК" if mode == "shared" else "Изолированный (isolated) — отдельная сессия"
        is_running = telegram_bridge.is_running
        
        status_text = "[bold green]АКТИВЕН[/bold green]" if is_running else "[bold yellow]ОСТАНОВЛЕН[/bold yellow]"
        masked_tok = f"{token[:8]}...{token[-4:]}" if token and len(token) > 12 else ("НЕ задан" if not token else "***")
        
        bot_line = ""
        if is_running and telegram_bridge._bot_info:
            bot_line = f"  • Бот: [bold]@{telegram_bridge._bot_info.get('username', '?')}[/bold]\n"
        print_system(
            f"[bold cyan]Telegram Bridge:[/bold cyan]\n"
            f"  • Статус: {status_text}\n"
            f"{bot_line}"
            f"  • Режим: [bold #58a6ff]{mode}[/bold #58a6ff] ({mode_label})\n"
            f"  • Токен бота: [dim]{masked_tok}[/dim]\n"
            f"  • Chat ID: {chat_id or '[dim]НЕ привязан[/dim]'}\n"
            f"\nКоманды управления:\n"
            f"  • `/telegram setup` — пошаговый мастер настройки и привязки\n"
            f"  • `/telegram test` — отправить на телефон проверочное сообщение\n"
            f"  • `/telegram mode [shared|isolated]` — переключить режим работы\n"
            f"  • `/telegram start` — запустить Telegram-мост\n"
            f"  • `/telegram stop` — остановить Telegram-мост\n"
            f"  • `/telegram unlink` — отвязать чат (бот перестанет слушаться)"
        )
        return

    if sub == "test":
        token, chat_id = get_telegram_token(), get_telegram_chat_id()
        if not token or not chat_id:
            print_error("Бот не настроен или чат не привязан. Запустите /telegram setup")
            return
        from src.telegram.bot import send_welcome
        client = telegram_bridge.client if telegram_bridge.is_running and telegram_bridge.client else TelegramClient(token)
        if client.send_message(chat_id, "✅ <b>Связь работает.</b>\nЭто проверочное сообщение от Argent.") is None:
            print_error(f"Telegram не принял сообщение: {client.last_error or 'нет ответа'}")
            return
        send_welcome(client, chat_id)
        print_system("[green]✓ Проверочное сообщение отправлено — посмотрите в Telegram.[/green]")
        return

    if sub == "unlink":
        if not get_telegram_chat_id():
            print_system("Чат и так не привязан.")
            return
        if not questionary.confirm("Отвязать чат? Бот перестанет принимать от него команды.", default=False).ask():
            print_system("Отменено.")
            return
        set_telegram_chat_id(None)
        print_system("Чат отвязан. Привязать заново — /telegram setup")
        return

    if sub == "mode":
        if len(parts) > 2:
            target_mode = parts[2].lower()
            if target_mode in ("shared", "companion"):
                set_telegram_mode("shared")
                print_system("Режим Telegram: [bold #58a6ff]shared[/bold #58a6ff] (Компаньон — единая сессия с терминалом).")
                return
            elif target_mode in ("isolated", "standalone"):
                set_telegram_mode("isolated")
                print_system("Режим Telegram: [bold #58a6ff]isolated[/bold #58a6ff] (Изолированный — отдельная сессия).")
                return
            else:
                print_error(f"Неизвестный режим: '{target_mode}'. Допустимые: shared, isolated")
                return
        else:
            current_mode = get_telegram_mode()
            choice_shared = "shared (Компаньон — единая сессия с терминалом и дублирование на экран ПК)"
            choice_isolated = "isolated (Изолированный — отдельная сессия в фоне без вывода в консоль)"
            choice = questionary.select(
                "Выберите режим работы Telegram-моста:",
                choices=[choice_shared, choice_isolated],
                default=choice_shared if current_mode == "shared" else choice_isolated
            ).ask()
            if choice:
                new_mode = "shared" if choice.startswith("shared") else "isolated"
                set_telegram_mode(new_mode)
                print_system(f"Режим Telegram изменён на: [bold #58a6ff]{new_mode}[/bold #58a6ff]")
            return

    if sub == "start":
        if telegram_bridge.is_running:
            print_system("Telegram-мост уже запущен.")
            return
        token = get_telegram_token()
        if not token:
            print_error("Токен бота не настроен. Запустите /telegram setup")
            return
        chat_id = get_telegram_chat_id()
        if not chat_id:
            print_system("[yellow]⚠ Chat ID ещё не привязан. Любые команды будут отклоняться до привязки через /telegram setup.[/yellow]")
        ok, msg = telegram_bridge.start()
        if ok:
            set_telegram_enabled(True)
            print_system(f"[green]✓ {msg}[/green]")
        else:
            print_error(f"Не удалось запустить: {msg}")
        return

    if sub == "stop":
        if not telegram_bridge.is_running:
            print_system("Telegram-мост не запущен.")
            return
        telegram_bridge.stop()
        set_telegram_enabled(False)
        print_system("Telegram-мост остановлен.")
        return

    if sub == "setup":
        print_system("[bold cyan]── Мастер настройки Telegram Bridge ──[/bold cyan]")
        current_token = get_telegram_token()
        if current_token:
            masked = f"{current_token[:8]}...{current_token[-4:]}" if len(current_token) > 12 else "***"
            change_token = questionary.confirm(f"Токен бота уже задан ({masked}). Изменить его?", default=False).ask()
            if change_token:
                new_token = questionary.password("Введите новый Telegram Bot Token (из @BotFather):").ask()
                if new_token:
                    current_token = new_token.strip()
        else:
            current_token = questionary.password("Введите Telegram Bot Token (получите у @BotFather в Telegram):").ask()

        if not current_token:
            print_system("Настройка прервана.")
            return

        client = TelegramClient(current_token)
        me = client.get_me()
        if not me:
            print_error("Ошибка: не удалось связаться с Telegram API с этим токеном. Проверьте правильность токена.")
            return

        set_telegram_token(current_token)
        bot_username = me.get("username", "bot")
        print_system(f"[green]✓ Бот @{bot_username} успешно авторизован![/green]")

        current_chat = get_telegram_chat_id()
        print_system("\nДля безопасности Argent принимает команды только от вашего аккаунта Telegram.")
        choices = [
            "Автоматически (отправьте /start боту в Telegram прямо сейчас)",
            "Ввести Chat ID вручную",
        ]
        if current_chat:
            choices.append(f"Оставить текущий ({current_chat})")
        else:
            choices.append("Пропустить привязку (сделать позже)")

        pair_method = questionary.select("Как привязать ваш Chat ID?", choices=choices).ask()

        if pair_method and pair_method.startswith("Автоматически"):
            print_system(f"\n[cyan]Жду сообщение от вас... Откройте Telegram и напишите [bold]@{bot_username}[/bold] любое сообщение или /start[/cyan]")
            paired = False
            offset = None
            start_time = time.time()
            while time.time() - start_time < 45:
                updates = client.get_updates(offset=offset, timeout=3)
                for u in updates:
                    offset = u.get("update_id", 0) + 1
                    msg = u.get("message")
                    if msg:
                        user_chat_id = msg.get("chat", {}).get("id")
                        user_name = msg.get("from", {}).get("username") or msg.get("from", {}).get("first_name", "User")
                        if user_chat_id:
                            set_telegram_chat_id(user_chat_id)
                            from src.telegram.bot import send_welcome
                            client.send_message(user_chat_id, "🎉 <b>Argent успешно привязан к этому чату!</b>")
                            send_welcome(client, user_chat_id)
                            print_system(f"[green]✓ Успешно привязан пользователь @{user_name} (Chat ID: {user_chat_id})[/green]")
                            paired = True
                            break
                if paired:
                    break
            if not paired:
                print_system("[yellow]Таймаут ожидания. Вы можете настроить Chat ID позже или ввести вручную.[/yellow]")

        elif pair_method and pair_method.startswith("Ввести Chat ID"):
            manual_id = questionary.text("Введите ваш Telegram Chat ID (узнать можно у @userinfobot):").ask()
            if manual_id:
                set_telegram_chat_id(manual_id.strip())
                print_system(f"[green]✓ Chat ID {manual_id.strip()} сохранён.[/green]")

        choice_shared = "shared (Компаньон — единая сессия с терминалом и дублирование в консоль ПК)"
        choice_isolated = "isolated (Изолированный — отдельная фоновая сессия)"
        mode_choice = questionary.select(
            "Выберите режим работы бота:",
            choices=[choice_shared, choice_isolated],
            default=choice_shared
        ).ask()
        if mode_choice:
            set_telegram_mode("shared" if mode_choice.startswith("shared") else "isolated")


        should_start = questionary.confirm("Запустить Telegram-мост прямо сейчас?", default=True).ask()
        if should_start:
            ok, msg = telegram_bridge.start()
            if ok:
                set_telegram_enabled(True)
                print_system(f"[green]✓ {msg}[/green]")
            else:
                print_error(f"Ошибка запуска: {msg}")
        return

    print_error(f"Неизвестная подкоманда: {sub}. Доступные: status, setup, test, mode, start, stop, unlink")



