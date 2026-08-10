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
    print_system(f"Модель изменена: {new_model}"
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
    """Handle slash commands. Returns True if REPL should exit."""
    cmd = command.lower().strip()
    if cmd in ("/exit", "/quit"):
        export_chat_history(agent, auto=True)
        return True
    elif cmd == "/clear":
        agent.clear_history()
        print_system("История диалога очищена.")
    elif cmd == "/model":
        current = get_current_model()
        new_model = select_model(current)
        if new_model and new_model != current:
            switch_model(agent, new_model)
        else:
            print_system("Модель не изменена.")
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

            print_system(f"Провайдер изменён: {new_prov}{options_text}")
            if new_prov != current_prov:
                # Every provider, ollama included: a model id almost never
                # exists on two of them, so keeping the old name means every
                # request 404s until the user works out they must run /model.
                print_system(f"Выберите модель {new_prov.upper()}:")
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
            print_system("Использование: /stop <pid>")
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
            "**Argent Coder — команды**\n"
            "\n**Модель и провайдер**\n"
            "- `/provider` — Провайдер API (Ollama / Z.ai / OpenRouter / KoboldCPP), ключ и адрес\n"
            "- `/model` — Активная модель (окно контекста и ярус по ней)\n"
            "- `/aux` — Дешёвая/локальная модель для служебных задач (сводки, /commit)\n"
            "- `/temp [значение]` — Температура генерации (0.0 – 2.0)\n"
            "- `/thinking` — Вырезать ли блоки рассуждений из истории\n"
            "\n**Писать и править код**\n"
            "- `/work [--auto] [задача]` — Менять или чинить СУЩЕСТВУЮЩИЙ код\n"
            "- `/vibe` — Вайб-режим: безопасное одобряется само, чекпоинт каждый ход (откат — /rewind)\n"
            "- `/auto [задача]` — Полностью автономный режим (экспериментальный)\n"
            "- `/tasks` — Автоматизации по расписанию, без присмотра\n"
            "- `/project [запрос]` — Большой проект с нуля\n"
            "- `/rooms <задача>` — Экспериментальный движок «комнаты и рельсы»\n"
            "- `/init` — Изучить проект и создать .argent/AGENTS.md (память о проекте)\n"
            "- `/commit` — Сгенерировать сообщение коммита и закоммитить\n"
            "\n**Файлы и изменения**\n"
            "- `/changes` — Что модель изменила за сессию\n"
            "- `/diff <файл>` — Дифф файла против снимка до правки\n"
            "- `/undo <файл>` — Вернуть файл к снимку до правки\n"
            "- `/rewind` — Машина времени: откат всего дерева к любому чекпоинту\n"
            "- `/cd [путь]` — Сменить (или показать) рабочую директорию\n"
            "\n**Знания и поиск**\n"
            "- `/rag_toggle` — Индексация для семантического поиска\n"
            "- `/auto_retrieve` — Подставлять результаты поиска в каждый запрос\n"
            "- `/kb [list|add|remove|toggle|index]` — Внешние базы знаний (документация движка, библиотеки)\n"
            "- `/kb_toggle` — Загружать внешние базы знаний при старте\n"
            "- `/research [тема]` — Автономное веб-исследование → заметки\n"
            "- `/search` — Настройки поиска: ключи Brave и Ollama, языки, reranker\n"
            "- `/skills` — Список навыков\n"
            "- `/skill import <источник>` — Установить навык из GitHub или папки\n"
            "\n**Безопасность и качество**\n"
            "- `/guard [off|warn|block]` — Защита от опасных команд\n"
            "- `/critic [текст]` — Разбор плана независимым критиком\n"
            "- `/critic on|off|model <имя>` — Автокритика диффа перед /commit\n"
            "- `/goal [текст|clear]` — Цель и прогресс: показать, задать, сбросить\n"
            "\n**Сессии**\n"
            "- `/save [метка]` — Экспорт диалога и сохранение сессии\n"
            "- `/sessions [запрос]` · `/load <номер|метка|фрагмент>` — Список и восстановление\n"
            "- `/copy <n>` — Скопировать блок кода №n в буфер\n"
            "- `/clear` — Очистить историю диалога\n"
            "\n**Инструменты и расширения**\n"
            "- `/tools` — Включить/выключить инструменты\n"
            "- `/hooks [путь]` — Каталог плагинов\n"
            "- `/plugin [auto on|off]` — Плагины: список и авто-создание их самим ИИ\n"
            "- `/mcp` — MCP-серверы (add/remove/start/stop/test)\n"
            "- `/browser [режим|браузер]` — Автоматизация браузера\n"
            "\n**Фон и диагностика**\n"
            "- `/jobs` · `/stop <pid>` — Фоновые процессы\n"
            "- `/doctor` — Самодиагностика окружения\n"
            "- `/stats` — Диагностика сессии (модель, бюджет контекста, плагины, MCP)\n"
            "- `/logs [модуль] [n]` — Логи (например /logs tools 20, /logs error)\n"
            "- `/verbose` · `/debug` — Индикаторы статуса / подробные логи\n"
            "\n**Прочее**\n"
            "- `/help` — Эта справка   ·   `/exit` (или `/quit`) — Выход\n"
        )
        
        custom_cmds = hook_manager.get_custom_commands()
        if custom_cmds:
            help_text += "\n**Команды плагинов:**\n"
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
        print_system(f"Подробные логи в чате (режим отладки): {state}")
    elif cmd == "/thinking":
        current = get_strip_reasoning()
        new_val = not current
        set_strip_reasoning(new_val)
        state = "[bold green]ON[/bold green]" if new_val else "[bold red]OFF[/bold red]"
        print_system(f"Принудительное удаление блоков рассуждений из истории: {state}")
    elif cmd == "/temp" or cmd.startswith("/temp ") or cmd == "/temperature" or cmd.startswith("/temperature "):
        parts = command.strip().split(" ", 1)
        if len(parts) > 1:
            val_str = parts[1].strip()
            try:
                val = float(val_str)
                if 0.0 <= val <= 2.0:
                    set_temperature(val)
                    print_system(f"Температура модели: {val}")
                else:
                    print_error("Температура должна быть от 0.0 до 2.0.")
            except ValueError:
                print_error("Температура должна быть числом.")
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
                    print_system("Температура модели: 0.2")
                elif choice.startswith("0.7"):
                    set_temperature(0.7)
                    print_system("Температура модели: 0.7")
                elif choice.startswith("1.0"):
                    set_temperature(1.0)
                    print_system("Температура модели: 1.0")
                elif choice == "Custom Value...":
                    custom_val = questionary.text("Enter custom temperature (0.0 to 2.0):").ask()
                    if custom_val:
                        try:
                            val = float(custom_val)
                            if 0.0 <= val <= 2.0:
                                set_temperature(val)
                                print_system(f"Температура модели: {val}")
                            else:
                                print_error("Температура должна быть от 0.0 до 2.0.")
                        except ValueError:
                            print_error("Введите число.")
    else:
        custom_cmds = hook_manager.get_custom_commands()
        base_cmd = command.strip().split(" ")[0].lstrip("/")
        if base_cmd in custom_cmds:
            try:
                args = command.strip().split(" ")[1:]
                custom_cmds[base_cmd](*args)
            except Exception as e:
                print_error(f"Пользовательская команда '/{base_cmd}' упала: {e}")
        else:
            print_error(unknown_command_message(command))
    return False


def known_commands() -> list:
    """Every slash command the REPL dispatches, read from the source.

    Derived rather than hand-listed: a second list of command names would drift
    from the dispatch the first time one is added, and a "did you mean" that
    suggests a command that no longer exists is worse than no suggestion.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent
    src = ""
    for name in ("main.py", "command_handler.py"):
        try:
            src += (root / name).read_text(encoding="utf-8")
        except OSError:
            continue
    found = set(re.findall(r'user_input(?:\.strip\(\))?\.startswith\("(/[a-z_]+)"', src))
    found |= set(re.findall(r'user_input\.strip\(\)\s*==\s*"(/[a-z_]+)"', src))
    found |= set(re.findall(r'cmd\s*==\s*"(/[a-z_]+)"', src))
    found |= set(re.findall(r'cmd\.startswith\("(/[a-z_]+)"', src))
    for group in re.findall(r'cmd in \(([^)]+)\)', src):
        found |= set(re.findall(r'"(/[a-z_]+)"', group))
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
SEARCH_MENU_BRAVE = "Brave API-ключ (независимый индекс поиска)"
SEARCH_MENU_OLLAMA = "Ollama API-ключ (независимый индекс поиска)"
SEARCH_MENU_LANGS = "Языки поисковых запросов"
SEARCH_MENU_MODEL = "Модель reranker'а"


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
        get_ollama_api_key, set_ollama_api_key,
        get_reranker_model, set_reranker_model,
        get_search_languages, set_search_languages,
    )
    from src.research.rerank import MULTILINGUAL_MODEL, _DEFAULT_MODEL
    from src.research.search import engine_labels

    key = get_brave_api_key()
    ollama_key = get_ollama_api_key()
    langs = get_search_languages()
    model = get_reranker_model() or _DEFAULT_MODEL
    multilingual = model != _DEFAULT_MODEL

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
        choices=[SEARCH_MENU_BRAVE, SEARCH_MENU_OLLAMA, SEARCH_MENU_LANGS,
                 SEARCH_MENU_MODEL, "Отмена"]).ask()

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
