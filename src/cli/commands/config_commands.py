"""Configuration and utility command handlers for Argent CLI.
Integrates with existing command_handler implementations for /model, /provider, /help,
/mcp, /kb, /search, /think, /aux, /telegram, /lmtrust, /tokens, /prompt, etc.
"""

import questionary
from config import (
    get_current_model, set_current_model,
    get_provider, set_provider,
    get_zai_api_key, set_zai_api_key,
    get_zai_endpoint, set_zai_endpoint,
    ZAI_ENDPOINT_CODING, ZAI_ENDPOINT_GENERAL,
    get_openrouter_api_key, set_openrouter_api_key,
    get_xkiro_api_key, set_xkiro_api_key,
    get_wormsoft_api_key, set_wormsoft_api_key,
    get_koboldcpp_url, set_koboldcpp_url,
)
from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import console, print_system, print_error, select_model, print_context_usage, _select_from_list


@registry.register("/ui", help_text="Стиль интерфейса TUI: premium (оформленный) или classic (как в терминале)", category="config", is_prefix=True)
def handle_ui(ctx: CommandContext) -> CommandResult:
    import os
    from config import UI_STYLES, get_ui_style, set_ui_style

    parts = ctx.user_input.strip().split(maxsplit=1)
    current = get_ui_style()
    if len(parts) < 2:
        print_system(
            f"Стиль интерфейса: [bold cyan]{current}[/bold cyan]. "
            "Переключить: /ui premium | /ui classic\n"
            "premium — оформленные панели, заголовки, код-карточки; classic — как в терминале. "
            "Касается Rust-интерфейса (TUI); обычный терминальный REPL всегда classic."
        )
        return CommandResult.continue_loop()
    wanted = parts[1].strip().lower()
    if wanted not in UI_STYLES:
        print_error(f"Неизвестный стиль: {wanted}. Доступно: {', '.join(UI_STYLES)}")
        return CommandResult.continue_loop()
    set_ui_style(wanted)
    if os.environ.get("ARGENT_TUI") == "1":
        try:
            from src.bridge.repl_bridge import ReplBridge
            if ReplBridge.active_bridge is not None:
                ReplBridge.active_bridge.emit({"type": "ui_style", "style": get_ui_style()})
        except Exception:
            pass
        print_system(f"Стиль интерфейса: [bold cyan]{get_ui_style()}[/bold cyan] — применён.")
    else:
        print_system(f"Стиль интерфейса сохранён: [bold cyan]{wanted}[/bold cyan]. Он применяется в TUI; терминальный REPL всегда classic.")
    return CommandResult.continue_loop()


@registry.register("/exit", aliases=["/quit", "/q"], help_text="Выход из Argent с сохранением сессии", category="config")
def handle_exit(ctx: CommandContext) -> CommandResult:
    from command_handler import export_chat_history
    export_chat_history(ctx.agent, auto=True)
    return CommandResult.exit_repl()


@registry.register("/model", aliases=["/models"], help_text="Выбор и переключение LLM модели (например: /model glm-5.3-flash:cloud)", category="config", is_prefix=True)
def handle_model(ctx: CommandContext) -> CommandResult:
    from command_handler import switch_model
    parts = ctx.user_input.strip().split(maxsplit=1)
    if len(parts) > 1:
        new_model = parts[1].strip()
        switch_model(ctx.agent, new_model)
    else:
        current = get_current_model()
        import os, sys
        new_model = None
        if (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()) or os.environ.get("ARGENT_TUI") == "1":
            try:
                new_model = select_model(current)
            except Exception:
                new_model = None

        if new_model and new_model != current:
            switch_model(ctx.agent, new_model)
        else:
            from providers import create_provider
            try:
                prov = create_provider()
                models = prov.list_models()
                _say_catalog_problem(prov)
                from claude_catalog import label_for
                models_preview = ", ".join(label_for(prov.name, m) for m in models[:10]) + ("..." if len(models) > 10 else "")
                current = label_for(prov.name, current)
                print_system(f"Текущая модель: [bold cyan]{current}[/bold cyan]\n"
                             f"Доступные модели ({len(models)}): {models_preview}\n"
                             f"Для переключения используйте: /model <имя_модели>")
            except Exception:
                print_system(f"Текущая модель: {current}. Для переключения используйте: /model <имя_модели>")
    return CommandResult.continue_loop()


def _say_catalog_problem(provider) -> None:
    """Tell the person why the model list is not the live one from the provider, if it is not."""
    catalog = getattr(provider, "_last_catalog", None)
    problem = getattr(catalog, "problem", "") if catalog is not None else ""
    if not problem:
        return
    hint = " Укажите workspace: /key workspace <ID>." if "workspace" in problem.lower() else ""
    print_system(f"[yellow]Список моделей Anthropic недоступен ({problem}).{hint} "
                 f"Показан {'сохранённый' if catalog.source == 'cache' else 'встроенный'} список.[/yellow]")


@registry.register("/provider", help_text="Выбор провайдера API (например: /provider zai)", category="config", is_prefix=True)
def handle_provider(ctx: CommandContext) -> CommandResult:
    from command_handler import switch_model
    current_prov = get_provider()
    choices = ["ollama", "zai", "openrouter", "claude", "deepseek", "xkiro", "wormsoft", "koboldcpp"]
    parts = ctx.user_input.strip().split(maxsplit=1)
    if len(parts) > 1:
        req_prov = parts[1].strip().lower()
        if req_prov in choices:
            new_prov = req_prov
        else:
            print_error(f"Неизвестный провайдер '{req_prov}'. Доступные: {', '.join(choices)}")
            return CommandResult.continue_loop()
    else:
        new_prov = _select_from_list("Выберите API провайдера:", choices, default=current_prov)

        if not new_prov:
            print_system(f"Текущий провайдер: [bold cyan]{current_prov}[/bold cyan]\n"
                         f"Доступные: {', '.join(choices)}\n"
                         f"Для переключения используйте: /provider <имя>")
            return CommandResult.continue_loop()

    if new_prov:
        set_provider(new_prov)
        ctx.agent.set_provider(new_prov)
        options_text = ""
        if new_prov == "zai":
            current_key = get_zai_api_key()
            if not current_key:
                try:
                    new_key = questionary.password("Enter Z.AI API Key:").ask()
                    if new_key:
                        set_zai_api_key(new_key)
                        options_text = " (API Key saved)"
                except Exception:
                    pass
        elif new_prov == "openrouter":
            current_key = get_openrouter_api_key()
            if not current_key:
                try:
                    new_key = questionary.password("Enter OpenRouter API Key:").ask()
                    if new_key:
                        set_openrouter_api_key(new_key)
                        options_text = " (API Key saved)"
                except Exception:
                    pass
        elif new_prov == "claude":
            from config import get_claude_api_key, set_claude_api_key
            current_key = get_claude_api_key()
            if not current_key:
                try:
                    new_key = questionary.password("Enter Claude API Key:").ask()
                    if new_key:
                        set_claude_api_key(new_key)
                        options_text = " (API Key saved)"
                except Exception:
                    pass
        elif new_prov == "xkiro":
            from config import get_xkiro_api_key, set_xkiro_api_key
            current_key = get_xkiro_api_key()
            if not current_key:
                try:
                    new_key = questionary.password("Enter xKiro API Key:").ask()
                    if new_key:
                        set_xkiro_api_key(new_key)
                        options_text = " (API Key saved)"
                except Exception:
                    pass
        elif new_prov == "wormsoft":
            from config import get_wormsoft_api_key, set_wormsoft_api_key
            current_key = get_wormsoft_api_key()
            if not current_key:
                try:
                    new_key = questionary.password("Enter WormSoft API Key:").ask()
                    if new_key:
                        set_wormsoft_api_key(new_key)
                        options_text = " (API Key saved)"
                except Exception:
                    pass
        elif new_prov == "deepseek":
            from config import get_deepseek_api_key, set_deepseek_api_key
            current_key = get_deepseek_api_key()
            if not current_key:
                try:
                    new_key = questionary.password("Enter DeepSeek API Key:").ask()
                    if new_key:
                        set_deepseek_api_key(new_key)
                        options_text = " (API Key saved)"
                except Exception:
                    pass
        elif new_prov == "koboldcpp":
            current_url = get_koboldcpp_url()
            options_text = f" (URL: {current_url})"

        print_system(f"Провайдер изменён: {new_prov}{options_text}")
        if new_prov != current_prov:
            from config import get_model_for_provider
            saved_model = get_model_for_provider(new_prov)
            from providers import starting_model
            # A model name from the PREVIOUS provider means nothing here: start from one this provider knows.
            prompt_model = starting_model(new_prov, saved_model, get_current_model(new_prov))
            print_system(f"Выберите модель {new_prov.upper()} (активная: {prompt_model}):")
            new_model = select_model(prompt_model)
            if new_model and new_model != get_current_model():
                switch_model(ctx.agent, new_model)
            elif prompt_model:
                switch_model(ctx.agent, prompt_model)

    return CommandResult.continue_loop()


@registry.register("/compact", aliases=["/compress"], help_text="Прогрессивная дистилляция и сжатие контекста (/compact model — выбрать бесплатную модель OpenRouter для сжатия)", category="config", is_prefix=True)
def handle_compact(ctx: CommandContext) -> CommandResult:
    agent = ctx.agent
    parts = ctx.user_input.strip().split()
    if len(parts) > 1 and parts[1].lower() in ("model", "модель"):
        from command_handler import _handle_aux_command
        _handle_aux_command(preset_provider="openrouter")
        return CommandResult.continue_loop()
    if len(agent.messages) <= 2:
        print_system("История диалога слишком коротка для сжатия.")
    else:
        from src.agent.trimmer import soft_trim_with_summarization, age_tool_results
        agent.messages = age_tool_results(agent.messages)
        target_msgs = max(4, int(agent.max_history_messages * 0.5))
        target_budget = max(1024, int(agent.max_context_tokens * 0.5))
        before_len = len(agent.messages)
        agent.messages = soft_trim_with_summarization(
            agent.messages, agent.model_name, target_msgs, target_budget
        )
        after_len = len(agent.messages)
        if hasattr(agent, "reset_turn_records"):
            agent.reset_turn_records()
        print_system(f"Контекст оптимизирован: {before_len} → {after_len} сообщений (Прогрессивная дистилляция).")
    return CommandResult.continue_loop()


def _parse_duration(text: str) -> float | None:
    """'90' / '90с' / '15м' / '2ч' / 'off' -> seconds (None when it is not a duration)."""
    t = text.strip().lower()
    if t in ("off", "0", "выкл", "нет", "no"):
        return 0.0
    units = {"с": 1, "s": 1, "м": 60, "m": 60, "min": 60, "мин": 60, "ч": 3600, "h": 3600}
    for suffix in sorted(units, key=len, reverse=True):
        if t.endswith(suffix) and t[:-len(suffix)].replace(".", "", 1).isdigit():
            return float(t[:-len(suffix)]) * units[suffix]
    return float(t) if t.replace(".", "", 1).isdigit() else None


@registry.register("/network", aliases=["/сеть"], help_text="Сколько ждать, если пропал интернет или провайдер перегружен (/network 15м | off)", category="config", is_prefix=True)
def handle_network(ctx: CommandContext) -> CommandResult:
    from config import get_network_patience, set_network_patience
    from src.agent.netstate import clock_text
    parts = ctx.user_input.strip().split(maxsplit=1)
    if len(parts) > 1:
        seconds = _parse_duration(parts[1])
        if seconds is None:
            print_error("Использование: /network 15м | 90с | 2ч | off")
            return CommandResult.continue_loop()
        set_network_patience(seconds)
    cur = get_network_patience()
    if cur <= 0:
        print_system("Ожидание связи: [bold red]выключено[/bold red] — при обрыве ход заканчивается ошибкой.")
    else:
        print_system(f"Ожидание связи: [bold cyan]до {clock_text(cur)}[/bold cyan]. Если связь пропала, Argent "
                     "ждёт её возвращения и сам повторяет запрос; диалог при этом сохранён.")
    return CommandResult.continue_loop()


@registry.register("/think", aliases=["/thinking"], help_text="Настройки размышлений модели (ON/OFF/AUTO, effort, budget, strip, rethink, summary/full)", category="config", is_prefix=True)
def handle_think(ctx: CommandContext) -> CommandResult:
    from config import (
        get_thinking_mode, set_thinking_mode,
        get_reasoning_effort, set_reasoning_effort,
        get_thinking_budget, set_thinking_budget,
        get_thinking_rethink, set_thinking_rethink,
        get_thinking_display_mode, set_thinking_display_mode,
        get_strip_reasoning, set_strip_reasoning,
        get_thinking_display_label,
    )
    parts = ctx.user_input.strip().split()
    if len(parts) > 1:
        sub = parts[1].lower()
        if sub in ("summary", "compact", "short"):
            set_thinking_display_mode("summary")
            print_system("Отображение рассуждений: [bold green]summary[/bold green] (компактная строка с таймером в стиле Claude Code).")
        elif sub in ("full", "tree", "expanded", "verbose"):
            set_thinking_display_mode("full")
            print_system("Отображение рассуждений: [bold yellow]full[/bold yellow] (полное дерево рассуждений).")
        elif sub in ("on", "enable", "enabled", "1", "true"):
            set_thinking_mode("on")
            print_system(f"Режим генерации рассуждений: [bold green]ON[/bold green] [{get_thinking_display_label()}].")
        elif sub in ("off", "disable", "disabled", "0", "false"):
            set_thinking_mode("off")
            print_system(f"Режим генерации рассуждений: [bold red]OFF[/bold red] [{get_thinking_display_label()}].")
        elif sub in ("auto", "default", "reset"):
            set_thinking_mode("auto")
            print_system(f"Режим генерации рассуждений: [bold cyan]AUTO[/bold cyan] [{get_thinking_display_label()}].")
        elif sub in ("low", "medium", "high", "xhigh"):
            set_reasoning_effort(sub)
            print_system(f"Глубина размышлений модели установлена: [bold cyan]{sub}[/bold cyan] [{get_thinking_display_label()}].")
        elif sub == "rethink":
            if len(parts) > 2:
                arg = parts[2].lower()
                if arg in ("on", "1", "true", "enable"):
                    set_thinking_rethink(True)
                elif arg in ("off", "0", "false", "disable"):
                    set_thinking_rethink(False)
                else:
                    print_error("Использование: /think rethink [on|off]")
                    return CommandResult.continue_loop()
            else:
                set_thinking_rethink(not get_thinking_rethink())
            st = "[bold green]ВКЛ[/bold green]" if get_thinking_rethink() else "[bold red]ВЫКЛ[/bold red]"
            print_system(f"Протокол самопроверки (<rethink>): {st} [{get_thinking_display_label()}].")
        elif sub == "budget":
            if len(parts) > 2:
                b_str = parts[2].lower()
                if b_str in ("none", "off", "reset", "0", "null"):
                    set_thinking_budget(None)
                    print_system("Лимит токенов на рассуждения: [bold cyan]без ограничений[/bold cyan].")
                else:
                    try:
                        b = int(b_str)
                        if b < 0:
                            print_error("Бюджет токенов не может быть отрицательным.")
                        else:
                            set_thinking_budget(b if b > 0 else None)
                            print_system(f"Лимит токенов на рассуждения: [bold cyan]{b if b > 0 else 'без ограничений'}[/bold cyan].")
                    except ValueError:
                        print_error("Бюджет токенов должен быть целым числом (например: /think budget 4096) или 'none'.")
            else:
                cur_b = get_thinking_budget()
                print_system(f"Текущий бюджет токенов на рассуждения: [bold cyan]{cur_b or 'без ограничений'}[/bold cyan].")
        elif sub == "strip":
            if len(parts) > 2:
                arg = parts[2].lower()
                if arg in ("on", "1", "true", "enable"):
                    set_strip_reasoning(True)
                elif arg in ("off", "0", "false", "disable"):
                    set_strip_reasoning(False)
                else:
                    print_error("Использование: /think strip [on|off]")
                    return CommandResult.continue_loop()
            else:
                set_strip_reasoning(not get_strip_reasoning())
            st = "[bold green]ВКЛ[/bold green]" if get_strip_reasoning() else "[bold red]ВЫКЛ[/bold red]"
            print_system(f"Очистка блоков рассуждений из истории: {st}")
        else:
            print_error("Неизвестный параметр. Допустимо: /think [on|off|auto|low|medium|high|xhigh|summary|full|budget|rethink|strip]")
    else:
        cur_mode = get_thinking_mode()
        cur_effort = get_reasoning_effort() or "default"
        cur_disp = get_thinking_display_mode()
        cur_rethink = "ON" if get_thinking_rethink() else "OFF"
        cur_badge = get_thinking_display_label()

        choices = [
            f"1. Режим генерации: {cur_mode.upper()} (ON / OFF / AUTO)",
            f"2. Глубина усилий (effort): {cur_effort} (low / medium / high / xhigh / default)",
            f"3. Отображение в UI: {cur_disp} (summary / full)",
            f"4. Самопроверка (<rethink>): {cur_rethink} (переключить)",
            "5. Оставить без изменений",
        ]
        choice = questionary.select(
            f"Настройки размышлений модели (сейчас: [{cur_badge}]):",
            choices=choices,
            default=choices[4],
        ).ask()
        if not choice or "без изменений" in choice:
            return CommandResult.continue_loop()

        if choice.startswith("1."):
            m_choice = questionary.select(
                f"Выберите режим размышлений (сейчас: {cur_mode}):",
                choices=["auto — По умолчанию модели", "on — Принудительно включить", "off — Принудительно выключить"],
                default="auto — По умолчанию модели" if cur_mode == "auto" else ("on — Принудительно включить" if cur_mode == "on" else "off — Принудительно выключить")
            ).ask()
            if m_choice:
                if "on" in m_choice:
                    set_thinking_mode("on")
                elif "off" in m_choice:
                    set_thinking_mode("off")
                else:
                    set_thinking_mode("auto")
                print_system(f"Режим размышлений установлен: [bold cyan]{get_thinking_mode().upper()}[/bold cyan]")
        elif choice.startswith("2."):
            e_choice = questionary.select(
                f"Выберите глубину размышлений (сейчас: {cur_effort}):",
                choices=["low", "medium", "high", "xhigh", "default (сбросить)"],
                default=cur_effort if cur_effort in ("low", "medium", "high", "xhigh") else "default (сбросить)"
            ).ask()
            if e_choice:
                if e_choice.startswith("default"):
                    set_reasoning_effort(None)
                    print_system("Глубина размышлений сброшена на дефолт.")
                else:
                    set_reasoning_effort(e_choice)
                    print_system(f"Глубина размышлений: [bold cyan]{e_choice}[/bold cyan]")
        elif choice.startswith("3."):
            d_choice = questionary.select(
                f"Отображение рассуждений (сейчас: {cur_disp}):",
                choices=["summary — Компактная строка с таймером", "full — Полное дерево рассуждений"],
                default="summary — Компактная строка с таймером" if cur_disp == "summary" else "full — Полное дерево рассуждений"
            ).ask()
            if d_choice:
                val = "summary" if "summary" in d_choice else "full"
                set_thinking_display_mode(val)
                print_system(f"Отображение рассуждений: [bold cyan]{val}[/bold cyan]")
        elif choice.startswith("4."):
            new_r = not get_thinking_rethink()
            set_thinking_rethink(new_r)
            st = "[bold green]ВКЛ[/bold green]" if new_r else "[bold red]ВЫКЛ[/bold red]"
            print_system(f"Самопроверка (<rethink>): {st}")

    return CommandResult.continue_loop()


@registry.register("/effort", help_text="Глубина размышлений модели (low / medium / high / xhigh / off)", category="config", is_prefix=True)
def handle_effort(ctx: CommandContext) -> CommandResult:
    from config import get_reasoning_effort, set_reasoning_effort, get_thinking_display_label
    parts = ctx.user_input.strip().split()
    if len(parts) > 1:
        val = parts[1].lower()
        if val in ("low", "medium", "high", "xhigh"):
            set_reasoning_effort(val)
            print_system(f"Глубина размышлений модели установлена: [bold cyan]{val}[/bold cyan] [{get_thinking_display_label()}].")
        elif val in ("off", "none", "reset", "default", "def"):
            set_reasoning_effort(None)
            print_system("Глубина размышлений сброшена на [bold cyan]значение модели по умолчанию[/bold cyan].")
        else:
            print_error("Недопустимое значение. Выберите: low, medium, high, xhigh или off")
    else:
        cur = get_reasoning_effort() or "по умолчанию модели"
        choices = [
            "low — Краткая глубина размышлений",
            "medium — Сбалансированная глубина размышлений",
            "high — Глубокие размышления",
            "xhigh — Максимально глубокие размышления",
            "off — Сбросить на значение по умолчанию",
            f"Оставить текущую ({cur})",
        ]
        choice = questionary.select(
            f"Выберите глубину размышлений модели (сейчас: {cur}):",
            choices=choices,
            default=f"Оставить текущую ({cur})",
        ).ask()
        if choice and not choice.startswith("Оставить"):
            if choice.startswith("low"):
                set_reasoning_effort("low")
                print_system("Глубина размышлений: [bold cyan]low[/bold cyan]")
            elif choice.startswith("medium"):
                set_reasoning_effort("medium")
                print_system("Глубина размышлений: [bold cyan]medium[/bold cyan]")
            elif choice.startswith("high"):
                set_reasoning_effort("high")
                print_system("Глубина размышлений: [bold cyan]high[/bold cyan]")
            elif choice.startswith("xhigh"):
                set_reasoning_effort("xhigh")
                print_system("Глубина размышлений: [bold cyan]xhigh[/bold cyan]")
            elif choice.startswith("off"):
                set_reasoning_effort(None)
                print_system("Глубина размышлений сброшена на дефолт.")
    return CommandResult.continue_loop()


@registry.register("/rethink", help_text="Режим самопроверки модели (<rethink>): on / off / status", category="config", is_prefix=True)
def handle_rethink(ctx: CommandContext) -> CommandResult:
    from config import get_thinking_rethink, set_thinking_rethink, get_thinking_display_label
    parts = ctx.user_input.strip().split()
    if len(parts) > 1:
        sub = parts[1].lower()
        if sub in ("on", "1", "true", "enable"):
            set_thinking_rethink(True)
            print_system(f"Режим самопроверки (<rethink>): [bold green]ВКЛЮЧЁН[/bold green] [{get_thinking_display_label()}].")
        elif sub in ("off", "0", "false", "disable"):
            set_thinking_rethink(False)
            print_system(f"Режим самопроверки (<rethink>): [bold red]ВЫКЛЮЧЕН[/bold red] [{get_thinking_display_label()}].")
        elif sub in ("status", "state"):
            cur = get_thinking_rethink()
            status_str = "[bold green]ВКЛ[/bold green]" if cur else "[bold red]ВЫКЛ[/bold red]"
            print_system(f"Статус самопроверки (<rethink>): {status_str} [{get_thinking_display_label()}].")
        else:
            print_error("Использование: /rethink [on|off|status]")
    else:
        new_state = not get_thinking_rethink()
        set_thinking_rethink(new_state)
        status_str = "[bold green]ВКЛЮЧЁН[/bold green]" if new_state else "[bold red]ВЫКЛЮЧЕН[/bold red]"
        print_system(f"Режим самопроверки (<rethink>): {status_str} [{get_thinking_display_label()}].")
    return CommandResult.continue_loop()


@registry.register("/mcp", help_text="Управление серверами MCP", category="config", is_prefix=True)
def handle_mcp(ctx: CommandContext) -> CommandResult:
    from command_handler import _handle_mcp_command
    _handle_mcp_command(ctx.user_input)
    return CommandResult.continue_loop()


@registry.register("/telegram", help_text="Управление мостом Telegram", category="config", is_prefix=True)
def handle_telegram(ctx: CommandContext) -> CommandResult:
    from command_handler import _handle_telegram_command
    _handle_telegram_command(ctx.user_input)
    return CommandResult.continue_loop()


@registry.register("/kb", help_text="База знаний (Knowledge Base)", category="config", is_prefix=True)
def handle_kb(ctx: CommandContext) -> CommandResult:
    from command_handler import _handle_kb_command
    _handle_kb_command(ctx.user_input)
    return CommandResult.continue_loop()


@registry.register("/lmtrust", aliases=["/blindspot"], help_text="Тесты доверия LMTrust", category="config", is_prefix=True)
def handle_lmtrust(ctx: CommandContext) -> CommandResult:
    from command_handler import _handle_lmtrust_command
    _handle_lmtrust_command(ctx.user_input)
    return CommandResult.continue_loop()


@registry.register("/jobs", help_text="Список фоновых задач", category="config")
def handle_jobs(ctx: CommandContext) -> CommandResult:
    from tools.command_ops import list_background_commands
    print_system(list_background_commands())
    return CommandResult.continue_loop()


@registry.register("/stop", help_text="Остановить фоновую задачу", category="config", is_prefix=True)
def handle_stop(ctx: CommandContext) -> CommandResult:
    from tools.command_ops import stop_background_command, list_background_commands
    parts = ctx.user_input.strip().split()
    if len(parts) < 2:
        print_system(list_background_commands())
        print_system("Использование: /stop <pid>")
    else:
        print_system(stop_background_command(parts[1]))
    return CommandResult.continue_loop()


@registry.register("/aux", help_text="Настройка вспомогательной модели", category="config")
def handle_aux(ctx: CommandContext) -> CommandResult:
    from command_handler import _handle_aux_command
    _handle_aux_command()
    return CommandResult.continue_loop()


@registry.register("/fast", help_text="Быстрый режим Fast-Path (инструменты без обращения к LLM)", category="config", is_prefix=True)
def handle_fast(ctx: CommandContext) -> CommandResult:
    from config import (
        get_fast_dispatch_backend, set_fast_dispatch_backend,
        get_fast_mode, set_fast_mode
    )
    parts = ctx.user_input.strip().split()
    arg = parts[1].lower() if len(parts) > 1 else ""

    if arg in ("on", "1", "true"):
        set_fast_mode(True)
        backend = get_fast_dispatch_backend()
        print_system(f"Быстрый режим (Fast-Path): [bold green]ВКЛ[/bold green] (активный бэкенд: [bold cyan]{backend}[/bold cyan])")
    elif arg in ("off", "0", "false"):
        set_fast_mode(False)
        print_system("Быстрый режим (Fast-Path): [bold red]ВЫКЛ[/bold red] (все запросы направляются в LLM)")
    elif arg in ("leeth", "micromodel", "needle"):
        set_fast_dispatch_backend(arg)
        print_system(f"Быстрый режим (Fast-Path): переключен на [bold cyan]{arg}[/bold cyan]")
    elif arg == "model":
        from config import LEETH_BACKENDS, get_leeth_backend, set_leeth_backend

        choice = parts[2].lower() if len(parts) > 2 else ""
        if choice in LEETH_BACKENDS:
            set_leeth_backend(choice)
            print_system(f"Модель Leeth: [bold cyan]{choice}[/bold cyan]" + (" (единая откалиброванная модель src/leeth2; веса: .argent/models/leeth2/final, при их отсутствии работает прежняя)" if choice == "leeth2" else " (400M + mmBERT-роутер)"))
        else:
            print_system(f"Модель Leeth сейчас: [bold cyan]{get_leeth_backend()}[/bold cyan]. Варианты: /fast model legacy | /fast model leeth2")
    elif arg == "status":
        backend = get_fast_dispatch_backend()
        active = backend != "off"
        status_str = f"[bold green]ВКЛ[/bold green] (бэкенд: [bold cyan]{backend}[/bold cyan])" if active else "[bold red]ВЫКЛ[/bold red]"
        print_system(f"⚡ Fast-Path статус: {status_str}")
        print_system("  /fast on|off         — включить/выключить перехват рутинных инструментов")
        print_system("  /fast leeth          — использовать Leeth (1.5ms, 23.7M params, INT8 ONNX)")
        print_system("  /fast micromodel     — использовать MicroModel router")
        print_system("  /fast needle         — использовать Needle 3 coprocessor")
        print_system("  /fast model legacy|leeth2 — какая модель отвечает на вопросы Leeth (прежний стек или Leeth 2)")
    else:
        cur = get_fast_mode()
        set_fast_mode(not cur)
        backend = get_fast_dispatch_backend()
        if not cur:
            print_system(f"Быстрый режим (Fast-Path): [bold green]ВКЛ[/bold green] (бэкенд: [bold cyan]{backend}[/bold cyan])")
        else:
            print_system("Быстрый режим (Fast-Path): [bold red]ВЫКЛ[/bold red] (все запросы направляются в LLM)")
    return CommandResult.continue_loop()


@registry.register("/search", help_text="Настройки веб-поиска", category="config", is_prefix=True)
def handle_search(ctx: CommandContext) -> CommandResult:
    from command_handler import _handle_search_command
    _handle_search_command(ctx.user_input)
    return CommandResult.continue_loop()


@registry.register("/checkpoint", help_text="Создать страховочный git-чекпоинт", category="config", is_prefix=True)
def handle_checkpoint(ctx: CommandContext) -> CommandResult:
    from tools.misc_tools import git_checkpoint
    parts = ctx.user_input.strip().split(maxsplit=1)
    msg = parts[1].strip() if len(parts) > 1 else "ручной чекпоинт"
    print_system(git_checkpoint(msg))
    return CommandResult.continue_loop()


@registry.register("/audit", help_text="Аудит безопасности проекта", category="config", is_prefix=True)
def handle_audit(ctx: CommandContext) -> CommandResult:
    from tools.security_tools import audit_security
    parts = ctx.user_input.strip().split(maxsplit=1)
    level = parts[1].strip() if len(parts) > 1 else "standard"
    print_system(audit_security(level=level))
    return CommandResult.continue_loop()


@registry.register("/doctor", help_text="Самодиагностика окружения и зависимостей", category="config")
def handle_doctor(ctx: CommandContext) -> CommandResult:
    from doctor import print_diagnostics
    print_diagnostics()
    return CommandResult.continue_loop()



@registry.register("/tokens", aliases=["/context"], help_text="Использование токенов контекста", category="config")
def handle_tokens(ctx: CommandContext) -> CommandResult:
    from config import get_current_model
    b = ctx.agent.get_context_breakdown()
    print_context_usage(b.get("total", 0), b.get("max", 1), get_current_model(), breakdown=b)
    return CommandResult.continue_loop()


@registry.register("/prompt", help_text="Просмотр текущего системного промпта", category="config")
def handle_prompt(ctx: CommandContext) -> CommandResult:
    sp = ctx.agent.build_system_prompt()
    from rich.syntax import Syntax
    console.print(Syntax(sp, "markdown", theme="monokai", word_wrap=True))
    return CommandResult.continue_loop()


@registry.register("/key", help_text="Установить API ключ", category="config")
def handle_key(ctx: CommandContext) -> CommandResult:
    prov = get_provider()
    if prov == "zai":
        k = questionary.password("Введите Z.AI API Key:").ask()
        if k:
            set_zai_api_key(k)
            print_system("Z.AI API Key сохранён.")
    elif prov == "openrouter":
        k = questionary.password("Введите OpenRouter API Key:").ask()
        if k:
            set_openrouter_api_key(k)
            print_system("OpenRouter API Key сохранён.")
    elif prov == "xkiro":
        k = questionary.password("Введите xKiro API Key:").ask()
        if k:
            set_xkiro_api_key(k)
            print_system("xKiro API Key сохранён.")
    elif prov == "wormsoft":
        k = questionary.password("Введите WormSoft API Key:").ask()
        if k:
            set_wormsoft_api_key(k)
            print_system("WormSoft API Key сохранён.")
    elif prov == "claude":
        from config import get_claude_workspace_id, set_claude_api_key, set_claude_workspace_id
        words = ctx.user_input.strip().split()
        if len(words) > 1 and words[1].lower() == "workspace":
            # An organisation-level key needs the workspace named on every call.
            if len(words) > 2:
                set_claude_workspace_id(None if words[2].lower() in ("off", "none", "-") else words[2])
                print_system(f"Workspace Claude: {get_claude_workspace_id() or 'не задан'}")
            else:
                print_system(f"Workspace Claude: {get_claude_workspace_id() or 'не задан'}. "
                             "Задать: /key workspace <ID>, снять: /key workspace off")
            return CommandResult.continue_loop()
        k = questionary.password("Введите Claude API Key:").ask()
        if k:
            set_claude_api_key(k)
            print_system("Claude API Key сохранён.")
    elif prov == "deepseek":
        from config import set_deepseek_api_key
        k = questionary.password("Введите DeepSeek API Key:").ask()
        if k:
            set_deepseek_api_key(k)
            print_system("DeepSeek API Key сохранён.")
    else:
        print_system(f"Для провайдера {prov} API-ключ не требуется.")
    return CommandResult.continue_loop()


@registry.register("/endpoint", help_text="Переключить эндпоинт Z.AI", category="config")
def handle_endpoint(ctx: CommandContext) -> CommandResult:
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
        print_system("Z.AI Endpoint set to Coding Plan.")
    else:
        set_zai_endpoint(ZAI_ENDPOINT_GENERAL)
        print_system("Z.AI Endpoint set to General API.")
@registry.register("/temperature", aliases=["/temp"], help_text="Настройка температуры генерации (0.0–2.0)", category="config", is_prefix=True)
def handle_temperature(ctx: CommandContext) -> CommandResult:
    from config import get_temperature, set_temperature
    parts = ctx.user_input.strip().split(maxsplit=1)
    if len(parts) > 1:
        val_str = parts[1].strip()
        if val_str.lower() in ("default", "reset", "def"):
            set_temperature(0.3)
            print_system("Температура модели сброшена к дефолту: [bold cyan]0.3[/bold cyan]")
            return CommandResult.continue_loop()
        try:
            val = float(val_str)
            if 0.0 <= val <= 2.0:
                set_temperature(val)
                print_system(f"Температура модели установлена: [bold cyan]{val}[/bold cyan]")
            else:
                print_error("Температура должна быть в диапазоне от 0.0 до 2.0.")
        except ValueError:
            print_error("Температура должна быть числом (например: 0.2, 0.7, 1.0) или 'default'.")
        return CommandResult.continue_loop()

    current_temp = get_temperature()
    choices = [
        "0.2 (Детерминированная — для кода и строгих инструкций)",
        "0.3 (По умолчанию в Argent — сбалансированная)",
        "0.7 (Баланс креативности и логики)",
        "1.0 (Креативная генерация)",
        "Сбросить к дефолту (0.3)",
        f"Оставить текущую ({current_temp})",
        "Своё значение…",
    ]
    choice = questionary.select(
        f"Выберите температуру модели (сейчас: {current_temp}):",
        choices=choices,
        default=f"Оставить текущую ({current_temp})",
    ).ask()

    if not choice or choice.startswith("Оставить"):
        return CommandResult.continue_loop()

    if choice.startswith("0.2"):
        set_temperature(0.2)
        print_system("Температура модели: [bold cyan]0.2[/bold cyan]")
    elif choice.startswith("0.3") or choice.startswith("Сбросить"):
        set_temperature(0.3)
        print_system("Температура модели: [bold cyan]0.3[/bold cyan]")
    elif choice.startswith("0.7"):
        set_temperature(0.7)
        print_system("Температура модели: [bold cyan]0.7[/bold cyan]")
    elif choice.startswith("1.0"):
        set_temperature(1.0)
        print_system("Температура модели: [bold cyan]1.0[/bold cyan]")
    elif choice.startswith("Своё"):
        custom = questionary.text(f"Введите температуру (0.0–2.0):", default=str(current_temp)).ask()
        if custom:
            try:
                val = float(custom.strip())
                if 0.0 <= val <= 2.0:
                    set_temperature(val)
                    print_system(f"Температура модели: [bold cyan]{val}[/bold cyan]")
                else:
                    print_error("Температура должна быть от 0.0 до 2.0.")
            except ValueError:
                print_error("Неверный формат числа.")
    return CommandResult.continue_loop()


# --- Plain switches and one number -------------------------------------------------
# /verbose, /results, /debug and /max_tokens were dispatched by the old
# command_handler chain and fell out when commands moved to the registry: the
# README, the Tab menu and the config setters all still existed, and typing the
# command answered "unknown command". /max_tokens was the only way to change
# max_generation_tokens, which the agent and the providers both read.

def _explicit_toggle(argument: str):
    """True / False for an explicit on|off word, None for anything else."""
    word = argument.strip().lower()
    if word in ("on", "1", "true", "yes", "вкл", "да"):
        return True
    if word in ("off", "0", "false", "no", "выкл", "нет"):
        return False
    return None


def _switch(ctx: CommandContext, get, set_, label: str, note_off: str = "") -> CommandResult:
    """Body shared by the on/off commands: `/x` flips, `/x on` and `/x off` set."""
    parts = ctx.user_input.strip().split(maxsplit=1)
    if len(parts) > 1:
        new_val = _explicit_toggle(parts[1])
        if new_val is None:
            print_error(f"Использование: {parts[0]} [on|off] — без аргумента переключает.")
            return CommandResult.continue_loop()
    else:
        new_val = not get()
    set_(new_val)
    state = "[bold green]ВКЛ[/bold green]" if new_val else "[bold red]ВЫКЛ[/bold red]"
    print_system(f"{label}: {state}")
    if not new_val and note_off:
        print_system(f"[dim]{note_off}[/dim]")
    return CommandResult.continue_loop()


@registry.register("/verbose", help_text="Переключить индикаторы статуса (спиннеры)", category="config", is_prefix=True)
def handle_verbose(ctx: CommandContext) -> CommandResult:
    from config import get_verbose_status, set_verbose_status
    return _switch(ctx, get_verbose_status, set_verbose_status, "Индикаторы статуса")


@registry.register("/results", help_text="Полный вывод инструментов вместо сводки", category="config", is_prefix=True)
def handle_results(ctx: CommandContext) -> CommandResult:
    from config import get_show_tool_results, set_show_tool_results
    return _switch(ctx, get_show_tool_results, set_show_tool_results,
                   "Полный вывод результатов инструментов",
                   note_off="Остаётся строка-сводка; ошибки показываются целиком всегда.")


@registry.register("/debug", help_text="Подробные логи инструментов (режим отладки)", category="config", is_prefix=True)
def handle_debug(ctx: CommandContext) -> CommandResult:
    from config import get_debug_mode, set_debug_mode
    return _switch(ctx, get_debug_mode, set_debug_mode, "Подробные логи в чате (режим отладки)")


MAX_TOKENS_DEFAULT = 8192
MAX_TOKENS_FLOOR = 256
MAX_TOKENS_CEILING = 1024 * 1024


def _parse_token_count(raw: str):
    """'16384', '16k', '1m' or '16_384' as an int; None when it is not a count."""
    text = raw.strip().lower().replace("_", "").replace(",", "")
    factor = 1
    if text.endswith("k"):
        factor, text = 1024, text[:-1]
    elif text.endswith("m"):
        factor, text = 1024 * 1024, text[:-1]
    return int(text) * factor if text.isdigit() else None


@registry.register("/max_tokens", aliases=["/maxtokens"],
                   help_text="Лимит токенов на один ответ (например 16384, 32768)",
                   category="config", is_prefix=True)
def handle_max_tokens(ctx: CommandContext) -> CommandResult:
    from config import get_max_generation_tokens, set_max_generation_tokens
    parts = ctx.user_input.strip().split(maxsplit=1)
    current = get_max_generation_tokens()
    if len(parts) < 2:
        print_system(
            f"Лимит токенов на один ответ: [bold cyan]{current}[/bold cyan]. "
            f"Изменить: /max_tokens <число> (например 16384 или 32k); "
            f"/max_tokens default — вернуть {MAX_TOKENS_DEFAULT}.")
        return CommandResult.continue_loop()

    word = parts[1].strip().lower()
    value = MAX_TOKENS_DEFAULT if word in ("default", "reset", "def") else _parse_token_count(word)
    if value is None:
        print_error("Нужно целое число токенов (например 16384 или 32k) или 'default'.")
        return CommandResult.continue_loop()
    if not MAX_TOKENS_FLOOR <= value <= MAX_TOKENS_CEILING:
        print_error(f"Лимит должен быть от {MAX_TOKENS_FLOOR} до {MAX_TOKENS_CEILING} токенов.")
        return CommandResult.continue_loop()

    set_max_generation_tokens(value)
    print_system(f"Лимит токенов на один ответ: [bold cyan]{value}[/bold cyan]")
    window = getattr(ctx.agent, "max_context_tokens", 0) or 0
    if window and value >= window:
        print_system(f"[yellow]⚠ Это не меньше окна контекста модели ({window}): "
                     f"ответ не поместится рядом с историей.[/yellow]")
    return CommandResult.continue_loop()


@registry.register("/sampling", aliases=["/params", "/sampler"], help_text="Тонкая настройка сэмплеров (top_p, min_p, top_k, temp, repeat_penalty)", category="config", is_prefix=True)
def handle_sampling(ctx: CommandContext) -> CommandResult:
    from rich.table import Table
    from config import (
        get_temperature, set_temperature,
        get_top_p, set_top_p,
        get_top_k, set_top_k,
        get_min_p, set_min_p,
        get_repeat_penalty, set_repeat_penalty,
        reset_sampling_params,
    )

    args = ctx.user_input.strip().split()
    # /sampling preset <name>
    if len(args) > 1 and args[1].lower() in ("preset", "presets", "profile"):
        from config import SAMPLING_PRESETS, apply_sampling_preset
        if len(args) > 2:
            preset_name = args[2]
            try:
                p = apply_sampling_preset(preset_name)
                print_system(
                    f"Применён пресет [bold cyan]{preset_name}[/bold cyan]: "
                    f"temp={p['temperature']}, min_p={p['min_p']}, top_p={p['top_p']}, "
                    f"top_k={p['top_k']}, repeat_penalty={p['repeat_penalty']}"
                )
            except ValueError as e:
                print_error(str(e))
            return CommandResult.continue_loop()
        else:
            names = ", ".join(f"[bold cyan]{k}[/bold cyan]" for k in SAMPLING_PRESETS.keys())
            print_system(f"Использование: /sampling preset <name>\nДоступные пресеты: {names}")
            return CommandResult.continue_loop()

    # /sampling reset
    if len(args) > 1 and args[1].lower() in ("reset", "default", "defaults"):
        reset_sampling_params()
        print_system("Все параметры сэмплирования сброшены к дефолтам модели (temperature=0.3).")
        return CommandResult.continue_loop()

    # /sampling set <param> <val> or /sampling <param> <val>
    # e.g.: /sampling min_p 0.05
    if len(args) >= 3:
        target_param = args[1].lower()
        if target_param == "set":
            target_param = args[2].lower()
            val_str = args[3] if len(args) > 3 else ""
        else:
            val_str = args[2]

        if target_param in ("temperature", "temp", "t"):
            if val_str.lower() in ("default", "reset", "def"):
                set_temperature(0.3)
                print_system("Параметр [bold]temperature[/bold] сброшен к дефолту: [bold cyan]0.3[/bold cyan]")
                return CommandResult.continue_loop()
            try:
                v = float(val_str)
                if 0.0 <= v <= 2.0:
                    set_temperature(v)
                    print_system(f"Параметр [bold]temperature[/bold] установлен: [bold cyan]{v}[/bold cyan]")
                else:
                    print_error("temperature должна быть от 0.0 до 2.0")
            except ValueError:
                print_error("Неверное число для temperature (или используйте 'default')")
            return CommandResult.continue_loop()

        elif target_param in ("top_p", "topp", "p"):
            if val_str.lower() in ("none", "null", "off", "reset"):
                set_top_p(None)
                print_system("Параметр [bold]top_p[/bold] сброшен к дефолту модели.")
            else:
                try:
                    v = float(val_str)
                    if 0.0 < v <= 1.0:
                        set_top_p(v)
                        print_system(f"Параметр [bold]top_p[/bold] установлен: [bold cyan]{v}[/bold cyan]")
                    else:
                        print_error("top_p должен быть от 0.0 (исключая) до 1.0")
                except ValueError:
                    print_error("Неверное число для top_p")
            return CommandResult.continue_loop()

        elif target_param in ("min_p", "minp"):
            if val_str.lower() in ("none", "null", "off", "reset"):
                set_min_p(None)
                print_system("Параметр [bold]min_p[/bold] сброшен к дефолту модели.")
            else:
                try:
                    v = float(val_str)
                    if 0.0 <= v <= 1.0:
                        set_min_p(v)
                        print_system(f"Параметр [bold]min_p[/bold] установлен: [bold cyan]{v}[/bold cyan]")
                    else:
                        print_error("min_p должен быть от 0.0 до 1.0 (рекомендуется 0.05)")
                except ValueError:
                    print_error("Неверное число для min_p")
            return CommandResult.continue_loop()

        elif target_param in ("top_k", "topk", "k"):
            if val_str.lower() in ("none", "null", "off", "reset"):
                set_top_k(None)
                print_system("Параметр [bold]top_k[/bold] сброшен к дефолту модели.")
            else:
                try:
                    v = int(val_str)
                    if v >= 1:
                        set_top_k(v)
                        print_system(f"Параметр [bold]top_k[/bold] установлен: [bold cyan]{v}[/bold cyan]")
                    else:
                        print_error("top_k должен быть целым числом >= 1")
                except ValueError:
                    print_error("Неверное целое число для top_k")
            return CommandResult.continue_loop()

        elif target_param in ("repeat_penalty", "repetition_penalty", "rep_pen", "penalty"):
            if val_str.lower() in ("none", "null", "off", "reset"):
                set_repeat_penalty(None)
                print_system("Параметр [bold]repeat_penalty[/bold] сброшен к дефолту модели.")
            else:
                try:
                    v = float(val_str)
                    if 0.5 <= v <= 2.0:
                        set_repeat_penalty(v)
                        print_system(f"Параметр [bold]repeat_penalty[/bold] установлен: [bold cyan]{v}[/bold cyan]")
                    else:
                        print_error("repeat_penalty обычно находится в диапазоне 1.0–1.3")
                except ValueError:
                    print_error("Неверное число для repeat_penalty")
            return CommandResult.continue_loop()

    # Display current parameters
    temp = get_temperature()
    top_p = get_top_p()
    min_p = get_min_p()
    top_k = get_top_k()
    rep_pen = get_repeat_penalty()

    table = Table(title="Параметры сэмплирования LLM (Sampling Parameters)", border_style="dim #30363d", header_style="bold #79c0ff")
    table.add_column("Параметр", style="bold #58a6ff", min_width=18)
    table.add_column("Текущее значение", style="bold white", min_width=16)
    table.add_column("Описание и рекомендации", style="dim")

    table.add_row(
        "temperature (/temp)",
        f"[bold cyan]{temp}[/bold cyan]",
        "Степень случайности (0.0-2.0). 0.2 для кода, 0.7 для креатива."
    )
    table.add_row(
        "top_p (Nucleus)",
        f"[cyan]{top_p}[/cyan]" if top_p is not None else "[dim]модель по умолчанию[/dim]",
        "Кумулятивная вероятность (0.0-1.0). Отсекает маловероятный хвост."
    )
    table.add_row(
        "min_p (Dynamic)",
        f"[bold green]{min_p}[/bold green]" if min_p is not None else "[dim]модель по умолчанию[/dim]",
        "Минимальный порог относительно топового токена (рек. 0.05 для кода)."
    )
    table.add_row(
        "top_k",
        f"[cyan]{top_k}[/cyan]" if top_k is not None else "[dim]модель по умолчанию[/dim]",
        "Ограничение числа кандидатов токенов (например, 40 или 50)."
    )
    table.add_row(
        "repeat_penalty",
        f"[cyan]{rep_pen}[/cyan]" if rep_pen is not None else "[dim]модель по умолчанию[/dim]",
        "Штраф за повторы токенов (например, 1.05–1.15)."
    )
    console.print(table)

    # Interactive menu
    action = questionary.select(
        "Настройка параметров сэмплирования:",
        choices=[
            "Применить пресет (Code / Precise / Creative / Default)",
            f"Изменить temperature (сейчас {temp})",
            f"Изменить top_p (сейчас {top_p if top_p is not None else 'default'})",
            f"Изменить min_p (сейчас {min_p if min_p is not None else 'default'})",
            f"Изменить top_k (сейчас {top_k if top_k is not None else 'default'})",
            f"Изменить repeat_penalty (сейчас {rep_pen if rep_pen is not None else 'default'})",
            "Сбросить всё к дефолтам модели (Reset all)",
            "Готово (выйти)",
        ]
    ).ask()

    if not action or action.startswith("Готово"):
        return CommandResult.continue_loop()

    if "пресет" in action.lower():
        from config import apply_sampling_preset
        preset_choice = questionary.select(
            "Выберите готовый пресет сэмплирования:",
            choices=[
                "code: temp=0.2, min_p=0.05, repeat_penalty=1.05 (Идеально для кода и скриптов)",
                "precise: temp=0.0 (Детерминированный, жадный argmax для строгих фактов)",
                "creative: temp=0.8, top_p=0.95, top_k=50, repeat_penalty=1.1 (Для историй и брейншторма)",
                "default: temp=0.3, без дополнительных сэмплеров (Базовый дефолт Argent)",
            ]
        ).ask()
        if preset_choice:
            p_name = preset_choice.split(":")[0].strip()
            apply_sampling_preset(p_name)
            print_system(f"Успешно применён пресет [bold cyan]{p_name}[/bold cyan]!")
        return CommandResult.continue_loop()

    if action.startswith("Сбросить"):
        reset_sampling_params()
        print_system("Все сэмплеры сброшены к дефолтам модели.")
        return CommandResult.continue_loop()

    if "temperature" in action:
        val = questionary.text("Введите temperature (0.0-2.0):", default=str(temp)).ask()
        if val:
            try:
                v = float(val.strip())
                if 0.0 <= v <= 2.0:
                    set_temperature(v)
                    print_system(f"temperature: [bold cyan]{v}[/bold cyan]")
            except ValueError:
                print_error("Неверное число.")
    elif "top_p" in action:
        val = questionary.text("Введите top_p (0.0-1.0 или пусто для сброса):", default=str(top_p) if top_p is not None else "").ask()
        if val is not None:
            if not val.strip():
                set_top_p(None)
                print_system("top_p: сброшен к дефолту модели.")
            else:
                try:
                    v = float(val.strip())
                    if 0.0 < v <= 1.0:
                        set_top_p(v)
                        print_system(f"top_p: [bold cyan]{v}[/bold cyan]")
                except ValueError:
                    print_error("Неверное число.")
    elif "min_p" in action:
        val = questionary.text("Введите min_p (0.0-1.0, например 0.05, или пусто для сброса):", default=str(min_p) if min_p is not None else "").ask()
        if val is not None:
            if not val.strip():
                set_min_p(None)
                print_system("min_p: сброшен к дефолту модели.")
            else:
                try:
                    v = float(val.strip())
                    if 0.0 <= v <= 1.0:
                        set_min_p(v)
                        print_system(f"min_p: [bold cyan]{v}[/bold cyan]")
                except ValueError:
                    print_error("Неверное число.")
    elif "top_k" in action:
        val = questionary.text("Введите top_k (целое число, например 40, или пусто для сброса):", default=str(top_k) if top_k is not None else "").ask()
        if val is not None:
            if not val.strip():
                set_top_k(None)
                print_system("top_k: сброшен к дефолту модели.")
            else:
                try:
                    v = int(val.strip())
                    if v >= 1:
                        set_top_k(v)
                        print_system(f"top_k: [bold cyan]{v}[/bold cyan]")
                except ValueError:
                    print_error("Неверное целое число.")
    elif "repeat_penalty" in action:
        val = questionary.text("Введите repeat_penalty (например 1.1, или пусто для сброса):", default=str(rep_pen) if rep_pen is not None else "").ask()
        if val is not None:
            if not val.strip():
                set_repeat_penalty(None)
                print_system("repeat_penalty: сброшен к дефолту модели.")
            else:
                try:
                    v = float(val.strip())
                    if 0.5 <= v <= 2.0:
                        set_repeat_penalty(v)
                        print_system(f"repeat_penalty: [bold cyan]{v}[/bold cyan]")
                except ValueError:
                    print_error("Неверное число.")

    return CommandResult.continue_loop()


@registry.register("/snapshot", help_text="Режим снимков хода (compact, verbose, off)", category="config", is_prefix=True)
def handle_snapshot(ctx: CommandContext) -> CommandResult:
    from config import get_turn_snapshot_mode, set_turn_snapshot_mode
    parts = ctx.user_input.strip().split(maxsplit=1)
    if len(parts) > 1:
        arg = parts[1].strip().lower()
        if arg in ("compact", "verbose", "off"):
            set_turn_snapshot_mode(arg)
            desc = {
                "compact": "компактная строка: время, токены, кеш, инструменты, контекст, стоимость",
                "verbose": "подробная многострочная разбивка по этапам и анатомии контекста",
                "off": "минималистичный тайминг без расширенных метрик",
            }.get(arg, "")
            print_system(f"✓ Режим снимков хода изменён на: [bold cyan]{arg}[/bold cyan] ({desc})")
        else:
            print_error(f"Неизвестный режим: '{arg}'. Доступно: compact, verbose, off")
    else:
        current = get_turn_snapshot_mode()
        desc = {
            "compact": "компактная строка: время, токены, кеш, инструменты, контекст, стоимость",
            "verbose": "подробная многострочная разбивка по этапам и анатомии контекста",
            "off": "минималистичный тайминг без расширенных метрик",
        }.get(current, "")
        print_system(f"Текущий режим снимков хода: [bold cyan]{current}[/bold cyan] ({desc})\n"
                     f"Использование: /snapshot [compact | verbose | off]")
    return CommandResult.continue_loop()


@registry.register(
    "/local_model",
    aliases=["/delegate", "/agent_tier"],
    help_text="Управление агентами делегирования (on/off, tier, cloud, fallback, path, timeout)",
    category="config",
    is_prefix=True,
)
def handle_local_model(ctx: CommandContext) -> CommandResult:
    from config import (
        is_local_agent_enabled, set_local_agent_enabled,
        get_local_model_tier, set_local_model_tier,
        get_local_model_path, set_local_model_path,
        get_local_agent_cloud_provider, set_local_agent_cloud_provider,
        get_local_agent_cloud_model, set_local_agent_cloud_model,
        is_local_agent_cloud_fallback, set_local_agent_cloud_fallback,
        get_local_agent_timeout, set_local_agent_timeout,
    )
    from local_inference import SystemResourceAuditor

    parts = ctx.user_input.strip().split()
    if len(parts) > 1:
        sub = parts[1].lower()

        if sub in ("on", "enable", "enabled", "1", "true"):
            set_local_agent_enabled(True)
            tier = get_local_model_tier()
            t_name = {0: "T0 (Cloud)", 1: "T1 (≤0.5B)", 2: "T2 (≤1.5B)", 3: "T3 (≤3B)"}.get(tier, f"T{tier}")
            print_system(f"✓ Агенты делегирования: [bold green]ВКЛЮЧЕНЫ[/bold green]\n"
                         f"  Текущий тир: [bold cyan]{t_name}[/bold cyan]\n"
                         f"  Инструмент delegate_local доступен для вызова моделью.")
            return CommandResult.continue_loop()

        elif sub in ("off", "disable", "disabled", "0", "false"):
            set_local_agent_enabled(False)
            print_system("✓ Агенты делегирования: [bold red]ВЫКЛЮЧЕНЫ[/bold red]\n"
                         "  Инструмент delegate_local скрыт от модели (0 токенов оверхеда).")
            return CommandResult.continue_loop()

        elif sub in ("tier", "t"):
            if len(parts) > 2:
                t_arg = parts[2].lower()
                if t_arg in ("cloud", "0", "t0", "c"):
                    set_local_model_tier(0)
                    print_system("✓ Тир делегирования установлен: [bold cyan]T0 (Cloud)[/bold cyan] (0 MB RAM/VRAM).")
                elif t_arg in ("1", "t1"):
                    set_local_model_tier(1)
                    print_system("✓ Тир делегирования установлен: [bold cyan]T1 (≤0.5B, ~350MB RAM)[/bold cyan].")
                elif t_arg in ("2", "t2", "1-2"):
                    set_local_model_tier(2)
                    print_system("✓ Тир делегирования установлен: [bold cyan]T2 (≤1.5B, ~1GB RAM)[/bold cyan].")
                elif t_arg in ("3", "t3", "1-3"):
                    set_local_model_tier(3)
                    print_system("✓ Тир делегирования установлен: [bold cyan]T3 (≤3B, ~2GB RAM)[/bold cyan].")
                else:
                    print_error("Неверный тир. Допустимо: /local_model tier [cloud | 1 | 2 | 3]")
            else:
                cur_t = get_local_model_tier()
                t_name = {0: "T0 (Cloud)", 1: "T1 (≤0.5B)", 2: "T2 (≤1.5B)", 3: "T3 (≤3B)"}.get(cur_t, f"T{cur_t}")
                print_system(f"Текущий тир: [bold cyan]{t_name}[/bold cyan]. Допустимо: cloud, 1, 2, 3")
            return CommandResult.continue_loop()

        elif sub in ("cloud", "c"):
            if len(parts) >= 4:
                prov = parts[2].lower()
                m_name = parts[3]
                set_local_agent_cloud_provider(prov)
                set_local_agent_cloud_model(m_name)
                print_system(f"✓ Cloud-модель установлена: [bold cyan]{prov} / {m_name}[/bold cyan]")
            elif len(parts) == 3:
                m_name = parts[2]
                set_local_agent_cloud_model(m_name)
                print_system(f"✓ Cloud-модель установлена: [bold cyan]{m_name}[/bold cyan] ({get_local_agent_cloud_provider()})")
            else:
                prov = get_local_agent_cloud_provider()
                m = get_local_agent_cloud_model()
                print_system(f"Текущая Cloud-модель для T0: [bold cyan]{prov} / {m}[/bold cyan]\n"
                             f"Использование: /local_model cloud <provider> <model>")
            return CommandResult.continue_loop()

        elif sub in ("fallback", "fb"):
            if len(parts) > 2:
                arg = parts[2].lower()
                if arg in ("on", "1", "true", "enable"):
                    set_local_agent_cloud_fallback(True)
                    print_system("✓ Fallback на Cloud: [bold green]ВКЛ[/bold green] (при нехватке RAM задачи уйдут в T0).")
                elif arg in ("off", "0", "false", "disable"):
                    set_local_agent_cloud_fallback(False)
                    print_system("✓ Fallback на Cloud: [bold red]ВЫКЛ[/bold red].")
                else:
                    print_error("Использование: /local_model fallback [on | off]")
            else:
                st = "[bold green]ВКЛ[/bold green]" if is_local_agent_cloud_fallback() else "[bold red]ВЫКЛ[/bold red]"
                print_system(f"Статус Cloud Fallback: {st}")
            return CommandResult.continue_loop()

        elif sub in ("path", "p"):
            if len(parts) > 2:
                p_arg = " ".join(parts[2:])
                if p_arg.lower() in ("reset", "none", "off", "clear"):
                    set_local_model_path(None)
                    print_system("✓ Пользовательский путь к модели сброшен на значения по умолчанию.")
                else:
                    set_local_model_path(p_arg)
                    print_system(f"✓ Путь к пользовательской модели: [bold cyan]{p_arg}[/bold cyan]")
            else:
                cur_p = get_local_model_path()
                print_system(f"Пользовательский путь: [bold cyan]{cur_p or 'не задан'}[/bold cyan]")
            return CommandResult.continue_loop()

        elif sub in ("timeout", "t_out"):
            if len(parts) > 2:
                try:
                    sec = float(parts[2])
                    set_local_agent_timeout(sec)
                    print_system(f"✓ Таймаут делегирования: [bold cyan]{sec:.1f}с[/bold cyan]")
                except ValueError:
                    print_error("Таймаут должен быть числом секунд (например: /local_model timeout 30)")
            else:
                cur_to = get_local_agent_timeout()
                print_system(f"Текущий таймаут: [bold cyan]{cur_to:.1f}с[/bold cyan]")
            return CommandResult.continue_loop()

    # Display full status
    is_enabled = is_local_agent_enabled()
    tier = get_local_model_tier()
    t_name = {0: "T0 (Cloud)", 1: "T1 (≤0.5B)", 2: "T2 (≤1.5B)", 3: "T3 (≤3B)"}.get(tier, f"T{tier}")
    cloud_p = get_local_agent_cloud_provider()
    cloud_m = get_local_agent_cloud_model()
    fb = "ВКЛ" if is_local_agent_cloud_fallback() else "ВЫКЛ"
    custom_p = get_local_model_path() or "авто (Ollama / GGUF)"
    timeout = get_local_agent_timeout()

    ram_free = SystemResourceAuditor.get_free_ram_gb()
    ram_total = SystemResourceAuditor.get_total_ram_gb()
    gpu = SystemResourceAuditor.get_gpu_vram_info()
    gpu_str = f"{gpu['free_mb']}MB / {gpu['total_mb']}MB" if gpu["available"] else "не обнаружен"

    st_badge = "[bold green]ВКЛЮЧЕНЫ[/bold green]" if is_enabled else "[bold red]ВЫКЛЮЧЕНЫ[/bold red]"

    print_system(
        f"Агенты делегирования (Argent Brain): {st_badge}\n"
        f"  │ Тир: [bold cyan]{t_name}[/bold cyan]\n"
        f"  │ Cloud-модель (T0): [bold cyan]{cloud_p} / {cloud_m}[/bold cyan]\n"
        f"  │ Cloud Fallback: [bold cyan]{fb}[/bold cyan]\n"
        f"  │ Локальная модель: [bold cyan]{custom_p}[/bold cyan]\n"
        f"  │ Таймаут: [bold cyan]{timeout:.0f}с[/bold cyan]\n"
        f"  │ Ресурсы системы: RAM {ram_free:.1f}GB / {ram_total:.1f}GB · GPU VRAM {gpu_str}\n\n"
        f"Команды: /local_model [on | off | tier <cloud|1|2|3> | cloud <prov> <mod> | fallback on|off | timeout <sec>]"
    )
    return CommandResult.continue_loop()


@registry.register("/needle", help_text="Сопроцессор Needle 3 (быстрый локальный dispatch)", category="config", is_prefix=True)
def handle_needle(ctx: CommandContext) -> CommandResult:
    from config import (
        get_needle_enabled, set_needle_enabled,
        get_needle_fast_path, set_needle_fast_path,
    )
    parts = ctx.user_input.strip().split(maxsplit=2)
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "on":
        set_needle_enabled(True)
        print_system("⚡ Needle 3 coprocessor: ВКЛ")
        # Auto-initialize on enable to report load time
        try:
            from src.agent.needle_coprocessor import coprocessor
            import time
            t0 = time.perf_counter()
            ok = coprocessor.initialize()
            elapsed = (time.perf_counter() - t0) * 1000
            if ok:
                print_system(f"  Модель загружена за {elapsed:.0f} ms")
            else:
                print_system(f"  ⚠ Не удалось инициализировать: {coprocessor._init_error}")
        except Exception as e:
            print_system(f"  ⚠ Ошибка инициализации: {e}")
    elif sub == "off":
        set_needle_enabled(False)
        set_needle_fast_path(False)
        print_system("Needle 3 coprocessor: ВЫКЛ")
    elif sub == "fast":
        fast_arg = parts[2].lower() if len(parts) > 2 else ""
        if fast_arg == "on":
            if not get_needle_enabled():
                set_needle_enabled(True)
                print_system("⚡ Needle 3 coprocessor: ВКЛ (авто)")
            set_needle_fast_path(True)
            print_system("⚡ Fast-Path режим: ВКЛ (перехват read_file, git_status, git_diff, grep_search)")
        elif fast_arg == "off":
            set_needle_fast_path(False)
            print_system("Fast-Path режим: ВЫКЛ")
        else:
            cur = get_needle_fast_path()
            print_system(f"Fast-Path режим: {'ВКЛ' if cur else 'ВЫКЛ'}")
            print_system("  /needle fast on  — включить мгновенный перехват рутинных инструментов")
            print_system("  /needle fast off — выключить")
    elif sub == "test":
        test_query = parts[2].strip() if len(parts) > 2 else ""
        if not test_query:
            print_error("Укажите запрос: /needle test \"прочитай config.py\"")
            return CommandResult.continue_loop()
        try:
            from src.agent.needle_coprocessor import coprocessor
            if not coprocessor.is_available:
                print_error("Пакет cactus-needle не установлен.")
                return CommandResult.continue_loop()
            coprocessor.initialize()
            result = coprocessor.try_fast_dispatch(test_query)
            if result:
                print_system(
                    f"[⚡ Needle 3 Fast-Path: {result.elapsed_ms:.1f} ms]\n"
                    f"  Tool: {result.tool_name}\n"
                    f"  Args: {result.arguments}\n"
                    f"  Confidence: {result.confidence:.2f}"
                )
            else:
                print_system("  → Запрос не распознан как рутинный инструмент (будет передан в основную LLM)")
        except Exception as e:
            print_error(f"Ошибка тестового запуска: {e}")
    elif sub == "status":
        enabled = get_needle_enabled()
        fast = get_needle_fast_path()
        print_system(f"Needle 3 Coprocessor")
        print_system(f"  Включён:    {'да' if enabled else 'нет'}")
        print_system(f"  Fast-Path:  {'да' if fast else 'нет'}")
        try:
            from src.agent.needle_coprocessor import NEEDLE_AVAILABLE, coprocessor
            print_system(f"  Пакет:      {'установлен' if NEEDLE_AVAILABLE else 'НЕ УСТАНОВЛЕН'}")
            print_system(f"  Инициализирован: {'да' if coprocessor._initialized else 'нет'}")
            if coprocessor._init_error:
                print_system(f"  Ошибка:     {coprocessor._init_error}")
        except ImportError:
            print_system("  Пакет:      НЕ УСТАНОВЛЕН")
    else:
        print_system("⚡ Needle 3 — ультрабыстрый CPU-сопроцессор для вызова инструментов")
        print_system("  /needle on           — включить сопроцессор")
        print_system("  /needle off          — выключить")
        print_system("  /needle fast on|off  — режим мгновенного перехвата рутинных команд")
        print_system("  /needle test <запрос> — тестовый запуск с замером мс")
        print_system("  /needle status       — состояние")

    return CommandResult.continue_loop()


@registry.register("/micro", help_text="Собственные микромодели и активное обучение (Fast-Path)", category="config", is_prefix=True)
def handle_micro(ctx: CommandContext) -> CommandResult:
    from config import (
        get_fast_dispatch_backend, set_fast_dispatch_backend,
        get_micromodel_enabled, set_micromodel_enabled,
    )
    parts = ctx.user_input.strip().split(maxsplit=2)
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "on":
        set_fast_dispatch_backend("micromodel")
        print_system("⚡ Fast-Path бэкенд переключён на: MicroModel (ВКЛ)")
    elif sub == "off":
        set_fast_dispatch_backend("off")
        print_system("Fast-Path перехват: ВЫКЛ")
    elif sub == "backend":
        b_arg = parts[2].lower() if len(parts) > 2 else ""
        if b_arg in ("leeth", "micromodel", "needle", "off"):
            set_fast_dispatch_backend(b_arg)
            print_system(f"⚡ Fast-Path бэкенд установлен: {b_arg}")
        else:
            cur = get_fast_dispatch_backend()
            print_system(f"Текущий бэкенд: {cur}")
            print_system("  /micro backend leeth      — Leeth System One (1.5ms, 23.7M, INT8 ONNX)")
            print_system("  /micro backend micromodel — собственная обученная микромодель (1.8ms)")
            print_system("  /micro backend needle     — Cactus Needle 3 coprocessor")
            print_system("  /micro backend off        — отключить быстрый перехват")
    elif sub == "test":
        test_query = parts[2].strip() if len(parts) > 2 else ""
        if not test_query:
            print_error("Укажите запрос: /micro test \"прочитай config.py\"")
            return CommandResult.continue_loop()
        try:
            from src.micromodels.dispatcher import dispatcher
            from src.micromodels.runtime import runtime

            intent, conf, all_p, ms = runtime.predict_intent(test_query)
            disp_res = dispatcher.try_fast_dispatch(test_query)

            print_system(f"Результат анализа микромодели ({ms:.2f} ms):")
            print_system(f"  Намерение:  {intent} (уверенность: {conf*100:.1f}%)")

            # Top 3 candidates
            top3 = sorted(all_p.items(), key=lambda x: -x[1])[:3]
            candidates_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in top3)
            print_system(f"  Топ классы: {candidates_str}")

            if disp_res:
                print_system(
                    f"  Статус:     ⚡ FAST-DISPATCH\n"
                    f"  Инструмент: {disp_res.tool_name}\n"
                    f"  Аргументы:  {disp_res.arguments}\n"
                    f"  Время:      {disp_res.elapsed_ms:.2f} ms"
                )
            else:
                print_system("  Статус:     Будет передан в основную LLM (reasoning / confidence < 85% / NO_TOOL)")
        except Exception as e:
            print_error(f"Ошибка проверки микромодели: {e}")
    elif sub == "train":
        print_system("🔄 Запуск слияния данных активного обучения и дообучения микромодели...")
        try:
            from src.micromodels.data_collector import collector
            ok, msg = collector.merge_and_retrain()
            if ok:
                print_system(f"✓ {msg}")
            else:
                print_system(f"ℹ {msg}")
        except Exception as e:
            print_error(f"Ошибка обучения: {e}")
    elif sub == "queue":
        try:
            from src.micromodels.data_collector import collector
            q_file = collector.active_queue_file
            if not q_file.exists():
                print_system("Очередь активного обучения пуста.")
            else:
                import json
                lines = []
                with open(q_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            lines.append(json.loads(line))
                print_system(f"Неуверенные запросы в очереди ({len(lines)} шт.):")
                for r in lines[-5:]:
                    print_system(f"  [{r.get('timestamp')}] \"{r.get('query')}\" -> {r.get('predicted_intent')} ({r.get('confidence')*100:.1f}%)")
        except Exception as e:
            print_error(f"Ошибка чтения очереди: {e}")
    elif sub == "status":
        from src.micromodels.runtime import runtime
        from src.micromodels.data_collector import collector
        runtime.initialize()
        backend_name = get_fast_dispatch_backend()
        stats = collector.get_stats()

        print_system("Собственные Микромодели Argent (TinyIntentEncoder + Specialists)")
        print_system(f"  Fast-Path бэкенд:   {backend_name} ({'АКТИВЕН' if backend_name == 'micromodel' else 'пассивен'})")
        print_system(f"  Inference движок:   {runtime.backend}")
        print_system(f"  Параметры модели:   ~1.81M (INT8 квантование)")
        print_system(f"  Средняя задержка:   ~1.6-1.8 ms (CPU)")
        print_system(f"  Порог уверенности:  {runtime.confidence_threshold*100:.0f}%")
        print_system(f"  Активное обучение:  {stats['uncertain_queries_queued']} неуверенных запросов")
        print_system(f"  Собрано живых пар:  {stats['live_samples_recorded']} примеров")
    else:
        print_system("⚡ MicroModels — собственные узкоспециализированные ИИ модели Argent")
        print_system("  /micro status             — состояние модели, движка и метрик")
        print_system("  /micro test <запрос>      — анализ запроса (интент, аргументы, латентность)")
        print_system("  /micro backend [name]     — выбор бэкенда (micromodel | needle | off)")
        print_system("  /micro train              — дообучить модель на новых данных")
        print_system("  /micro queue              — просмотр запросов в очереди активного обучения")
        print_system("  /micro on | off           — включить/выключить перехват микромоделями")

    return CommandResult.continue_loop()


@registry.register("/help", aliases=["/?"], help_text="Справка по доступным командам", category="config")
def handle_help(ctx: CommandContext) -> CommandResult:
    from rich.table import Table
    table = Table(title="Команды Argent CLI (Claude Code style)", border_style="dim #30363d", header_style="bold #79c0ff")
    table.add_column("Команда", style="bold #58a6ff", min_width=18)
    table.add_column("Категория", style="dim", min_width=10)
    table.add_column("Описание", style="white")

    entries = sorted(registry._entries, key=lambda e: (e.category, e.name))
    seen = set()
    for e in entries:
        if e.name in seen:
            continue
        seen.add(e.name)
        aliases = f" [dim]({', '.join(e.aliases)})[/dim]" if e.aliases else ""
        table.add_row(f"{e.name}{aliases}", e.category, e.help_text)

    console.print(table)
    return CommandResult.continue_loop()
