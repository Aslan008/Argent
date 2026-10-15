"""Session-related command handlers for Argent CLI (/save, /sessions, /load, /resume, /clear, /rewind, /export)."""

import os
import questionary
from rich.markup import escape
from rich.table import Table

from config import get_provider
from file_tracker import clear_session_snapshots
from memory_manager import memory
from session import (
    save_session, load_session, list_sessions, delete_session, get_last_session,
    find_session, format_relative_time, format_short_path,
)
from ui import console, print_system, print_error
from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry


def restore_session(agent, meta: dict) -> bool:
    """Put a saved session back, telling the user what does NOT match."""
    data = load_session(meta["id"])
    if not data or not data.get("messages"):
        print_error("Не удалось прочитать сессию.")
        return False

    agent.messages = data["messages"]
    agent.session_id = data.get("id") or meta["id"]
    from src.agent.compaction_log import reset_compaction_history
    reset_compaction_history()

    # Loaded in the middle of another conversation: the screen is replaced, not appended to. Done before
    # the notes below, which would otherwise be wiped with the old transcript.
    try:
        from src.agent import history_replay
        history_replay.show(agent.messages, replace=True, saved_at=(data.get("saved_at") or "")[:16])
    except Exception:
        pass

    # Reset transient agent caches from previous session
    if hasattr(agent, "_read_cache"):
        agent._read_cache.clear()
    if hasattr(agent, "loop_guard"):
        agent.loop_guard.reset()
    if hasattr(agent, "_checkpoint_turn_indices"):
        agent._checkpoint_turn_indices.clear()
    if hasattr(agent, "reset_turn_records"):
        agent.reset_turn_records()
    from src.agent import todos
    # the checklist of the conversation just left goes; the one saved WITH this session comes back
    todos.restore((data.get("state") or {}).get("todos"))
    if hasattr(agent, "_turn_checkpoint_done"):
        agent._turn_checkpoint_done = False

    # Repair any orphaned tool calls from interrupted session
    if hasattr(agent, "repair_history"):
        try:
            if agent.repair_history():
                print_system("[yellow]Прошлый ход был прерван (сбой, обрыв связи или закрытие окна).[/yellow] "
                             "Напишите «продолжай» — работа пойдёт с последнего сохранённого шага.")
        except Exception as e:
            from logger import get_logger
            get_logger("session").warning("repair_history during restore failed: %s", e)

    saved_time = (data.get("saved_at") or "")[:16]
    print_system(f"Восстановлено {len(data['messages'])} сообщений ({saved_time}).")

    saved_model = data.get("model") or ""
    current_model = getattr(agent, "model_name", "")
    if saved_model and saved_model != current_model:
        print_system(f"[yellow]Сессия велась на '{saved_model}', сейчас активна "
                     f"'{current_model}'.[/yellow] История содержит вызовы "
                     f"инструментов, которые текущая модель может не повторить — "
                     f"смените модель через /model, если это важно.")

    saved_cwd = data.get("cwd") or ""
    current = os.getcwd()
    if saved_cwd and os.path.normcase(os.path.abspath(saved_cwd)) != os.path.normcase(current):
        print_system(f"[yellow]Сессия сохранена в:[/yellow] {saved_cwd}\n"
                     f"[yellow]   Вы сейчас в:       [/yellow] {current}")
        if not os.path.isdir(saved_cwd):
            print_system("[dim]Директории сессии больше нет — пути из истории "
                         "не совпадут с текущим проектом.[/dim]")
            return True
        try:
            go = questionary.confirm("Перейти в директорию сессии?", default=False).ask()
        except Exception:
            go = False
        if go:
            os.chdir(saved_cwd)
            try:
                memory._mgr = None
            except Exception:
                pass
            print_system(f"Рабочая директория: {os.getcwd()}")
        else:
            print_system("[dim]Остаёмся здесь. Учтите: пути в истории — от другого "
                         "проекта, как и рабочая память в .argent/memory.json.[/dim]")
    return True


@registry.register("/save", help_text="Сохранить текущую сессию диалога", category="session", is_prefix=True)
def handle_save(ctx: CommandContext) -> CommandResult:
    agent = ctx.agent
    user_input = ctx.user_input
    parts = user_input.split(" ", 1)
    name = parts[1].strip() if len(parts) > 1 else None

    from command_handler import export_chat_history
    export_chat_history(agent, filename=name, auto=False)

    try:
        saved = save_session(agent.messages, {
            "model": agent.model_name,
            "provider": get_provider(),
        }, session_id=agent.session_id, label=name)
        if saved:
            agent.session_id = saved
            print_system(f"Сессия сохранена: {saved}" + (f" (метка: {name})" if name else ""))
        else:
            print_system("[dim]Сессия не сохранена — в истории нет ни одного вашего сообщения.[/dim]")
    except Exception as e:
        print_error(f"Не удалось сохранить сессию: {e}")

    return CommandResult.continue_loop()


# Track the most recently displayed session list so /load <N> or /sessions rm <N>
# refers to the exact numbered rows the user saw in the table.
_last_displayed_sessions: list[dict] | None = None


@registry.register("/sessions", aliases=["/session"], help_text="Список и управление сохранёнными сессиями", category="session", is_prefix=True)
def handle_sessions(ctx: CommandContext) -> CommandResult:
    global _last_displayed_sessions
    raw_cmd = ctx.user_input.strip()
    prefix = next((p for p in ("/sessions", "/session") if raw_cmd.startswith(p)), "/sessions")
    query = raw_cmd[len(prefix):].strip()

    if query.startswith("rm ") or query.startswith("delete "):
        ref = query.split(maxsplit=1)[1].strip()
        target = find_session(ref, sessions=_last_displayed_sessions) if _last_displayed_sessions else None
        if not target:
            target = find_session(ref)
        if not target:
            print_error(f"Сессия '{ref}' не найдена.")
            return CommandResult.continue_loop()
        sid = target.get("id")
        preview = (target.get("preview") or "")[:40]
        lbl = f"[{target['label']}] " if target.get("label") else ""
        try:
            confirm = questionary.confirm(
                f"Удалить сессию {lbl}{sid} ({preview}...)?",
                default=True
            ).ask()
        except Exception:
            confirm = False
        if confirm:
            if delete_session(sid):
                print_system(f"Сессия {sid} удалена.")
                if _last_displayed_sessions:
                    _last_displayed_sessions = [s for s in _last_displayed_sessions if s.get("id") != sid]
            else:
                print_error(f"Не удалось удалить сессию {sid}.")
        return CommandResult.continue_loop()

    sessions = list_sessions(query or None)
    is_dir_query = (query or "").strip().lower() in (".", "here", "cwd", "./", ".\\")
    if not sessions:
        if is_dir_query:
            print_system("В текущей директории нет сохранённых сессий.")
        else:
            print_system(f"Ничего не найдено по запросу '{query}'." if query else "Сохранённых сессий нет.")
        return CommandResult.continue_loop()

    if is_dir_query:
        title = f"Сессии в текущей директории ({format_short_path(os.getcwd())})"
    elif query:
        title = f"Сессии по запросу '{query}'"
    else:
        title = "Сохранённые сессии"

    table = Table(
        title=title,
        title_style="bold #58a6ff",
        border_style="dim #30363d",
        header_style="bold #79c0ff",
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("№", style="bold #58a6ff", justify="right", width=4)
    table.add_column("Метка / Превью", style="white", min_width=24, max_width=45)
    table.add_column("Директория (Проект)", style="dim", min_width=18, max_width=32)
    table.add_column("Модель / Провайдер", style="cyan", min_width=16, max_width=24)
    table.add_column("Сообщ.", style="dim", justify="right", width=7)
    table.add_column("Сохранено", style="dim #8b949e", min_width=12)

    here = os.path.normcase(os.path.abspath(os.getcwd()))
    displayed = sessions[:20]
    _last_displayed_sessions = displayed

    for i, s in enumerate(displayed):
        idx_str = str(i + 1)
        lbl = s.get("label")
        prev = (s.get("preview") or "").replace("\n", " ").strip()
        if lbl:
            label_prev = f"[bold #7ee787]{escape(lbl)}[/bold #7ee787]\n[dim italic]{escape(prev[:40])}{'…' if len(prev) > 40 else ''}[/dim italic]"
        else:
            label_prev = f"{escape(prev[:45])}{'…' if len(prev) > 45 else ''}"

        cwd = s.get("cwd") or ""
        if cwd and os.path.normcase(os.path.abspath(cwd)) == here:
            dir_str = f"[bold green]● здесь[/bold green]\n[dim]({escape(format_short_path(cwd))})[/dim]"
        elif cwd:
            dir_str = f"[dim]{escape(format_short_path(cwd))}[/dim]"
        else:
            dir_str = "[dim]—[/dim]"

        prov = s.get("provider") or ""
        mdl = s.get("model") or "?"
        model_str = f"[cyan]{prov + ':' if prov else ''}{mdl}[/cyan]"
        count_str = str(s.get("message_count") or 0)
        saved_str = format_relative_time(s.get("saved_at") or "")

        table.add_row(idx_str, label_prev, dir_str, model_str, count_str, saved_str)

    console.print(table)
    hint = "Восстановить: /load <номер|метка> (или просто /load) · Удалить: /sessions rm <номер>"
    if len(sessions) > 20:
        print_system(f"[dim]Показано 20 из {len(sessions)} сессий. {hint}[/dim]")
    else:
        print_system(f"[dim]{hint}[/dim]")
    return CommandResult.continue_loop()


@registry.register("/load", aliases=["/resume"], help_text="Восстановить сессию", category="session", is_prefix=True)
def handle_load(ctx: CommandContext) -> CommandResult:
    global _last_displayed_sessions
    user_input = ctx.user_input
    agent = ctx.agent
    parts = user_input.strip().split(maxsplit=1)
    cmd_name = parts[0].lower()

    if len(parts) < 2 and cmd_name == "/resume":
        target = get_last_session(os.getcwd()) or get_last_session()
        if target:
            restore_session(agent, target)
            return CommandResult.continue_loop()
        print_system("Нет доступных прошлых сессий для продолжения.")
        return CommandResult.continue_loop()

    if len(parts) < 2:
        sessions = list_sessions()
        if not sessions:
            print_system("Сохранённых сессий нет.")
            return CommandResult.continue_loop()
        here = os.path.normcase(os.path.abspath(os.getcwd()))
        choices = []
        for i, s in enumerate(sessions[:25]):
            lbl = f"{s['label']} " if s.get("label") else ""
            cwd = s.get("cwd") or ""
            is_here = "● здесь" if cwd and os.path.normcase(os.path.abspath(cwd)) == here else f"{format_short_path(cwd)}"
            time_str = format_relative_time(s.get("saved_at") or "")
            prev = (s.get("preview") or "").replace("\n", " ").strip()[:40]
            choices.append(f"[{i+1}] {lbl}{is_here} ({time_str}, {s.get('message_count', 0)} msgs) — \"{prev}\"")
        choices.append("Отмена")
        try:
            sel = questionary.select("Выберите сессию для восстановления:", choices=choices).ask()
        except Exception:
            sel = None
        if not sel or sel == "Отмена":
            return CommandResult.continue_loop()
        idx = int(sel.split("]")[0].replace("[", "").strip())
        target = find_session(str(idx))
        if target:
            restore_session(agent, target)
        return CommandResult.continue_loop()

    target = find_session(parts[1], sessions=_last_displayed_sessions) if _last_displayed_sessions else None
    if target is None:
        target = find_session(parts[1])
    if target is None:
        print_error(f"Сессия '{parts[1]}' не найдена. Список: /sessions")
        return CommandResult.continue_loop()
    restore_session(agent, target)
    return CommandResult.continue_loop()


@registry.register("/clear", aliases=["/reset", "/new"], help_text="Очистить историю диалога", category="session")
def handle_clear(ctx: CommandContext) -> CommandResult:
    agent = ctx.agent
    if hasattr(agent, "clear_history"):
        agent.clear_history()
    else:
        agent.messages.clear()
        agent.session_id = None

    import command_handler
    command_handler.print_system("История диалога очищена. Новый диалог начат с чистого листа.")
    return CommandResult.continue_loop(turn_counter=0)



@registry.register("/rewind", help_text="Откат к контрольной точке (чекпоинту)", category="session", is_prefix=True)
def handle_rewind(ctx: CommandContext) -> CommandResult:
    user_input = ctx.user_input
    agent = ctx.agent
    raw_args = user_input.strip()[len("/rewind"):].strip()
    tokens = raw_args.split()

    if "-h" in tokens or "--help" in tokens:
        print_system(
            "[bold]Команда /rewind — Машина времени Argent[/bold]\n\n"
            "Использование:\n"
            "  /rewind              — интерактивный выбор из последних чекпоинтов\n"
            "  /rewind <sha|текст>  — откат к чекпоинту по SHA или поисковой фразе\n"
            "  /rewind -y <sha>     — откат без подтверждения\n"
            "  /rewind --list [N]   — показать список последних N чекпоинтов\n"
            "  /rewind --limit <N>  — открыть меню с увеличенным количеством чекпоинтов"
        )
        return CommandResult.continue_loop()

    from src.agent.checkpoints import list_checkpoints, rewind_to, CheckpointError
    from file_tracker import clear_session_snapshots

    if "--list" in tokens or "-l" in tokens:
        limit = 15
        for t in tokens:
            if t.isdigit():
                limit = int(t)
                break
        cps = list_checkpoints(limit)
        if not cps:
            print_system("Нет доступных чекпоинтов Argent.")
        else:
            lines = ["Доступные чекпоинты:"]
            for c in cps:
                lines.append(f"  - [cyan]{c['sha']}[/cyan]  {c['label']}  [dim]({c['age']})[/dim]")
            print_system("\n".join(lines))
        return CommandResult.continue_loop()

    auto_yes = "-y" in tokens or "--yes" in tokens
    limit = 25
    filtered_tokens = []
    for t in tokens:
        if t in ("-y", "--yes"):
            continue
        if t.startswith("--limit="):
            try:
                limit = int(t.split("=", 1)[1])
            except ValueError:
                pass
            continue
        filtered_tokens.append(t)

    query = " ".join(filtered_tokens).strip()
    cps = list_checkpoints(limit)
    if not cps:
        print_system("Нет чекпоинтов Argent в истории текущей ветки. "
                     "Они создаются автоматически перед первой правкой каждого хода.")
        return CommandResult.continue_loop()

    target_cp = None
    if query:
        q_lower = query.lower()
        for c in cps:
            if c["sha"].lower().startswith(q_lower):
                target_cp = c
                break
        if target_cp is None:
            for c in cps:
                if q_lower in c["label"].lower():
                    target_cp = c
                    break
        if target_cp is None:
            print_error(f"Чекпоинт по запросу '{query}' не найден среди последних {len(cps)} записей.")
            lines = ["Доступные чекпоинты:"]
            for c in cps[:10]:
                lines.append(f"  - [cyan]{c['sha']}[/cyan]  {c['label']}  [dim]({c['age']})[/dim]")
            print_system("\n".join(lines))
            return CommandResult.continue_loop()
    else:
        choices = [f"{c['sha']}  {c['label']}  ({c['age']})" for c in cps] + ["Отмена"]
        sel = None
        try:
            sel = questionary.select("Откатить рабочее дерево к какому чекпоинту?", choices=choices).ask()
        except Exception:
            pass

        if not sel or sel == "Отмена":
            return CommandResult.continue_loop()
        sel_sha = sel.split()[0]
        target_cp = next((c for c in cps if c["sha"] == sel_sha), None)
        if not target_cp:
            return CommandResult.continue_loop()

    sha = target_cp["sha"]
    label = target_cp["label"]

    if not auto_yes:
        confirmed = False
        try:
            confirmed = questionary.confirm(
                f"Откатить ВСЁ до чекпоинта {sha} ({label})? (незакоммиченное будет сохранено в git stash)",
                default=False,
            ).ask()
        except Exception:
            pass
        if not confirmed:
            print_system("Откат отменён.")
            return CommandResult.continue_loop()

    try:
        result_msg = rewind_to(sha)
        print_system(result_msg)
        if hasattr(agent, "on_rewind"):
            agent.on_rewind(sha, label)
        clear_session_snapshots(session_id=getattr(agent, "session_id", "default") or "default")
    except CheckpointError as e:
        print_error(str(e))

    return CommandResult.continue_loop()


@registry.register("/back", help_text="Вернуться к состоянию до последнего сообщения: диалог и файлы",
                   category="session", is_prefix=True)
def handle_back(ctx: CommandContext) -> CommandResult:
    """The button under the last message in the TUI sends this. One step per call, so
    repeating it walks back through the conversation."""
    agent = ctx.agent
    tokens = ctx.user_input.strip()[len("/back"):].split()
    if "-h" in tokens or "--help" in tokens:
        print_system("[bold]/back[/bold] — вернуть диалог И файлы к состоянию до последнего вашего сообщения.\n"
                     "  /back      — с подтверждением\n  /back -y   — без подтверждения\n"
                     "Правки отменяемого хода сохраняются в `git stash`. Повторный /back возвращает ещё на сообщение назад.")
        return CommandResult.continue_loop()
    if not hasattr(agent, "plan_last_turn_rewind"):
        print_error("Эта модель агента не поддерживает возврат к сообщению.")
        return CommandResult.continue_loop()

    plan = agent.plan_last_turn_rewind()
    if not plan.get("ok"):
        print_error(plan.get("error", "Возврат невозможен."))
        return CommandResult.continue_loop()

    if "-y" not in tokens:
        preview = " ".join((plan["text"] or "").split())
        preview = preview if len(preview) <= 70 else preview[:67] + "…"
        confirmed = False
        try:
            confirmed = questionary.confirm(f"Вернуться к состоянию до сообщения «{preview}»? {plan['detail']}", default=True).ask()
        except Exception:
            pass
        if not confirmed:
            print_system("Возврат отменён.")
            return CommandResult.continue_loop()

    result = agent.rewind_last_turn()
    if not result.get("ok"):
        print_error(result.get("error", "Не удалось вернуться."))
        return CommandResult.continue_loop()
    try:
        clear_session_snapshots(session_id=getattr(agent, "session_id", "default") or "default")
    except Exception:
        pass
    bridge = None
    if os.environ.get("ARGENT_TUI") == "1":
        try:
            from src.bridge.repl_bridge import ReplBridge
            bridge = ReplBridge.active_bridge
        except Exception:
            bridge = None
    if bridge is not None:
        # The full-screen UI drops the message from its transcript and puts the text back in the prompt.
        bridge.emit({"type": "rewound", "text": result["text"], "detail": result["detail"]})
    else:
        print_system(f"↩ Вернулись к состоянию до сообщения. {escape(result['detail'])}")
        print_system("Текст сообщения (можно отредактировать и отправить заново):\n" + escape(result["text"]))
    return CommandResult.continue_loop()


@registry.register("/export", help_text="Экспорт истории диалога", category="session", is_prefix=True)
def handle_export(ctx: CommandContext) -> CommandResult:
    from command_handler import export_chat_history
    parts = ctx.user_input.split(" ", 1)
    name = parts[1].strip() if len(parts) > 1 else None
    export_chat_history(ctx.agent, filename=name, auto=False)
    return CommandResult.continue_loop()


@registry.register(
    "/copy",
    aliases=["/cp"],
    help_text="Скопировать последний ответ ассистента или код в буфер обмена (/copy [code|all])",
    category="session",
    is_prefix=True,
)
def handle_copy(ctx: CommandContext) -> CommandResult:
    from src.cli.input_queue import set_clipboard_text
    from ui import print_system, print_error
    import re

    target = "last"
    parts = ctx.user_input.strip().split()
    if len(parts) > 1:
        target = parts[1].lower().strip()

    messages = getattr(ctx.agent, "messages", []) or []
    if not messages:
        print_system("История сообщений пуста.")
        return CommandResult.continue_loop()

    if target in ("code", "код"):
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = str(m.get("content") or "")
                blocks = re.findall(r"```[^\n]*\n(.*?)```", content, re.DOTALL)
                if blocks:
                    code_text = "\n\n".join(b.strip() for b in blocks if b.strip())
                    if set_clipboard_text(code_text):
                        print_system(f"Блоки кода ({len(code_text)} симв.) скопированы в буфер обмена.")
                    else:
                        print_error("Не удалось записать в буфер обмена.")
                    return CommandResult.continue_loop()
        print_system("В последнем ответе ассистента нет блоков кода ```...```.")
        return CommandResult.continue_loop()

    elif target in ("all", "все"):
        full = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if content and role in ("user", "assistant"):
                full.append(f"### {role.capitalize()}:\n{content}\n")
        all_text = "\n".join(full)
        if set_clipboard_text(all_text):
            print_system(f"Весь диалог ({len(all_text)} симв.) скопирован в буфер обмена.")
        else:
            print_error("Не удалось записать в буфер обмена.")
        return CommandResult.continue_loop()

    else:
        for m in reversed(messages):
            if m.get("role") == "assistant":
                content = str(m.get("content") or "").strip()
                if content:
                    if set_clipboard_text(content):
                        print_system(f"Ответ ассистента ({len(content)} симв.) скопирован в буфер обмена.")
                    else:
                        print_error("Не удалось записать в буфер обмена.")
                    return CommandResult.continue_loop()
        print_system("Нет ответа ассистента для копирования.")
        return CommandResult.continue_loop()
