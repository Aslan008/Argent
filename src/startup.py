"""What happens between "Argent started" and "the first prompt".

The terminal REPL (main.py) and the TUI's Python bridge both open with the same
conversation: offer to continue the previous session, say what survives of the
working memory, warn when the restored session belongs to another directory or
model — and only THEN start the MCP servers, because a server's working directory
is resolved when it is spawned and the restore may change directory.

This used to live in main.py, so the TUI could not reuse it: it printed through the
console and asked through prompt_toolkit. The TUI got a one-line "last session:
/resume" notice instead, and its MCP servers were already starting while you were
still deciding. Here the sequence is written once and the front end supplies only
what differs: where lines of text go (`StartupIO`). Questions go through
`questionary`, which the TUI bridge already routes to its own modals.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import questionary

from session import get_last_session, load_session


@dataclass
class StartupIO:
    """Where a front end wants the startup text to go."""

    system: Callable[[str], None]
    error: Callable[[str], None]
    #: A titled horizontal rule, for front ends that draw one (the REPL does).
    rule: Optional[Callable[[str], None]] = None

    def heading(self, title: str) -> None:
        if self.rule is not None:
            self.rule(title)
        else:
            self.system(f"[bold cyan]{title}[/bold cyan]")


def console_io() -> StartupIO:
    """The terminal's own printers (looked up at call time, so they can be patched)."""
    import ui

    return StartupIO(system=lambda text: ui.print_system(text),
                     error=lambda text: ui.print_error(text),
                     rule=lambda title: ui.console.rule(f"[bold cyan]{title}[/bold cyan]"))


# Choices of the "С чего начать?" question. Named so a front end can recognise them.
RESTORE = "Продолжить прошлую сессию"
KEEP_MEMORY = "Новый чат, но сохранить рабочую память проекта"
CLEAN = "Начать с чистого листа (забыть всё выше)"


def carried_over_memory() -> list:
    """What the working memory would drag into a brand-new chat.

    .argent/memory.json is per-project and loaded at construction, so a fresh
    launch silently inherits the previous run's objective, its list of
    completed steps and its recorded failures — all of which feed the objective
    anchor in the prompt. Nothing said so, which is why declining to restore a
    session did not actually give you a clean slate.
    """
    from memory_manager import memory

    data = memory.data or {}
    parts = []
    objective = (data.get("objective") or "").strip()
    if objective:
        parts.append(f"цель «{objective[:60]}»")
    for key, word in (("completed", "шагов сделано"),
                      ("key_facts", "фактов"),
                      ("errors_encountered", "ошибок запомнено")):
        count = len(data.get(key) or [])
        if count:
            parts.append(f"{count} {word}")
    return parts


def previous_session() -> Optional[dict]:
    """The last saved session worth offering, or None."""
    last = get_last_session()
    if last and last.get("preview") and last.get("preview") != "[corrupted session]":
        return last
    return None


def offer_previous_context(agent, io: Optional[StartupIO] = None) -> None:
    """One question about everything that survives from the last run.

    There used to be two independent things and one question: the saved
    conversation was offered, while the working memory came back regardless.
    Answering "no" therefore did NOT give a clean start — which is why /clear
    was pressed 115 times, more than every other command combined. Now the
    choice covers both, so "начать с чистого листа" is literally true.
    """
    from memory_manager import memory

    io = io or console_io()
    last = previous_session()
    has_session = last is not None
    carried = carried_over_memory()
    if not has_session and not carried:
        return

    if has_session:
        where = last.get("cwd") or ""
        same_place = (where and os.path.normcase(os.path.abspath(where))
                      == os.path.normcase(os.getcwd()))
        suffix = "" if same_place else f" [в {Path(where).name}]" if where else ""
        saved_time = (last.get("saved_at") or "")[:16]
        io.system(f"Прошлая сессия ({saved_time}){suffix}: "
                  f"\"{last.get('preview', '')}\"")
    if carried:
        # Named out loud: this is the part that used to arrive uninvited.
        io.system(f"[dim]Рабочая память проекта: {', '.join(carried)}.[/dim]")

    choices = ([RESTORE] if has_session else []) + ([KEEP_MEMORY] if carried else []) + [CLEAN]
    try:
        choice = questionary.select("С чего начать?", choices=choices,
                                    default=choices[0]).ask()
    except (KeyboardInterrupt, EOFError):
        choice = None

    if choice == RESTORE:
        restore_session(agent, last, io)
    elif choice == CLEAN:
        memory.clear()
        io.system("[dim]Рабочая память очищена — контекста прошлых разговоров нет.[/dim]")


def restore_session(agent, meta: dict, io: Optional[StartupIO] = None) -> bool:
    """Put a saved session back, telling the user what does NOT match.

    Restoring the messages is the easy half. The hard half is that the system
    prompt gets rebuilt for the CURRENT directory and model while the
    conversation still talks about the old project's files — and the working
    memory in .argent/memory.json belongs to wherever you are standing now.
    Nothing downstream can detect that, so it has to be said here.
    """
    io = io or console_io()
    data = load_session(meta["id"])
    if not data or not data.get("messages"):
        io.error("Не удалось прочитать сессию.")
        return False

    agent.messages = data["messages"]
    agent.session_id = data.get("id") or meta["id"]
    if hasattr(agent, "reset_turn_records"):
        agent.reset_turn_records()
    from src.agent.compaction_log import reset_compaction_history
    reset_compaction_history()
    from src.agent import todos
    todos.restore((data.get("state") or {}).get("todos"))

    # Reset transient agent caches from previous session
    if hasattr(agent, "_read_cache"):
        agent._read_cache.clear()
    if hasattr(agent, "loop_guard"):
        agent.loop_guard.reset()
    if hasattr(agent, "_checkpoint_turn_indices"):
        agent._checkpoint_turn_indices.clear()
    if hasattr(agent, "_turn_checkpoint_done"):
        agent._turn_checkpoint_done = False

    # Repair any orphaned tool calls from interrupted session
    interrupted = False
    if hasattr(agent, "repair_history"):
        try:
            interrupted = bool(agent.repair_history())
        except Exception as e:
            from logger import get_logger
            get_logger("session").warning("repair_history during restore failed: %s", e)

    saved_time = (data.get("saved_at") or "")[:16]
    # The conversation itself goes back on the screen first, then the notes about it.
    try:
        from src.agent import history_replay
        history_replay.show(agent.messages, replace=False, saved_at=saved_time)
    except Exception:
        pass
    io.system(f"Восстановлено {len(data['messages'])} сообщений "
              f"({saved_time}).")
    if interrupted:
        io.system("[yellow]Прошлый ход был прерван (сбой, обрыв связи или закрытие окна).[/yellow] "
                  "Напишите «продолжай» — работа пойдёт с последнего сохранённого шага.")

    saved_model = data.get("model") or ""
    current_model = getattr(agent, "model_name", "")
    if saved_model and saved_model != current_model:
        io.system(f"[yellow]Сессия велась на '{saved_model}', сейчас активна "
                  f"'{current_model}'.[/yellow] История содержит вызовы "
                  f"инструментов, которые текущая модель может не повторить — "
                  f"смените модель через /model, если это важно.")

    saved_cwd = data.get("cwd") or ""
    current = os.getcwd()
    if saved_cwd and os.path.normcase(os.path.abspath(saved_cwd)) != os.path.normcase(current):
        io.system(f"[yellow]Сессия сохранена в:[/yellow] {saved_cwd}\n"
                  f"[yellow]   Вы сейчас в:       [/yellow] {current}")
        if not os.path.isdir(saved_cwd):
            io.system("[dim]Директории сессии больше нет — пути из истории "
                      "не совпадут с текущим проектом.[/dim]")
            return True
        # Never silent: after a cd, files land somewhere the user did not ask
        # for. That is the one class of action Argent always confirms.
        try:
            go = questionary.confirm("Перейти в директорию сессии?", default=False).ask()
        except Exception:
            go = False
        if go:
            os.chdir(saved_cwd)
            try:
                from memory_manager import memory
                memory._mgr = None
            except Exception:
                pass
            io.system(f"Рабочая директория: {os.getcwd()}")
        else:
            io.system("[dim]Остаёмся здесь. Учтите: пути в истории — от другого "
                      "проекта, как и рабочая память в .argent/memory.json.[/dim]")
    return True


def offer_first_run_setup(run_slash: Callable[[str], object], io: Optional[StartupIO] = None) -> bool:
    """First launch ever (no config file yet): walk through provider and model.

    `run_slash` executes a slash command in the caller's own way (the REPL's
    handler, or the bridge's dispatcher). Returns True when the walk-through ran.
    """
    from config import CONFIG_FILE

    if Path(CONFIG_FILE).exists():
        return False
    io = io or console_io()
    io.heading("Первый запуск")
    io.system("Похоже, это первый запуск Argent. Настроим провайдера и модель — это займёт минуту.")
    try:
        if questionary.confirm("Настроить сейчас?", default=True).ask():
            run_slash("/provider")
            run_slash("/model")
            io.system("[green]Готово. Изменить в любой момент: /provider, /model.[/green]")
        else:
            io.system("Пропущено. Настроить позже можно командами /provider и /model.")
    except (KeyboardInterrupt, EOFError):
        io.system("Настройка пропущена. Позже: /provider и /model.")
    return True
