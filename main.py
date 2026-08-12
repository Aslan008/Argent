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
    get_current_model, set_current_model,
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
from session import (
    save_session, load_session, list_sessions, delete_session, get_last_session,
    find_session,
)
from file_tracker import snapshot, get_diff, undo, get_pending_changes, undo_all
from prompts import (
    build_spec_prompt, build_work_investigation_prompt,
    build_work_planning_prompt, build_planning_prompt, build_architecture_prompt
)
from src.cli.cli_ui import render_response_stream
from src.project.orchestrator import ProjectOrchestrator
from src.agent.shell import run_text

# Tools that belong to a MODE, not to ordinary chat: the project brain is
# driven by the orchestrator's own state machine, and offering its steps in a
# normal conversation invites the model to write a spec nobody asked for.
#
# Stated as a DENYLIST on purpose. It used to be a hand-written allowlist, and
# every tool added after it was written silently vanished from chat — measured,
# 27 of 66, including move_file, find_definition, list_mcp_tools and
# view_image. The model would call a name it had been told about, get "not
# available", and be offered a nonsense substitute. A denylist states intent
# once; anything new is available unless someone decides otherwise.
CHAT_DENIED_TOOLS = {
    "add_project_task", "complete_project_task", "list_project_tasks",
    "write_project_spec", "write_project_architecture", "write_file_spec",
    "plan_work_changes", "add_work_task",
}


def chat_allowed_tools() -> list:
    """The toolset for a regular chat turn: everything real, minus the modes."""
    from tools.schemas import get_available_tools
    avail = get_available_tools() or []
    return [name for name in avail if name not in CHAT_DENIED_TOOLS]



def offer_safety_checkpoint(task: str) -> None:
    """Before unattended /auto work: offer one git checkpoint if the tree is
    dirty. A clean tree needs no insurance; outside git there is nothing to do.
    Rollback path: git_rollback (only ever touches 'Argent Checkpoint' commits)."""
    try:
        is_git = run_text("git rev-parse --is-inside-work-tree",
                          shell=True, capture_output=True)
        if not is_git or getattr(is_git, "returncode", -1) != 0:
            return
        dirty_res = run_text("git status --porcelain", shell=True, capture_output=True)
        dirty = getattr(dirty_res, "stdout", "").strip() if dirty_res else ""
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


def _parse_task_tools(spec: str):
    """Parse the optional `tools: a, b` segment of /tasks add.

    Names are validated against the real registry: a typo would otherwise
    silently narrow the toolset to nothing and the task would fail every night
    for a reason nobody could see.
    """
    from tools.schemas import AVAILABLE_TOOLS

    spec = (spec or "").strip()
    if not spec:
        return [], []
    if spec.lower().startswith("tools:"):
        spec = spec.split(":", 1)[1]
    names = [n.strip() for n in spec.replace(";", ",").split(",") if n.strip()]
    unknown = [n for n in names if not isinstance(AVAILABLE_TOOLS, dict) or n not in AVAILABLE_TOOLS]
    return names, unknown


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


def offer_previous_context(agent) -> None:
    """One question about everything that survives from the last run.

    There used to be two independent things and one question: the saved
    conversation was offered, while the working memory came back regardless.
    Answering "no" therefore did NOT give a clean start — which is why /clear
    was pressed 115 times, more than every other command combined. Now the
    choice covers both, so "начать с чистого листа" is literally true.
    """
    from memory_manager import memory

    last = get_last_session()
    has_session = bool(last and last.get("preview"))
    carried = carried_over_memory()
    if not has_session and not carried:
        return

    if has_session:
        where = last.get("cwd") or ""
        same_place = (where and os.path.normcase(os.path.abspath(where))
                      == os.path.normcase(os.getcwd()))
        suffix = "" if same_place else f" [в {Path(where).name}]" if where else ""
        saved_time = (last.get("saved_at") or "")[:16]
        print_system(f"Прошлая сессия ({saved_time}){suffix}: "
                     f"\"{last.get('preview', '')}\"")
    if carried:
        # Named out loud: this is the part that used to arrive uninvited.
        print_system(f"[dim]Рабочая память проекта: {', '.join(carried)}.[/dim]")

    RESTORE = "Продолжить прошлую сессию"
    KEEP_MEMORY = "Новый чат, но сохранить рабочую память проекта"
    CLEAN = "Начать с чистого листа (забыть всё выше)"
    choices = ([RESTORE] if has_session else []) + ([KEEP_MEMORY] if carried else []) + [CLEAN]
    try:
        choice = questionary.select("С чего начать?", choices=choices,
                                    default=choices[0]).ask()
    except (KeyboardInterrupt, EOFError):
        choice = None

    if choice == RESTORE:
        restore_session(agent, last)
    elif choice == CLEAN:
        memory.clear()
        print_system("[dim]Рабочая память очищена — контекста прошлых разговоров нет.[/dim]")


def restore_session(agent, meta: dict) -> bool:
    """Put a saved session back, telling the user what does NOT match.

    Restoring the messages is the easy half. The hard half is that the system
    prompt gets rebuilt for the CURRENT directory and model while the
    conversation still talks about the old project's files — and the working
    memory in .argent/memory.json belongs to wherever you are standing now.
    Nothing downstream can detect that, so it has to be said here.
    """
    data = load_session(meta["id"])
    if not data or not data.get("messages"):
        print_error("Не удалось прочитать сессию.")
        return False

    agent.messages = data["messages"]
    agent.session_id = data.get("id") or meta["id"]
    saved_time = (data.get("saved_at") or "")[:16]
    print_system(f"Восстановлено {len(data['messages'])} сообщений "
                 f"({saved_time}).")

    saved_model = data.get("model") or ""
    current_model = getattr(agent, "model_name", "")
    if saved_model and saved_model != current_model:
        print_system(f"[yellow]⚠ Сессия велась на '{saved_model}', сейчас активна "
                     f"'{current_model}'.[/yellow] История содержит вызовы "
                     f"инструментов, которые текущая модель может не повторить — "
                     f"смените модель через /model, если это важно.")

    saved_cwd = data.get("cwd") or ""
    current = os.getcwd()
    if saved_cwd and os.path.normcase(os.path.abspath(saved_cwd)) != os.path.normcase(current):
        print_system(f"[yellow]⚠ Сессия сохранена в:[/yellow] {saved_cwd}\n"
                     f"[yellow]   Вы сейчас в:       [/yellow] {current}")
        if not os.path.isdir(saved_cwd):
            print_system("[dim]Директории сессии больше нет — пути из истории "
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
            print_system(f"Рабочая директория: {os.getcwd()}")
        else:
            print_system("[dim]Остаёмся здесь. Учтите: пути в истории — от другого "
                         "проекта, как и рабочая память в .argent/memory.json.[/dim]")
    return True


def handle_tasks_command(user_input: str, agent) -> None:
    """/tasks — scheduled automations that run while Argent is open.

    The terminal manages them; the scheduler itself lives in the server process
    (python argent_server.py), so a definition added here starts firing as soon
    as that is running.
    """
    from src.automation.schedule import ScheduleError, parse_schedule
    from src.automation.store import (
        Automation, load_automations, load_runs, remove_automation, upsert_automation,
    )

    parts = user_input.strip().split(maxsplit=2)
    sub = parts[1] if len(parts) > 1 else "list"
    arg = parts[2] if len(parts) > 2 else ""

    if sub not in ("list", "add", "on", "off", "rm", "remove", "delete", "run", "runs", "memory", "forget"):
        print_error("Команды: /tasks list | add <имя> | <расписание> | <задача> | "
                    "on <имя> | off <имя> | rm <имя> | run <имя> | runs | "
                    "memory [имя] | forget <имя>")
        return

    if sub == "list":
        items = load_automations()
        if not items:
            print_system(
                "Автоматизаций нет. Пример:\n"
                "  /tasks add отчёт | daily at 09:00 | Собери метрики командой "
                "`npm run stats` и запиши сводку в reports/daily.md\n"
                "Запускаются, пока работает argent_server.py.")
            return
        print_system("[bold cyan]Автоматизации:[/bold cyan]")
        for a in items:
            state = "вкл" if a.enabled else "ВЫКЛ"
            last = f", последний: {a.last_run[:16]} ({a.last_status})" if a.last_run else ""
            tools = f", инструменты: {', '.join(a.allowed_tools)}" if a.allowed_tools else ""
            print_system(f"  • {a.name} [{state}] — {a.schedule}{last}{tools}")
            print_system(f"      {a.task[:110]}")
        return

    if sub == "add":
        pieces = [p.strip() for p in arg.split("|")]
        if len(pieces) < 3 or not all(pieces[:3]):
            print_error("Формат: /tasks add <имя> | <расписание> | <задача> [| tools: a, b]\n"
                        "Расписание: 'every 30m', 'every 2h' или 'daily at 09:00'.")
            return
        name, schedule, task = pieces[0], pieces[1], pieces[2]
        try:
            parse_schedule(schedule)
        except ScheduleError as e:
            print_error(str(e))
            return

        # The toolset is the capability boundary of an unattended run, so it has
        # to be reachable from here — a limit you can only set by hand-editing
        # JSON is a limit nobody sets.
        tools_part = pieces[3] if len(pieces) > 3 else ""
        if tools_part.strip().lower().startswith("tools:") and not tools_part.split(":", 1)[1].strip():
            print_error("Предупреждение: блок tools пуст (вы указали '| tools:', но не перечислили инструменты).")
        tools, unknown = _parse_task_tools(tools_part)
        if unknown:
            print_error(f"Неизвестные инструменты: {', '.join(unknown)}. "
                        f"Список — /tools.")
            return

        upsert_automation(Automation(name=name, task=task, schedule=schedule,
                                     allowed_tools=tools))
        print_system(f"Автоматизация '{name}' сохранена ({schedule}).\n"
                     f"Она выполняется БЕЗ участия человека: любое действие, требующее "
                     f"подтверждения, будет отклонено и записано в журнал — "
                     f"смотрите /tasks runs.")
        if tools:
            print_system(f"Доступные ей инструменты: {', '.join(tools)}.")
        else:
            print_system("[yellow]Инструменты не ограничены[/yellow] — задача получит "
                         "весь набор. Сузьте его: `| tools: search_web, read_webpage`.")
        return

    if sub in ("on", "off"):
        items = load_automations()
        target = next((a for a in items if a.name == arg.strip()), None)
        if target is None:
            print_error(f"Автоматизация '{arg.strip()}' не найдена.")
            return
        target.enabled = (sub == "on")
        upsert_automation(target)
        print_system(f"'{target.name}': {'включена' if target.enabled else 'выключена'}.")
        return

    if sub in ("rm", "remove", "delete"):
        print_system(f"Удалена: {arg.strip()}" if remove_automation(arg.strip())
                     else f"Автоматизация '{arg.strip()}' не найдена.")
        return

    if sub == "run":
        from src.automation.runner import run_and_record
        from src.automation.store import get_automation
        target = get_automation(arg.strip())
        if target is None:
            print_error(f"Автоматизация '{arg.strip()}' не найдена.")
            return
        print_system(f"Запускаю '{target.name}' сейчас (без подтверждений)...")
        # No toast: you are looking at the output right now.
        result = run_and_record(target, agent=agent, notify=False)
        print_system(f"Статус: {result['status']}\n{result['summary'][:1500]}")
        if result.get("new_items"):
            print_system(f"[green]Новых элементов запомнено: {result['new_items']}[/green]")
        if result["denied_actions"]:
            print_system("[yellow]Отклонено (нужен человек):[/yellow] " +
                         "; ".join(d["action"] for d in result["denied_actions"]))
        return

    if sub == "memory":
        from src.automation.memory import stats
        counts = stats(arg.strip() or None)
        counts = {k: v for k, v in counts.items() if v}
        if not counts:
            print_system(
                "Память пуста. Она наполняется, когда задача вызывает "
                "filter_new_items — так мониторинг сообщает только о новом, "
                "а не повторяет один и тот же список каждый прогон.")
            return
        print_system("[bold cyan]Запомнено элементов:[/bold cyan]")
        for name, count in sorted(counts.items()):
            print_system(f"  • {name}: {count}")
        print_system("[dim]Сброс: /tasks forget <имя>[/dim]")
        return

    if sub == "forget":
        from src.automation.memory import forget
        target_name = arg.strip()
        if not target_name:
            print_error("Формат: /tasks forget <имя>")
            return
        dropped = forget(target_name)
        print_system(f"Забыто элементов: {dropped}. Следующий прогон '{target_name}' "
                     f"снова сочтёт всё новым."
                     if dropped else f"Для '{target_name}' ничего не запомнено.")
        return

    if sub == "runs":
        runs = load_runs(limit=15, name=arg.strip() or None)
        if not runs:
            print_system("Прогонов ещё не было.")
            return
        print_system("[bold cyan]Последние прогоны:[/bold cyan]")
        for r in runs:
            denied = f", отклонено: {len(r['denied_actions'])}" if r.get("denied_actions") else ""
            fresh = f", новых: {r['new_items']}" if r.get("new_items") else ""
            print_system(f"  {r['started'][:16]}  {r['name']}  [{r['status']}] "
                         f"{r['seconds']}s{fresh}{denied}")
            if r.get("summary"):
                print_system(f"      {r['summary'][:150]}")
        return

    print_error("Команды: /tasks list | add <имя> | <расписание> | <задача> | "
                "on <имя> | off <имя> | rm <имя> | run <имя> | runs | "
                "memory [имя] | forget <имя>")


def toggle_vibe_mode(vibe_mode: bool) -> bool:
    """The vibecoder switch. Curates EXISTING knobs — no new machinery:
    auto-approve safe actions (destructive ones still prompt) and guarantee
    per-turn checkpoints, so every step stays rewindable via /rewind."""
    current_policy = getattr(approval, "get_policy", lambda: approval.POLICY_ASK)()
    vibe_mode = (current_policy != approval.POLICY_AUTO)
    from src.agent.checkpoints import set_auto_checkpoint
    if vibe_mode:
        set_auto_checkpoint(True)
        approval.set_policy(approval.POLICY_AUTO)
        print_system(
            "🌴 Vibe-режим ВКЛ: безопасные действия одобряются автоматически "
            "(опасные — по-прежнему спросят), перед первой правкой каждого хода "
            "создаётся чекпоинт. Откат в любой момент: /rewind."
        )
    else:
        approval.set_policy(approval.POLICY_ASK)
        print_system("Vibe-режим ВЫКЛ: обычный режим подтверждений возвращён.")
    return vibe_mode


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
            items = list(ACTIVE_PROCESSES.items()) if isinstance(ACTIVE_PROCESSES, dict) else list(ACTIVE_PROCESSES)
            for item in items:
                try:
                    proc_info = item[1] if isinstance(item, tuple) else item
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
                }, session_id=getattr(agent, "session_id", None))
            except Exception:
                pass

    atexit.register(cleanup)

    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        from version import __version__ as _ver
    except Exception:
        _ver = "?"
    console.rule(f"[bold cyan]Argent Coder[/bold cyan] [dim]v{_ver}[/dim]")
    print_system("Argent Coder — автономная среда разработки.")
    print_system("Список команд: /help")
    
    agent = ArgentAgent()
    orchestrator = ProjectOrchestrator(agent)
    
    builtin_cmds = [
        # Base commands
        '/help', '/provider', '/model', '/clear', '/init', '/research', '/rag_toggle', '/auto_retrieve',
        '/kb', '/kb_toggle',
        '/hooks', '/plugin', '/tools', '/save', '/project', '/work', '/commit',
        '/sessions', '/load', '/copy', '/logs', '/skills', '/skill import', '/auto', '/vibe', '/tasks', '/verbose', '/results', '/debug', '/browser', '/exit', '/quit',
        '/mcp', '/thinking', '/temp', '/temperature',
        '/cd', '/undo', '/diff', '/changes', '/rewind', '/stats', '/aux', '/search', '/doctor', '/jobs', '/stop', '/goal', '/critic', '/rooms',
        
        # Subcommands and parameter variations
        '/mcp list', '/mcp add', '/mcp remove', '/mcp test', '/mcp start', '/mcp stop',
        '/kb list', '/kb add', '/kb remove', '/kb toggle', '/kb index',
        '/browser user', '/browser isolated', '/browser chrome', '/browser yandex', '/browser edge', '/browser brave', '/browser auto',
        '/hooks auto', '/plugin auto',
        '/temp 0.2', '/temp 0.7', '/temp 1.0',
        '/temperature 0.2', '/temperature 0.7', '/temperature 1.0',
        '/work --auto',
        '/critic on', '/critic off', '/critic model', '/critic status',
        '/tasks list', '/tasks add', '/tasks on', '/tasks off', '/tasks rm', '/tasks run', '/tasks runs',
        '/guard', '/guard off', '/guard warn', '/guard block',
        '/rooms resume', '/rooms list', '/rooms show', '/rooms spawn on', '/rooms spawn off',
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
    # Prime the context-meter cache so the bottom toolbar shows a real figure
    # from the very first keystroke, not only after the first turn.
    try:
        agent.get_context_usage()
    except Exception:
        pass
    
    # First-run onboarding: if Argent has never been configured, walk the user
    # through provider + model setup (reuses the /provider and /model flows).
    from config import CONFIG_FILE
    if not Path(CONFIG_FILE).exists():
        console.rule("[bold cyan]Первый запуск[/bold cyan]")
        print_system("Похоже, это первый запуск Argent. Настроим провайдера и модель — это займёт минуту.")
        try:
            if questionary.confirm("Настроить сейчас?", default=True).ask():
                handle_slash_command("/provider", agent)
                handle_slash_command("/model", agent)
                print_system("[green]Готово. Изменить в любой момент: /provider, /model.[/green]")
            else:
                print_system("Пропущено. Настроить позже можно командами /provider и /model.")
        except (KeyboardInterrupt, EOFError):
            print_system("Настройка пропущена. Позже: /provider и /model.")

    print_system(f"Провайдер: {get_provider().upper()}")
    print_system(f"Модель: {get_current_model()}")
    print_system(f"Рабочая директория: {os.getcwd()}")
    if not Path(".argent/AGENTS.md").exists() and not Path("AGENTS.md").exists():
        print_system("[dim]Подсказка: нет AGENTS.md — выполните /init, чтобы Argent изучил проект и создал память о нём.[/dim]")
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
        # "Running with 140 tools" and "the model can use them" are independent
        # states that both look like ON; say so when they disagree.
        from src.agent.mcp_prompt import model_access_warning
        warning = model_access_warning()
        if warning:
            print_system(f"[yellow]⚠ {warning}[/yellow]")

    offer_previous_context(agent)

    # Trigger Startup Hook
    hook_manager.call_hook("on_startup")

    is_project_mode = False
    is_auto_mode = False
    vibe_mode = False
    auto_sleep_time = 0
    auto_wake_context = ""
    auto_continue_input = None
    last_task_id = None
    task_retries = 0
    project_iterations = 0
    MAX_TASK_RETRIES = 3
    turn_counter = 0
    
    from config import get_auto_rag, get_auto_kb
    if get_auto_kb():
        try:
            from rag_engine import init_external_kbs
            kb_res = init_external_kbs()
            if "Successfully" in kb_res:
                print_system(kb_res)
        except ImportError:
            pass

    if get_auto_rag():
        import threading
        cwd = os.getcwd()
        print_system(f"Авто-RAG включён. Индексирую {cwd} в фоне…")
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
            
            if auto_sleep_time > 0 or isinstance(auto_wake_context, dict):
                import queue
                from src.agent.events import EVENT_QUEUE
                
                class EventInterruptedException(Exception):
                    def __init__(self, event):
                        self.event = event

                def sleep_with_events(secs):
                    import time
                    end = time.monotonic() + secs
                    while time.monotonic() < end:
                        try:
                            if EVENT_QUEUE is not None:
                                ev = EVENT_QUEUE.get(timeout=min(0.5, max(0.1, end - time.monotonic())))
                            else:
                                time.sleep(0.1)
                                continue
                            if ev.get("type") == "bg_done":
                                raise EventInterruptedException(ev)
                        except queue.Empty:
                            pass

                # Waiting ON a condition: poll locally instead of waking the
                # model to look. Each wake costs a full turn, so a ten-poll
                # "is it done yet?" loop becomes one sleep and one wake.
                if isinstance(auto_wake_context, dict):
                    spec, auto_wake_context = auto_wake_context, ""
                    until = spec.get("until", "condition")
                    timeout = spec.get("timeout", 60)
                    reason = spec.get("reason", "no reason")
                    print_system(f"*[Heartbeat] Жду условие: {until} "
                                 f"(до {timeout} сек., Ctrl+C для прерывания)*")
                    from src.agent.wait_conditions import wait_for
                    try:
                        outcome = wait_for(until, timeout=timeout,
                                           sleep=sleep_with_events)
                    except EventInterruptedException as e:
                        ev = e.event
                        print_system(f"*[Событие] Фоновая команда {ev.get('pid', '?')} завершилась. Ожидание прервано.*")
                        auto_wake_context = f"[Heartbeat прерван] Фоновая команда {ev.get('pid', '?')} завершилась с кодом {ev.get('exit_code', '?')}. Проверьте её вывод с помощью read_background_command."
                        auto_sleep_time = 0
                    except KeyboardInterrupt:
                        print_system("Ожидание прервано.")
                        auto_sleep_time = 0
                        auto_wake_context = ""
                        continue
                    
                    if not auto_wake_context: # If not interrupted by event
                        auto_sleep_time = 0
                        if outcome and isinstance(outcome, dict) and "met" in outcome:
                            verdict = "ВЫПОЛНЕНО" if outcome["met"] else "НЕ выполнено"
                            out_reason = outcome.get("reason", "unknown")
                            waited = outcome.get("waited", 0)
                            print_system(f"*[Heartbeat] {verdict}: {out_reason} "
                                         f"({waited} сек.)*")
                            auto_wake_context = (
                                f"[Heartbeat пробуждение] Ожидалось условие: {until}\n"
                                f"Результат: {out_reason} (ждали {waited} сек.).\n"
                                f"Исходная причина ожидания: {reason}\n"
                                + ("Условие выполнено — продолжай." if outcome["met"] else
                                   "Условие НЕ выполнено. Не считай ожидаемое событие произошедшим: "
                                   "проверь состояние сам и реши, ждать ли дальше, "
                                   "действовать иначе или завершить работу через `end_auto_mode`.")
                            )
                        else:
                            print_error("Ошибка wait_for: возвращен неверный формат")
                            auto_wake_context = "[Heartbeat пробуждение] Ошибка: проверка условия вернула неверный формат. Проверьте результат сами."

                if auto_sleep_time > 0:
                    print_system(f"*[Heartbeat] Переход в сон на {auto_sleep_time} сек. (может быть прерван событиями, Ctrl+C для отмены)*")
                    try:
                        # Chunked sleep so Ctrl+C aborts within a tick, not after
                        # the full (possibly very long) heartbeat delay.
                        sleep_with_events(auto_sleep_time)
                    except EventInterruptedException as e:
                        ev = e.event
                        print_system(f"*[Событие] Фоновая команда {ev['pid']} завершилась. Ожидание прервано.*")
                        auto_wake_context = f"[Heartbeat прерван] Фоновая команда {ev['pid']} завершилась с кодом {ev['exit_code']}. Проверьте её вывод с помощью read_background_command."
                    except KeyboardInterrupt:
                        print_system("Состояние Heartbeat прервано.")
                        auto_sleep_time = 0
                        auto_wake_context = ""
                        continue
                    auto_sleep_time = 0
                
                if not is_auto_mode:
                    auto_continue_input = auto_wake_context if auto_wake_context else "[Heartbeat завершен] Продолжай."
                    auto_wake_context = ""

            if is_auto_mode:
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

            if user_input.startswith("/init"):
                user_input = (
                    "Onboard to THIS project and create a professional `.argent/AGENTS.md` — it is "
                    "your persistent memory about this codebase and is loaded into your context every "
                    "session, so it must be accurate and concise.\n\n"
                    "STRICT workflow:\n"
                    "1. Call `analyze_project()` first to get the structure, languages, entry points and commands.\n"
                    "2. Read the few key files you need (README, the main entry point, core config) to "
                    "understand the real architecture — do not guess.\n"
                    "3. Write `.argent/AGENTS.md` with `write_file`. Keep it DENSE and under ~150 lines, with:\n"
                    "   - `## Overview` — what the project is, in 2-3 sentences.\n"
                    "   - `## Architecture` — main modules/layers and how they relate.\n"
                    "   - `## Build / Run / Test` — exact commands.\n"
                    "   - `## Conventions` — code style, patterns and rules a contributor must follow.\n"
                    "   - `## Key files` — the most important files and what each does.\n"
                    "4. Tell the user what you wrote and remind them they can edit the file by hand.\n"
                    "Do NOT dump full file listings or trivia — this file costs context every turn, keep it tight."
                )
                active_tools = [
                    "analyze_project", "read_file", "get_file_outline", "list_directory",
                    "grep_search", "search_files", "write_file",
                ]
                print_system("Изучаю проект и создаю .argent/AGENTS.md...")
            elif user_input.startswith("/research"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Укажите тему. Пример: /research Unity DOTS")
                    continue
                topic = parts[1].strip()
                active_tools = ["run_deep_research"]
                user_input = (
                    f"Please act as an autonomous Research Agent for the topic: '{topic}'.\n\n"
                    f"Your STRICT workflow is:\n"
                    f"1. MUST CALL `run_deep_research(objective='{topic}')` right now to let the sub-agent gather massive information.\n"
                    f"2. Read the final synthesized report returned by the sub-agent.\n"
                    f"3. Present the findings directly to me in a highly structured, readable, and detailed format right here in the chat.\n"
                    f"Do NOT write the findings to a file unless I explicitly ask you to. Just give me the info.\n"
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
            elif user_input.strip() == "/vibe":
                vibe_mode = toggle_vibe_mode(vibe_mode)
                ui_state["mode"] = "VIBE" if vibe_mode else "CHAT"
                continue
            elif user_input.startswith("/browser"):
                parts = user_input.strip().split()
                if len(parts) == 1:
                    # Show current status and detected browsers
                    from config import get_browser_mode, get_browser_name
                    from browser_detect import detect_browsers
                    mode = get_browser_mode()
                    name = get_browser_name()
                    browsers = detect_browsers()
                    print_system(f"Режим браузера: [bold cyan]{mode}[/bold cyan]")
                    print_system(f"Selected Browser: [bold cyan]{name}[/bold cyan]")
                    if browsers:
                        print_system("Detected browsers:")
                        for b in browsers:
                            print_system(f"  - [bold yellow]{b.key}[/bold yellow]: {b.exe_path}")
                    else:
                        print_system("[dim]Браузеров на основе Chromium не найдено.[/dim]")
                    print_system("\nUsage: /browser user | isolated | chrome | yandex | edge | brave | auto")
                elif parts[1] in ("user", "isolated"):
                    from config import set_browser_mode
                    set_browser_mode(parts[1])
                    if parts[1] == "user":
                        print_system("Режим браузера: [bold green]ВАШ[/bold green] — модель работает в вашем браузере через CDP.")
                    else:
                        print_system("Режим браузера: [bold yellow]ИЗОЛИРОВАННЫЙ[/bold yellow] — модель работает в Playwright Chromium.")
                elif parts[1] in ("chrome", "yandex", "edge", "brave", "auto"):
                    from config import set_browser_name, set_browser_mode
                    set_browser_name(parts[1])
                    set_browser_mode("user")
                    print_system(f"Браузер: [bold green]{parts[1]}[/bold green] (режим «ваш браузер» включён).")
                else:
                    print_error(f"Неизвестный вариант: {parts[1]}. Допустимые: user, isolated, chrome, yandex, edge, brave, auto")
                continue

            elif user_input.strip() == "/rag_toggle":
                from config import get_auto_rag, set_auto_rag
                current = get_auto_rag()
                new_state = not current
                set_auto_rag(new_state)
                
                status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
                print_system(f"Авто-RAG (индексация при запуске): {status}")
                
                if new_state:
                    print_system("RAG включится при следующем запуске Argent.")
                    print_system("If you want to start it now without restarting, type `/work` and the agent can use semantic search if it's already active.")
                else:
                    try:
                        from rag_engine import disable_rag
                        disable_rag()
                        print_system("Семантический поиск (RAG) отключён для текущей сессии.")
                    except ImportError:
                        pass
                continue

            elif user_input.strip() == "/kb_toggle":
                from config import get_auto_kb, set_auto_kb
                current = get_auto_kb()
                new_state = not current
                set_auto_kb(new_state)
                
                status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
                print_system(f"Авто-KB (загрузка внешних баз знаний при запуске): {status}")
                
                if new_state:
                    print_system("Внешние базы знаний загрузятся при следующем запуске.")
                continue

            elif user_input.strip() == "/auto_retrieve":
                from config import get_auto_retrieve, set_auto_retrieve
                new_state = not get_auto_retrieve()
                set_auto_retrieve(new_state)
                status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
                print_system(f"Авто-подстановка (результаты semantic_search в контекст на каждый запрос): {status}")
                if new_state:
                    print_system("Требует включённого RAG. Полезно слабым моделям, которые сами не догадываются искать в документации.")
                continue

            elif user_input.startswith("/skill import") or user_input.startswith("/skills import"):
                from skill_manager import skill_manager
                # split off the leading "/skill import" / "/skills import" verb
                arg = user_input.split("import", 1)[1].strip().strip('"')
                if not arg:
                    print_error(
                        "Использование: /skill import <источник>\n"
                        "  <source> can be:\n"
                        "    • a GitHub repo:        owner/repo  or  https://github.com/owner/repo\n"
                        "    • a specific skill:     https://github.com/owner/repo/tree/main/skills/<name>\n"
                        "    • a local path:         a SKILL.md folder, a SKILL.md, or a .md file"
                    )
                else:
                    print_system("Импортирую… (если источник удалённый — клонирую, это займёт время)")
                    print_system(skill_manager.import_skill(arg))
                continue
            elif user_input.startswith("/skills"):
                from skill_manager import skill_manager
                skills = skill_manager.list_skills()
                if not skills:
                    print_system("Навыков нет. Создайте инструментом `create_skill` или импортируйте: /skill import.")
                else:
                    print_system("[bold cyan]Доступные навыки:[/bold cyan]")
                    for skill in skills:
                        tag = " [dim](bundle)[/dim]" if skill.get("kind") == "folder" else ""
                        print_system(f"- [bold yellow]{skill['name']}[/bold yellow]{tag}: {skill['description']}")
                continue
            elif user_input.startswith("/goal"):
                from memory_manager import memory
                arg = user_input[len("/goal"):].strip()
                if not arg:
                    obj = (memory.data or {}).get("objective") or "[not set]"
                    task = (memory.data or {}).get("current_task") or "[not set]"
                    done = (memory.data or {}).get("completed") or []
                    files = (memory.data or {}).get("files_modified") or []
                    print_system("[bold cyan]Goal[/bold cyan]")
                    print_system(f"  OBJECTIVE: {obj}")
                    print_system(f"  ТЕКУЩАЯ ЗАДАЧА: {task}")
                    print_system(f"  PROGRESS: {len(done)} step(s) done, {len(files)} file(s) touched")
                    print_system("Задать цель: [bold]/goal <текст>[/bold], сбросить: [bold]/goal clear[/bold].")
                elif arg.lower() in ("clear", "reset", "done"):
                    memory.clear()
                    print_system("Цель и рабочая память очищены.")
                else:
                    memory.set_objective(arg)
                    print_system(f"Цель задана: [bold yellow]{arg}[/bold yellow]")
                continue

            elif user_input.startswith("/rooms"):
                arg = user_input[len("/rooms"):].strip()
                if not arg:
                    print_system("Использование: /rooms <задача> — экспериментальный движок «комнаты и рельсы».")
                    print_system("           /rooms list — список комнат;  /rooms show <имя> — граф комнаты;  /rooms resume — продолжить прерванный прогон.")
                    print_system("Пример: /rooms почини падающие тесты")
                    continue
                from src.rooms.session import build_library, default_journal, run_rooms
                lib = build_library()
                if lib.load_errors:
                    print_error(f"Комнаты не прошли валидацию: {lib.load_errors}")
                    continue

                low = arg.lower()
                if low == "list":
                    for name in lib.names():
                        st = lib.stats.get(name, {})
                        src = lib.sources.get(name, "?")
                        q = " [КАРАНТИН]" if st.get("quarantined") else ""
                        print_system(f"  {name} [{src}] — {st.get('successes', 0)}✓/{st.get('failures', 0)}✗{q}")
                    continue
                if low.startswith("show"):
                    name = arg[4:].strip()
                    room = lib.get(name)
                    if not room:
                        print_error(f"Нет комнаты '{name}'. /rooms list — список.")
                    else:
                        from src.rooms.library import describe_room
                        print_system(describe_room(room))
                    continue
                if low.startswith("spawn"):
                    from config import get_rooms_spawn, set_rooms_spawn
                    sub = arg[5:].strip().lower()
                    if sub in ("on", "off"):
                        set_rooms_spawn(sub == "on")
                        print_system(f"spawn_room (ИИ пишет новые комнаты): [bold]{'ON' if sub == 'on' else 'OFF'}[/bold]")
                    else:
                        print_system(f"spawn_room: [bold]{'ON' if get_rooms_spawn() else 'OFF'}[/bold] — ИИ может предлагать новые комнаты (через валидатор + твоё одобрение). Вкл: /rooms spawn on")
                    continue

                journal = default_journal()
                is_resume = arg.lower() == "resume"
                if is_resume and journal.last() is None:
                    print_system("Нет прерванного прогона для возобновления.")
                    continue
                print_system(f"[bold cyan]Rooms[/bold cyan]: {'возобновление' if is_resume else arg}")
                print_system(f"Комнаты: {', '.join(lib.names())}")
                if not questionary.confirm(
                    "Автономный прогон (реальные инструменты + суб-агенты). Продолжить?",
                    default=False,
                ).ask():
                    print_system("Отменено.")
                    continue

                def _human(node, state):
                    ans = questionary.text(f"[human_pause: {node.id}] {node.prompt or 'Ваш ввод:'}").ask()
                    if ans:
                        state.data[f"{node.id}.output"] = ans

                try:
                    result = run_rooms(
                        "" if is_resume else arg, library=lib, human_prompt=_human,
                        journal=journal, resume=is_resume,
                    )
                    print_system(f"[bold]Итог:[/bold] {result.outcome}")
                    print_system("Маршрут: " + " -> ".join(f"{r}:{e}" for r, e in result.history))
                    if result.outcome == "escalated":
                        print_system("Прогон эскалирован к человеку. При необходимости продолжите: /rooms resume")
                except Exception as e:
                    print_error(f"Запуск rooms не удался: {e}")
                continue
            elif user_input.startswith("/critic"):
                from memory_manager import memory
                from src.agent.critic import parse_critic_model
                from config import (
                    get_critic_auto, set_critic_auto, get_critic_model,
                    set_critic_model, get_critic_provider, set_critic_provider,
                )
                arg = user_input[len("/critic"):].strip()
                low = arg.lower()
                if low in ("on", "off"):
                    set_critic_auto(low == "on")
                    print_system(f"Auto-critic before /commit is now: [bold]{'ON' if low == 'on' else 'OFF'}[/bold]")
                elif low == "status":
                    m = get_critic_model() or "(current model)"
                    p = get_critic_provider() or "(current provider)"
                    print_system(f"Auto-critic: [bold]{'ON' if get_critic_auto() else 'OFF'}[/bold] | model: {m} | provider: {p}")
                elif low == "model" or low.startswith("model "):
                    spec = arg[len("model"):].strip()
                    if not spec:
                        m = get_critic_model() or "(current model)"
                        p = get_critic_provider() or "(current provider)"
                        print_system(f"Critic model: [bold]{m}[/bold] | provider: {p}")
                        print_system("Set with [bold]/critic model <name>[/bold] or [bold]/critic model <provider>:<model>[/bold]; reset with [bold]/critic model clear[/bold].")
                    elif spec.lower() == "clear":
                        set_critic_model("")
                        set_critic_provider("")
                        print_system("Модель критика сброшена на текущую.")
                    else:
                        prov, mdl = parse_critic_model(spec)
                        set_critic_model(mdl)
                        set_critic_provider(prov)
                        print_system(f"Модель критика: [bold yellow]{mdl}[/bold yellow]" + (f" (provider: {prov})" if prov else ""))
                else:
                    target = arg
                    what = "plan / idea"
                    if not target:
                        target = next(
                            (m.get("content") for m in reversed(getattr(agent, "messages", []) or [])
                             if m.get("role") == "assistant" and (m.get("content") or "").strip()),
                            "",
                        ).strip()
                        what = "assistant's latest plan / answer"
                    if not target:
                        print_error("Использование: /critic <план/идея> | /critic on|off | /critic model <имя> | /critic status")
                    else:
                        from agent import run_plan_critique
                        goal = memory.data.get("objective") or ""
                        print_system("[dim]Запускаю независимого критика (чистый контекст, только чтение)…[/dim]")
                        try:
                            print_system(run_plan_critique(target, goal=goal, what=what) or "[critic returned nothing]")
                        except Exception as e:
                            print_error(f"Критик упал: {e}")
                continue
            elif user_input.startswith("/guard"):
                from config import get_command_guard, set_command_guard
                arg = user_input[len("/guard"):].strip().lower()
                if arg in ("off", "warn", "block"):
                    set_command_guard(arg)
                    print_system(f"Защита от опасных команд: [bold]{arg}[/bold]")
                    if arg == "block":
                        print_system("Catastrophic commands (rm -rf /, mkfs, fork bombs, curl|sh…) will be refused without prompting.")
                    elif arg == "off":
                        print_system("Остаётся только прежнее подтверждение разрушительных команд.")
                else:
                    print_system(f"Command risk-gate: [bold]{get_command_guard()}[/bold]  (off | warn | block)")
                    print_system("warn = confirm risky commands with the reason; block = refuse catastrophic ones; off = legacy. Set with [bold]/guard <level>[/bold].")
                continue
            # /plugin too: the interface calls them «плагины» everywhere, so
            # that is the word people reach for. It was typed and did not exist.
            elif user_input.startswith("/hooks") or user_input.startswith("/plugin"):
                parts = user_input.split(" ")
                if len(parts) == 1:
                    status = "ENABLED" if get_autonomous_plugins_enabled() else "DISABLED"
                    print_system(f"Каталог плагинов: [bold cyan]{get_hooks_dir()}[/bold cyan]")
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
                            print_system("Плагинов в каталоге нет.")
                    else:
                        print_error(f"Каталог плагинов не найден: {hooks_dir}")
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
                        print_system(f"Автономное создание плагинов: [bold yellow]{status}[/bold yellow]")
                else:
                    new_path = user_input.split(" ", 1)[1].strip()
                    set_hooks_dir(new_path)
                    hook_manager.reload_plugins(new_path)
                    print_system(f"Каталог плагинов изменён: [bold green]{new_path}[/bold green]")
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
                name = parts[1].strip() if len(parts) > 1 else None
                export_chat_history(agent, filename=name, auto=False)
                # The same name labels the session, so /load can take it back.
                try:
                    saved = save_session(agent.messages, {
                        "model": agent.model_name,
                        "provider": get_provider(),
                    }, session_id=agent.session_id, label=name)
                    if saved:
                        agent.session_id = saved
                        print_system(f"Сессия сохранена: {saved}"
                                     + (f" (метка: {name})" if name else ""))
                    else:
                        print_system("[dim]Сессия не сохранена — в истории нет "
                                     "ни одного вашего сообщения.[/dim]")
                except Exception as e:
                    print_error(f"Не удалось сохранить сессию: {e}")
                continue
            elif user_input.strip().startswith("/sessions"):
                query = user_input.strip()[len("/sessions"):].strip()
                sessions = list_sessions(query or None)
                if not sessions:
                    print_system(f"Ничего не найдено по запросу '{query}'." if query
                                 else "Сохранённых сессий нет.")
                else:
                    header = (f"[bold cyan]Сессии по запросу '{query}':[/bold cyan]"
                              if query else "[bold cyan]Saved Sessions:[/bold cyan]")
                    console.print(header)
                    here = os.path.normcase(os.getcwd())
                    for i, s in enumerate(sessions[:20]):
                        date = (s.get("saved_at") or "")[:16]
                        model = s.get("model") or "?"
                        preview = s.get("preview") or ""
                        count = s.get("message_count") or 0
                        label = f" [magenta]{s['label']}[/magenta]" if s.get("label") else ""
                        cwd = s.get("cwd") or ""
                        # Only the odd one out is worth the width — the project
                        # you are in right now needs no announcement.
                        where = ("" if not cwd or os.path.normcase(os.path.abspath(cwd)) == here
                                 else f" [dim]@{Path(cwd).name}[/dim]")
                        console.print(f"  [dim][{i+1}][/dim] {date} [dim]|[/dim] [cyan]{model}[/cyan]"
                                      f"{label}{where} [dim]|[/dim] {count} msgs "
                                      f"[dim]|[/dim] [italic]\"{preview}\"[/italic]")
                    print_system("Восстановить: /load <номер | метка | часть текста>")
                continue
            elif user_input.startswith("/load"):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2:
                    print_error("Usage: /load <номер | метка | часть текста>")
                    continue
                target = find_session(parts[1])
                if target is None:
                    print_error(f"Сессия '{parts[1]}' не найдена. Список: /sessions")
                    continue
                restore_session(agent, target)
                continue

            elif user_input.startswith("/copy"):
                try:
                    import pyperclip
                except ImportError:
                    print_error("Модуль pyperclip не установлен (возможно это headless/SSH среда).")
                    continue
                blocks = get_code_blocks()
                parts = user_input.strip().split()
                if not blocks:
                    print_system("В последнем ответе нет блоков кода.")
                elif len(parts) < 2:
                    print_system("Использование: /copy <номер>")
                    print_system(f"Available blocks: {', '.join(f'[{b['index']}] {b['lang']}' for b in blocks)}")
                else:
                    try:
                        idx = int(parts[1])
                        block = next((b for b in blocks if b['index'] == idx), None)
                        if block:
                            pyperclip.copy(block['code'])
                            print_system(f"Copied block [{idx}] ({block['lang']}, {len(block['code'].splitlines())} lines) to clipboard.")
                        else:
                            print_error(f"Блок {idx} не найден. Доступно: 1-{len(blocks)}")
                    except ValueError:
                        print_error("Использование: /copy <номер>")
                continue

            elif user_input.startswith("/logs"):
                try:
                    log_dir = Path.home() / ".argent" / "logs"
                except Exception as e:
                    print_error(f"Не удалось определить домашнюю директорию: {e}")
                    continue
                parts = user_input.strip().split()
                
                if len(parts) > 1 and parts[1] == "clear":
                    for f in log_dir.glob("*.log"):
                        f.write_text("", encoding="utf-8")
                    print_system("Все логи очищены.")
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
                    print_system("Файлов логов нет в ~/.argent/logs/")
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
                    print_system("Записей в логах нет.")
                continue
            
            elif user_input.strip() == "/stats":
                from config import (
                    get_context_window, get_model_size_category,
                    get_model_category_override
                )
                
                model = get_current_model()
                provider = get_provider()
                ctx = get_context_window()
                msg_count = len(getattr(agent, "messages", []) or [])
                
                category = get_model_size_category(model)
                override = get_model_category_override()
                cat_source = "manual" if override else "auto"
                
                mcp_servers = mcp_client.get_servers()
                active_mcp = [s.get('name') for s in mcp_servers if s.get('running') and s.get('name')]
                
                plugins = list(hook_manager.plugins.keys()) if isinstance(getattr(hook_manager, "plugins", None), dict) else []
                
                stats_msg = (
                    f"[bold cyan]Argent Diagnostics:[/bold cyan]\n"
                    f"  [dim]Directory:[/dim] {os.getcwd()}\n"
                    f"  [dim]Model:[/dim] {model} ({provider})\n"
                    f"  [dim]Classification:[/dim] {category} ({cat_source})\n"
                    f"  [dim]Context Window:[/dim] {ctx} tokens\n"
                    f"  [dim]History:[/dim] {msg_count} messages\n"
                )

                # Context budget breakdown — where the prompt tokens go.
                try:
                    b = agent.get_context_breakdown()
                    tools_note = f" ({b.get('tool_count', 0)} tools)" if b.get('tool_count') else " (in-prompt catalog)"
                    percent = b.get('percent', 0)
                    stats_msg += (
                        f"  [dim]Context budget:[/dim] {b.get('total', 0)}/{b.get('max', 0)} tokens ({percent:.0f}%)\n"
                        f"    [dim]- system prompt:[/dim] {b.get('system', 0)} tok\n"
                        f"    [dim]- tool schemas:[/dim] {b.get('tools', 0)} tok{tools_note}\n"
                        f"    [dim]- history:[/dim] {b.get('history', 0)} tok\n"
                    )
                except Exception:
                    pass

                if active_mcp:
                    stats_msg += f"  [dim]MCP Servers:[/dim] {', '.join(active_mcp)}\n"
                if plugins:
                    stats_msg += f"  [dim]Plugins:[/dim] {', '.join(plugins)}\n"
                    
                print_system(stats_msg)
                continue

            elif user_input.startswith("/cd"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_system(f"Рабочая директория: [bold cyan]{os.getcwd()}[/bold cyan]")
                    print_system("Использование: /cd <путь>")
                    continue
                target_dir = parts[1].strip()
                try:
                    resolved = Path(target_dir).expanduser().resolve()
                    if not resolved.exists():
                        print_error(f"Директории не существует: {resolved}")
                        continue
                    if not resolved.is_dir():
                        print_error(f"Это не директория: {resolved}")
                        continue
                    os.chdir(resolved)
                    print_system(f"Рабочая директория изменена: [bold green]{os.getcwd()}[/bold green]")
                except Exception as e:
                    print_error(f"Не удалось сменить директорию: {e}")
                continue

            elif user_input.startswith("/undo"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Использование: /undo <путь_к_файлу>")
                    continue
                result = undo(parts[1].strip())
                print_system(result)
                continue

            elif user_input.startswith("/diff"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Использование: /diff <путь_к_файлу>")
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
                    print_system("В этой сессии файлы не менялись.")
                else:
                    print_system("[bold cyan]Tracked File Changes:[/bold cyan]")
                    for ch in changes:
                        print_system(f"  - {ch['key']} ({ch['snapshot_count']} snapshots)")
                    print_system("\nПосмотреть изменения: /diff <путь>, откатить: /undo <путь>.")
                continue

            elif user_input.startswith("/tasks"):
                handle_tasks_command(user_input, agent)
                continue

            elif user_input.strip() == "/rewind":
                from src.agent.checkpoints import list_checkpoints, rewind_to, CheckpointError
                cps = list_checkpoints(15)
                if not cps:
                    print_system("Нет чекпоинтов Argent в истории текущей ветки. "
                                 "Они создаются автоматически перед первой правкой каждого хода.")
                    continue
                choices = [f"{c['sha']}  {c['label']}  ({c['age']})" for c in cps] + ["❌ Отмена"]
                sel = questionary.select("Откатить рабочее дерево к какому чекпоинту?",
                                         choices=choices).ask()
                if not sel or sel.startswith("❌"):
                    continue
                sha = sel.split()[0]
                confirmed = questionary.confirm(
                    f"Откатить ВСЁ до чекпоинта {sha}? (незакоммиченное будет сохранено в git stash)",
                    default=False,
                ).ask()
                if not confirmed:
                    continue
                try:
                    print_system(rewind_to(sha))
                except CheckpointError as e:
                    print_error(str(e))
                continue

            elif user_input.startswith("/project"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Опишите проект. Пример: /project Сделай игру «Змейка» на Python")
                    continue
                proj_prompt = parts[1].strip()
                is_project_mode, user_input = orchestrator.start_project(proj_prompt)
                

                
            elif user_input.startswith("/work"):
                parts = user_input.split(" ", 1)
                if len(parts) < 2:
                    print_error("Опишите задачу. Пример: /work Исправь null reference в player.py")
                    continue
                work_prompt = parts[1].strip()
                
                auto_mode = False
                if work_prompt.startswith("--auto"):
                    auto_mode = True
                    work_prompt = work_prompt.replace("--auto", "", 1).strip()
                    if not work_prompt:
                        print_error("После --auto нужно описать задачу.")
                        continue
                
                is_project_mode, user_input = orchestrator.start_work(work_prompt, auto_mode=auto_mode)

            elif user_input.startswith("/commit"):
                try:
                    git_check = run_text("git rev-parse --git-dir", capture_output=True)
                    if git_check.returncode != 0:
                        print_error("Это не git-репозиторий. Перейдите в проект с git.")
                        continue

                    staged_diff = run_text(["git", "diff", "--cached"], capture_output=True).stdout
                    if not staged_diff.strip():
                        print_error("No staged changes found. Use 'git add' first.")
                        continue

                    from config import get_critic_auto
                    if get_critic_auto():
                        from agent import run_plan_critique
                        from memory_manager import memory
                        print_system("[dim]Критик смотрит подготовленный дифф перед коммитом…[/dim]")
                        try:
                            review = run_plan_critique(
                                staged_diff, goal=memory.data.get("objective") or "",
                                what="staged git diff (about to be committed)",
                            )
                            print_system("[bold magenta]── Critic ──[/bold magenta]")
                            print_system(review or "[critic returned nothing]")
                        except Exception as e:
                            print_error(f"Критик упал (коммит продолжается): {e}")

                    print_system("Generating commit message based on staged changes...")
                    
                    commit_prompt = (
                        "You are a Senior Developer. Generate a concise, professional Git commit message following Conventional Commits "
                        "specification based on the following diff. Only output the commit message, nothing else.\n\n"
                        f"{staged_diff}"
                    )
                    
                    gen_message = ""
                    try:
                        from providers import create_service_provider
                        provider, svc_model = create_service_provider()
                        res = provider.sync_chat(
                            model=svc_model,
                            messages=[{"role": "user", "content": commit_prompt}]
                        )
                        if isinstance(res, str):
                            gen_message = res.strip().strip('"').strip("'")
                        elif isinstance(res, dict):
                            gen_message = res.get("text", str(res))
                        elif res:
                            gen_message = str(res)
                    except Exception as e:
                        print_error(f"Не удалось сгенерировать сообщение коммита: {e}")
                        continue
                    
                    if not gen_message:
                        print_error("Не удалось сгенерировать сообщение коммита.")
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
                            print_system("Коммит создан с вашим сообщением.")
                        else:
                            print_system("Commit aborted.")
                            
                except FileNotFoundError:
                    print_error("Git не установлен или не найден в PATH.")
                except Exception as e:
                    print_error(f"Ошибка при коммите: {e}")
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
                    result = run_text(
                        cmd_parts,
                        shell=False,
                        capture_output=True,
                    )
                    out = result.stdout.strip()
                    err = result.stderr.strip()
                    
                    if out:
                        print(f"{out}")
                    if err:
                        print_error(f"{err}")
                        
                    if result.returncode != 0:
                        print_error(f"Команда завершилась с кодом {result.returncode}.")
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
                    print_error(f"Не удалось выполнить команду: {e}")
                    continue
                
            if active_tools is None:
                if is_project_mode:
                    active_tools = orchestrator.get_active_tools() or chat_allowed_tools()
                else:
                    # Regular chat mode
                    active_tools = chat_allowed_tools()

            # Sync the approval policy with the current mode: in autonomous and
            # vibe modes safe actions are auto-approved, destructive ones still prompt.
            approval.set_policy(approval.POLICY_AUTO if (is_auto_mode or vibe_mode) else approval.POLICY_ASK)

            # Reflect the active mode in the status bar.
            ui_state["mode"] = ("AUTO" if is_auto_mode
                                else "PROJECT" if is_project_mode
                                else "VIBE" if vibe_mode
                                else "CHAT")

            response_chunks = agent.process_user_input(user_input, allowed_tools=active_tools)
            stream_res = render_response_stream(
                agent, response_chunks, is_auto_mode=is_auto_mode
            )
            if isinstance(stream_res, tuple) and len(stream_res) >= 4:
                streamed_text, is_auto_mode, auto_sleep_time, auto_wake_context = stream_res[:4]
            else:
                streamed_text = str(stream_res)
                auto_sleep_time = 0
                auto_wake_context = ""
            
            # Show context usage after response
            usage = agent.get_context_usage()
            print_context_usage(usage["tokens"], usage["max"], usage["percent"])
            
            # Auto-save every 5 turns, back over the SAME session — otherwise a
            # long conversation files a new snapshot of itself every five turns
            # and crowds the other forty-nine out of the store.
            turn_counter += 1
            if turn_counter % 5 == 0:
                try:
                    saved_id = save_session(getattr(agent, "messages", []), {
                        "model": getattr(agent, "model_name", "unknown"),
                        "provider": get_provider(),
                    }, session_id=getattr(agent, "session_id", None))
                    if saved_id:
                        agent.session_id = saved_id
                except Exception as e:
                    from logger import get_logger
                    get_logger("session").warning("auto-save failed: %s", e)
            
            # Trigger Post Response Hook
            if getattr(agent, "messages", None) and agent.messages[-1].get("role") in ("assistant", "model"):
                hook_manager.call_hook("post_response", agent.messages[-1].get("content", ""))
                    
            # === Project Brain: State Machine ===
            if is_project_mode:
                step_res = orchestrator.step()
                if isinstance(step_res, tuple) and len(step_res) == 2:
                    is_project_mode, auto_continue_input = step_res
            
        except KeyboardInterrupt:
            # Heal the history: an interrupted turn may have left tool_calls
            # without matching tool results, which would poison the next request.
            try:
                if hasattr(agent, "repair_history"):
                    agent.repair_history()
            except Exception as e:
                print_error(f"Не удалось восстановить историю: {e}")
            if is_auto_mode:
                is_auto_mode = False
                print_system("\n[bold yellow]Выполнение прервано пользователем (Ctrl+C). Выход из автоматического режима.[/bold yellow]")
            continue
        except EOFError:
            export_chat_history(agent, auto=True)
            break
        except Exception as e:
            print_error(f"Ошибка главного цикла: {e}")
            continue
    
    print_system("Goodbye!")

if __name__ == "__main__":
    main()