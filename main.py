import sys
import os
import time
import atexit
import warnings

# Suppress noisy dependency mismatch warning from requests/urllib3 in user environment
warnings.filterwarnings("ignore", message=r".*urllib3.*or chardet.*charset_normalizer.*")

from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import prompt as ptk_prompt # Renamed to avoid conflict with our own prompt
from src.cli.cli_prompt import build_prompt_session
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
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
    print_context_usage, print_user_prompt, print_banner, s, c
)
from rich.markup import escape
from command_handler import export_chat_history, handle_slash_command
from hook_manager import hook_manager
from rag_engine import enable_rag_for_project, disable_rag
from pathlib import Path
from session import (
    save_session, load_session, list_sessions, delete_session, get_last_session,
    find_session, format_relative_time, format_short_path,
)
from file_tracker import snapshot, get_diff, undo, get_pending_changes, undo_all
from src.cli.cli_ui import render_response_stream
from src.cli.cd_path import clean_cd_path
from src.agent.shell import run_text
from src.startup import (
    carried_over_memory, offer_first_run_setup, offer_previous_context, restore_session,
)


def chat_allowed_tools() -> list:
    """The toolset for a regular chat turn: all available tools."""
    from tools.schemas import get_available_tools
    from config import get_disabled_tools
    disabled = set(get_disabled_tools() or [])
    tools = get_available_tools()
    if isinstance(tools, dict):
        tools = list(tools.keys())
    return [name for name in (tools or []) if name not in disabled]



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
            _ps(git_checkpoint(f"перед задачей: {task[:60]}"))
    except Exception:
        pass


def is_vibe_mode() -> bool:
    """Returns True if autonomous vibe mode is active."""
    return approval.get_policy() == approval.POLICY_AUTO


def set_vibe_mode(enabled: bool, notify: bool = True) -> bool:
    """Explicitly enables or disables vibe mode, keeping approval policy,
    auto-checkpoints, and connected bridges (Telegram) in sync."""
    target_mode = bool(enabled)
    from src.agent.checkpoints import set_auto_checkpoint
    if target_mode:
        set_auto_checkpoint(True)
        approval.set_policy(approval.POLICY_AUTO)
        if notify:
            print_system(
                "🌴 Vibe-режим ВКЛ: безопасные действия одобряются автоматически, "
                "деструктивные — отклоняются во избежание потери данных, "
                "перед первой правкой каждого хода создаётся чекпоинт. Откат: /rewind."
            )
    else:
        approval.set_policy(approval.POLICY_ASK)
        if notify:
            print_system("Vibe-режим ВЫКЛ: обычный режим подтверждений возвращён.")

    # Sync with shared Telegram bridge if active
    try:
        from src.telegram.bot import telegram_bridge
        if getattr(telegram_bridge, "_terminal_session", None) is not None:
            telegram_bridge._terminal_session.vibe = target_mode
    except Exception:
        pass

    return target_mode


def toggle_vibe_mode(vibe_mode: bool | None = None) -> bool:
    """The vibecoder switch. Curates EXISTING knobs — no new machinery:
    auto-approve safe actions, auto-deny destructive ones, and guarantee
    per-turn checkpoints, so every step stays rewindable via /rewind."""
    if vibe_mode is None:
        vibe_mode = is_vibe_mode()
    return set_vibe_mode(not vibe_mode)


def parse_vibe_command(cmd: str) -> tuple[str, str | None] | None:
    """Parse a /vibe command string into (action, payload) or None if not a /vibe command.

    Actions:
    - 'toggle': '/vibe' or '/vibe   ' -> payload is None
    - 'on': '/vibe on', '/vibe 1', '/vibe true' -> payload is None
    - 'off': '/vibe off', '/vibe 0', '/vibe false' -> payload is None
    - 'task': '/vibe <task text>' -> payload is task text
    """
    if not cmd:
        return None
    stripped = cmd.strip()
    if stripped != "/vibe" and not stripped.startswith(("/vibe ", "/vibe\t")):
        return None

    rest = stripped[5:].strip()
    if not rest:
        return "toggle", None
    norm = rest.lower()
    if norm in ("on", "1", "true"):
        return "on", None
    if norm in ("off", "0", "false"):
        return "off", None
    return "task", rest


def main():
    if "--tui" in sys.argv:
        try:
            import argent_tui
            from src.bridge.repl_bridge import ReplBridge
            bridge = ReplBridge()
            argent_tui.run_tui(bridge)
            return
        except Exception as e:
            print_error(f"Не удалось запустить Rust TUI: {e}")
            print_error("Переключение на стандартный CLI...")

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
            from tools.command_ops import ACTIVE_PROCESSES, _kill_process_tree, _close_proc_pipes
            items = list(ACTIVE_PROCESSES.items()) if isinstance(ACTIVE_PROCESSES, dict) else list(ACTIVE_PROCESSES)
            for item in items:
                try:
                    proc_info = item[1] if isinstance(item, tuple) else item
                    proc = proc_info.get("process")
                    if proc:
                        _kill_process_tree(proc)
                        _close_proc_pipes(proc)
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
        # Stop Telegram bridge
        try:
            from src.telegram.bot import telegram_bridge
            if telegram_bridge.is_running:
                telegram_bridge.stop()
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
    print_banner(_ver)
    print_system("Argent Coder — автономная среда разработки.")
    print_system("Список команд: /help")
    
    agent = ArgentAgent()

    # Attach terminal agent to Telegram bridge for shared mode
    try:
        from src.telegram.bot import telegram_bridge
        telegram_bridge.attach_terminal_agent(agent)
    except Exception:
        pass
    
    def get_all_commands():
        # Tab completion: the shared catalog (registry + described subcommands)
        # plus plugin commands. The same catalog feeds the TUI.
        from src.cli.commands import completion_catalog
        all_set = {c["name"] for c in completion_catalog()}
        for name in hook_manager.get_custom_commands().keys():
            all_set.add(f"/{name}")
        return sorted(all_set)

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
    offer_first_run_setup(lambda command: handle_slash_command(command, agent))

    print_system(f"Провайдер: {get_provider().upper()}")
    print_system(f"Модель: {get_current_model()}")
    print_system(f"Рабочая директория: {os.getcwd()}")
    if not Path(".argent/AGENTS.md").exists() and not Path("AGENTS.md").exists():
        print_system("[dim]Подсказка: нет AGENTS.md — выполните /init, чтобы Argent изучил проект и создал память о нём.[/dim]")
    offer_previous_context(agent)

    # MCP servers spawn AFTER the session restore: offer_previous_context may
    # chdir to the saved session directory, and a server's CWD is resolved at
    # spawn time — spawning earlier would anchor it to the launch folder.
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

    # Trigger Startup Hook
    hook_manager.call_hook("on_startup")
    from src.agent.shell_hooks import session_started, SHELL_HOOKS
    _started_output = session_started()
    if _started_output:
        print_system(f"[dim]{_started_output}[/dim]")
    if SHELL_HOOKS.status()["untrusted"]:
        print_system("[yellow]Хуки проекта (.argent/hooks.json) не запускаются, пока вы им не доверите: "
                     "/hooks trust[/yellow]")

    # /tasks schedules fire while Argent is open, and nothing else runs the
    # schedule: the GUI server that used to is gone.
    try:
        from src.automation import notice
        from src.automation.scheduler import start_background_scheduler
        from src.automation.store import load_automations
        if start_background_scheduler(on_event=notice.enqueue_event) is not None:
            scheduled = [a for a in load_automations() if a.enabled]
            if scheduled:
                print_system(f"[dim]⏰ Автоматизации по расписанию: {len(scheduled)} "
                             f"(/tasks — сводка; запускаются, пока Argent открыт)[/dim]")
            digest = notice.startup_digest()
            if digest:
                print_system(digest)
    except Exception as e:
        print_system(f"[yellow]Планировщик /tasks не запущен: {e}[/yellow]")

    vibe_mode = False
    auto_sleep_time = 0
    auto_wake_context = ""
    auto_continue_input = None
    turn_counter = 0
    last_sigint_prompt_time = 0.0
    
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

    from config import get_telegram_enabled, get_telegram_token
    if get_telegram_enabled() and get_telegram_token():
        try:
            from src.telegram.bot import telegram_bridge
            ok, msg = telegram_bridge.start()
            if ok:
                print_system(f"Telegram-мост: [bold green]ВКЛ[/bold green] ({msg})")
        except Exception as e:
            print_system(f"[yellow]⚠ Не удалось запустить Telegram-мост: {e}[/yellow]")

    while True:
        try:
            print() # Visual spacing

            try:
                from src.automation.notice import drain_pending
                for line in drain_pending():
                    print_system(line)
            except Exception:
                pass

            if auto_sleep_time > 0 or isinstance(auto_wake_context, dict):
                import queue
                import re
                from src.agent.events import EVENT_QUEUE
                
                class EventInterruptedException(Exception):
                    def __init__(self, event):
                        self.event = event

                def make_event_sleeper(target_pids=None, is_condition_wait=False):
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
                                    pid = str(ev.get("pid", ""))
                                    exit_code = ev.get("exit_code", 0)

                                    # Filter out stale events for processes that were already completed
                                    # and had output retrieved by read_background_command earlier:
                                    try:
                                        from tools.command_ops import COMPLETED_PROCESSES, ACTIVE_PROCESSES_LOCK
                                        with ACTIVE_PROCESSES_LOCK:
                                            if pid in COMPLETED_PROCESSES:
                                                continue
                                    except Exception:
                                        pass

                                    # 1. If any background process failed with an error, ALWAYS wake the agent immediately
                                    if exit_code != 0:
                                        raise EventInterruptedException(ev)

                                    # 2. If waiting on a specific process condition:
                                    if target_pids:
                                        if pid in target_pids:
                                            raise EventInterruptedException(ev)
                                        # An unrelated process finished cleanly: ignore and continue waiting for target PID
                                        continue

                                    # 3. If waiting on a file or generic condition (is_condition_wait=True):
                                    # An unrelated process finished cleanly: do not wake model, let wait_for continue polling
                                    if is_condition_wait:
                                        continue

                                    # 4. If plain timer sleep (auto_sleep_time > 0):
                                    # Wake on any process completion
                                    raise EventInterruptedException(ev)
                            except queue.Empty:
                                pass
                    return sleep_with_events

                # Waiting ON a condition: poll locally instead of waking the
                # model to look. Each wake costs a full turn, so a ten-poll
                # "is it done yet?" loop becomes one sleep and one wake.
                if isinstance(auto_wake_context, dict):
                    spec, auto_wake_context = auto_wake_context, ""
                    until = spec.get("until", "condition")
                    timeout = spec.get("timeout", 60)
                    reason = spec.get("reason", "no reason")
                    from src.agent.wait_conditions import wait_for
                    from src.cli.cli_ui import _HeartbeatWaitDisplay
                    
                    # Extract target PIDs from condition if any (e.g. process_finished(6))
                    target_pids = set(re.findall(r'process_(?:finished|running)\(\s*["\']?\s*(\d+)\s*["\']?\s*\)', until))
                    condition_sleeper = make_event_sleeper(target_pids=target_pids if target_pids else None, is_condition_wait=True)
                    hb_display = _HeartbeatWaitDisplay(condition=until, timeout=timeout, target_pids=target_pids)

                    try:
                        with Live(hb_display, console=console, refresh_per_second=6, transient=True):
                            outcome = wait_for(until, timeout=timeout,
                                               sleep=condition_sleeper)
                    except EventInterruptedException as e:
                        ev = e.event
                        print_system(f"*[Событие] Фоновая команда {ev.get('pid', '?')} завершилась. Ожидание прервано.*")
                        auto_wake_context = f"[Heartbeat прерван] Фоновая команда {ev.get('pid', '?')} завершилась с кодом {ev.get('exit_code', '?')}. Проверьте её вывод с помощью read_background_command."
                        auto_sleep_time = 0
                    except KeyboardInterrupt:
                        print_system("Ожидание прервано пользователем (Ctrl+C). Фоновые процессы продолжают работу в фоне.")
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
                            from src.agent.wait_conditions import condition_wake_message
                            auto_wake_context = condition_wake_message(
                                until, reason, outcome["met"], out_reason, waited)
                        else:
                            print_error("Ошибка wait_for: возвращен неверный формат")
                            auto_wake_context = "[Heartbeat пробуждение] Ошибка: проверка условия вернула неверный формат. Проверьте результат сами."

                if auto_sleep_time > 0:
                    from src.cli.cli_ui import _HeartbeatWaitDisplay

                    timer_sleeper = make_event_sleeper(is_condition_wait=False)
                    hb_display = _HeartbeatWaitDisplay(timeout=auto_sleep_time, is_timer=True)
                    try:
                        # Chunked sleep so Ctrl+C aborts within a tick, not after
                        # the full (possibly very long) heartbeat delay.
                        with Live(hb_display, console=console, refresh_per_second=6, transient=True):
                            timer_sleeper(auto_sleep_time)
                    except EventInterruptedException as e:
                        ev = e.event
                        print_system(f"*[Событие] Фоновая команда {ev['pid']} завершилась. Ожидание прервано.*")
                        auto_wake_context = f"[Heartbeat прерван] Фоновая команда {ev['pid']} завершилась с кодом {ev['exit_code']}. Проверьте её вывод с помощью read_background_command."
                    except KeyboardInterrupt:
                        print_system("Состояние Heartbeat прервано пользователем (Ctrl+C).")
                        auto_sleep_time = 0
                        auto_wake_context = ""
                        continue
                    auto_sleep_time = 0
                
                auto_continue_input = auto_wake_context if auto_wake_context else "[Heartbeat завершен] Продолжай."
                auto_wake_context = ""

            from src.cli.input_queue import get_input_queue, get_live_input_state, TurnInputListener
            input_queue = get_input_queue()
            live_input_state = get_live_input_state()

            if auto_continue_input:
                user_input = auto_continue_input
                auto_continue_input = None
                print_user_prompt(user_input)
            elif input_queue.has_items():
                user_input = input_queue.pop()
                rem = input_queue.count()
                suffix = f" [dim]({rem} в очереди)[/dim]" if rem > 0 else ""
                print_user_prompt(f"[Очередь] {user_input}{suffix}")
            else:
                draft_default = live_input_state.clear() if live_input_state.current_text else ""
                try:
                    from prompt_toolkit.formatted_text import HTML
                    prompt_sym_col = c.get("user_prompt_symbol", "#58a6ff")
                    user_input = prompt_session.prompt(
                        HTML(f'<style fg="{prompt_sym_col}"><b>❯</b></style> '),
                        default=draft_default,
                    )
                    last_sigint_prompt_time = 0.0
                except KeyboardInterrupt:
                    now = time.time()
                    if now - last_sigint_prompt_time < 1.5:
                        print_system("\nВыход из Argent...")
                        export_chat_history(agent, auto=True)
                        break
                    last_sigint_prompt_time = now
                    print_system("[dim](Нажмите Ctrl+C ещё раз для выхода)[/dim]")
                    continue
                except EOFError:
                    export_chat_history(agent, auto=True)
                    break
                    
                if not user_input.strip():
                    continue
                    
                user_input = hook_manager.call_modifier_hook("pre_prompt", user_input)
                if not user_input.strip():
                    continue
                if not user_input.startswith(("/", "!")):
                    from src.agent.shell_hooks import prompt_submitted
                    user_input, hook_note = prompt_submitted(user_input)
                    if user_input is None:
                        print_system(f"[yellow]Сообщение остановлено хуком: {hook_note}[/yellow]")
                        continue

                if not user_input.startswith("/"):
                    from src.cli.cli_prompt import expand_file_mentions
                    expanded_input, attached_files = expand_file_mentions(user_input)
                    if attached_files:
                        print_system(f"[dim]📎 Прикреплены файлы: {', '.join('@' + f for f in attached_files)}[/dim]")
                        user_input = expanded_input

                # Reset project mode on new manual input
                is_project_mode = False
                project_iterations = 0
            
            active_tools = None

            # Dispatch slash commands and shell escapes via unified CommandRegistry
            if user_input.startswith("/") or user_input.startswith("!"):
                from src.cli.commands import dispatch_command, CommandContext
                ctx = CommandContext(
                    agent=agent,
                    user_input=user_input,
                    ui_state=ui_state,
                    turn_counter=turn_counter,
                    is_project_mode=is_project_mode,
                    project_iterations=project_iterations,
                    vibe_mode=vibe_mode,
                    prompt_session=prompt_session,
                )
                cmd_res = dispatch_command(user_input, ctx)
                if cmd_res.handled:
                    if cmd_res.turn_counter is not None:
                        turn_counter = cmd_res.turn_counter
                    if cmd_res.vibe_mode is not None:
                        vibe_mode = cmd_res.vibe_mode
                    if cmd_res.is_project_mode is not None:
                        is_project_mode = cmd_res.is_project_mode
                    if cmd_res.project_iterations is not None:
                        project_iterations = cmd_res.project_iterations
                    if cmd_res.should_exit:
                        break
                    if cmd_res.should_continue:
                        continue
                    if cmd_res.new_input:
                        user_input = cmd_res.new_input
                    if cmd_res.active_tools is not None:
                        active_tools = cmd_res.active_tools
                else:
                    if user_input.startswith("/"):
                        from command_handler import unknown_command_message
                        print_error(unknown_command_message(user_input))
                        continue

            if active_tools is None:
                # Regular chat / vibe mode: None allows agent.py to apply tier-based slimming
                active_tools = None

            # Sync the approval policy with the current mode: in vibe mode
            # safe actions are auto-approved, destructive ones are auto-denied.
            vibe_mode = is_vibe_mode()
            approval.set_policy(approval.POLICY_AUTO if vibe_mode else approval.POLICY_ASK)

            # Reflect the active mode in the status bar.
            ui_state["mode"] = ("VIBE" if vibe_mode else "CHAT")

            # If Telegram bridge is running in shared mode, notify that work has started
            try:
                from config import get_telegram_mode
                from src.telegram.bot import telegram_bridge
                if telegram_bridge.is_running and get_telegram_mode() == "shared":
                    telegram_bridge.notify_terminal_turn_started(user_input, wait=False)
            except Exception:
                pass

            with TurnInputListener(agent, state=live_input_state):
                response_chunks = agent.process_user_input(user_input, allowed_tools=active_tools)
                try:
                    # Companion mode: the phone watches the tools run.
                    from src.telegram.bot import telegram_bridge as _tg
                    if _tg.mirrors_terminal:
                        response_chunks = _tg.tee_chunks(response_chunks)
                except Exception:
                    pass
                stream_res = render_response_stream(
                    agent, response_chunks
                )

            if isinstance(stream_res, tuple) and len(stream_res) >= 4:
                streamed_text, _, auto_sleep_time, auto_wake_context = stream_res[:4]
            elif isinstance(stream_res, tuple) and len(stream_res) >= 3:
                streamed_text, auto_sleep_time, auto_wake_context = stream_res[0], stream_res[1], stream_res[2]
            else:
                streamed_text = str(stream_res)
                auto_sleep_time = 0
                auto_wake_context = ""

            if getattr(agent, "_stop_requested", False):
                agent.reset_stop()

            # Play completion sound if enabled
            try:
                from src.cli.sound import play_sound
                play_sound("done")
            except Exception:
                pass

            # If Telegram bridge is running in shared mode, send completion notification
            try:
                from config import get_telegram_mode
                from src.telegram.bot import telegram_bridge
                if telegram_bridge.is_running and get_telegram_mode() == "shared":
                    telegram_bridge.notify_terminal_turn_completed(user_input, streamed_text, wait=False)
            except Exception:
                pass
            
            # Show context usage after response

            usage = agent.get_context_usage()
            print_context_usage(usage["tokens"], usage["max"], usage["percent"])
            
            # Auto-save immediately on turn 1 and every 5 turns back over the SAME session,
            # so early crashes or exits never lose work.
            turn_counter += 1
            if turn_counter == 1 or turn_counter % 5 == 0:
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
            from src.agent.shell_hooks import turn_stopped
            stop_output = turn_stopped()
            if stop_output:
                print_system(f"[dim]{stop_output}[/dim]")

        except KeyboardInterrupt:
            try:
                from src.cli.sound import play_sound
                play_sound("error")
            except Exception:
                pass
            print_system("\n[dim yellow]⚡ Ход прерван пользователем (Ctrl+C)[/dim yellow]")
            # Heal the history: an interrupted turn may have left tool_calls
            # without matching tool results, which would poison the next request.
            try:
                if hasattr(agent, "repair_history"):
                    agent.repair_history()
            except Exception as e:
                print_error(f"Не удалось восстановить историю: {e}")
            continue
        except EOFError:
            export_chat_history(agent, auto=True)
            break
        except Exception as e:
            try:
                from src.cli.sound import play_sound
                play_sound("error")
            except Exception:
                pass
            print_error(f"Ошибка главного цикла: {e}")
            continue

    
    print_system("Goodbye!")

if __name__ == "__main__":
    main()
