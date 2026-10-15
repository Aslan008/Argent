"""Agent and environment command handlers for Argent CLI.
Includes /vibe, /tasks, /sound, /critic, /guard, /hooks, /tools, /logs, /stats,
/rag_toggle, /kb_toggle, /auto_retrieve, /skills, /goal, /rooms, /browser.
"""

import os
from pathlib import Path
import questionary
from rich.markup import escape

import approval
from config import (
    get_hooks_dir, set_hooks_dir, get_autonomous_plugins_enabled, set_autonomous_plugins_enabled,
    get_disabled_tools, set_disabled_tools,
    get_current_model, get_provider,
)
from hook_manager import hook_manager
from mcp_client import mcp_client
from memory_manager import memory
from src.agent.shell import run_text
from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import console, print_system, print_error


# --- Vibe Mode Helpers ---

def is_vibe_mode() -> bool:
    """Returns True if autonomous vibe mode is active."""
    return approval.get_policy() == approval.POLICY_AUTO


def set_vibe_mode(enabled: bool, notify: bool = True) -> bool:
    """Explicitly enables or disables vibe mode, keeping approval policy,
    auto-checkpoints, and connected bridges in sync."""
    target_mode = bool(enabled)
    from src.agent.checkpoints import set_auto_checkpoint
    if target_mode:
        set_auto_checkpoint(True)
        approval.set_policy(approval.POLICY_AUTO)
        if notify:
            print_system(
                "Vibe-режим ВКЛ: безопасные действия одобряются автоматически, "
                "деструктивные — отклоняются во избежание потери данных, "
                "перед первой правкой каждого хода создаётся чекпоинт. Откат: /rewind."
            )
    else:
        approval.set_policy(approval.POLICY_ASK)
        if notify:
            print_system("Vibe-режим ВЫКЛ: обычный режим подтверждений возвращён.")

    try:
        from src.telegram.bot import telegram_bridge
        if getattr(telegram_bridge, "_terminal_session", None) is not None:
            telegram_bridge._terminal_session.vibe = target_mode
    except Exception:
        pass

    return target_mode


def toggle_vibe_mode(vibe_mode: bool | None = None) -> bool:
    if vibe_mode is None:
        vibe_mode = is_vibe_mode()
    return set_vibe_mode(not vibe_mode)


def parse_vibe_command(cmd: str) -> tuple[str, str | None] | None:
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


def offer_safety_checkpoint(task: str) -> None:
    try:
        is_git = run_text("git rev-parse --is-inside-work-tree", shell=True, capture_output=True)
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
            print_system(git_checkpoint(f"перед задачей: {task[:60]}"))
    except Exception:
        pass


@registry.register("/vibe", help_text="Включить/выключить автономный vibe-режим", category="agent", is_prefix=True)
def handle_vibe(ctx: CommandContext) -> CommandResult:
    res = parse_vibe_command(ctx.user_input)
    if res is None:
        return CommandResult.continue_loop()
    vibe_action, vibe_payload = res

    ui_state = ctx.ui_state if isinstance(ctx.ui_state, dict) else {}

    if vibe_action == "on":
        vibe_mode = set_vibe_mode(True)
        ui_state["mode"] = "VIBE"
        return CommandResult.continue_loop(vibe_mode=vibe_mode)
    elif vibe_action == "off":
        vibe_mode = set_vibe_mode(False)
        ui_state["mode"] = "CHAT"
        return CommandResult.continue_loop(vibe_mode=vibe_mode)
    elif vibe_action == "toggle":
        vibe_mode = toggle_vibe_mode(ctx.vibe_mode)
        ui_state["mode"] = "VIBE" if vibe_mode else "CHAT"
        return CommandResult.continue_loop(vibe_mode=vibe_mode)
    elif vibe_action == "task":
        task_text = vibe_payload
        vibe_mode = set_vibe_mode(True)
        ui_state["mode"] = "VIBE"
        offer_safety_checkpoint(task_text)
        return CommandResult.pass_to_agent(new_input=task_text, vibe_mode=vibe_mode)

    return CommandResult.continue_loop()


# --- Tasks Command ---

def _parse_task_tools(spec: str):
    from tools.schemas import AVAILABLE_TOOLS
    spec = (spec or "").strip()
    if not spec:
        return [], []
    if spec.lower().startswith("tools:"):
        spec = spec.split(":", 1)[1]
    names = [n.strip() for n in spec.replace(";", ",").split(",") if n.strip()]
    unknown = [n for n in names if not isinstance(AVAILABLE_TOOLS, dict) or n not in AVAILABLE_TOOLS]
    return names, unknown


def handle_tasks_command(user_input: str, agent=None) -> None:
    """`/tasks ...` — the work lives in src/automation/tasks_cli.py; this only
    supplies the host's printers and the running scheduler (if any)."""
    from src.automation import tasks_cli
    from src.automation.scheduler import get_scheduler
    tasks_cli.handle(user_input, out=print_system, err=print_error, live=get_scheduler())


@registry.register("/tasks", help_text="Автоматизации по расписанию: сводка, add, cmd, remind, new, run, inbox, pause",
                   category="agent", is_prefix=True)
def handle_tasks(ctx: CommandContext) -> CommandResult:
    handle_tasks_command(ctx.user_input, ctx.agent)
    return CommandResult.continue_loop()


@registry.register("/remind", help_text="Напоминание в нужное время: /remind через 2 часа | позвонить Ане",
                   category="agent", is_prefix=True)
def handle_remind(ctx: CommandContext) -> CommandResult:
    argument = ctx.user_input.strip()[len("/remind"):].strip()
    handle_tasks_command(f"/tasks remind {argument}" if argument else "/tasks remind", ctx.agent)
    return CommandResult.continue_loop()


# --- Sound Command ---

@registry.register("/sound", help_text="Управление звуковыми эффектами", category="agent", is_prefix=True)
def handle_sound(ctx: CommandContext) -> CommandResult:
    from src.cli.sound import handle_sound_command
    handle_sound_command(ctx.user_input)
    return CommandResult.continue_loop()


# --- Critic Command ---

@registry.register("/critic", help_text="Независимый критик планов и кода", category="agent", is_prefix=True)
def handle_critic(ctx: CommandContext) -> CommandResult:
    from src.agent.critic import parse_critic_model
    from config import (
        get_critic_auto, set_critic_auto, get_critic_model,
        set_critic_model, get_critic_provider, set_critic_provider,
    )
    user_input = ctx.user_input
    agent = ctx.agent
    arg = user_input[len("/critic"):].strip()
    low = arg.lower()

    if low in ("on", "off"):
        set_critic_auto(low == "on")
        print_system(f"Auto-critic before /commit is now: [bold]{'ON' if low == 'on' else 'OFF'}[/bold]")
        return CommandResult.continue_loop()
    elif low == "status":
        m = get_critic_model() or "(current model)"
        p = get_critic_provider() or "(current provider)"
        print_system(f"Auto-critic: [bold]{'ON' if get_critic_auto() else 'OFF'}[/bold] | model: {m} | provider: {p}")
        return CommandResult.continue_loop()
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
        return CommandResult.continue_loop()
    else:
        target = arg
        what = "plan / idea"
        if not target:
            for m in reversed(getattr(agent, "messages", []) or []):
                if m.get("role") == "assistant":
                    c = (m.get("content") or "").strip()
                    if c:
                        target = c
                        break
                    th = (m.get("thinking") or "").strip()
                    if th:
                        target = th
                        break
            what = "assistant's latest plan / answer"
        if not target:
            plan_path = Path(".argent") / "artifacts" / "implementation_plan.md"
            if plan_path.is_file():
                try:
                    target = plan_path.read_text(encoding="utf-8", errors="replace").strip()
                    what = "implementation_plan.md artifact"
                except Exception:
                    pass
        if not target:
            print_error("В истории нет сообщений ассистента и артефактов для критики. Укажите план явно: /critic <текст>")
        else:
            from agent import run_plan_critique
            from src.agent.critic import parse_critic_verdict, CriticVerdict
            goal = memory.data.get("objective") or ""
            print_system("[dim]Запускаю независимого критика (чистый контекст, только чтение)…[/dim]")
            try:
                review = run_plan_critique(target, goal=goal, what=what)
                verdict = parse_critic_verdict(review)
                verdict_badge = {
                    CriticVerdict.SOLID: "[bold green]VERDICT: SOLID[/bold green]",
                    CriticVerdict.FIXABLE: "[bold yellow]VERDICT: FIXABLE[/bold yellow]",
                    CriticVerdict.RETHINK: "[bold red]VERDICT: RETHINK[/bold red]",
                }.get(verdict, "")
                if verdict_badge:
                    print_system(f"[bold #79c0ff]── Critic ──[/bold #79c0ff] {verdict_badge}")
                print_system(review or "[critic returned nothing]")
            except Exception as e:
                print_error(f"Критик упал: {e}")

    return CommandResult.continue_loop()


# --- Ultracode / Advisor Commands ---

@registry.register("/ultracode", help_text="Ultracode: большая задача планом, с проверками и критиком",
                   category="agent", is_prefix=True)
def handle_ultracode(ctx: CommandContext) -> CommandResult:
    import config
    from src.agent.workflow import calibration, runtime, store
    arg = ctx.user_input.strip()[len("/ultracode"):].strip()
    low = arg.lower()
    word, _, rest = arg.partition(" ")
    word, rest = word.lower(), rest.strip()

    if not arg or word == "status":
        state = "[bold green]ВКЛ[/bold green]" if runtime.enabled() else "[bold]выкл[/bold]"
        level = config.get_ultracode_level()
        print_system(f"Ultracode: {state} (только в этой сессии) · уровень: {level} · "
                     f"лимит времени: {config.get_ultracode_max_minutes()} мин")
        print_system(runtime.status())
        if not runtime.enabled():
            print_system("[dim]/ultracode on — включить; /ultracode run <задача> — запустить сразу[/dim]")
    elif word == "on":
        runtime.set_enabled(True)
        from src.agent.workflow import advisor as adv
        target = adv.resolve(config.get_provider(), config.get_current_model())
        print_system("Ultracode [bold green]включён[/bold green]: для больших задач модель запустит воркфлоу сама "
                     "(или /ultracode run <задача>).")
        print_system(f"Советник: [bold]{target.label}[/bold]"
                     + ("" if target.configured else " (не настроен — советует основная модель: "
                                                    "/advisor model <провайдер>:<модель>)"))
    elif word == "off":
        runtime.set_enabled(False)
        print_system("Ultracode выключен.")
    elif word == "run":
        if not rest:
            print_error("Использование: /ultracode run <задача целиком, с проверками>")
        else:
            runtime.set_enabled(True)
            print_system(runtime.start(rest))
    elif word in ("stop", "pause"):
        print_system(runtime.stop())
    elif word == "resume":
        print_system(runtime.resume(rest))
    elif word == "level" and rest:
        value = rest.lower()
        if value == "auto" or (value.isdigit() and 0 <= int(value) <= 3):
            config.set_ultracode_level(value if value == "auto" else int(value))
            names = ", ".join(f"{k} — {v}" for k, v in sorted(calibration.LEVEL_NAMES.items(), reverse=True))
            print_system(f"Уровень Ultracode: [bold]{value}[/bold] ({names})")
        else:
            print_error("Уровень: auto, 0, 1, 2 или 3.")
    elif word in ("budget", "minutes") and rest.isdigit():
        config.set_ultracode_max_minutes(int(rest))
        print_system(f"Лимит времени воркфлоу: {config.get_ultracode_max_minutes()} мин.")
    elif word == "calibrate":
        profile = calibration.calibrate(config.get_provider(), config.get_current_model(), force=True)
        level, why = calibration.choose_level(profile, config.get_ultracode_level())
        speed = f"{profile.tok_per_s:.1f} ток/с" if profile.tok_per_s else "не измерено"
        print_system(f"Калибровка: {profile.model} · {'локальная' if profile.local else 'облачная'} · {speed} "
                     f"→ уровень {level} ({calibration.LEVEL_NAMES[level]}): {why}")
    elif word == "night":
        from src.agent.workflow import night
        sub, _, tail = rest.partition(" ")
        if sub.lower() in ("cancel", "отмена") and tail.strip():
            print_system(night.cancel(tail.strip()))
        elif "|" in rest:
            when, _, goal = rest.partition("|")
            runtime.set_enabled(True)
            print_system(night.prepare(goal.strip(), when.strip()))
        else:
            waiting = night.scheduled()
            for s in waiting:
                print_system(f"{s.id} · запланирован · {s.goal[:70]}")
            print_error("Использование: /ultracode night <когда> | <задача целиком, с проверками>   "
                        "(например: завтра в 2:30 | перепиши модуль и прогони тесты)   "
                        "или: /ultracode night cancel <id>" + ("" if waiting else "   (запланированных нет)"))
    elif word == "keyword" and rest.lower() in ("on", "off"):
        config.set_ultracode_keyword_enabled(rest.lower() == "on")
        print_system("Слово «ultracode» в сообщении "
                     + ("[bold green]включает[/bold green] режим." if rest.lower() == "on" else "режим не включает."))
    elif word == "list":
        runs = store.list_runs()
        if not runs:
            print_system("Воркфлоу в этом проекте ещё не запускались.")
        for r in runs[-10:]:
            print_system(f"{r.id} · {r.status} · уровень {r.level} · {r.goal[:70]}")
    else:
        print_error("Использование: /ultracode [status] | on | off | run <задача> | stop | resume [id] | "
                    "level auto|0-3 | budget <минуты> | keyword on|off | night <когда> | <задача> | calibrate | list")
    return CommandResult.continue_loop()


@registry.register("/advisor", help_text="Советник Ultracode: более сильная модель для решающих моментов",
                   category="agent", is_prefix=True)
def handle_advisor(ctx: CommandContext) -> CommandResult:
    import config
    from src.agent.critic import parse_critic_model
    from src.agent.workflow import advisor as adv
    arg = ctx.user_input.strip()[len("/advisor"):].strip()
    word, _, rest = arg.partition(" ")
    word, rest = word.lower(), rest.strip()

    if not arg or word == "status":
        target = adv.resolve(config.get_provider(), config.get_current_model())
        where = "облако (данные уходят наружу)" if target.cloud else "ваш компьютер"
        print_system(f"Советник: [bold]{target.label}[/bold] · {'настроен' if target.configured else 'не настроен'} · "
                     f"работает на: {where}")
        if target.leaves_pc:
            ok = config.is_advisor_cloud_consented()
            print_system("Согласие на отправку плана и патчей в облако: "
                         + ("[green]дано[/green]" if ok else "[yellow]не дано — спросим при запуске[/yellow]"))
        print_system("[dim]/advisor model <провайдер>:<модель> · /advisor clear · /advisor consent on|off[/dim]")
    elif word == "model" and rest:
        provider, model = parse_critic_model(rest)
        config.set_advisor_model(model)
        config.set_advisor_provider(provider)
        print_system(f"Модель советника: [bold yellow]{model}[/bold yellow]"
                     + (f" (провайдер: {provider})" if provider else ""))
        target = adv.resolve(config.get_provider(), config.get_current_model())
        if target.leaves_pc and not config.is_advisor_cloud_consented():
            print_system("[yellow]Это облачная модель, а основная — локальная: план и патчи уйдут наружу. "
                         "Разрешить заранее: /advisor consent on (иначе спросим при запуске).[/yellow]")
    elif word in ("clear", "reset"):
        config.set_advisor_model("")
        config.set_advisor_provider("")
        print_system("Советник сброшен: советует основная модель.")
    elif word == "consent" and rest.lower() in ("on", "off"):
        config.set_advisor_cloud_consented(rest.lower() == "on")
        print_system("Отправка плана и патчей облачному советнику: [bold]%s[/bold]"
                     % ("разрешена" if rest.lower() == "on" else "запрещена (будем спрашивать)"))
    else:
        print_error("Использование: /advisor [status] | model <провайдер>:<модель> | clear | consent on|off")
    return CommandResult.continue_loop()


# --- Status Command ---

@registry.register("/status", help_text="Одним экраном: что включено, что идёт, что ждёт вас", category="agent")
def handle_status(ctx: CommandContext) -> CommandResult:
    from src.cli import status_overview
    for line in status_overview.render(status_overview.collect()):
        print_system(line)
    return CommandResult.continue_loop()


# --- Lanes Command ---

@registry.register("/lanes", help_text="Очередь запросов к локальным моделям и нагрузка на ПК", category="agent",
                   is_prefix=True)
def handle_lanes(ctx: CommandContext) -> CommandResult:
    import config
    from src.agent import governor
    arg = ctx.user_input.strip()[len("/lanes"):].strip().lower()
    words = arg.split()

    def flag(value: str):
        return {"on": True, "off": False}.get(value)

    if not words or words[0] == "status":
        for line in governor.status_lines():
            print_system(line)
        print_system("[dim]Настройка: /lanes on|off · /lanes parallel <1-8> · /lanes nice on|off · "
                     "/lanes threads on|off[/dim]")
    elif words[0] in ("on", "off"):
        config.set_local_governor_enabled(words[0] == "on")
        print_system("Очередь локальных моделей: [bold]%s[/bold]" % ("включена" if words[0] == "on" else "выключена"))
    elif words[0] in ("parallel", "parallelism") and len(words) == 2 and words[1].isdigit():
        config.set_local_parallelism(int(words[1]))
        print_system(f"Одновременных запросов к одному серверу: [bold]{config.get_local_parallelism()}[/bold]. "
                     "Больше 1 имеет смысл, если сервер сам умеет параллелить (OLLAMA_NUM_PARALLEL) и хватает видеопамяти.")
    elif words[0] == "nice" and len(words) == 2 and flag(words[1]) is not None:
        config.set_lower_server_priority_enabled(flag(words[1]))
        if not flag(words[1]):
            from src.agent.local_backend import TUNER
            TUNER.restore()
        print_system("Пониженный приоритет процесса модели: [bold]%s[/bold]" % ("включён" if flag(words[1]) else "выключен"))
    elif words[0] == "threads" and len(words) == 2 and flag(words[1]) is not None:
        config.set_limit_server_threads_enabled(flag(words[1]))
        print_system("Лимит потоков Ollama (оставлять ядра для вас): [bold]%s[/bold]" % ("включён" if flag(words[1]) else "выключен"))
    else:
        print_error("Использование: /lanes [status] | on | off | parallel <1-8> | nice on|off | threads on|off")
    return CommandResult.continue_loop()


# --- Guard Command ---

@registry.register("/guard", help_text="Управление защитой от опасных шелл-команд", category="agent", is_prefix=True)
def handle_guard(ctx: CommandContext) -> CommandResult:
    from config import get_command_guard, set_command_guard
    arg = ctx.user_input[len("/guard"):].strip().lower()
    if arg in ("off", "warn", "block"):
        set_command_guard(arg)
        print_system(f"Защита от опасных команд: [bold]{arg}[/bold]")
        if arg == "block":
            print_system("Catastrophic commands (rm -rf /, mkfs, fork bombs, curl|sh…) will be refused without prompting.")
        elif arg == "off":
            print_system("Остаётся только прежнее подтверждение разрушительных команд.")
    else:
        print_system(f"Command risk-gate: [bold]{get_command_guard()}[/bold]  (off | warn | block)")
    return CommandResult.continue_loop()


@registry.register("/loopguard", aliases=["/loop_guard", "/antiloop"], help_text="Управление защитой от зацикливания инструментов LoopGuard (on | off | auto)", category="agent", is_prefix=True)
def handle_loopguard(ctx: CommandContext) -> CommandResult:
    from config import get_loop_guard_mode, set_loop_guard_mode
    parts = ctx.user_input.strip().split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else ""

    if arg in ("on", "enable", "enabled", "1", "true"):
        set_loop_guard_mode("on")
        print_system("LoopGuard (защита от циклов инструментов): [bold green]ВКЛЮЧЕН (принудительно для всех моделей)[/bold green]")
    elif arg in ("off", "disable", "disabled", "0", "false"):
        set_loop_guard_mode("off")
        print_system("LoopGuard (защита от циклов инструментов): [bold red]ОТКЛЮЧЕН[/bold red]")
    elif arg in ("auto", "default", "reset"):
        set_loop_guard_mode("auto")
        print_system("LoopGuard: [bold cyan]АВТО[/bold cyan] (по умолчанию для стратегии выбранной модели)")
    else:
        current = get_loop_guard_mode()
        print_system(f"Текущий режим LoopGuard: [bold]{current.upper()}[/bold]  (on | off | auto)")
        print_system("Использование: [bold]/loopguard on[/bold] — принудительно включить, [bold]/loopguard off[/bold] — выключить, [bold]/loopguard auto[/bold] — авто.")
    return CommandResult.continue_loop()


# --- Hooks and Plugins ---

@registry.register("/hooks", aliases=["/plugin"], help_text="Управление плагинами и хуками", category="agent", is_prefix=True)
def handle_hooks(ctx: CommandContext) -> CommandResult:
    parts = ctx.user_input.split(" ")
    if len(parts) == 1:
        status = "ENABLED" if get_autonomous_plugins_enabled() else "DISABLED"
        print_system(f"Каталог плагинов: [bold cyan]{get_hooks_dir()}[/bold cyan]")
        print_system(f"Autonomous Plugin Creation: [bold yellow]{status}[/bold yellow]")
        
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
    elif parts[1].lower() in ("list", "config", "trust", "untrust"):
        _shell_hooks_command(parts[1].lower(), [p.lower() for p in parts[2:]])
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
        new_path = ctx.user_input.split(" ", 1)[1].strip()
        set_hooks_dir(new_path)
        hook_manager.reload_plugins(new_path)
        print_system(f"Каталог плагинов изменён: [bold green]{new_path}[/bold green]")
    return CommandResult.continue_loop()


def _shell_hooks_command(sub: str, flags: list) -> None:
    """`/hooks list | trust | untrust` — the hooks written as JSON (see src.agent.shell_hooks)."""
    from src.agent.shell_hooks import SHELL_HOOKS, is_trusted, trust_file

    project_file = next(path for source, path in SHELL_HOOKS.files() if source == "project")

    if sub == "trust":
        if not project_file.is_file():
            print_error(f"В проекте нет {project_file}")
            return
        specs, problems = [], []
        try:
            import json
            from src.agent.shell_hooks import parse_specs
            specs, problems = parse_specs(json.loads(project_file.read_text(encoding="utf-8")), "project")
        except (OSError, ValueError) as e:
            print_error(f"Файл не читается: {e}")
            return
        print_system(f"Хуки проекта ([bold]{project_file}[/bold]) выполняют команды на вашем компьютере:")
        for spec in specs:
            print_system(f"  • {spec.describe()}")
        for problem in problems:
            print_system(f"  [yellow]! {problem}[/yellow]")
        if "-y" not in flags:
            try:
                if not questionary.confirm("Доверять этому файлу? (изменённый файл потребует доверия заново)",
                                           default=False).ask():
                    print_system("Не доверяю — хуки проекта не запускаются.")
                    return
            except (KeyboardInterrupt, EOFError):
                return
        trust_file(project_file)
        print_system("[green]Файл хуков проекта доверен и будет запускаться.[/green]")
        return

    if sub == "untrust":
        import config
        trusted = dict(config.get_trusted_files() or {})
        if trusted.pop(str(project_file.resolve()), None) is None:
            print_system("Файл и так не был доверенным.")
        else:
            config.set_trusted_files(trusted)
            print_system("Хуки проекта больше не запускаются.")
        return

    status = SHELL_HOOKS.status()
    if not status["specs"] and not status["untrusted"]:
        print_system("Хуков в JSON нет. Создайте .argent/hooks.json (проект) или ~/.argent/hooks.json (везде):")
        print_system('  {"hooks": {"PostToolUse": [{"matcher": "write_file|replace_in_file", "if": "*.py", '
                     '"command": "ruff check \\"$ARGENT_FILE\\""}]}}')
        print_system("События: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop. "
                     "Код выхода 2 в PreToolUse блокирует вызов.")
    for spec in status["specs"]:
        print_system(f"  [{spec.source}] {spec.describe()}")
    for note in status["notes"]:
        print_system(f"  [yellow]! {note}[/yellow]")
    if status["untrusted"]:
        print_system(f"[yellow]Хуки проекта ({status['untrusted']}) НЕ запускаются: файл не доверен. "
                     f"Прочитайте его и выполните /hooks trust[/yellow]")


# --- Tools ---

@registry.register("/tools", help_text="Включение/отключение инструментов", category="agent")
def handle_tools(ctx: CommandContext) -> CommandResult:
    from tools import AVAILABLE_TOOLS
    all_tools = list(AVAILABLE_TOOLS.keys())
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
    return CommandResult.continue_loop()


# --- Logs & Stats ---

@registry.register("/logs", help_text="Просмотр и очистка логов", category="agent", is_prefix=True)
def handle_logs(ctx: CommandContext) -> CommandResult:
    try:
        log_dir = Path.home() / ".argent" / "logs"
    except Exception as e:
        print_error(f"Не удалось определить домашнюю директорию: {e}")
        return CommandResult.continue_loop()

    parts = ctx.user_input.strip().split()
    if len(parts) > 1 and parts[1] == "clear":
        for f in log_dir.glob("*.log"):
            f.write_text("", encoding="utf-8")
        print_system("Все логи очищены.")
        return CommandResult.continue_loop()

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
        return CommandResult.continue_loop()

    if module_filter:
        log_files = [f for f in log_files if f.stem == module_filter]
        if not log_files:
            available = ", ".join(f.stem for f in sorted(log_dir.glob("*.log")))
            print_error(f"No log '{module_filter}'. Available: {available}")
            return CommandResult.continue_loop()

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

    return CommandResult.continue_loop()


@registry.register("/stats", help_text="Диагностика и статистика контекста", category="agent")
def handle_stats(ctx: CommandContext) -> CommandResult:
    from config import (
        get_context_window, get_model_size_category,
        get_model_category_override, get_moe_specs
    )
    agent = ctx.agent
    model = get_current_model()
    provider = get_provider()
    ctx_window = get_context_window()
    msg_count = len(getattr(agent, "messages", []) or [])

    category = get_model_size_category(model)
    override = get_model_category_override()
    cat_source = "manual" if override else "auto"

    moe_details = ""
    specs = get_moe_specs(model)
    if specs or override == "moe" or category == "moe":
        if specs:
            moe_details = f" [MoE: {specs[0]:.0f}B total / {specs[1]:.0f}B active]"
        else:
            moe_details = " [MoE mode]"

    mcp_servers = mcp_client.get_servers()
    active_mcp = [s.get('name') for s in mcp_servers if s.get('running') and s.get('name')]
    plugins = list(hook_manager.plugins.keys()) if isinstance(getattr(hook_manager, "plugins", None), dict) else []

    stats_msg = (
        f"[bold cyan]Argent Diagnostics:[/bold cyan]\n"
        f"  [dim]Directory:[/dim] {os.getcwd()}\n"
        f"  [dim]Model:[/dim] {model} ({provider})\n"
        f"  [dim]Classification:[/dim] {category} ({cat_source}){moe_details}\n"
        f"  [dim]Context Window:[/dim] {ctx_window} tokens\n"
        f"  [dim]History:[/dim] {msg_count} messages\n"
    )

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
    return CommandResult.continue_loop()


# --- RAG / KB / Auto-Retrieve ---

@registry.register("/rag_toggle", help_text="Переключить авто-RAG индексацию", category="agent")
def handle_rag_toggle(ctx: CommandContext) -> CommandResult:
    from config import get_auto_rag, set_auto_rag
    current = get_auto_rag()
    new_state = not current
    set_auto_rag(new_state)
    status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
    print_system(f"Авто-RAG (индексация при запуске): {status}")
    if new_state:
        print_system("RAG включится при следующем запуске Argent.")
    else:
        try:
            from rag_engine import disable_rag
            disable_rag()
            print_system("Семантический поиск (RAG) отключён для текущей сессии.")
        except ImportError:
            pass
    return CommandResult.continue_loop()


@registry.register("/kb_toggle", help_text="Переключить авто-загрузку внешних KB", category="agent")
def handle_kb_toggle(ctx: CommandContext) -> CommandResult:
    from config import get_auto_kb, set_auto_kb
    current = get_auto_kb()
    new_state = not current
    set_auto_kb(new_state)
    status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
    print_system(f"Авто-KB (загрузка внешних баз знаний при запуске): {status}")
    if new_state:
        print_system("Внешние базы знаний загрузятся при следующем запуске.")
    return CommandResult.continue_loop()


@registry.register("/auto_retrieve", help_text="Переключить авто-подстановку семантического поиска", category="agent")
def handle_auto_retrieve(ctx: CommandContext) -> CommandResult:
    from config import get_auto_retrieve, set_auto_retrieve
    new_state = not get_auto_retrieve()
    set_auto_retrieve(new_state)
    status = "[bold green]ENABLED[/bold green]" if new_state else "[bold red]DISABLED[/bold red]"
    print_system(f"Авто-подстановка (результаты semantic_search в контекст на каждый запрос): {status}")
    return CommandResult.continue_loop()


# --- Skills ---

@registry.register("/skill import", aliases=["/skills import"], help_text="Импортировать навык из git или папки", category="agent", is_prefix=True)
def handle_skill_import(ctx: CommandContext) -> CommandResult:
    from skill_manager import skill_manager
    arg = ctx.user_input.split("import", 1)[1].strip().strip('"')
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
    return CommandResult.continue_loop()


@registry.register("/skills", aliases=["/skill"], help_text="Список доступных навыков", category="agent", is_prefix=True)
def handle_skills(ctx: CommandContext) -> CommandResult:
    if "import" in ctx.user_input:
        return handle_skill_import(ctx)
    from skill_manager import skill_manager
    skills = skill_manager.list_skills()
    if not skills:
        print_system("Навыков нет. Создайте инструментом `create_skill` или импортируйте: /skill import.")
    else:
        print_system("[bold cyan]Доступные навыки:[/bold cyan]")
        for skill in skills:
            tag = " [dim](bundle)[/dim]" if skill.get("kind") == "folder" else ""
            print_system(f"- [bold yellow]{skill['name']}[/bold yellow]{tag}: {skill['description']}")
    return CommandResult.continue_loop()


# --- Goal & Memory ---

@registry.register("/goal", help_text="Посмотреть или установить цель проекта", category="agent", is_prefix=True)
def handle_goal(ctx: CommandContext) -> CommandResult:
    arg = ctx.user_input[len("/goal"):].strip()
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
    return CommandResult.continue_loop()


# --- Rooms ---

@registry.register("/rooms", help_text="Комнаты: повторяемые рецепты с рельсами (/rooms без аргументов — справка)",
                   category="agent", is_prefix=True)
def handle_rooms(ctx: CommandContext) -> CommandResult:
    """The whole command lives in src/rooms/cli.py; this only supplies the terminal."""
    arg = ctx.user_input[len("/rooms"):].strip()
    from src.rooms import cli as rooms_cli
    from src.rooms.session import questionary_confirm

    def _confirm(question: str) -> bool:
        return bool(questionary.confirm(question, default=False).ask())

    def _choose(question: str, options: list):
        return questionary.select(question, choices=options).ask()

    def _human(node, state):
        from src.rooms.ask import ask_human
        ask_human(node, state, print_system,
                  lambda q: questionary.text(q).ask(),
                  lambda q, options: questionary.select(q, choices=options).ask())

    def _ask_text(question: str) -> str:
        return questionary.text(question).ask() or ""

    io = rooms_cli.Io(out=print_system, err=print_error, confirm=_confirm, choose=_choose,
                      human=_human, confirm_step=questionary_confirm, ask_text=_ask_text)
    try:
        rooms_cli.handle(arg, io)
    except (KeyboardInterrupt, EOFError):
        print_system("Прервано.")
    except Exception as e:
        print_error(f"/rooms: {e}")
    return CommandResult.continue_loop()


# --- Browser ---

@registry.register("/browser", help_text="Настройка и управление браузером", category="agent", is_prefix=True)
def handle_browser(ctx: CommandContext) -> CommandResult:
    parts = ctx.user_input.strip().split()
    if len(parts) == 1:
        from config import get_browser_mode, get_browser_name, get_browser_state_mode, get_effective_browser_state_mode
        from browser_detect import detect_browsers
        mode = get_browser_mode()
        name = get_browser_name()
        state_mode = get_browser_state_mode()
        effective = get_effective_browser_state_mode()
        browsers = detect_browsers()
        print_system(f"Режим браузера: [bold cyan]{mode}[/bold cyan]")
        print_system(f"Selected Browser: [bold cyan]{name}[/bold cyan]")
        state_label = state_mode if state_mode != "auto" else f"auto → {effective}"
        print_system(f"State extraction: [bold cyan]{state_label}[/bold cyan]")
        if browsers:
            print_system("Detected browsers:")
            for b in browsers:
                print_system(f"  - [bold yellow]{b.key}[/bold yellow]: {b.exe_path}")
        else:
            print_system("[dim]Браузеров на основе Chromium не найдено.[/dim]")
        print_system("\nUsage: /browser user | isolated | chrome | yandex | edge | brave | auto | a11y | dom")
    elif parts[1] in ("user", "isolated"):
        from config import set_browser_mode
        set_browser_mode(parts[1])
        if parts[1] == "user":
            print_system("Режим браузера: [bold green]ВАШ[/bold green] — модель работает в вашем браузере через CDP.")
        else:
            print_system("Режим браузера: [bold yellow]ИЗОЛИРОВАННЫЙ[/bold yellow] — модель работает в Playwright Chromium.")
    elif parts[1] in ("chrome", "yandex", "edge", "brave"):
        from config import set_browser_name, set_browser_mode
        set_browser_name(parts[1])
        set_browser_mode("user")
        print_system(f"Браузер: [bold green]{parts[1]}[/bold green] (режим «ваш браузер» включён).")
    elif parts[1] in ("a11y", "dom"):
        from config import set_browser_state_mode
        set_browser_state_mode(parts[1])
        label = "Accessibility Tree (семантическое дерево)" if parts[1] == "a11y" else "DOM (плоский список элементов)"
        print_system(f"State extraction: [bold green]{parts[1]}[/bold green] — {label}")
    elif parts[1] == "auto":
        from config import set_browser_state_mode
        set_browser_state_mode("auto")
        print_system("State extraction: [bold green]auto[/bold green] — выбирается по tier модели (a11y для крупных, dom для мелких).")
    else:
        print_error(f"Неизвестный вариант: {parts[1]}. Допустимые: user, isolated, chrome, yandex, edge, brave, auto, a11y, dom")
    return CommandResult.continue_loop()
