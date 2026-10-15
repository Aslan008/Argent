"""`/status` — one screen: what is switched on, what is running, what is waiting for the person.

Every subsystem has its own command (/lanes, /ultracode, /memory, /desktop, /tasks, /rooms, /network ...);
this is the single place that reads them all. It only READS state — nothing is started, connected or
changed. Each section is gathered on its own, so one broken subsystem shows up as a line saying so
instead of taking the whole overview down.

`collect()` returns [(section, level, text)], level being "ok" | "warn" | "off" | "attention":
    ok         works / switched on and healthy
    off        switched off (nothing to do)
    warn       works, but something deserves a look
    attention  waits for the person (an unfinished run, a question, a paused schedule)
"""

from typing import Callable, List, Tuple

from rich.markup import escape

OK, OFF, WARN, ATTENTION = "ok", "off", "warn", "attention"

Row = Tuple[str, str, str]


def _on(flag: bool) -> str:
    return "вкл" if flag else "выкл"


# ---- sections ---------------------------------------------------------------------------------------

def _model() -> Row:
    import config
    provider, model = config.get_provider(), config.get_current_model()
    return ("Модель", OK, f"{provider} · {model}")


def _local_queue() -> Row:
    from src.agent import governor
    from src.agent.local_backend import classify
    import config
    backend = classify(config.get_provider(), config.get_current_model())
    if not backend.local:
        return ("Очередь локальных моделей", OFF, "основная модель не на этом ПК, очередь не нужна")
    cfg = governor.settings()
    if not cfg["enabled"]:
        return ("Очередь локальных моделей", WARN,
                f"{backend.family} на этом ПК, очередь ВЫКЛ (/lanes on — чтобы модели не душили ПК)")
    text = f"вкл · одновременно запросов: {cfg['parallelism']} · {backend.family}"
    try:
        from src.agent.pressure import monitor
        mon = monitor()
        mon.sample()
        if mon.overloaded:
            return ("Очередь локальных моделей", WARN, text + f" · ПК перегружен: {mon.reason}")
    except Exception:
        pass
    return ("Очередь локальных моделей", OK, text)


def _ultracode() -> Row:
    import config
    from src.agent.workflow import runtime
    if runtime.enabled():
        return ("Ultracode", OK, f"вкл в этой сессии · уровень {config.get_ultracode_level()} · {runtime.status()}")
    return ("Ultracode", OFF, "выкл (/ultracode on)")


def _automations() -> Row:
    from src.automation.store import load_automations, running_elsewhere
    items = load_automations()
    if not items:
        return ("Автоматизации", OFF, "нет (/tasks add)")
    enabled = [a for a in items if a.enabled]
    paused = [a for a in items if not a.enabled and getattr(a, "disabled_reason", "")]
    failed = [a for a in enabled if str(getattr(a, "last_status", "") or "").lower() in ("error", "failed", "fail")]
    running = running_elsewhere()
    text = f"{len(enabled)} включено из {len(items)}"
    if running:
        text += " · сейчас идут: " + ", ".join(running[:3])
    if paused:
        text += f" · остановлено само: {', '.join(a.name for a in paused[:3])}"
        return ("Автоматизации", ATTENTION, text)
    if failed:
        text += f" · последний запуск с ошибкой: {', '.join(a.name for a in failed[:3])}"
        return ("Автоматизации", WARN, text)
    return ("Автоматизации", OK, text)


def _rooms() -> Row:
    from src.rooms.session import build_library, default_journal
    lib = build_library()
    names = lib.active_rooms() if hasattr(lib, "active_rooms") else lib.rooms
    mine = sum(1 for n in names if lib.sources.get(n) == "user")
    text = f"{len(names)} комнат (своих: {mine})"
    quarantined = [n for n in lib.rooms if n not in names]
    level = OK
    if quarantined:
        text += f" · на карантине: {', '.join(sorted(quarantined)[:3])}"
        level = WARN
    try:
        if default_journal().unfinished():
            text += " · есть НЕЗАВЕРШЁННЫЙ прогон (/rooms resume)"
            level = ATTENTION
    except Exception:
        pass
    try:
        from src.rooms.triggers import list_triggers
        triggers = list_triggers()
        live = [t for t in triggers if t["enabled"]]
        if triggers:
            text += f" · триггеров: {len(live)} из {len(triggers)} включено"
    except Exception:
        pass
    return ("Комнаты", level, text)


def _memory() -> Row:
    import config
    if not config.is_memory_enabled():
        return ("Память", OFF, "выкл (/memory)")
    text = "вкл"
    if config.is_dream_global_enabled():
        text += " · консолидация глобальной памяти: вкл"
    return ("Память", OK, text)


def _desktop() -> Row:
    from src.automation import desktop_guard
    if not desktop_guard.enabled():
        return ("Управление ПК", OFF, "выкл (/desktop on) — действия только с вашего разрешения")
    apps = desktop_guard.allowed_apps()
    return ("Управление ПК", ATTENTION if not apps else OK,
            f"ВКЛ · разрешённых программ: {len(apps)}" + (" (список пуст — ничего не сработает)" if not apps else "")
            + f" · аварийная остановка: {desktop_guard.stop_chord()}")


def _telegram() -> Row:
    import config
    if not config.get_telegram_enabled():
        return ("Telegram", OFF, "выкл (/telegram)")
    return ("Telegram", OK, f"вкл · режим {config.get_telegram_mode()}")


def _waiting() -> Row:
    """Background commands and waits that are still alive."""
    from src.agent import watches
    pids = watches.running_background_pids()
    if not pids:
        return ("Фоновые команды", OFF, "нет запущенных")
    return ("Фоновые команды", OK, f"запущено: {len(pids)} (pid {', '.join(sorted(pids)[:5])})")


def _todos() -> Row:
    from src.agent.todos import TODOS
    items = TODOS.get("main")
    if not items:
        return ("План дел", OFF, "пусто")
    done = sum(1 for t in items if t.status == "completed")
    return ("План дел", OK, f"{done} из {len(items)} выполнено")


def _mcp() -> Row:
    import config
    servers = config.get_mcp_servers()
    if not servers:
        return ("MCP", OFF, "серверов нет")
    names = [s.get("name", "?") for s in servers]
    return ("MCP", OK, f"{len(servers)}: {', '.join(names[:5])}" + ("…" if len(names) > 5 else ""))


def _browser() -> Row:
    from browser_detect import bundled_chromium_status
    st = bundled_chromium_status()
    if st["known"] and st["missing"]:
        return ("Браузер Playwright", WARN,
                f"нет {', '.join(st['missing'])} — playwright install chromium (или режим 'user')")
    return ("Браузер Playwright", OK, "Chromium нужной версии на месте" if st["known"] else "не проверялся")


SECTIONS: List[Callable[[], Row]] = [
    _model, _local_queue, _ultracode, _rooms, _automations, _memory, _desktop, _telegram,
    _waiting, _todos, _mcp, _browser,
]


def collect() -> List[Row]:
    """Gather every section; a section that cannot be read says so."""
    rows = []
    for section in SECTIONS:
        try:
            rows.append(section())
        except Exception as exc:     # one broken subsystem must not hide the others
            title = section.__name__.strip("_").replace("_", " ")
            rows.append((title, WARN, f"не удалось прочитать: {exc}"))
    return rows


_MARK = {OK: "[green]●[/green]", OFF: "[dim]○[/dim]", WARN: "[yellow]▲[/yellow]", ATTENTION: "[bold red]![/bold red]"}


def render(rows: List[Row]) -> List[str]:
    """Lines for print_system: the things that need the person first, then the rest."""
    width = max((len(name) for name, _, _ in rows), default=0)
    lines = []
    waiting = [r for r in rows if r[1] == ATTENTION]
    for name, level, text in rows:
        lines.append(f"{_MARK[level]} [bold]{escape(name.ljust(width))}[/bold]  {escape(text)}")
    if waiting:
        lines.append("")
        lines.append("[bold red]Ждёт вас:[/bold red] " + "; ".join(escape(name) for name, _, _ in waiting))
    elif any(level == WARN for _, level, _ in rows):
        lines.append("")
        lines.append("[yellow]Есть на что посмотреть (▲). Подробности — командами подсистем и /doctor.[/yellow]")
    return lines
