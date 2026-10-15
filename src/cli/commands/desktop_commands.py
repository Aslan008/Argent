"""/desktop: switch on the agent's hands on the desktop and say where they may work (see src/automation)."""

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import print_error, print_system

_USAGE = ("/desktop [status] | on | off | allow <программа.exe> | deny <программа.exe> | windows | "
          "key <сочетание> | audit [N]")


def _status() -> str:
    import config
    from src.automation import desktop_guard, uia
    state = "[green]включено[/green]" if desktop_guard.enabled() else "[yellow]выключено[/yellow]"
    lines = [f"Управление рабочим столом: {state} (действует только в этой сессии)"]
    if not uia.available():
        lines.append("[red]Нет пакета uiautomation или это не Windows: инструменты недоступны.[/red]")
    apps = config.get_desktop_allowed_apps()
    lines.append("Разрешённые программы: " + (", ".join(apps) if apps else "[dim]нет — добавьте: /desktop allow notepad.exe[/dim]"))
    lines.append(f"Аварийная остановка: [bold]{desktop_guard.stop_chord()}[/bold] (в любом окне)")
    series = desktop_guard.SERIES
    if series.stopped:
        lines.append(f"[red]Остановлено: {series.stop_reason}. /desktop on — включить снова.[/red]")
    lines.append(f"Действий в серии: {series.count}/{desktop_guard.MAX_SERIES} · журнал: {desktop_guard.audit_path()}")
    return "\n".join(lines)


@registry.register("/desktop", help_text="Управление программами на рабочем столе (UI Automation): "
                                         "on / off / allow / deny / windows / key / audit",
                   category="agent", is_prefix=True)
def handle_desktop(ctx: CommandContext) -> CommandResult:
    import config
    from src.automation import desktop_guard, uia

    arg = ctx.user_input.strip()[len("/desktop"):].strip()
    word, _, rest = arg.partition(" ")
    word, rest = word.lower(), rest.strip()

    if word in ("", "status"):
        print_system(_status())
    elif word == "on":
        if not uia.available():
            print_error("Управление рабочим столом работает только на Windows с пакетом uiautomation.")
        else:
            desktop_guard.set_enabled(True)
            apps = config.get_desktop_allowed_apps()
            print_system("Управление рабочим столом [bold]включено[/bold] до конца сессии. "
                         + (f"Программы: {', '.join(apps)}." if apps else
                            "Список программ пуст: добавьте нужную командой /desktop allow <программа.exe>.")
                         + f" Аварийная остановка: {desktop_guard.stop_chord()}.")
    elif word == "off":
        desktop_guard.set_enabled(False)
        print_system("Управление рабочим столом выключено.")
    elif word == "allow" and rest:
        name = desktop_guard.normalize_app(rest)
        if name in config_forbidden():
            print_error(f"{name}: в менеджерах паролей и системных диалогах Argent не действует.")
        else:
            apps = config.get_desktop_allowed_apps()
            if name not in apps:
                config.set_desktop_allowed_apps(apps + [name])
            print_system(f"Разрешено действовать в: [bold]{name}[/bold]. Каждое действие всё равно спросит вас.")
    elif word in ("deny", "remove") and rest:
        name = desktop_guard.normalize_app(rest)
        config.set_desktop_allowed_apps([a for a in config.get_desktop_allowed_apps() if a != name])
        print_system(f"{name} убран из списка.")
    elif word == "windows":
        if not uia.available():
            print_error("Нужен Windows и пакет uiautomation.")
        else:
            allowed = config.get_desktop_allowed_apps()
            for w in uia.windows():
                mark = "[green]разрешена[/green]" if w.process.lower() in allowed else "[dim]нет[/dim]"
                print_system(f"{w.process or '?':24} {mark}  {w.title[:70]}")
    elif word == "key" and rest:
        try:
            uia.parse_keys(rest)
        except ValueError as e:
            print_error(f"Не подходит: {e}. Пример: ctrl+alt+shift+f12")
        else:
            config.set_desktop_stop_chord(rest)
            print_system(f"Аварийная остановка теперь: [bold]{rest.lower()}[/bold]")
    elif word == "audit":
        try:
            count = int(rest) if rest else 15
        except ValueError:
            count = 15
        rows = desktop_guard.read_audit(max(1, min(count, 100)))
        if not rows:
            print_system("Журнал пуст.")
        for row in rows:
            extra = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("at", "event") and v not in ("", None))
            print_system(f"{row.get('at', '')}  {row.get('event', '')}  {extra}"[:200])
    else:
        print_error("Использование: " + _USAGE)
    return CommandResult.continue_loop()


def config_forbidden() -> set:
    from src.automation import desktop_guard
    return set(desktop_guard._FORBIDDEN_PROCESSES)
