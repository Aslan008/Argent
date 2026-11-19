"""/desktop: the agent's hands on the desktop — the brake, and what they can do (see src/automation)."""

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from argent.ui import print_error, print_system

_USAGE = "/desktop [status] | on | off | windows | key <сочетание> | audit [N]"

_LIMITS = ("Действовать можно в любом окне, кроме: своего терминала Argent, полей паролей, диалогов UAC "
           "и окон, похожих на пароль или оплату.")


def _status() -> str:
    from src.automation import desktop_guard, uia
    state = "[green]включено[/green]" if desktop_guard.enabled() else "[yellow]выключено[/yellow]"
    lines = [f"Управление рабочим столом: {state}"]
    if not uia.available():
        lines.append("[red]Нет пакета uiautomation или это не Windows: инструменты недоступны.[/red]")
    lines.append(_LIMITS)
    lines.append(f"Аварийная остановка: [bold]{desktop_guard.stop_chord()}[/bold] (в любом окне)")
    series = desktop_guard.SERIES
    if series.stopped:
        lines.append(f"[red]Остановлено: {series.stop_reason}. /desktop on — включить снова.[/red]")
    lines.append(f"Действий в серии: {series.count}/{desktop_guard.MAX_SERIES} · журнал: {desktop_guard.audit_path()}")
    return "\n".join(lines)


@registry.register("/desktop", help_text="Управление программами на рабочем столе (UI Automation): "
                                         "status / on / off / windows / key / audit",
                   category="agent", is_prefix=True)
def handle_desktop(ctx: CommandContext) -> CommandResult:
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
            print_system("Управление рабочим столом [bold]включено[/bold] (по умолчанию оно включено; "
                         "эта команда нужна после /desktop off или аварийной остановки). "
                         + _LIMITS + f" Аварийная остановка: {desktop_guard.stop_chord()}.")
    elif word == "off":
        desktop_guard.set_enabled(False)
        print_system("Управление рабочим столом выключено до /desktop on.")
    elif word == "windows":
        if not uia.available():
            print_error("Нужен Windows и пакет uiautomation.")
        else:
            for w in uia.windows():
                verdict = desktop_guard.check_window(w.process, w.title,
                                                     desktop_guard.allowed_apps() or None)
                mark = "[green]можно[/green]" if verdict else f"[dim]нельзя: {verdict.reason}[/dim]"
                print_system(f"{w.process or '?':24} {mark}  {w.title[:70]}")
    elif word == "key" and rest:
        try:
            uia.parse_keys(rest)
        except ValueError as e:
            print_error(f"Не подходит: {e}. Пример: ctrl+alt+shift+f12")
        else:
            import argent.config as config
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
