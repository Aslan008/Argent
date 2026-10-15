"""`/tasks` — everything about scheduled work, in one place.

    /tasks                     what is running, what is next, what needs you
    /tasks list | show <имя>   the definitions
    /tasks add <имя> | <расписание> | <задача> [| ключ: значение ...]
    /tasks cmd <имя> | <расписание> | <команда> [| ...]   a command, no model
    /tasks remind <когда> | <текст>                      a reminder, no model
    /tasks new <шаблон> [аргумент]  |  /tasks templates   ready-made ones
    /tasks edit <имя> | ключ: значение ...
    /tasks on|off <имя|all>   /tasks rm <имя>
    /tasks run <имя>   /tasks stop <имя>
    /tasks runs [имя]  /tasks last <имя>   /tasks inbox
    /tasks pause [2h]  /tasks resume
    /tasks service install|remove|status               run even when Argent is closed
    /tasks memory [имя]  /tasks forget <имя>

The parsing and the text live here, away from the command registry, so they can be
tested without a terminal: ``handle`` takes the two functions it prints with.
"""

import os
import re
from datetime import datetime, timedelta

from rich.markup import escape

from src.automation import manage, store, templates
from src.automation.schedule import (
    ScheduleError, describe_schedule, humanize_when, next_fire, parse_schedule,
)

# --- option parsing ------------------------------------------------------------------

_KEY_ALIASES = {
    "schedule": "schedule", "расписание": "schedule", "when": "schedule", "когда": "schedule",
    "task": "task", "задача": "task",
    "mode": "mode", "режим": "mode",
    "notify": "notify", "уведомлять": "notify", "сообщать": "notify",
    "channels": "channels", "channel": "channels", "каналы": "channels", "канал": "channels",
    "tools": "tools", "инструменты": "tools",
    "commands": "commands", "cmds": "commands", "команды": "commands",
    "report": "report", "отчёт": "report", "отчет": "report",
    "timeout": "timeout", "таймаут": "timeout",
    "turns": "turns", "ходы": "turns",
    "retries": "retries", "повторы": "retries",
    "catchup": "catchup", "догон": "catchup",
    "max_failures": "max_failures", "лимит_сбоев": "max_failures",
    "name": "name", "имя": "name",
}

_NOTIFY_WORDS = {
    "auto": "auto", "авто": "auto", "по-умолчанию": "auto",
    "always": "always", "всегда": "always", "каждый": "always",
    "failures": "failures", "failure": "failures", "сбои": "failures", "только сбои": "failures",
    "changes": "changes", "change": "changes", "изменения": "changes", "при изменении": "changes",
    "never": "never", "никогда": "never", "выкл": "never", "нет": "never",
}
_MODE_WORDS = {"agent": "agent", "агент": "agent", "command": "command", "cmd": "command",
               "команда": "command", "remind": "remind", "reminder": "remind",
               "напоминание": "remind"}
_NOTHING = {"", "off", "none", "нет", "-", "no", "выкл"}
_ALL_TOOLS = {"all", "все", "*"}

_KEYED = re.compile(r"^\s*([\wЀ-ӿ]+)\s*:\s*(.*)$", re.S)


def _is_keyed(segment: str) -> bool:
    match = _KEYED.match(segment)
    return bool(match) and match.group(1).lower() in _KEY_ALIASES


def parse_options(segments) -> tuple:
    """(fields, errors) from `key: value` segments. Unknown keys are errors — a
    typo that was silently ignored would leave a limit nobody set."""
    fields, errors = {}, []
    for segment in segments:
        match = _KEYED.match(segment)
        if not match or match.group(1).lower() not in _KEY_ALIASES:
            errors.append(f"не понял «{segment.strip()}»: ожидалось «ключ: значение» "
                          f"({', '.join(sorted({v for v in _KEY_ALIASES.values()}))})")
            continue
        key, value = _KEY_ALIASES[match.group(1).lower()], match.group(2).strip()
        low = value.lower()
        if key == "notify":
            if low not in _NOTIFY_WORDS:
                errors.append(f"notify: «{value}» неизвестно (auto, always, failures, changes, never)")
                continue
            fields[key] = _NOTIFY_WORDS[low]
        elif key == "mode":
            if low not in _MODE_WORDS:
                errors.append(f"mode: «{value}» неизвестно (agent, command, remind)")
                continue
            fields[key] = _MODE_WORDS[low]
        elif key == "tools":
            fields[key] = [] if low in _ALL_TOOLS | _NOTHING else value
        elif key == "commands":
            fields[key] = [] if low in _NOTHING else value
        elif key == "channels":
            names = [] if low in _NOTHING else [("telegram" if w.strip().lower() in ("телеграм", "tg")
                                                 else w.strip().lower()) for w in value.split(",")]
            fields[key] = names
        elif key == "report":
            fields[key] = "" if low in _NOTHING else value
        else:
            fields[key] = value
    return fields, errors


def split_add(arg: str):
    """(name, schedule, task, option segments) from `name | schedule | task | key: v ...`.

    Everything between the schedule and the first `key:` segment is the task, so a
    command with a pipe in it (`dir | find "x"`) stays in one piece.
    """
    pieces = arg.split("|")
    # The name is optional: `schedule | task` is enough, and is told apart from
    # `name | schedule | task` by which of the first two pieces reads as a schedule.
    nameless = len(pieces) >= 2 and _reads_as_schedule(pieces[0]) and not _reads_as_schedule(pieces[1])
    if nameless:
        pieces = [""] + pieces
    if len(pieces) < 3:
        return None
    rest = pieces[2:]
    cut = next((i for i, seg in enumerate(rest) if _is_keyed(seg)), len(rest))
    task = " | ".join(p.strip() for p in rest[:cut]).strip()
    return pieces[0].strip(), pieces[1].strip(), task, rest[cut:]


def _reads_as_schedule(text: str) -> bool:
    try:
        parse_schedule(text.strip())
    except ScheduleError:
        return False
    return True


def _auto_name(task: str) -> str:
    base = templates._slug(" ".join(task.split()[:4]), 28) or "задача"
    taken = {a.name for a in store.load_automations()}
    name, number = base, 2
    while name in taken:
        name, number = f"{base}-{number}", number + 1
    return name


# --- presentation --------------------------------------------------------------------

_STATUS_COLOR = {"ok": "green", "error": "red", "denied": "yellow", "skipped": "dim"}


def _stamp(iso) -> str:
    return (iso or "")[:16].replace("T", " ")


def _status(status) -> str:
    if not status:
        return "[dim]ещё не запускалась[/dim]"
    return f"[{_STATUS_COLOR.get(status, 'white')}]{status}[/]"


def _state_tag(a) -> str:
    if a.enabled:
        return "[green]вкл[/green]"
    if a.disabled_reason:
        return f"[yellow]ВЫКЛ — {escape(a.disabled_reason)}[/yellow]"
    return "[yellow]ВЫКЛ[/yellow]"


def _next_text(a, now) -> str:
    if not a.enabled:
        return "—"
    return humanize_when(next_fire(a.schedule, a.last_run_dt(), now, a.catchup_minutes * 60), now)


def _mode_tag(a) -> str:
    return {"command": "команда", "remind": "напоминание"}.get(a.mode, "агент")


def list_lines(items, now: datetime) -> list:
    lines = []
    for a in items:
        lines.append(f"  • [bold]{escape(a.name)}[/bold]  {_state_tag(a)}  [dim]{_mode_tag(a)}[/dim]  "
                     f"{escape(describe_schedule(a.schedule))}")
        last = f"последний: {_stamp(a.last_run)} {_status(a.last_status)}" if a.last_run else "ещё не запускалась"
        lines.append(f"      следующий: {_next_text(a, now)} · {last} · прогонов: {a.run_count}")
        lines.append(f"      [dim]{escape(a.task[:110])}[/dim]")
    return lines


def dashboard_lines(now: datetime = None, live=None) -> list:
    """The `/tasks` overview: running, next up, what needs attention."""
    now = now or datetime.now()
    items = store.load_automations()
    if not items:
        return [
            "Автоматизаций пока нет. Начните с готового шаблона или с одной строки:",
            "  /tasks templates                          — что есть готового",
            "  /tasks new morning                        — утренняя сводка по проекту",
            "  /tasks remind через 2 часа | позвонить Ане",
            "  /tasks add по будням в 9:00 | Собери сводку по git за вчера",
            "[dim]Запускаются по расписанию, пока Argent открыт; чтобы шли и при закрытом — /tasks service install[/dim]",
        ]
    enabled = [a for a in items if a.enabled]
    head = f"[bold cyan]⏰ Автоматизации:[/bold cyan] {len(items)} (включено {len(enabled)})"
    paused = store.pause_until(now)
    if paused is not None:
        head += "  [yellow]⏸ на паузе " + (
            "до /tasks resume" if paused == datetime.max else f"до {humanize_when(paused, now)}") + "[/yellow]"
    lines = [head]

    running = list((live.status()["running"] if live else {}).items())
    elsewhere = [n for n in store.running_elsewhere() if n not in dict(running)]
    for name, info in running:
        lines.append(f"  [cyan]▶ идёт:[/cyan] {escape(name)} (с {info['since']:%H:%M})")
    for name in elsewhere:
        lines.append(f"  [cyan]▶ идёт в другом окне:[/cyan] {escape(name)}")

    upcoming = sorted(((next_fire(a.schedule, a.last_run_dt(), now, a.catchup_minutes * 60), a)
                       for a in enabled), key=lambda pair: (pair[0] is None, pair[0] or now))
    shown = [(when, a) for when, a in upcoming if when is not None][:5]
    if shown:
        lines.append("  Ближайшие:")
        for when, a in shown:
            lines.append(f"    {humanize_when(when, now):<16} {escape(a.name)}  "
                         f"[dim]{escape(describe_schedule(a.schedule))}[/dim]")

    sleeping = [a for a in items if not a.enabled]
    for a in sleeping[:4]:
        lines.append(f"  [yellow]■[/yellow] {escape(a.name)} — [yellow]{escape(a.disabled_reason or 'выключена вручную')}[/yellow]")

    unseen = store.unseen_runs(store.last_seen() or now - timedelta(days=1))
    if unseen:
        lines.append(f"  [bold yellow]● Новое для вас: {len(unseen)}[/bold yellow] — /tasks inbox")
    recent = store.load_runs(limit=4)
    if recent:
        lines.append("  Недавно:")
        for r in recent:
            tail = f" · нового: {r['new_items']}" if r.get("new_items") else ""
            lines.append(f"    {_stamp(r.get('started'))}  {escape(r['name'])}  {_status(r['status'])}{tail}")
    lines.append("[dim]/tasks list · show · add · new · run · runs · inbox · pause · help[/dim]")
    return lines


def show_lines(a, now: datetime) -> list:
    lines = [f"[bold cyan]{escape(a.name)}[/bold cyan]  {_state_tag(a)}"]
    lines += [f"  {line}" for line in manage.summary_lines(a, now)]
    lines.append(f"  задача: {escape(a.task)}")
    if a.mode == "agent":
        lines.append(f"  лимит ходов: {a.max_turns}" + (f", таймаут: {a.timeout_minutes} мин" if a.timeout_minutes else ""))
    elif a.timeout_minutes:
        lines.append(f"  таймаут: {a.timeout_minutes} мин")
    lines.append(f"  повторов при сбое: {a.retries} · допуск опоздания: {a.catchup_minutes} мин · "
                 f"пауза после {a.max_failures or '∞'} сбоев подряд")
    lines.append(f"  прогонов: {a.run_count}" + (f" · сбоев подряд: {a.fail_streak}" if a.fail_streak else ""))
    memory = _memory_count(a.name)
    if memory:
        lines.append(f"  запомнено элементов: {memory} (сброс: /tasks forget {escape(a.name)})")
    for run in store.load_runs(limit=3, name=a.name):
        lines.append(f"  [dim]{_stamp(run['started'])}[/dim] {_status(run['status'])} {run['seconds']}s "
                     f"{escape(' '.join((run.get('summary') or '').split())[:110])}")
    return lines


def _memory_count(name: str) -> int:
    try:
        from src.automation.memory import stats
        return stats(name).get(name, 0)
    except Exception:
        return 0


def result_lines(result: dict) -> list:
    status = result.get("status", "ok")
    seconds = (result["finished"] - result["started"]).total_seconds() if (
        result.get("finished") and result.get("started")
        and not isinstance(result["started"], str)) else None
    head = f"Статус: {_status(status)}" + (f" · {seconds:.0f} с" if seconds is not None else "")
    lines = [head]
    summary = (result.get("summary") or "").strip()
    if summary:
        lines.append(escape(summary[:1800]))
    if result.get("new_items"):
        lines.append(f"[green]Новых элементов запомнено: {result['new_items']}[/green]")
    if result.get("commands"):
        lines.append("[dim]Разрешённые команды, которые выполнялись: " +
                     escape("; ".join(result["commands"])) + "[/dim]")
    if result.get("denied_actions"):
        lines.append("[yellow]Отклонено (нужен человек):[/yellow] " +
                     escape("; ".join(d["action"] for d in result["denied_actions"])))
    if result.get("report"):
        lines.append(f"[dim]Отчёт записан: {escape(str(result['report']))}[/dim]")
    return lines


def templates_lines() -> list:
    lines = ["[bold cyan]Готовые шаблоны[/bold cyan]  (/tasks new <шаблон> [аргумент])"]
    for t in templates.TEMPLATES.values():
        arg = f" <{t.arg_hint.split(',')[0]}>" if t.arg_hint else ""
        lines.append(f"  [bold]{t.id}[/bold]{escape(arg)} — {t.title}")
        lines.append(f"      [dim]{escape(t.description)}[/dim]")
    lines.append("[dim]Потом любую можно подправить: /tasks edit <имя> | schedule: ... | notify: always[/dim]")
    return lines


HELP = [
    "[bold cyan]/tasks[/bold cyan] — работа по расписанию, пока вас нет",
    "  /tasks                          сводка: что идёт, что ближайшее, что требует вас",
    "  /tasks add расписание | задача      — этого достаточно; имя придумается само",
    "  /tasks add [имя |] расписание | задача [| tools: a, b] [| commands: git log*; npm test]",
    "                                  [| notify: always] [| report: .argent/reports/{date}.md]",
    "  /tasks cmd имя | расписание | команда   — команда без модели (бэкап, тесты, проверка)",
    "  /tasks remind через 2 часа | позвонить Ане   — напоминание без модели",
    "  /tasks new <шаблон> [аргумент]  /tasks templates   — готовые",
    "  /tasks edit имя | ключ: значение   (schedule, task, notify, tools, commands, report,",
    "                                      channels, timeout, turns, retries, catchup, max_failures)",
    "  /tasks list · show имя · on/off имя|all · rm имя",
    "  /tasks run имя · stop имя · runs [имя] · last имя · inbox",
    "  /tasks pause [2h|до 18:00] · resume",
    "  /tasks service install|remove|status   — запускать и при закрытом Argent (Windows)",
    "  /tasks memory [имя] · forget имя",
    "[bold]Расписание[/bold]: every 30m · каждые 2 часа · daily at 09:00 · по будням в 9:00 ·",
    "  пн,ср,пт в 18:30 · every 1h between 09:00 and 19:00 · monthly on 15 at 09:00 ·",
    "  через 2 часа · завтра в 8:00 · при запуске",
    "[bold]notify[/bold]: auto (сбой/новое) · failures · changes · always · never",
    "Запущенная без человека задача отклоняет всё, что требует подтверждения; "
    "команды, которые ей можно, перечисляются в commands:",
]


# --- the command ---------------------------------------------------------------------

_SUBCOMMANDS = {
    "status": "status", "статус": "status", "": "status",
    "list": "list", "ls": "list", "список": "list",
    "show": "show", "info": "show",
    "add": "add", "new-task": "add",
    "cmd": "cmd", "command": "cmd",
    "remind": "remind", "напомни": "remind",
    "new": "new", "template": "new",
    "templates": "templates", "шаблоны": "templates",
    "edit": "edit", "изменить": "edit",
    "on": "on", "enable": "on", "вкл": "on",
    "off": "off", "disable": "off", "выкл": "off",
    "rm": "rm", "remove": "rm", "delete": "rm", "удалить": "rm",
    "run": "run", "запустить": "run",
    "stop": "stop", "стоп": "stop",
    "runs": "runs", "history": "runs", "история": "runs",
    "last": "last", "последний": "last",
    "inbox": "inbox", "входящие": "inbox",
    "pause": "pause", "пауза": "pause",
    "resume": "resume", "продолжить": "resume",
    "service": "service", "служба": "service",
    "memory": "memory", "память": "memory",
    "forget": "forget", "забыть": "forget",
    "help": "help", "?": "help", "помощь": "help",
}


def _duration_until(text: str, now: datetime):
    """When a pause ends: '2h', '30m', 'до 18:00', 'until 18:00'. None on nonsense."""
    text = (text or "").strip().lower()
    match = re.match(r"^(?:until|до)\s+(\d{1,2})[:.](\d{2})$", text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if hour > 23 or minute > 59:
            return None
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        return target if target > now else target + timedelta(days=1)
    try:
        spec = parse_schedule(text if re.match(r"^(in|через)\b", text) else f"in {text}", now=now)
    except ScheduleError:
        return None
    return datetime.fromisoformat(spec["at"]) if spec["kind"] == "once" else None


def handle(user_input: str, out, err, live=None, now: datetime = None) -> None:
    """Run one `/tasks ...` command. ``out`` prints markup, ``err`` prints an error."""
    now = now or datetime.now()
    parts = user_input.strip().split(maxsplit=2)
    word = parts[1].lower() if len(parts) > 1 else ""
    arg = parts[2].strip() if len(parts) > 2 else ""
    sub = _SUBCOMMANDS.get(word)
    if sub is None:
        err(f"Неизвестная подкоманда «{word}». Команды: /tasks list | show | add | cmd | remind | "
            f"new | edit | on | off | rm | run | stop | runs | last | inbox | pause | resume | "
            f"service | memory | forget (подробно: /tasks help)")
        return

    def say(lines):
        for line in lines:
            out(line)

    def find(query: str):
        if not query:
            err("Укажите имя автоматизации (список: /tasks list).")
            return None
        found, candidates = store.find_automation(query)
        if found is None:
            err(manage._not_found(query, candidates))
        return found

    if sub == "help":
        say(HELP)
    elif sub == "status":
        say(dashboard_lines(now, live))
    elif sub == "list":
        items = store.load_automations()
        if not items:
            say(dashboard_lines(now, live))
        else:
            out("[bold cyan]Автоматизации:[/bold cyan]")
            say(list_lines(items, now))
    elif sub == "show":
        found = find(arg)
        if found:
            say(show_lines(found, now))
    elif sub in ("add", "cmd"):
        _add(sub, arg, out, err, now)
    elif sub == "remind":
        _remind(arg, out, err, now)
    elif sub == "new":
        _new(arg, out, err, now)
    elif sub == "templates":
        say(templates_lines())
    elif sub == "edit":
        _edit(arg, out, err, now)
    elif sub in ("on", "off"):
        _toggle(sub == "on", arg, out, err)
    elif sub == "rm":
        found = find(arg)
        if found:
            store.remove_automation(found.name)
            out(f"Удалена: {escape(found.name)}")
            if _memory_count(found.name):
                out(f"[dim]Память удалённой ({_memory_count(found.name)} элементов) осталась: "
                    f"/tasks forget {escape(found.name)}[/dim]")
    elif sub == "run":
        found = find(arg)
        if found:
            _run(found, out, err, live)
    elif sub == "stop":
        found = find(arg)
        if found:
            if live is not None and live.stop_run(found.name):
                out(f"Останавливаю «{escape(found.name)}»…")
            else:
                err(f"«{found.name}» сейчас не выполняется в этом окне.")
    elif sub == "runs":
        _runs(arg, out, err)
    elif sub == "last":
        found = find(arg)
        if found:
            _last(found, out)
    elif sub == "inbox":
        _inbox(out, now)
    elif sub == "pause":
        until = _duration_until(arg, now) if arg else None
        if arg and until is None:
            err("Не понял срок. Примеры: /tasks pause 2h · /tasks pause до 18:00 · /tasks pause")
            return
        store.set_paused(until)
        out("⏸ Все автоматизации на паузе " + (f"до {humanize_when(until, now)}." if until else
                                               "— пока не скажете: /tasks resume.") +
            " [dim]Ручной запуск (/tasks run) работает.[/dim]")
    elif sub == "resume":
        store.resume()
        out("▶ Автоматизации снова идут по расписанию.")
    elif sub == "service":
        _service(arg, out, err)
    elif sub == "memory":
        _memory(arg, out)
    elif sub == "forget":
        _forget(arg, out, err)


def _announce_saved(a, notes, out, now, verb="сохранена"):
    out(f"[green]✓[/green] Автоматизация «{escape(a.name)}» {verb}.")
    for line in manage.summary_lines(a, now):
        out(f"  {escape(line)}")
    for note in notes:
        out(f"  [dim]ℹ {escape(note)}[/dim]")
    for warning in manage.warnings_for(a):
        out(f"  [yellow]⚠ {escape(warning)}[/yellow]")
    if a.mode == "agent":
        out("  [dim]Выполняется БЕЗ участия человека: всё, что требует подтверждения, "
            "будет отклонено и записано (/tasks runs).[/dim]")
    out(f"  [dim]Проверить сейчас: /tasks run {escape(a.name)}[/dim]")


def _add(sub: str, arg: str, out, err, now: datetime) -> None:
    parsed = split_add(arg)
    if parsed is None or not all(parsed[1:3]):
        what = "команда" if sub == "cmd" else "задача"
        err(f"Формат: /tasks {sub} <расписание> | <{what}>   (имя и остальное — по желанию: "
            f"/tasks {sub} [имя |] <расписание> | <{what}> [| ключ: значение ...])\n"
            "Примеры расписания: every 30m · каждые 2 часа · по будням в 9:00 · через 2 часа. "
            "Справка: /tasks help")
        return
    name, schedule, task, segments = parsed
    name = name or _auto_name(task)
    options, problems = parse_options(segments)
    if problems:
        err("\n".join(problems))
        return
    fields = {"name": name, "schedule": schedule, "task": task, **options}
    if sub == "cmd":
        fields["mode"] = "command"
    try:
        a, notes = manage.create(fields, now)
    except manage.ManageError as e:
        err("\n".join(e.problems))
        return
    _announce_saved(a, notes, out, now)


def _remind(arg: str, out, err, now: datetime) -> None:
    pieces = arg.split("|")
    if len(pieces) < 2 or not pieces[0].strip() or not pieces[1].strip():
        err("Формат: /tasks remind <когда> | <текст>   например: /tasks remind через 2 часа | позвонить Ане\n"
            "Когда: через 30 минут · завтра в 9:00 · 2026-10-15 18:00 · по будням в 9:00")
        return
    when, text = pieces[0].strip(), pieces[1].strip()
    options, problems = parse_options(pieces[2:])
    if problems:
        err("\n".join(problems))
        return
    taken = {a.name for a in store.load_automations()}
    base = "напоминание-" + (templates._slug(text, 20) or "без-текста")
    name, number = base, 2
    while name in taken:
        name, number = f"{base}-{number}", number + 1
    fields = {"name": name, "schedule": when, "task": text, "mode": "remind", "notify": "always", **options}
    try:
        a, notes = manage.create(fields, now)
    except manage.ManageError as e:
        err("\n".join(e.problems))
        return
    out(f"[green]✓[/green] Напомню: «{escape(text)}» — {escape(describe_schedule(a.schedule))} "
        f"({humanize_when(next_fire(a.schedule, None, now), now)}). [dim]Имя: {escape(a.name)}[/dim]")
    for warning in manage.warnings_for(a):
        out(f"  [yellow]⚠ {escape(warning)}[/yellow]")
    out("  [dim]Напоминание не использует модель и покажется здесь, на рабочем столе "
        "и в Telegram (если добавить `| channels: telegram`).[/dim]")


def _new(arg: str, out, err, now: datetime) -> None:
    head, _, tail = arg.partition("|")
    words = head.split(maxsplit=1)
    if not words:
        for line in templates_lines():
            out(line)
        return
    template = templates.get_template(words[0])
    if template is None:
        err(f"Нет шаблона «{words[0]}». Список: /tasks templates")
        return
    value = words[1].strip() if len(words) > 1 else ""
    options, problems = parse_options([s for s in tail.split("|") if s.strip()] if tail else [])
    if problems:
        err("\n".join(problems))
        return
    try:
        fields = templates.fields_for(template, value)
    except ValueError as e:
        err(str(e))
        return
    fields.update(options)
    fields.setdefault("name", templates.default_name(
        template, value, {a.name for a in store.load_automations()}))
    try:
        a, notes = manage.create(fields, now)
    except manage.ManageError as e:
        err("\n".join(e.problems))
        return
    _announce_saved(a, notes, out, now)


def _edit(arg: str, out, err, now: datetime) -> None:
    name, _, tail = arg.partition("|")
    if not name.strip() or not tail.strip():
        err("Формат: /tasks edit <имя> | ключ: значение [| ключ: значение ...]\n"
            "Ключи: schedule, task, mode, notify, tools, commands, report, channels, timeout, "
            "turns, retries, catchup, max_failures")
        return
    options, problems = parse_options([s for s in tail.split("|") if s.strip()])
    if problems:
        err("\n".join(problems))
        return
    try:
        a, notes = manage.update(name.strip(), options, now)
    except manage.ManageError as e:
        err("\n".join(e.problems))
        return
    _announce_saved(a, notes, out, now, verb="обновлена")


def _toggle(enable: bool, arg: str, out, err) -> None:
    if not arg:
        err("Укажите имя или all.")
        return
    if arg.strip().lower() in ("all", "все"):
        changed = 0
        for a in store.load_automations():
            if a.enabled != enable:
                try:
                    manage.set_enabled(a.name, enable)
                    changed += 1
                except manage.ManageError as e:
                    err("\n".join(e.problems))
        out(f"{'Включено' if enable else 'Выключено'}: {changed}.")
        return
    try:
        a, note = manage.set_enabled(arg.strip(), enable)
    except manage.ManageError as e:
        err("\n".join(e.problems))
        return
    out(f"«{escape(a.name)}»: {'включена' if enable else 'выключена'}. {escape(note)}".strip())


def _run(a, out, err, live) -> None:
    # In the TUI the run goes to the background and the answer arrives as a notice;
    # a command prompt would rather wait for it.
    if live is not None and os.environ.get("ARGENT_TUI") == "1":
        ok, why = live.run_now(a.name)
        if ok:
            out(f"▶ «{escape(a.name)}» запущена в фоне — результат появится здесь, когда будет готов "
                f"[dim](отдельный процесс, ваш чат не затрагивается)[/dim]")
        else:
            err(why)
        return
    if not store.claim_run(a.name):
        err(f"«{a.name}» уже выполняется.")
        return
    try:
        from src.automation.runner import SubprocessRunner
        from ui import console
        out(f"Запускаю «{escape(a.name)}» сейчас (без подтверждений, в отдельном процессе)…")
        with console.status("[bold cyan]Выполняется…[/bold cyan]"):
            result = SubprocessRunner()(a, trigger="manual")
    finally:
        store.release_run(a.name)
    for line in result_lines(result):
        out(line)


def _runs(arg: str, out, err) -> None:
    name = None
    if arg:
        found, candidates = store.find_automation(arg)
        if found is None:
            err(manage._not_found(arg, candidates))
            return
        name = found.name
    runs = store.load_runs(limit=15, name=name)
    if not runs:
        out("Прогонов ещё не было.")
        return
    out("[bold cyan]Последние прогоны:[/bold cyan]")
    for r in runs:
        denied = f", отклонено: {len(r['denied_actions'])}" if r.get("denied_actions") else ""
        fresh = f", нового: {r['new_items']}" if r.get("new_items") else ""
        how = {"manual": " (вручную)", "retry": " (повтор)", "startup": " (при запуске)"}.get(r.get("trigger"), "")
        quiet = " [dim]— без новостей[/dim]" if r.get("silent") else ""
        out(f"  {_stamp(r['started'])}  {escape(r['name'])}  {_status(r['status'])} {r['seconds']}s"
            f"{fresh}{denied}{how}{quiet}")
        if r.get("summary") and not r.get("silent"):
            out(f"      {escape(' '.join(r['summary'].split())[:150])}")
    out("[dim]Полный отчёт последнего прогона: /tasks last <имя>[/dim]")


def _last(a, out) -> None:
    runs = store.load_runs(limit=1, name=a.name)
    if not runs:
        out(f"«{escape(a.name)}» ещё не запускалась.")
        return
    r = runs[-1]
    out(f"[bold cyan]{escape(a.name)}[/bold cyan] · {_stamp(r['started'])} · {_status(r['status'])} · {r['seconds']}s")
    out(escape(r.get("summary") or "(пустой отчёт)"))
    if r.get("commands"):
        out("[dim]Выполнялись разрешённые команды: " + escape("; ".join(r["commands"])) + "[/dim]")
    if r.get("denied_actions"):
        out("[yellow]Отклонено без человека:[/yellow] " + escape("; ".join(d["action"] for d in r["denied_actions"])))
    if r.get("report"):
        out(f"[dim]Отчёт записан: {escape(r['report'])}[/dim]")


def _inbox(out, now: datetime) -> None:
    since = store.last_seen() or now - timedelta(days=1)
    unseen = store.unseen_runs(since)
    if not unseen:
        out("Нового нет — всё, что требовало внимания, вы уже видели. [dim](/tasks runs — весь журнал)[/dim]")
        store.mark_seen(now)
        return
    out(f"[bold cyan]Пока вас не было ({len(unseen)}):[/bold cyan]")
    for r in unseen[-12:]:
        fresh = f" · нового: {r['new_items']}" if r.get("new_items") else ""
        out(f"  [bold]{escape(r['name'])}[/bold] · {_stamp(r['started'])} · {_status(r['status'])}{fresh}")
        body = (r.get("summary") or "").strip()
        if body:
            for line in body.splitlines()[:8]:
                out(f"      {escape(line[:200])}")
        if r.get("denied_actions"):
            out("      [yellow]отклонено:[/yellow] " + escape("; ".join(d["action"] for d in r["denied_actions"][:3])))
    if len(unseen) > 12:
        out(f"[dim]… и ещё {len(unseen) - 12} — /tasks runs[/dim]")
    store.mark_seen(now)


def _service(arg: str, out, err) -> None:
    from src.automation import service
    words = arg.split()
    action = words[0].lower() if words else "status"
    root = store.project_root()
    if action == "install":
        minutes = service.DEFAULT_MINUTES
        if len(words) > 1:
            try:
                minutes = int(words[1])
            except ValueError:
                err("Интервал — целое число минут: /tasks service install 15")
                return
        ok, message = service.install(root, minutes)
        (out if ok else err)(message)
    elif action in ("remove", "uninstall", "rm"):
        ok, message = service.remove(root)
        (out if ok else err)(message)
    elif action == "status":
        info = service.status(root)
        if not info["supported"]:
            out("Автозапуск при закрытом Argent: только Windows. Для других систем:\n  "
                + service.cron_line(root))
        elif info["installed"]:
            out(f"[green]✓[/green] «{escape(info['name'])}» установлена: расписание идёт и при закрытом Argent.")
        else:
            out("Не установлена: задачи идут, только пока Argent открыт. Включить: /tasks service install")
    else:
        err("Формат: /tasks service install [минуты] | remove | status")


def _memory(arg: str, out) -> None:
    from src.automation.memory import stats
    counts = {k: v for k, v in stats(arg or None).items() if v}
    if not counts:
        out("Память пуста. Она наполняется, когда задача вызывает filter_new_items — "
            "так мониторинг сообщает только о новом, а не повторяет один и тот же список каждый прогон.")
        return
    out("[bold cyan]Запомнено элементов:[/bold cyan]")
    for name, count in sorted(counts.items()):
        out(f"  • {escape(name)}: {count}")
    out("[dim]Сброс: /tasks forget <имя>[/dim]")


def _forget(arg: str, out, err) -> None:
    from src.automation.memory import forget
    if not arg:
        err("Формат: /tasks forget <имя>")
        return
    found, _ = store.find_automation(arg)
    target = found.name if found else arg
    dropped = forget(target)
    out(f"Забыто элементов: {dropped}. Следующий прогон «{escape(target)}» снова сочтёт всё новым."
        if dropped else f"Для «{escape(target)}» ничего не запомнено.")
