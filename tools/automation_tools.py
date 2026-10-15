"""manage_automations — let the agent set up scheduled work from a conversation.

"Watch the uv releases every morning" should not require learning `/tasks add`
syntax: the model builds the definition, the person approves it, and from then on
it runs by itself. Everything goes through src/automation/manage.py, so the tool
and the slash command accept and reject exactly the same things.

Two rules keep this from being a way for a model to give itself standing powers:

* Anything that changes or runs an automation asks the person first, showing the
  schedule, the task and — spelled out — the commands it would be allowed to run.
  A definition with pre-approved commands, or one that is itself a shell command,
  is submitted as a *destructive* approval: it is always asked, and refused
  outright in /vibe mode, where nobody is there to ask.
* An unattended run cannot use it (it is on the runner's denylist, and its
  approvals are refused by the gate), so a scheduled job can never reschedule
  itself or widen its own permissions.
"""

import threading
from datetime import datetime

from logger import get_logger

log = get_logger("automation")

ACTIONS = ("list", "show", "create", "update", "remove", "enable", "disable", "run", "runs",
           "templates")

_MUTATING = {"create", "update", "remove", "enable", "disable", "run"}


def _brief(a, now: datetime) -> str:
    from src.automation import manage
    state = "вкл" if a.enabled else f"ВЫКЛ ({a.disabled_reason or 'вручную'})"
    return (f"- {a.name} [{state}] {a.mode}: {manage.summary_lines(a, now)[0]}"
            f" | последний: {a.last_run or 'никогда'} {a.last_status or ''}".rstrip())


def _describe_for_approval(verb: str, a, notes) -> tuple:
    """(approval text, is_standing_power) for a definition about to be saved."""
    from src.automation.schedule import describe_schedule
    parts = [f"{verb} автоматизацию «{a.name}» — {describe_schedule(a.schedule)}"]
    if a.mode == "command":
        parts.append(f"ВЫПОЛНЯТЬ КОМАНДУ без модели и без подтверждений: {a.task}")
    elif a.mode == "remind":
        parts.append(f"напоминание: {a.task}")
    else:
        parts.append(f"задача: {a.task[:200]}")
        parts.append("инструменты: " + (", ".join(a.allowed_tools) or "все"))
        if a.allowed_commands:
            parts.append("МОДЕЛИ БУДЕТ РАЗРЕШЕНО выполнять без подтверждения: " + "; ".join(a.allowed_commands))
    if a.report_file:
        parts.append(f"отчёт в файл: {a.report_file}")
    if a.channels:
        parts.append("уведомления в: " + ", ".join(a.channels))
    standing = a.mode == "command" or bool(a.allowed_commands)
    return " | ".join(parts), standing


def _ask(text: str, destructive: bool) -> bool:
    from approval import request_approval
    return request_approval(text, destructive=destructive)


def _start_in_background(a) -> str:
    from src.automation import store
    from src.automation.scheduler import get_scheduler

    live = get_scheduler()
    if live is not None:
        ok, why = live.run_now(a.name)
        return "Запущена в фоне; результат — action=runs." if ok else f"Error: {why}"
    if not store.claim_run(a.name):
        return f"Error: «{a.name}» уже выполняется."

    def _work():
        try:
            from src.automation.runner import SubprocessRunner
            SubprocessRunner()(a, trigger="manual")
        finally:
            store.release_run(a.name)

    threading.Thread(target=_work, daemon=True, name=f"automation-{a.name}").start()
    return "Запущена в фоне; результат — action=runs."


def manage_automations(action: str, name: str = None, schedule: str = None, task: str = None,
                       mode: str = None, notify: str = None, tools: list = None,
                       commands: list = None, channels: list = None, report_file: str = None,
                       timeout_minutes: int = None, max_turns: int = None, retries: int = None,
                       template: str = None, template_arg: str = None, limit: int = 10) -> str:
    """Create, change, list, run or remove scheduled automations."""
    from src.automation import manage, store, templates
    from src.automation.schedule import ScheduleError

    action = (action or "").strip().lower()
    if action not in ACTIONS:
        return f"Error: action must be one of {', '.join(ACTIONS)}."

    if action in _MUTATING:
        import approval
        if approval._nobody_is_watching():
            return "Error: automations cannot be changed or run from an unattended run."

    now = datetime.now()

    if action == "templates":
        return "\n".join(
            f"- {t.id}: {t.title}. {t.description}"
            + (f" Аргумент: {t.arg_hint}." if t.arg_hint else "") for t in templates.TEMPLATES.values())

    if action == "list":
        items = store.load_automations()
        return "\n".join(_brief(a, now) for a in items) if items else "Автоматизаций нет."

    if action in ("show", "update", "remove", "enable", "disable", "run") and not name:
        return "Error: name is required."

    if action == "runs":
        found = None
        if name:
            found, candidates = store.find_automation(name)
            if found is None:
                return "Error: " + manage._not_found(name, candidates)
        runs = store.load_runs(limit=max(1, min(int(limit or 10), 50)), name=found.name if found else None)
        if not runs:
            return "Прогонов ещё не было."
        return "\n".join(
            f"- {r['started'][:16]} {r['name']} [{r['status']}] {r['seconds']}s"
            f"{' нового: ' + str(r['new_items']) if r.get('new_items') else ''}: "
            f"{' '.join((r.get('summary') or '').split())[:300]}" for r in runs)

    if action == "show":
        a, candidates = store.find_automation(name)
        if a is None:
            return "Error: " + manage._not_found(name, candidates)
        lines = [_brief(a, now), f"задача: {a.task}"] + manage.summary_lines(a, now)[1:]
        lines += [f"предупреждение: {w}" for w in manage.warnings_for(a)]
        return "\n".join(lines)

    fields ={"schedule": schedule, "task": task, "mode": mode, "notify": notify, "tools": tools,
              "commands": commands, "channels": channels, "report": report_file,
              "timeout": timeout_minutes, "turns": max_turns, "retries": retries}
    fields = {k: v for k, v in fields.items() if v is not None}

    try:
        if action == "create":
            if template:
                tpl = templates.get_template(template)
                if tpl is None:
                    return f"Error: unknown template '{template}' (action=templates lists them)."
                base_fields = templates.fields_for(tpl, template_arg or "")
                base_fields.update(fields)
                fields = base_fields
                name = name or templates.default_name(
                    tpl, template_arg or "", {a.name for a in store.load_automations()})
            if not name:
                return "Error: name is required."
            fields["name"] = name
            if store.get_automation(name) is not None:
                return f"Error: automation '{name}' already exists (use action=update)."
            preview, notes = manage.build(fields, None, now)
            text, standing = _describe_for_approval("создать", preview, notes)
            if not _ask(text, destructive=standing):
                return "Aborted: the user did not approve creating this automation."
            created, notes = manage.create(fields, now)
            extra = "".join(f"\n- {n}" for n in notes + manage.warnings_for(created))
            return (f"Создана «{created.name}»: " + "; ".join(manage.summary_lines(created, now)[:2])
                    + (f"\nЗаметки:{extra}" if extra else ""))

        found, candidates = store.find_automation(name)
        if found is None:
            return "Error: " + manage._not_found(name, candidates)

        if action == "update":
            if not fields:
                return "Error: nothing to change — pass at least one field."
            preview, notes = manage.build(fields, found, now)
            text, standing = _describe_for_approval("изменить", preview, notes)
            if not _ask(text, destructive=standing):
                return "Aborted: the user did not approve this change."
            updated, notes = manage.update(found.name, fields, now)
            extra = "".join(f"\n- {n}" for n in notes + manage.warnings_for(updated))
            return f"Обновлена «{updated.name}»." + (f"\nЗаметки:{extra}" if extra else "")

        if action == "remove":
            if not _ask(f"УДАЛИТЬ автоматизацию «{found.name}» ({found.schedule})", destructive=True):
                return "Aborted: the user did not approve removing it."
            store.remove_automation(found.name)
            return f"Удалена «{found.name}»."

        if action in ("enable", "disable"):
            verb = "включить" if action == "enable" else "выключить"
            if not _ask(f"{verb} автоматизацию «{found.name}»", destructive=False):
                return f"Aborted: the user did not approve."
            updated, note = manage.set_enabled(found.name, action == "enable")
            return f"«{updated.name}»: {'включена' if updated.enabled else 'выключена'}. {note}".strip()

        if action == "run":
            if not _ask(f"запустить автоматизацию «{found.name}» сейчас ({found.mode}: {found.task[:120]})",
                        destructive=found.mode == "command" or bool(found.allowed_commands)):
                return "Aborted: the user did not approve running it."
            return _start_in_background(found)
    except manage.ManageError as e:
        return "Error: " + "; ".join(e.problems)
    except ScheduleError as e:
        return f"Error: {e}"
    except ValueError as e:
        return f"Error: {e}"
    return "Error: unsupported action."
