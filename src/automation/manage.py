"""Creating and changing automations — one code path for everyone who does it.

The `/tasks` command and the agent's `manage_automations` tool both end up here,
so a definition means the same thing (and is checked the same way) whoever typed
it: schedules are normalised to their canonical form, tools are checked against
the real tool list, and a problem is a list of plain sentences, not a traceback.
"""

from datetime import datetime

from src.automation import store
from src.automation.schedule import (
    ScheduleError, describe_schedule, humanize_when, is_one_shot, next_fire, normalize_schedule,
)
from src.automation.store import Automation

KEYS = ("name", "schedule", "task", "mode", "notify", "channels", "tools", "commands",
        "report", "timeout", "turns", "retries", "catchup", "max_failures", "enabled")

NOTIFY_HELP = {
    "auto": "только сбой, отказ без человека или новое",
    "failures": "только сбой или отказ без человека",
    "changes": "когда отчёт изменился с прошлого раза (и при сбое)",
    "always": "после каждого прогона",
    "never": "никогда (журнал /tasks runs остаётся)",
}

MODE_HELP = {
    "agent": "модель выполняет задачу с ограниченным набором инструментов",
    "command": "команда выполняется как написана — без модели и токенов",
    "remind": "текст просто показывается в назначенное время",
}

# Faster than this and an agent job is a token fire-hose nobody is watching.
AGENT_INTERVAL_WARNING_SECONDS = 10 * 60


class ManageError(ValueError):
    """The definition has problems; ``problems`` lists every one of them."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace("\n", ";").split(";") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _as_int(value, field_name: str):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ManageError(f"{field_name}: нужно целое число, получено {value!r}") from None


def known_tool_names():
    try:
        from tools.schemas import AVAILABLE_TOOLS
        return set(AVAILABLE_TOOLS) if isinstance(AVAILABLE_TOOLS, dict) else None
    except Exception:
        return None


def _tool_list(value) -> list:
    if isinstance(value, str):
        value = [p for p in value.replace(";", ",").split(",")]
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def build(fields: dict, base: Automation = None, now: datetime = None):
    """(automation, notes): ``base`` with ``fields`` applied, validated.

    ``notes`` are things worth telling the person about what was decided for
    them (a tool added because a command needs it). Raises ManageError.
    """
    now = now or datetime.now()
    unknown = [k for k in fields if k not in KEYS]
    if unknown:
        raise ManageError(f"неизвестные параметры: {', '.join(unknown)}")

    a = Automation(name="", task="", schedule="") if base is None else Automation(**vars(base))
    notes, problems = [], []

    if "name" in fields and fields["name"] is not None:
        if base is not None and fields["name"] != base.name:
            raise ManageError("переименовать нельзя: создайте автоматизацию заново под новым именем")
        a.name = str(fields["name"])
    for key in ("task", "mode", "notify"):
        if fields.get(key) is not None:
            setattr(a, key, str(fields[key]).strip())
    if "schedule" in fields and fields["schedule"] is not None:
        try:
            a.schedule = normalize_schedule(str(fields["schedule"]), now)
        except ScheduleError as e:
            problems.append(str(e))
            a.schedule = str(fields["schedule"])      # so validation does not also say "empty"
    if "channels" in fields and fields["channels"] is not None:
        a.channels = [c.lower() for c in _as_list(fields["channels"])]
    if "tools" in fields and fields["tools"] is not None:
        a.allowed_tools = _tool_list(fields["tools"])
    if "commands" in fields and fields["commands"] is not None:
        a.allowed_commands = _as_list(fields["commands"])
    if "report" in fields and fields["report"] is not None:
        a.report_file = str(fields["report"]).strip()
    for key, attr in (("timeout", "timeout_minutes"), ("turns", "max_turns"),
                      ("retries", "retries"), ("catchup", "catchup_minutes"),
                      ("max_failures", "max_failures")):
        if key in fields:
            try:
                number = _as_int(fields[key], key)
            except ManageError as e:
                problems.extend(e.problems)
                continue
            if number is not None:
                setattr(a, attr, number)
    if "enabled" in fields and fields["enabled"] is not None:
        a.enabled = bool(fields["enabled"])

    if a.mode != "agent":
        # Tools and commands belong to the model; they mean nothing here.
        if a.allowed_tools or a.allowed_commands:
            notes.append("в режиме без модели инструменты и разрешённые команды не используются — убраны")
        a.allowed_tools, a.allowed_commands = [], []
    else:
        known = known_tool_names()
        if known is not None:
            bad = [t for t in a.allowed_tools if t not in known]
            if bad:
                problems.append(f"неизвестные инструменты: {', '.join(bad)} (список — /tools)")
        if (a.allowed_commands and a.allowed_tools
                and not {"run_command", "start_background_command"} & set(a.allowed_tools)):
            a.allowed_tools = a.allowed_tools + ["run_command"]
            notes.append("добавлен инструмент run_command: без него разрешённые команды бесполезны")

    if base is not None and a.schedule != base.schedule and (a.disabled_reason or "").startswith("выполнена"):
        a.enabled, a.disabled_reason = True, ""
        notes.append("новое время назначено — автоматизация снова включена")

    problems.extend(p for p in store.validate_automation(a) if p not in problems)
    if problems:
        raise ManageError(problems)
    return a, notes


def create(fields: dict, now: datetime = None):
    """Add a new automation. Returns (automation, notes)."""
    if "name" not in fields:
        raise ManageError("не указано имя")
    taken = store.get_automation(str(fields["name"]))
    if taken is not None:
        raise ManageError(f"автоматизация '{taken.name}' уже есть: изменить — /tasks edit")
    a, notes = build(fields, None, now)
    store.upsert_automation(a)
    return a, notes


def update(name: str, fields: dict, now: datetime = None):
    """Change an existing automation. Returns (automation, notes)."""
    base, candidates = store.find_automation(name)
    if base is None:
        raise ManageError(_not_found(name, candidates))
    a, notes = build(fields, base, now)

    def _apply(stored: Automation) -> None:
        # Only what this edit changed: a job that finished while the edit was
        # being typed has stamped its own bookkeeping on the stored copy.
        for key, value in vars(a).items():
            if value != getattr(base, key):
                setattr(stored, key, value)

    updated = store.update_automation(base.name, _apply)
    if updated is None:
        raise ManageError(f"автоматизация '{base.name}' исчезла во время правки")
    return updated, notes


def set_enabled(name: str, enabled: bool):
    """Switch one automation on or off. Returns (automation, note) — switching
    on clears the failure streak that paused it, since a person looked at it."""
    base, candidates = store.find_automation(name)
    if base is None:
        raise ManageError(_not_found(name, candidates))
    note = ""
    finished = (base.disabled_reason or "").startswith("выполнена") or (
        bool(base.last_run) and next_fire(base.schedule, base.last_run_dt(), datetime.now()) is None)
    if enabled and is_one_shot(base.schedule) and finished:
        raise ManageError(f"'{base.name}' — разовая и уже выполнена. Новое время: "
                          f"/tasks edit {base.name} | schedule: завтра в 09:00")

    def _flip(a: Automation) -> None:
        a.enabled = enabled
        if enabled:
            a.fail_streak, a.disabled_reason = 0, ""
        elif not a.disabled_reason:
            a.disabled_reason = "выключена вручную"

    updated = store.update_automation(base.name, _flip)
    if updated is None:
        raise ManageError(f"автоматизация '{base.name}' исчезла")
    if enabled and base.disabled_reason and not base.disabled_reason.startswith("выключена"):
        note = f"(была отключена: {base.disabled_reason})"
    return updated, note


def _not_found(name: str, candidates) -> str:
    if candidates:
        return f"'{name}' подходит под несколько: {', '.join(candidates)} — уточните имя"
    return f"автоматизация '{name}' не найдена (список — /tasks list)"


def warnings_for(a: Automation) -> list:
    out = []
    if a.mode == "agent":
        from src.automation.schedule import parse_schedule
        try:
            spec = parse_schedule(a.schedule, now=datetime.now())
        except ScheduleError:
            spec = {}
        if spec.get("kind") == "interval" and spec["seconds"] < AGENT_INTERVAL_WARNING_SECONDS:
            out.append("каждый прогон — это работа модели и расход токенов; чаще 10 минут "
                       "почти никогда не нужно (для проверки без модели есть режим command)")
        if not a.allowed_tools:
            out.append("инструменты не ограничены — задача получит весь набор; сузьте: "
                       "`| tools: search_web, read_webpage`")
        elif "run_command" in a.allowed_tools and not a.allowed_commands:
            out.append("run_command включён, но команд не разрешено — любая будет отклонена; "
                       "перечислите нужные: `| commands: git log*; npm test`")
        broad = [c for c in a.allowed_commands if c.strip().startswith("*")]
        if broad:
            out.append(f"слишком широкий шаблон команды: {', '.join(broad)}")
    if a.mode == "command":
        out.append("команда выполняется как написана — без модели и без подтверждений; "
                   "проверьте её руками один раз: /tasks run " + a.name)
    if "telegram" in a.channels:
        try:
            from config import get_telegram_chat_id, get_telegram_token
            if not (get_telegram_token() and get_telegram_chat_id()):
                out.append("канал telegram выбран, но бот не настроен (токен и chat id) — "
                           "уведомление туда не дойдёт")
        except Exception:
            pass
    return out


def summary_lines(a: Automation, now: datetime = None) -> list:
    """What a saved automation will do, in plain Russian."""
    now = now or datetime.now()
    when = humanize_when(next_fire(a.schedule, a.last_run_dt(), now, a.catchup_minutes * 60), now)
    lines = [f"{describe_schedule(a.schedule)} (следующий запуск: {when})",
             f"режим: {a.mode} — {MODE_HELP.get(a.mode, '')}",
             f"сообщать: {a.notify} — {NOTIFY_HELP.get(a.notify, '')}"]
    if a.allowed_tools:
        lines.append(f"инструменты: {', '.join(a.allowed_tools)}")
    if a.allowed_commands:
        lines.append(f"разрешённые команды: {'; '.join(a.allowed_commands)}")
    if a.report_file:
        lines.append(f"отчёт в файл: {a.report_file}")
    if a.channels:
        lines.append(f"доставка: рабочий стол + {', '.join(a.channels)}")
    return lines
