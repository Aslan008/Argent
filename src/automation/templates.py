"""Ready-made automations: pick one, give it its one argument, done.

Writing a schedule, a toolset and a task prompt that works unattended is the hard
part of setting an automation up; these are the ones that come up again and again,
already narrowed to the tools and commands they need. Nothing here is special —
`/tasks new` just fills in an ordinary definition you can edit afterwards.

Where a job is only "run this and tell me if it fails" it is a `command` job:
no model, no tokens, no provider that can be down at 03:00.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    description: str
    schedule: str
    task: str
    mode: str = "agent"
    tools: tuple = ()
    commands: tuple = ()
    notify: str = "auto"
    report_file: str = ""
    arg_hint: str = ""                  # empty: the template takes no argument
    extra: dict = field(default_factory=dict)

    @property
    def needs_arg(self) -> bool:
        return bool(self.arg_hint)


_READ_ONLY = ("read_file", "list_directory", "grep_search", "search_files")

TEMPLATES = {t.id: t for t in (
    Template(
        id="morning",
        title="Утренняя сводка по проекту",
        description="Будни в 09:00: что изменено и не закоммичено, что вошло за сутки, где висят TODO.",
        schedule="weekdays at 09:00",
        tools=("run_command",) + _READ_ONLY,
        commands=("git status*", "git log*", "git diff --stat*", "git branch*"),
        notify="always",
        report_file=".argent/reports/morning-{date}.md",
        task=("Подготовь короткую утреннюю сводку по этому репозиторию: что изменено, но не "
              "закоммичено (git status), какие коммиты появились за последние сутки "
              "(git log --since=yesterday --oneline), и где остались TODO/FIXME в изменённых "
              "файлах. 5-10 строк, без вступлений."),
    ),
    Template(
        id="tests",
        title="Тесты по расписанию",
        description="Раз в час в рабочее время запускает ваши тесты; сообщает, только если они упали. Без модели.",
        schedule="every 1h on weekdays between 09:00 and 19:00",
        mode="command",
        notify="failures",
        arg_hint="команда запуска тестов, например: pytest -q",
        task="{arg}",
    ),
    Template(
        id="command",
        title="Команда раз в день",
        description="Запускает команду каждый день в 03:00 (бэкап, очистка, обновление). Сообщает только о сбое. Без модели.",
        schedule="daily at 03:00",
        mode="command",
        notify="failures",
        arg_hint="команда, например: python scripts/backup.py",
        task="{arg}",
    ),
    Template(
        id="page",
        title="Слежение за страницей",
        description="Раз в час открывает страницу и сообщает, что на ней изменилось (новости, цены, версии).",
        schedule="every 1h",
        tools=("read_webpage", "search_web"),
        notify="changes",
        arg_hint="адрес страницы, например: https://example.com/pricing",
        task=("Открой страницу {arg} и сравни с тем, что было при прошлом прогоне (он указан выше). "
              "Если изменилось что-то существенное — новые записи, цены, версии, статус, — опиши "
              "это в 2-4 строках. Если ничего значимого, ответь NOTHING_NEW."),
    ),
    Template(
        id="releases",
        title="Новые релизы на GitHub",
        description="Раз в 6 часов проверяет релизы репозитория и сообщает только о новых.",
        schedule="every 6h",
        tools=("read_webpage",),
        notify="auto",
        arg_hint="репозиторий в виде владелец/имя, например: astral-sh/uv",
        task=("Прочитай https://github.com/{arg}/releases.atom и выпиши названия релизов. Передай "
              "их список в filter_new_items. Если новых нет — ответь NOTHING_NEW, иначе перечисли "
              "новые релизы с одной строкой о том, что в них важного."),
    ),
    Template(
        id="news",
        title="Дайджест новостей по теме",
        description="Каждый день в 08:30 ищет свежее по теме и присылает 3-5 новых материалов.",
        schedule="daily at 08:30",
        tools=("search_web", "read_webpage"),
        notify="always",
        arg_hint="тема, например: Rust async runtime",
        task=("Найди в сети самое свежее за последние сутки по теме «{arg}». Отбери 3-5 самых "
              "содержательных материалов, передай их адреса в filter_new_items и опиши только "
              "новые: заголовок, источник, одна строка сути. Если нового нет — NOTHING_NEW."),
    ),
    Template(
        id="todo",
        title="Недельный обзор TODO",
        description="По пятницам в 17:00 собирает TODO/FIXME по проекту и называет самые старые и важные.",
        schedule="fri at 17:00",
        tools=_READ_ONLY,
        notify="always",
        report_file=".argent/reports/todo-{date}.md",
        task=("Найди в проекте все TODO и FIXME (grep_search). Сгруппируй по файлам, назови десять "
              "самых важных или самых забытых и предложи, что закрыть на следующей неделе."),
    ),
    Template(
        id="deps",
        title="Устаревшие зависимости",
        description="По понедельникам в 10:00 проверяет устаревшие пакеты pip/npm; сообщает, если список изменился.",
        schedule="mon at 10:00",
        tools=("run_command", "read_file", "list_directory"),
        commands=("pip list --outdated*", "npm outdated*"),
        notify="changes",
        task=("Проверь устаревшие зависимости проекта: `pip list --outdated`, если есть Python-проект, "
              "и `npm outdated`, если есть package.json. Выпиши, что устарело больше чем на мажорную "
              "версию или закрывает уязвимость. Если всё свежее — NOTHING_NEW."),
    ),
)}


def get_template(template_id: str):
    return TEMPLATES.get((template_id or "").strip().lower())


def _slug(text: str, limit: int = 28) -> str:
    text = re.sub(r"^https?://", "", text.strip())
    text = re.sub(r"[^\w./-]+", "-", text, flags=re.UNICODE).strip("-.")
    return text[:limit].rstrip("-.")


def default_name(template: Template, arg: str, taken) -> str:
    """A readable name for a template instance that does not clash with an
    existing automation."""
    # A test or shell command is long and unreadable as a name; the template id says enough.
    named_by_arg = bool(arg) and template.mode == "agent"
    base = f"{template.id}-{_slug(arg)}" if named_by_arg and _slug(arg) else template.id
    name, number = base, 2
    taken = set(taken)
    while name in taken:
        name = f"{base}-{number}"
        number += 1
    return name


def fields_for(template: Template, arg: str = "") -> dict:
    """The definition fields this template stands for, with {arg} filled in."""
    if template.needs_arg and not (arg or "").strip():
        raise ValueError(f"шаблону '{template.id}' нужен аргумент: {template.arg_hint}")
    value = (arg or "").strip()

    def fill(text: str) -> str:
        return text.replace("{arg}", value)

    return {
        "schedule": template.schedule,
        "task": fill(template.task),
        "mode": template.mode,
        "tools": list(template.tools),
        "commands": [fill(c) for c in template.commands],
        "notify": template.notify,
        "report": template.report_file,
    }
