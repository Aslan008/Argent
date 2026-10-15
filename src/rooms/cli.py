"""The `/rooms` command: one dispatcher, kept free of the terminal.

Everything the command needs from the person — a yes/no, a choice, an answer to a
`human_pause` — arrives through `Io`, so the same code serves the REPL, the TUI
bridge and the tests. See HELP for the commands.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.rooms import history as run_history
from src.rooms.library import describe_room, short_description
from src.rooms.models import Room
from src.rooms.router import pick_room
from src.rooms.session import (
    available_tool_names, build_library, default_journal, execute_room_run, parse_params,
    user_rooms_dir,
)
from src.rooms.validator import validate_room

HELP = """\
/rooms — комнаты: повторяемые рецепты с рельсами (проверки процессом, лимиты, вопросы человеку)
  /rooms <задача>                  сама выберет комнату по задаче (если не уверена — спросит)
  /rooms run <комната> [k=v] [задача]   запустить конкретную; k=v — параметры (test_command="pytest -x")
  /rooms list                      комнаты, их статистика и теги
  /rooms show <комната>            граф комнаты
  /rooms route <задача>            какую комнату выберет роутер и почему (ничего не запускает)
  /rooms resume                    продолжить прерванный прогон
  /rooms last | history            итог последнего прогона | список прогонов
  /rooms build <описание>          составить комнату из описания словами (покажу схему, сохраню после «да»)
  /rooms edit <комната> <правка>   изменить комнату словами (стартовые сохраняются копией <имя>_custom)
  /rooms diagram <комната> [mermaid]  схема комнаты (текст или Mermaid)
  /rooms editor [stop]             визуальный редактор в браузере (перетаскивать узлы, соединять, проверять, сохранять)
  /rooms triggers [комната]        что запускает комнаты само: расписания и webhook
  /rooms triggers schedule <комната> <когда>   запуск по расписанию («каждый день в 03:00»)
  /rooms triggers webhook <комната> [имя]      запуск по HTTP-запросу (секрет показывается один раз)
  /rooms triggers remove|on|off <имя>          убрать | включить | выключить
  /rooms new <имя>                 пустая заготовка своей комнаты в .argent/rooms/
  /rooms validate <файл.json>      проверить комнату, не добавляя
  /rooms delete <имя>              удалить свою комнату
  /rooms unquarantine <имя>        вернуть комнату из карантина
  /rooms spawn on|off              ИИ может предлагать новые комнаты (через валидатор и твоё одобрение)
  /rooms hint on|off               подсказывать модели подходящую комнату (одна строка, только если явно подходит)
Пример: /rooms run debug_loop test_command="python -m pytest tests/unit -q"
"""

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

_SCAFFOLD = {
    "room": "",
    "description": "Что делает эта комната — одним предложением (по описанию и тегам её находит роутер).",
    "tags": [],
    "budget": {"max_iterations": 10, "max_tokens": 40000},
    "entry": "work",
    "defaults": {},
    "next": {"done": "END"},
    "report": "{{result}}",
    "nodes": [
        {"id": "work", "type": "agent", "role": "Coder",
         "budget": {"max_iterations": 6, "max_tokens": 20000},
         "writes": ["result"],
         "instruction": "Task: {{task}}\nDo the work. Answer with the JSON object: result=<what you did>."}
    ],
    "edges": [{"from": "work", "to": "EXIT:done"}],
}


@dataclass
class Io:
    out: Callable[[str], None]
    err: Callable[[str], None]
    confirm: Callable[[str], bool]
    choose: Callable[[str, list], Optional[str]]
    human: Optional[Callable] = None          # (node, state) -> None; fills state.data["<id>.output"]
    confirm_step: Optional[Callable] = None   # (node, state) -> bool; for `confirm: true` tool nodes
    ask_text: Optional[Callable] = None       # (question) -> str; a free-text answer (refining a draft room)


def _library(io: Io, library):
    lib = library if library is not None else build_library()
    if lib.load_errors:
        for name, errs in lib.load_errors.items():
            io.err(f"Комната «{name}» не загружена: {'; '.join(errs)}")
    return lib


def _short(text: str, n: int = 78) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def _cmd_list(io: Io, lib) -> None:
    if not lib.rooms:
        io.out("Комнат нет.")
        return
    for name in lib.names():
        st = lib.stats.get(name, {})
        q = " [КАРАНТИН]" if st.get("quarantined") else ""
        room = lib.rooms[name]
        io.out(f"  {name} [{lib.sources.get(name, '?')}] {st.get('successes', 0)}✓/{st.get('failures', 0)}✗{q}")
        io.out(f"      {short_description(room, 110)}")
        if room.tags:
            io.out(f"      теги: {', '.join(room.tags[:8])}")


def _cmd_route(io: Io, lib, task: str) -> None:
    if not task:
        io.err("Использование: /rooms route <задача>")
        return
    pick = pick_room(task, lib)
    if not pick.ranked:
        io.out("Ни одна комната не подошла по словам. Команда /rooms list покажет, что есть.")
        return
    io.out(f"Выбор: {pick.name} ({'уверенно' if pick.clear else 'неуверенно — спрошу'})")
    for name, score, why in pick.ranked[:5]:
        io.out(f"  {name}: {score}  (слова: {why})")


def _cmd_history(io: Io, record_dir) -> None:
    runs = run_history.list_runs(record_dir, limit=15)
    if not runs:
        io.out("Прогонов ещё не было.")
        return
    for r in runs:
        route = " -> ".join(f"{a}:{b}" for a, b in r.get("route", []))
        io.out(f"  {r.get('id')}  {r.get('outcome'):<9} {_short(r.get('task') or '', 40)}  [{route}]")


def _cmd_last(io: Io, record_dir, ref: str = "last") -> None:
    rec = run_history.load_run(ref or "last", record_dir)
    if rec is None:
        io.out("Прогонов ещё не было.")
        return
    io.out(f"Прогон {rec.get('id')} — {rec.get('outcome')}, {rec.get('seconds')} с")
    io.out(f"Задача: {rec.get('task') or '(продолжение)'}")
    io.out("Маршрут: " + " -> ".join(f"{a}:{b}" for a, b in rec.get("route", [])))
    for n in rec.get("nodes", []):
        io.out(f"  · {n.get('room')} › {n.get('node')} [{n.get('kind')}] {n.get('seconds')} с")
    tokens = (rec.get("usage") or {}).get("tokens")
    if tokens:
        io.out(f"Расход: ≈{int(tokens)} ток")


def _cmd_new(io: Io, lib, name: str, new_dir) -> None:
    if not _NAME.match(name or ""):
        io.err("Имя комнаты: латиница, цифры, _ (начинается с буквы), например: weekly_report")
        return
    if lib.get(name):
        io.err(f"Комната «{name}» уже есть.")
        return
    data = json.loads(json.dumps(_SCAFFOLD))
    data["room"] = name
    errors = lib.add_room(Room(**data), _tools(lib))
    if errors:
        io.err("Заготовка не прошла проверку: " + "; ".join(errors))
        return
    where = (lib.user_dir or user_rooms_dir()) / f"{name}.json"
    io.out(f"Создана заготовка: {where}")
    io.out("Допишите описание и теги (по ним роутер находит комнату), поправьте узлы; проверка: /rooms validate " + str(where))


def _tools(lib) -> set:
    try:
        return available_tool_names()
    except Exception:
        return set()


def _cmd_validate(io: Io, lib, path: str) -> None:
    from pathlib import Path
    if not path:
        io.err("Использование: /rooms validate <файл.json>")
        return
    p = Path(path.strip('"'))
    if not p.exists():
        io.err(f"Нет файла: {p}")
        return
    try:
        room = Room(**json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:
        io.err(f"Не разобралась схема: {e}")
        return
    errors = validate_room(room, _tools(lib), allow_spawn=(room.room in lib._spawn_rooms))
    if errors:
        io.err("Комната не годится:")
        for e in errors:
            io.err(f"  - {e}")
    else:
        io.out(f"Комната «{room.room}» корректна: узлов {len(room.nodes)}, рёбер {len(room.edges)}.")


def _cmd_delete(io: Io, lib, name: str) -> None:
    if lib.sources.get(name) != "user":
        io.err("Удалять можно только свои комнаты (из .argent/rooms/)." if name in lib.rooms
               else f"Нет комнаты «{name}».")
        return
    if not io.confirm(f"Удалить комнату «{name}»?"):
        io.out("Отменено.")
        return
    lib.remove_room(name)
    lib.save_stats()
    io.out(f"Комната «{name}» удалена.")


def _start_run(io: Io, lib, room_name: str, task: str, params: dict, runner, journal, record_dir) -> None:
    room = lib.get(room_name)
    if room is None:
        io.err(f"Нет комнаты «{room_name}». /rooms list — список.")
        return
    if lib.stats.get(room_name, {}).get("quarantined"):
        io.err(f"Комната «{room_name}» в карантине (часто заканчивалась неудачей). "
               f"Вернуть: /rooms unquarantine {room_name}")
        return
    if journal is not None and journal.unfinished():
        left = journal.last().get("room", "?")
        if not io.confirm(f"Есть прерванный прогон (комната «{left}»). Новый запуск сотрёт его журнал, и "
                          "продолжить его будет нельзя (/rooms resume). Запустить новый?"):
            io.out("Отменено. Продолжить прерванный: /rooms resume")
            return
    b = room.budget
    limit = f"до {b.max_iterations} шагов, ≈{b.max_tokens} ток" + (f", {b.max_seconds} с" if b.max_seconds else "")
    if not io.confirm(f"Автономный прогон комнаты «{room_name}» (реальные инструменты и суб-агенты; {limit}). Продолжить?"):
        io.out("Отменено.")
        return
    io.out(f"[bold cyan]Rooms[/bold cyan]: {room_name} — {task or '(без задачи)'}")
    try:
        summary = execute_room_run(
            task, room_name, params=params, library=lib, runner=runner, journal=journal,
            human_prompt=io.human, confirm=io.confirm_step, printer=io.out, record_dir=record_dir,
        )
    except Exception as e:
        io.err(f"Запуск комнаты не удался: {e}")
        return
    io.out("")
    io.out(summary.report)
    if summary.result.outcome == "escalated":
        io.out("Прогон остановлен до решения человека. Продолжить: /rooms resume")


_BUILD_NAME = re.compile(r"^([a-z][a-z0-9_]{1,40}):\s+(.+)$", re.S)
MAX_REFINEMENTS = 5


def _present(io: Io, lib, result, old=None) -> bool:
    """Show a build/edit result. True when there is a valid draft on the screen."""
    from src.rooms.diagram import render_flow
    if not result.ok:
        io.err(f"Не удалось составить комнату (попыток: {result.attempts}). Что не так:")
        for p in result.errors[:8]:
            io.err(f"  - {p}")
        return False
    io.out(render_flow(result.room))
    for note in result.notes:
        io.out(f"[yellow]Правило безопасности:[/yellow] {note}")
    if old is not None:
        io.out("Что изменилось: " + ("; ".join(result.changes) if result.changes else "ничего"))
    return True


def _save_flow(io: Io, lib, result, builder, old=None) -> None:
    """Draft -> (refine by words)* -> save. The person decides; nothing is saved without a yes."""
    for _ in range(MAX_REFINEMENTS + 1):
        if not _present(io, lib, result, old):
            return
        options = ["Сохранить", "Поправить словами", "Отмена"] if io.ask_text else ["Сохранить", "Отмена"]
        choice = io.choose(f"Комната «{result.room.room}» — что делать?", options)
        if choice == "Сохранить":
            errors = lib.add_room(result.room, _tools(lib))
            if errors:
                io.err("Не сохранена: " + "; ".join(errors))
                return
            lib.save_stats()
            where = (lib.user_dir or user_rooms_dir()) / f"{result.room.room}.json"
            io.out(f"Сохранена: {where}")
            params = " ".join(f'{k}="{v}"' for k, v in result.room.defaults.items())
            io.out(f"Запуск: /rooms run {result.room.room} {params} <задача>".replace("  ", " "))
            return
        if choice != "Поправить словами":
            io.out("Не сохранено.")
            return
        wish = io.ask_text("Что изменить?") if io.ask_text else ""
        if not wish or not wish.strip():
            io.out("Не сохранено.")
            return
        result = builder.edit(result.room, wish.strip(), result.room.room)
        old = None
    io.out("Слишком много правок подряд — не сохранено. Начните заново: /rooms build ...")


def _cmd_build(io: Io, lib, text: str, builder) -> None:
    text = (text or "").strip()
    if not text:
        io.err("Опишите, что комната должна делать и что делать в каком случае:\n"
               "  /rooms build после правки прогони тесты; если упали — две попытки починить, потом спроси меня\n"
               "  /rooms build my_check: проверь линтер и отправь итог мне в Telegram")
        return
    m = _BUILD_NAME.match(text)
    name, desc = (m.group(1), m.group(2)) if m else (None, text)
    io.out("Составляю комнату…")
    try:
        result = builder.build(desc, name)
    except Exception as e:
        io.err(f"Сборка не удалась: {e}")
        return
    _save_flow(io, lib, result, builder)


def _cmd_edit(io: Io, lib, name: str, instruction: str, builder) -> None:
    room = lib.get(name)
    if room is None:
        io.err(f"Нет комнаты «{name}». /rooms list — список.")
        return
    if not instruction.strip():
        io.err("Скажите, что изменить: /rooms edit <комната> <что изменить>")
        return
    new_name = name
    if lib.sources.get(name) == "starter":
        new_name = f"{name}_custom"
        io.out(f"Комната «{name}» входит в комплект и не меняется; правку сохраню как «{new_name}».")
    io.out("Вношу правку…")
    try:
        result = builder.edit(room, instruction.strip(), new_name)
    except Exception as e:
        io.err(f"Правка не удалась: {e}")
        return
    _save_flow(io, lib, result, builder, old=room)


def _cmd_diagram(io: Io, lib, rest: str) -> None:
    from src.rooms.diagram import render_flow, render_mermaid
    name, _, fmt = rest.partition(" ")
    room = lib.get(name)
    if room is None:
        io.err(f"Нет комнаты «{name}». /rooms list — список.")
        return
    io.out(render_mermaid(room) if fmt.strip().lower() == "mermaid" else render_flow(room))


def _open_in_browser(url: str) -> bool:
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _cmd_editor(io: Io, action: str) -> None:
    """Start/stop the local visual editor (a page in the browser; only this machine can reach it)."""
    from src.rooms import editor_server
    if action == "stop":
        io.out("Редактор остановлен." if editor_server.stop_editor() else "Редактор не запущен.")
        return
    srv = editor_server.running()
    if action == "status":
        io.out(f"Редактор: {srv.url}" if srv else "Редактор не запущен. /rooms editor — запустить.")
        return
    if srv is None:
        try:
            srv = editor_server.start_editor()
        except OSError as e:
            io.err(f"Не удалось запустить редактор: {e}")
            return
    opened = _open_in_browser(srv.url)
    io.out("Визуальный редактор комнат: " + srv.url + "\n"
           + ("Открыт в браузере. " if opened else "Откройте адрес в браузере. ")
           + "Доступен только с этого компьютера; адрес содержит одноразовый ключ — не пересылайте его.\n"
           "Остановить: /rooms editor stop")


def _cmd_triggers(io: Io, lib, rest: str) -> None:
    """List, add and remove what starts rooms by itself (a schedule or a webhook)."""
    from src.rooms import triggers, webhooks
    words = rest.split()
    verb = words[0].lower() if words else ""
    if verb in ("schedule", "расписание"):
        if len(words) < 3:
            io.err('Использование: /rooms triggers schedule <комната> <когда>, например: каждый день в 03:00')
            return
        room, when = words[1], " ".join(words[2:])
        if lib.get(room) is None:
            io.err(f"Нет комнаты «{room}». /rooms list — список.")
            return
        try:
            made = triggers.add_schedule(room, when)
        except ValueError as e:
            io.err(f"Не получилось: {e}")
            return
        io.out(f"Готово: «{made['name']}» — {made['schedule']}. Запускается без вас: вопрос человеку остановит "
               "комнату, шаги с подтверждением не выполняются.")
        return
    if verb == "webhook":
        if len(words) < 2:
            io.err("Использование: /rooms triggers webhook <комната> [имя]")
            return
        room, name = words[1], (words[2] if len(words) > 2 else None)
        if lib.get(room) is None:
            io.err(f"Нет комнаты «{room}». /rooms list — список.")
            return
        try:
            made, secret = triggers.add_webhook(room, name)
        except ValueError as e:
            io.err(f"Не получилось: {e}")
            return
        started = webhooks.start_if_configured()
        io.out(f"Готово: POST {made['url']}\n"
               f"Секрет (показан один раз, у нас хранится только хэш): {secret}\n"
               "Передавать в заголовке X-Argent-Secret. Слушает только этот компьютер (127.0.0.1)."
               + ("" if started else "\nСервер запустится вместе с Argent; отдельно: python -m src.rooms.webhooks serve"))
        return
    if verb in ("remove", "rm", "delete", "убрать", "on", "off"):
        name = " ".join(words[1:])
        found = [t for t in triggers.list_triggers() if t["name"] == name]
        if not found:
            io.err(f"Нет триггера «{name}». /rooms triggers — список.")
            return
        kind = found[0]["kind"]
        if verb in ("on", "off"):
            triggers.set_enabled(kind, name, verb == "on")
            io.out(f"«{name}»: {'включён' if verb == 'on' else 'выключен'}.")
        else:
            triggers.remove(kind, name)
            io.out(f"«{name}» убран.")
        return
    only = rest.strip() or None
    items = triggers.list_triggers(only)
    if not items:
        io.out("Триггеров нет. Добавить: /rooms triggers schedule <комната> <когда> | webhook <комната>")
        return
    for t in items:
        last = f" · последний запуск: {t['last_run']} ({t['last_status']})" if t.get("last_run") else ""
        io.out(f"  {t['room']}: {triggers.describe(t)}{last}")


def _is_command(sub: str, rest: str) -> bool:
    """Whether the first word, with what follows it, is a `/rooms` command rather than the start of a
    task. The words that take no argument are commands only when nothing follows."""
    if sub in ("list", "ls", "help", "?", "history", "hist", "resume"):
        return rest == ""
    if sub in ("last", "report"):
        return " " not in rest
    if sub in ("spawn", "hint"):
        return rest.lower() in ("", "on", "off", "status")
    if sub == "editor":
        return rest.lower() in ("", "stop", "status")
    return sub in ("show", "route", "new", "validate", "delete", "rm", "remove", "unquarantine", "run",
                   "build", "edit", "diagram", "graph", "triggers", "trigger")


def handle(arg: str, io: Io, *, library=None, runner=None, journal=None, record_dir=None, builder=None) -> None:
    arg = (arg or "").strip()
    if not arg:
        io.out(HELP)
        return
    head, _, rest = arg.partition(" ")
    sub = head.lower()
    rest = rest.strip()
    if not _is_command(sub, rest):
        sub = ""             # "list the files and explain them" is a task, not the `list` command

    if sub in ("help", "?"):
        io.out(HELP)
        return
    if sub == "spawn":
        from config import get_rooms_spawn, set_rooms_spawn
        flag = rest.lower()
        if flag in ("on", "off"):
            set_rooms_spawn(flag == "on")
            io.out(f"spawn_room (ИИ пишет новые комнаты): {'ON' if flag == 'on' else 'OFF'}")
        else:
            io.out(f"spawn_room: {'ON' if get_rooms_spawn() else 'OFF'} — ИИ может предлагать новые комнаты "
                   "(через валидатор и твоё одобрение). Вкл: /rooms spawn on")
        return
    if sub == "hint":
        from config import get_rooms_hint, set_rooms_hint
        flag = rest.lower()
        if flag in ("on", "off"):
            set_rooms_hint(flag == "on")
        io.out(f"Подсказка модели о подходящей комнате: {'ON' if get_rooms_hint() else 'OFF'} "
               "(когда сообщение явно подходит к комнате, модели добавляется одна строка). /rooms hint on|off")
        return
    if sub == "editor":
        _cmd_editor(io, rest.lower())
        return
    if sub in ("history", "hist"):
        _cmd_history(io, record_dir)
        return
    if sub in ("last", "report"):
        _cmd_last(io, record_dir, rest or "last")
        return

    lib = _library(io, library)

    if sub in ("list", "ls"):
        _cmd_list(io, lib)
    elif sub == "show":
        room = lib.get(rest)
        if room is None:
            io.err(f"Нет комнаты «{rest}». /rooms list — список.")
        else:
            from src.rooms.diagram import render_flow
            io.out(render_flow(room))
    elif sub == "route":
        _cmd_route(io, lib, rest)
    elif sub in ("diagram", "graph"):
        _cmd_diagram(io, lib, rest)
    elif sub in ("triggers", "trigger"):
        _cmd_triggers(io, lib, rest)
    elif sub == "build":
        from src.rooms.builder import production_builder
        _cmd_build(io, lib, rest, builder if builder is not None else production_builder(lib))
    elif sub == "edit":
        name, _, instruction = rest.partition(" ")
        if lib.get(name) is None:      # "edit the README ..." is a task, not this command
            _run_free_text(io, lib, arg, runner, journal, record_dir)
        else:
            from src.rooms.builder import production_builder
            _cmd_edit(io, lib, name, instruction, builder if builder is not None else production_builder(lib))
    elif sub == "new":
        _cmd_new(io, lib, rest, None)
    elif sub == "validate":
        _cmd_validate(io, lib, rest)
    elif sub in ("delete", "rm", "remove"):
        _cmd_delete(io, lib, rest)
    elif sub == "unquarantine":
        io.out(f"Комната «{rest}» возвращена в работу." if lib.unquarantine(rest) else f"Нет комнаты «{rest}».")
    elif sub == "resume":
        journal = journal if journal is not None else default_journal()
        if journal.last() is None:
            io.out("Нет прерванного прогона для возобновления.")
            return
        if not io.confirm("Продолжить прерванный автономный прогон?"):
            io.out("Отменено.")
            return
        try:
            summary = execute_room_run(
                "", journal.last().get("room", "analyze"), library=lib, runner=runner, journal=journal,
                resume=True, human_prompt=io.human, confirm=io.confirm_step, printer=io.out,
                record_dir=record_dir)
        except Exception as e:
            io.err(f"Возобновление не удалось: {e}")
            return
        io.out("")
        io.out(summary.report)
    elif sub == "run":
        name, _, tail = rest.partition(" ")
        task, params = parse_params(tail)
        if not name:
            io.err("Использование: /rooms run <комната> [k=v] [задача]")
            return
        _start_run(io, lib, name, task, params, runner,
                   journal if journal is not None else default_journal(), record_dir)
    else:
        _run_free_text(io, lib, arg, runner, journal, record_dir)


def _run_free_text(io: Io, lib, arg: str, runner, journal, record_dir) -> None:
    """A task in the person's words: the router picks the room; when it is not sure, the person does."""
    task, params = parse_params(arg)
    pick = pick_room(task, lib)
    name = pick.name
    if not pick.clear:
        options = [n for n, _, _ in pick.ranked[:3]]
        if "analyze" in lib.rooms and "analyze" not in options:
            options.append("analyze")
        if not options:
            io.err("Не нашла подходящую комнату. /rooms list — что есть, /rooms run <комната> <задача> — вручную, "
                   "/rooms build <описание> — составить новую.")
            return
        name = io.choose(f"Какую комнату запустить для: «{_short(task, 60)}»?", options)
        if not name:
            io.out("Отменено.")
            return
    else:
        io.out(f"Выбрана комната «{name}» (слова: {pick.ranked[0][2]})")
    _start_run(io, lib, name, task, params, runner,
               journal if journal is not None else default_journal(), record_dir)
