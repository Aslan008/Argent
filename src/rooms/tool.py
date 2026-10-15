"""The `rooms` tool: lets the model use a room instead of improvising a procedure.

`list` / `show` are free to call. `run` / `resume` execute real tools and
sub-agents autonomously, so each run is put to the PERSON first (the model never
approves its own run) with the room's limits on the screen; a declined run returns
a plain refusal the model can act on. Everything returns text and never raises.
"""

import json

from src.rooms import history as run_history
from src.rooms.library import describe_room, short_description
from src.rooms.router import pick_room
from src.rooms.session import (
    build_library, default_journal, execute_room_run, parse_params, questionary_confirm,
)

MAX_REPORT_CHARS = 6000


def _library():
    lib = build_library()
    return lib


def _coerce_params(params) -> dict:
    if not params:
        return {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            return parse_params(params)[1]
    if isinstance(params, dict):
        return {str(k): v for k, v in params.items() if str(k).isidentifier() and not str(k).startswith("_")}
    return {}


def _listing(lib) -> str:
    lines = ["Rooms (повторяемые рецепты; run — только с подтверждения человека):"]
    for name in lib.names():
        room = lib.rooms[name]
        st = lib.stats.get(name, {})
        flag = " [quarantined]" if st.get("quarantined") else ""
        params = f" params: {', '.join(room.defaults)}" if room.defaults else ""
        lines.append(f"- {name}{flag}: {short_description(room, 170)}.{params}")
    return "\n".join(lines)


def _ask_run(room, task, params) -> bool:
    import questionary
    b = room.budget
    limit = f"до {b.max_iterations} шагов, ≈{b.max_tokens} ток" + (f", {b.max_seconds} с" if b.max_seconds else "")
    shown = f"; параметры: {params}" if params else ""
    try:
        return bool(questionary.confirm(
            f"Модель просит запустить комнату «{room.room}» ({limit}){shown}. Задача: {task or '—'}. Разрешить?",
            default=False).ask())
    except (KeyboardInterrupt, EOFError):
        return False


def _human(node, state) -> None:
    import questionary
    ans = questionary.text(f"[{node.id}] {node.prompt or 'Ваш ввод:'}").ask()
    if ans:
        state.data[f"{node.id}.output"] = ans


def _ask_save(flow: str, name: str) -> bool:
    import questionary
    try:
        from ui import print_system
        print_system(flow)
    except Exception:
        pass
    try:
        return bool(questionary.confirm(
            f"Модель предлагает сохранить комнату «{name}» (схема и изменения выше). Сохранить?", default=False).ask())
    except (KeyboardInterrupt, EOFError):
        return False


def _build_action(action, lib, room, task, builder, ask_save) -> str:
    """Author or change a room for the person. It is saved only after THEY approve the drawn flow."""
    from src.rooms.builder import production_builder
    from src.rooms.diagram import render_flow
    from src.rooms.session import available_tool_names
    if not (task or "").strip():
        return "Error: `task` must say what the room should do (and what to do when a step fails or the answer is no)."
    builder = builder or production_builder(lib)
    old = None
    if action == "edit":
        old = lib.get(room)
        if old is None:
            return f"Error: no room '{room}' to edit.\n{_listing(lib)}"
        new_name = f"{room}_custom" if lib.sources.get(room) == "starter" else room
        result = builder.edit(old, task, new_name)
    else:
        result = builder.build(task, room or None)
    if not result.ok:
        return "Error: could not make a valid room: " + "; ".join(result.errors[:6])
    flow = render_flow(result.room)
    if result.notes:
        flow += "\n" + "\n".join(f"Правило безопасности: {n}" for n in result.notes)
    if old is not None:
        flow += "\nЧто изменилось: " + ("; ".join(result.changes) or "ничего")
    if not (ask_save or _ask_save)(flow, result.room.room):
        return "The person did NOT approve saving this room. Do not retry; ask what they want changed."
    errors = lib.add_room(result.room, available_tool_names())
    if errors:
        return "Error: the room was not saved: " + "; ".join(errors)
    lib.save_stats()
    params = ", ".join(result.room.defaults) or "none"
    return (f"Room '{result.room.room}' saved (parameters: {params}). Run it with "
            f"rooms(action='run', room='{result.room.room}', task=...) — the person is asked before every run.")


def _patch_action(action, lib, room, ops, save_as, ask_save) -> str:
    """Point changes to a room (or a new room made of them) — see src/rooms/patch.py. Applied to a copy,
    validated and hardened like any other room, and saved only after THE PERSON approves the drawn result."""
    from src.agent.plan_mode import _mutates
    from src.rooms.builder import NAME_RE, harden
    from src.rooms.diagram import diff_rooms, render_flow
    from src.rooms.patch import apply_ops, coerce_ops
    from src.rooms.session import available_tool_names
    from src.rooms.validator import validate_room
    ops, problem = coerce_ops(ops)
    if problem:
        return f"Error: {problem}"
    base = None
    if action == "create":
        name = (save_as or room or "").strip()
        if lib.get(name) is not None:
            return f"Error: room '{name}' already exists: change it with action='patch' or choose another name."
    else:
        base = lib.get(room)
        if base is None:
            return f"Error: no room '{room}' to patch.\n{_listing(lib)}"
        copy_name = f"{room}_custom"
        name = (save_as or "").strip() or (copy_name if lib.sources.get(room) == "starter" else room)
        if name != room and lib.get(name) is not None:
            hint = (f"patch that room (room='{name}')" if not save_as else "choose another `save_as`")
            return f"Error: room '{name}' already exists: {hint}."
    if not NAME_RE.match(name or ""):
        return f"Error: room name '{name}' must be snake_case (a-z, 0-9, _), starting with a letter."
    if lib.sources.get(name) == "starter":
        return f"Error: '{name}' is a packaged room and cannot be replaced: choose another name."
    result = apply_ops(base, ops, name)
    if not result.ok:
        return "Error: nothing was changed: " + "; ".join(result.errors[:6])
    problems = validate_room(result.room, set(available_tool_names()), allow_spawn=False)
    if not problems:
        problems = lib.reference_errors({**lib.rooms, result.room.room: result.room}).get(result.room.room, [])
    if problems:
        return "Error: the room would be invalid, nothing was changed: " + "; ".join(problems[:6])
    notes = harden(result.room, _mutates)
    flow = render_flow(result.room)
    flow += "\nТочечные правки модели:\n" + "\n".join(f"  {line}" for line in result.applied)
    if base is not None:
        flow += "\nЧто изменилось: " + ("; ".join(diff_rooms(base, result.room)) or "ничего")
    flow += "".join(f"\nПравило безопасности: {n}" for n in notes)
    flow += "".join(f"\nВнимание: {w}" for w in result.warnings)
    if not (ask_save or _ask_save)(flow, result.room.room):
        return "The person did NOT approve saving this room. Do not retry; ask what they want changed."
    errors = lib.add_room(result.room, available_tool_names())
    if errors:
        return "Error: the room was not saved: " + "; ".join(errors)
    lib.save_stats()
    out = [f"Room '{result.room.room}' saved ({len(result.applied)} change(s): " + "; ".join(result.applied) + ")."]
    out += [f"Safety rule: {n}" for n in notes] + [f"Warning: {w}" for w in result.warnings]
    out.append(f"Run it with rooms(action='run', room='{result.room.room}', task=...) — the person is asked before every run.")
    return "\n".join(out)


def rooms_action(action: str = "list", room: str = "", task: str = "", params=None, *,
                 library=None, runner=None, journal=None, record_dir=None,
                 ask_run=None, human=None, confirm_step=None, builder=None, ask_save=None,
                 ops=None, save_as: str = "") -> str:
    action = (action or "list").strip().lower()
    try:
        lib = library if library is not None else _library()
        if action == "list":
            return _listing(lib)
        if action == "show":
            r = lib.get(room)
            return describe_room(r) if r else f"Error: no room '{room}'.\n{_listing(lib)}"
        if action in ("last", "history"):
            if action == "last":
                rec = run_history.load_run(room or "last", record_dir)
                if rec is None:
                    return "No rooms runs yet."
                return json.dumps({k: rec.get(k) for k in ("id", "outcome", "task", "route", "usage", "seconds")},
                                  ensure_ascii=False)
            runs = run_history.list_runs(record_dir, limit=10)
            if not runs:
                return "No rooms runs yet."
            return "\n".join(f"{r['id']}  {r['outcome']}  {(r.get('task') or '')[:60]}" for r in runs)
        if action == "resume":
            journal = journal if journal is not None else default_journal()
            last = journal.last()
            if last is None:
                return "Nothing to resume: there is no interrupted rooms run."
            target = lib.get(last.get("room"))
            if target is None:
                return "Error: the interrupted run belongs to a room that no longer exists."
            if not (ask_run or _ask_run)(target, "(продолжение прерванного прогона)", {}):
                return "The person did NOT allow the run. Do not retry; ask what they want instead."
            summary = execute_room_run("", target.room, library=lib, runner=runner, journal=journal, resume=True,
                                       human_prompt=human or _human, confirm=confirm_step or questionary_confirm,
                                       record_dir=record_dir)
            return summary.report[:MAX_REPORT_CHARS]
        if action == "run":
            params = _coerce_params(params)
            if not room:
                pick = pick_room(task, lib)
                if not pick.clear:
                    cands = ", ".join(n for n, _, _ in pick.ranked[:3]) or "none matched"
                    return (f"Error: name the room (`room`). The task does not single one out (candidates: {cands}).\n"
                            + _listing(lib))
                room = pick.name
            target = lib.get(room)
            if target is None:
                return f"Error: no room '{room}'.\n{_listing(lib)}"
            if lib.stats.get(room, {}).get("quarantined"):
                return (f"Error: room '{room}' is quarantined (it kept failing). The person can restore it with "
                        f"/rooms unquarantine {room}.")
            if not (ask_run or _ask_run)(target, task, params):
                return "The person did NOT allow the run. Do not retry; ask what they want instead."
            journal = journal if journal is not None else default_journal()
            summary = execute_room_run(task, room, params=params, library=lib, runner=runner, journal=journal,
                                       human_prompt=human or _human, confirm=confirm_step or questionary_confirm,
                                       record_dir=record_dir)
            return summary.report[:MAX_REPORT_CHARS]
        if action in ("build", "edit"):
            return _build_action(action, lib, room, task, builder, ask_save)
        if action in ("patch", "create"):
            return _patch_action(action, lib, room, ops, save_as, ask_save)
        return "Error: action must be one of list, show, run, resume, build, edit, patch, create, last, history."
    except Exception as e:
        return f"Error: rooms {action} failed: {e}"
