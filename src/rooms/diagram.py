"""A room drawn so a person can read it: plain text for the terminal, Mermaid for a page.

Two readers, one source of truth (the Room model): `render_flow` is what `/rooms show`,
`/rooms build` and `/rooms edit` print before asking "save this?"; `render_mermaid` is
a flowchart that any Mermaid viewer (and the editor page) can draw; `diff_rooms` says in
words what an edit changed.
"""

from src.rooms.models import Edge, Node, Room

_ICON = {"tool": "⚙", "agent": "🤖", "condition": "◇", "human_pause": "✋", "spawn_room": "✚",
         "set": "＝", "notify": "🔔", "room": "▣", "mcp": "🔌", "skill": "📘", "plugin": "🧩",
         "leeth": "⚡", "foreach": "↻", "wait": "⏱", "loop": "⟳", "break": "⤓", "continue": "⤒",
         "power": "⏻", "hint": "💡", "forget": "🧹", "assert": "❗", "parallel": "⑂", "join": "⑃"}


def _node_line(n: Node) -> str:
    icon = _ICON.get(n.type, "·")
    if n.type == "tool":
        what = n.tool
        if n.args.get("command"):
            what = f"{n.tool}: {n.args['command']}"
        extra = []
        if n.sets:
            extra.append("вердикт процесса → " + ", ".join(n.sets))
        if n.confirm:
            extra.append("спрашивает")
        if not n.idempotent:
            extra.append("не повторять")
    elif n.type == "agent":
        what = f"агент {n.role or 'Coder'}" + (f" · инструменты: {', '.join(n.tools)}" if n.tools is not None else "")
        extra = [f"пишет: {', '.join(n.writes)}"] if n.writes else []
        if n.budget:
            extra.append(f"≤{n.budget.max_iterations} шагов, ≤{n.budget.max_tokens} ток")
    elif n.type == "set":
        what = "записать " + ", ".join(f"{k}={v!r}" for k, v in n.values.items())
        extra = []
    elif n.type == "notify":
        what = f"уведомить ({n.channel}): {n.message}"
        extra = []
    elif n.type == "room":
        what = f"подкомната «{n.room}»"
        extra = []
        if n.inputs:
            extra.append("вход: " + ", ".join(f"{k}={v}" for k, v in n.inputs.items()))
        if n.outputs:
            extra.append("выход: " + ", ".join(n.outputs))
        extra.append(f"итог → {n.into or n.id + '_exit'}")
    elif n.type == "human_pause":
        what = "вопрос человеку" + (f": {n.prompt}" if n.prompt else "")
        extra = [f"ответ → {n.into}"] if n.into else []
    elif n.type == "condition":
        what = f"проверка {n.condition}"
        extra = []
    elif n.type == "mcp":
        what = (f"MCP {n.server}: список инструментов" if (n.mode or "call") == "list"
                else f"MCP {n.server}.{n.tool}")
        extra = []
        if n.args:
            extra.append("аргументы: " + ", ".join(f"{k}={v!r}" for k, v in n.args.items()))
        if n.sets:
            extra.append("→ " + ", ".join(n.sets))
        if n.ignore:
            extra.append("игнорирует: " + ", ".join(n.ignore))
        if n.confirm:
            extra.append("спрашивает")
    elif n.type == "skill":
        what = f"прочитать скилл «{n.skill}»"
        extra = ["без разделов: " + ", ".join(n.ignore)] if n.ignore else []
    elif n.type == "plugin":
        what = f"плагин {n.plugin + ' · ' if n.plugin else ''}{'событие' if n.mode == 'hook' else 'команда'} {n.command}"
        extra = ["спрашивает"] if n.confirm else []
    elif n.type == "leeth":
        mode = n.mode or "delegate"
        what = {"delegate": "Leeth: поручить", "filter": "Leeth: отобрать по вопросу", "classify": "Leeth: разложить по категориям",
                "scan": "Leeth: оценить страницу", "fill": "Leeth: заполнить форму", "click": "Leeth: нажать найденное"}.get(mode, "Leeth")
        if n.instruction:
            what += f" — {n.instruction}"
        extra = []
        if n.input_from:
            extra.append(f"список: {n.input_from}")
        if n.sets:
            extra.append("→ " + ", ".join(n.sets))
        if n.confirm:
            extra.append("спрашивает")
    elif n.type == "foreach":
        what = f"для каждого из {n.input_from} → {n.item or 'item'}"
        extra = [f"не более {n.max_items}"] if n.max_items else []
    elif n.type == "wait":
        what = f"пауза {n.seconds:g} с"
        extra = []
    elif n.type == "loop":
        mode = n.mode or "for"
        what = {"for": f"цикл for {n.item or 'i'} от {n.start if n.start is not None else 0} до {n.stop} шаг {n.step if n.step is not None else 1}",
                "times": f"повторить {n.stop} раз", "while": f"цикл while {n.condition}",
                "do_while": f"цикл do-while {n.condition}", "until": f"повторять, пока не {n.condition}",
                "for_in": f"для каждого ключа в {n.input_from}"}.get(mode, "цикл")
        extra = [f"не более {n.max_items} раз"] if n.max_items else []
    elif n.type in ("break", "continue"):
        what = ("выйти из цикла " if n.type == "break" else "к следующему шагу цикла ") + str(n.loop)
        extra = []
    elif n.type == "power":
        what = {"shutdown": "выключить ПК", "restart": "перезагрузить ПК", "sleep": "спящий режим", "hibernate": "гибернация",
                "lock": "заблокировать ПК", "logout": "выйти из сессии", "cancel": "отменить выключение"}.get(n.mode, str(n.mode))
        extra = ([f"через {n.seconds:g} с"] if n.seconds else []) + (["спрашивает"] if n.confirm else [])
    elif n.type == "hint":
        what = {"next": "подсказка следующему агенту", "all": "подсказка всем агентам", "clear": "снять подсказки"}.get(n.mode or "next")
        if n.type == "hint" and (n.mode or "next") != "clear":
            what += f": {n.message[:60]}"
        extra = (["пишет модель"] if n.compose else []) + ([f"для: {', '.join(n.targets)}"] if n.targets else [])
    elif n.type == "forget":
        what = {"all": "забыть память и подсказки", "memory": "забыть память комнаты", "hints": "забыть подсказки"}.get(n.mode or "all")
        extra = [f"очистить поля: {', '.join(n.clear)}"] if n.clear else []
    elif n.type == "assert":
        what = f"проверить {n.condition}"
        extra = []
    elif n.type == "parallel":
        what = f"запустить все ветки одновременно; сходятся в {n.join}"
        extra = [f"не больше {n.workers} разом"] if n.workers else []
    elif n.type == "join":
        what = ("ждёт все ветки" if (n.wait_for or "all") == "all" else "идёт дальше с первой пришедшей") +             "; конфликт полей: " + {"error": "ошибка", "first": "первая ветка", "last": "последняя ветка",
                                    "list": "списком"}[n.merge or "error"]
        extra = ["сбой ветки не останавливает комнату"] if n.on_branch_fail == "continue" else []
    elif n.type == "notify" and n.compose:
        what = f"уведомление ({n.channel}), текст пишет модель: {n.message[:50]}"
        extra = []
    else:
        what = n.type
        extra = []
    if n.type == "agent":
        if n.sampling:
            extra.append("модель: " + ", ".join(f"{k}={v}" for k, v in n.sampling.items()))
        if n.memory:
            extra.append(f"память: {n.memory}")
        if n.hint:
            extra.append("есть подсказка")
        if n.skills:
            extra.append("скиллы: " + ", ".join(n.skills))
        if n.mcp is not None:
            extra.append("MCP: " + (", ".join(n.mcp) or "нет"))
        if n.ignore:
            extra.append("игнорирует: " + ", ".join(n.ignore))
    if n.disabled:
        extra.append("ОТКЛЮЧЁН")
    if n.retries:
        extra.append(f"повторов при сбое: {n.retries}")
    tail = f"  ({'; '.join(extra)})" if extra else ""
    return f"{icon} {n.id}  {what}{tail}"


def _edge_text(e: Edge) -> str:
    target = f"⛔ выход «{e.to.split(':', 1)[1]}»" if e.targets_exit() else e.to
    if e.kind == "if":
        return f"если {e.if_} → {target}"
    if e.kind == "while":
        return f"пока {e.while_} (до {e.max} раз, счётчик {e.id}) → {target}"
    if e.kind == "else":
        return f"иначе → {target}"
    if e.kind == "error":
        return f"при сбое узла → {target}"
    return f"→ {target}"


def render_flow(room: Room) -> str:
    b = room.budget
    limits = f"{b.max_iterations} шагов · {b.max_tokens} токенов" + (f" · {b.max_seconds} с" if b.max_seconds else "")
    lines = [f"Комната «{room.room}» — {room.description}".rstrip(" —"), f"Лимиты: {limits}"]
    if room.defaults:
        lines.append("Параметры: " + ", ".join(f"{k}={v!r}" for k, v in room.defaults.items()))
    if room.tags:
        lines.append("Теги: " + ", ".join(room.tags))
    r = room.rules
    if r is not None and r.active():
        bits = []
        if r.no_tools:
            bits.append("инструменты запрещены")
        if r.read_only:
            bits.append("только чтение")
        if r.deny_tools:
            bits.append("запрещено: " + ", ".join(r.deny_tools))
        if r.ignore:
            bits.append("игнорирует: " + ", ".join(r.ignore))
        if r.memory == "shared":
            bits.append("общая память агентов")
        if r.sampling:
            bits.append("модель: " + ", ".join(f"{k}={v}" for k, v in r.sampling.items()))
        if r.hint:
            bits.append("общая подсказка")
        lines.append("Правила комнаты: " + "; ".join(bits))
    lines.append("")
    outgoing = {}
    for e in room.edges:
        outgoing.setdefault(e.from_, []).append(e)
    for n in room.nodes:
        lines.append(("▶ " if n.id == room.entry else "  ") + _node_line(n))
        out = outgoing.get(n.id, [])
        for i, e in enumerate(out):
            lines.append(("      └ " if i == len(out) - 1 else "      ├ ") + _edge_text(e))
    if room.next:
        lines.append("")
        lines.append("Дальше: " + "; ".join(f"{k} → {v}" for k, v in room.next.items()))
    return "\n".join(lines)


def _mid(node_id: str) -> str:
    return "n_" + "".join(c if c.isalnum() else "_" for c in node_id)


def _label(text: str) -> str:
    return str(text).replace('"', "'").replace("\n", " ")[:80]


def render_mermaid(room: Room) -> str:
    """A `flowchart TD`; exits are drawn as their own nodes."""
    out = ["flowchart TD"]
    for n in room.nodes:
        shape = {"condition": ("{", "}"), "human_pause": ("([", "])"), "set": ("[/", "/]"),
                 "notify": (">", "]"), "room": ("[[", "]]")}.get(n.type, ("[", "]"))
        title = f"{n.id}\\n{n.type}" + (f": {n.tool}" if n.type == "tool" else f": {n.room}" if n.type == "room" else "")
        out.append(f'    {_mid(n.id)}{shape[0]}"{_label(title)}"{shape[1]}')
    exits = sorted({e.to.split(":", 1)[1] for e in room.edges if e.targets_exit()})
    for x in exits:
        out.append(f'    exit_{_mid(x)}(("{_label("EXIT " + x)}"))')
    for e in room.edges:
        dst = f"exit_{_mid(e.to.split(':', 1)[1])}" if e.targets_exit() else _mid(e.to)
        label = {"if": e.if_, "while": f"while {e.while_} ≤{e.max}", "else": "else", "error": "on error"}.get(e.kind, "")
        arrow = f'-- "{_label(label)}" -->' if label else "-->"
        out.append(f"    {_mid(e.from_)} {arrow} {dst}")
    return "\n".join(out)


def diff_rooms(old: Room, new: Room) -> list:
    """What an edit changed, in words (empty list == nothing)."""
    changes = []
    o = {n.id: n for n in old.nodes}
    n_ = {n.id: n for n in new.nodes}
    for nid in n_:
        if nid not in o:
            changes.append(f"+ узел {nid} ({n_[nid].type})")
    for nid in o:
        if nid not in n_:
            changes.append(f"− узел {nid}")
    for nid in o:
        if nid in n_ and o[nid].model_dump(exclude_none=True) != n_[nid].model_dump(exclude_none=True):
            changes.append(f"~ узел {nid} изменён")
    oe = {(e.from_, e.to, e.kind, e.if_ or e.while_ or "") for e in old.edges}
    ne = {(e.from_, e.to, e.kind, e.if_ or e.while_ or "") for e in new.edges}
    for e in sorted(ne - oe):
        changes.append(f"+ переход {e[0]} → {e[1]}" + (f" [{e[2]} {e[3]}]" if e[2] != "uncond" else ""))
    for e in sorted(oe - ne):
        changes.append(f"− переход {e[0]} → {e[1]}" + (f" [{e[2]} {e[3]}]" if e[2] != "uncond" else ""))
    if old.budget != new.budget:
        changes.append("~ лимиты комнаты изменены")
    if old.defaults != new.defaults:
        changes.append("~ параметры по умолчанию изменены")
    if old.description != new.description:
        changes.append("~ описание изменено")
    return changes
