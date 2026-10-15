"""Point operations on a room — what the model uses when it must change a room precisely.

`build` / `edit` ask a second model to rewrite a whole room from a description. That is the
right tool for "make me a room that ..." and the wrong one for "make the critic's
temperature 0.2" or "put a lint step between plan and code": a rewrite may touch what
nobody asked about. These operations change exactly the named thing, all together or not
at all, on a COPY of the room; the result still goes through the same validation, the same
house rules (confirm on steps that act) and the same approval by the person as any other
room (see src/rooms/tool.py).

An operation is an object with `op`:

  add_node     {node: {id, type, ...}}                         a new block (not connected yet)
  insert_node  {node: {...}, between: [from, to]}              a new block ON the arrow from -> to
  remove_node  {node, reconnect: true}                         delete a block and its arrows
  rename_node  {node, new_id}                                  new id; arrows and references follow
  set_param    {node, field, value}  or  {node, values: {..}}  any block parameter; value null resets it;
                                                               `args.file_path`, `sampling.temperature`,
                                                               `budget.max_tokens` reach into objects
  connect      {from, to, if | else | while+max+id | on_error} a new arrow
  disconnect   {from, to, if | else, all}                      remove arrow(s)
  set_edge     {from, to, if | else (selector), set: {...}}    change an arrow (condition, target, kind)
  set_room     {field, value}  or  {values: {..}}              description, budget, entry, defaults,
                                                               next, tags, report, rules (dotted paths too)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from src.rooms.models import Node, Room

MAX_OPS = 40
MAX_PATCH_NODES = 60
BLANK_BUDGET = {"max_iterations": 40, "max_tokens": 60000}
ROOM_FIELDS = ("description", "budget", "entry", "defaults", "next", "tags", "report", "rules")
EDGE_FIELDS = ("to", "if", "else", "while", "max", "id", "on_error")


class PatchError(Exception):
    pass


@dataclass
class PatchResult:
    room: Optional[Room] = None
    errors: list = field(default_factory=list)
    applied: list = field(default_factory=list)       # one readable line per operation
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.room is not None and not self.errors


def coerce_ops(ops) -> tuple:
    """(list of operation dicts, error text or None) from what a model passed as `ops`."""
    if ops is None or ops == "" or ops == []:
        return [], "`ops` is empty: pass a list of operations, e.g. [{\"op\": \"set_param\", \"node\": \"review\", \"field\": \"sampling.temperature\", \"value\": 0.2}]"
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except ValueError as e:
            return [], f"`ops` is not valid JSON: {e}"
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list) or not all(isinstance(o, dict) for o in ops):
        return [], "`ops` must be a list of objects, each with an `op` name"
    if len(ops) > MAX_OPS:
        return [], f"at most {MAX_OPS} operations in one call (got {len(ops)}); split them"
    return ops, None


# -- helpers --------------------------------------------------------------------------------------
def _node(data: dict, node_id) -> dict:
    for n in data["nodes"]:
        if n.get("id") == node_id:
            return n
    ids = ", ".join(str(n.get("id")) for n in data["nodes"]) or "none"
    raise PatchError(f"no node '{node_id}' (nodes: {ids})")


def _need(op: dict, *keys):
    for k in keys:
        if op.get(k) in (None, ""):
            raise PatchError(f"`{k}` is required")


def _is_exit(target) -> bool:
    return isinstance(target, str) and target.startswith("EXIT:")


def _check_target(data: dict, target) -> None:
    if _is_exit(target):
        if not target[5:]:
            raise PatchError("an exit arrow needs a name: EXIT:done")
        return
    _node(data, target)


def _kind(edge: dict) -> str:
    if edge.get("on_error"):
        return "error"
    if edge.get("while") is not None:
        return "while"
    if edge.get("if") is not None:
        return "if"
    if edge.get("else"):
        return "else"
    return "uncond"


def _label(edge: dict) -> str:
    kind = _kind(edge)
    tail = {"if": f" [if {edge.get('if')}]", "else": " [else]", "while": f" [while {edge.get('while')}]",
            "error": " [on_error]"}.get(kind, "")
    return f"{edge['from']} → {edge['to']}{tail}"


def _selected(data: dict, op: dict, *, many_ok: bool = False) -> list:
    """The arrows `from -> to` the operation names; a selector (`if` / `else`) narrows them."""
    _need(op, "from", "to")
    found = [e for e in data["edges"] if e.get("from") == op["from"] and e.get("to") == op["to"]]
    if "if" in op:
        found = [e for e in found if e.get("if") == op["if"]]
    if op.get("else"):
        found = [e for e in found if e.get("else")]
    if not found:
        have = [_label(e) for e in data["edges"] if e.get("from") == op["from"]] or ["no arrows leave it"]
        raise PatchError(f"no arrow {op['from']} → {op['to']}" + (f" with that selector" if "if" in op or op.get("else") else "")
                         + f" (from '{op['from']}': {'; '.join(have)})")
    if len(found) > 1 and not (many_ok or op.get("all")):
        raise PatchError(f"{len(found)} arrows {op['from']} → {op['to']} ({'; '.join(_kind(e) for e in found)}): "
                         "say which with `if` / `else`, or pass all=true")
    return found


def _set_path(container: dict, path: str, value) -> None:
    head, _, rest = path.partition(".")
    if not rest:
        if value is None:
            container.pop(head, None)
        else:
            container[head] = value
        return
    cur = container.get(head)
    if cur is None:
        cur = container[head] = {}
    if not isinstance(cur, dict):
        raise PatchError(f"'{head}' is not an object, so '{path}' cannot be set; give '{head}' whole")
    _set_path(cur, rest, value)


def _pairs(op: dict) -> list:
    """(field, value) pairs of a set_param / set_room operation."""
    if isinstance(op.get("values"), dict) and op["values"]:
        return list(op["values"].items())
    if op.get("field"):
        return [(str(op["field"]), op.get("value"))]
    raise PatchError("pass `field` and `value` (or `values`: {field: value})")


# -- operations -----------------------------------------------------------------------------------
def _add_node(data, op):
    spec = op.get("node")
    if not isinstance(spec, dict):
        raise PatchError("`node` must be an object with at least id and type")
    _need(spec, "id", "type")
    nid = str(spec["id"])
    if nid == "END" or nid.startswith("EXIT:"):
        raise PatchError(f"'{nid}' is reserved")
    if any(n.get("id") == nid for n in data["nodes"]):
        raise PatchError(f"node '{nid}' already exists (use set_param to change it)")
    spec = dict(spec)
    spec["id"] = nid
    data["nodes"].append(spec)
    return f"+ блок {nid} ({spec['type']})"


def _insert_node(data, op):
    _need(op, "between")
    pair = op["between"]
    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
        raise PatchError("`between` must be [from, to]")
    arrows = _selected(data, {"from": pair[0], "to": pair[1], "all": True})
    line = _add_node(data, op)
    nid = str(op["node"]["id"])
    for e in arrows:
        e["to"] = nid
    data["edges"].append({"from": nid, "to": pair[1]})
    return f"{line} на стрелке {pair[0]} → {pair[1]}"


def _remove_node(data, op):
    _need(op, "node")
    nid = op["node"]
    _node(data, nid)
    reconnect = op.get("reconnect", True)
    incoming = [e for e in data["edges"] if e.get("to") == nid]
    outgoing = [e for e in data["edges"] if e.get("from") == nid]
    target = None
    if reconnect and incoming and outgoing:
        if len(outgoing) != 1 or _kind(outgoing[0]) != "uncond":
            raise PatchError(f"'{nid}' has {len(outgoing)} outgoing arrows: connect what leads into it to the right "
                             "target with `connect`/`set_edge`, then remove it (or pass reconnect=false)")
        target = outgoing[0]["to"]
        for e in incoming:
            e["to"] = target
    if data.get("entry") == nid:
        if target is None or _is_exit(target):
            raise PatchError(f"'{nid}' is the entry: name another entry first (set_room field=entry)")
        data["entry"] = target
    data["nodes"] = [n for n in data["nodes"] if n.get("id") != nid]
    data["edges"] = [e for e in data["edges"] if e.get("from") != nid and e.get("to") != nid]
    return f"− блок {nid}" + (f" (стрелки замкнуты на {target})" if target else " (его стрелки удалены)")


def _rename_node(data, op, warnings):
    _need(op, "node", "new_id")
    old, new = op["node"], str(op["new_id"])
    n = _node(data, old)
    if new == "END" or new.startswith("EXIT:") or any(x.get("id") == new for x in data["nodes"]):
        raise PatchError(f"id '{new}' is reserved or taken")
    n["id"] = new
    for e in data["edges"]:
        if e.get("from") == old:
            e["from"] = new
        if e.get("to") == old:
            e["to"] = new
    if data.get("entry") == old:
        data["entry"] = new
    for other in data["nodes"]:
        for key in ("join", "loop"):
            if other.get(key) == old:
                other[key] = new
    text = json.dumps(data, ensure_ascii=False)
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(old)}[._]", text):
        warnings.append(f"после переименования {old} → {new} проверьте тексты, где старое имя могло стоять в "
                        f"{{{{{old}.output}}}} или в условиях: ссылки в тексте автоматически не переписываются")
    return f"блок {old} → {new}"


_NODE_FIELDS = set(Node.model_fields) - {"id"}


def _set_param(data, op):
    _need(op, "node")
    n = _node(data, op["node"])
    done = []
    pairs = _pairs(op)
    for path, value in pairs:
        head = str(path).partition(".")[0]
        if head == "id":
            raise PatchError("the id is changed with rename_node")
        if head not in _NODE_FIELDS:
            raise PatchError(f"no parameter '{head}' on a block (known: {', '.join(sorted(_NODE_FIELDS))})")
        _set_path(n, str(path), value)
        done.append(f"{path} = {json.dumps(value, ensure_ascii=False)[:60]}" if value is not None else f"{path} сброшен")
    if any(str(p).partition(".")[0] == "type" for p, _ in pairs) and "idempotent" not in dict(pairs):
        n.pop("idempotent", None)         # the default depends on the type
    return f"блок {op['node']}: " + "; ".join(done)


def _connect(data, op):
    _need(op, "from", "to")
    _node(data, op["from"])
    _check_target(data, op["to"])
    edge = {"from": op["from"], "to": op["to"]}
    for key in ("if", "while", "max", "id"):
        if op.get(key) is not None:
            edge[key] = op[key]
    for key in ("else", "on_error"):
        if op.get(key):
            edge[key] = True
    if any(_kind(e) == _kind(edge) and e.get("from") == edge["from"] and e.get("to") == edge["to"]
           and (e.get("if"), e.get("while")) == (edge.get("if"), edge.get("while")) for e in data["edges"]):
        raise PatchError(f"the arrow {_label(edge)} already exists")
    data["edges"].append(edge)
    return f"+ стрелка {_label(edge)}"


def _disconnect(data, op):
    found = _selected(data, op)
    ids = {id(e) for e in found}
    data["edges"] = [e for e in data["edges"] if id(e) not in ids]
    return "− стрелка " + "; ".join(_label(e) for e in found)


def _set_edge(data, op):
    changes = op.get("set")
    if not isinstance(changes, dict) or not changes:
        raise PatchError("`set` must be an object: which fields of the arrow to change, e.g. {\"if\": \"state.ok\"}")
    bad = [k for k in changes if k not in EDGE_FIELDS]
    if bad:
        raise PatchError(f"unknown arrow field {bad} (known: {list(EDGE_FIELDS)})")
    if "to" in changes:
        if changes["to"] in (None, ""):
            raise PatchError("`to` cannot be empty")
        _check_target(data, changes["to"])
    kinds = ("if", "else", "while", "on_error")
    new_kind = [k for k in kinds if changes.get(k) not in (None, False)]
    done = []
    for e in _selected(data, op):
        before = _label(e)
        if new_kind:                       # the kinds exclude each other: the one set now replaces the old
            for k in kinds:
                e.pop(k, None)
            if "while" not in new_kind:
                e.pop("max", None)
                e.pop("id", None)
        for key, value in changes.items():
            if value is None or value is False:
                e.pop(key, None)
            else:
                e[key] = value
        done.append(f"{before} → {_label(e)}")
    return "стрелка изменена: " + "; ".join(done)


def _set_room(data, op):
    done = []
    for path, value in _pairs(op):
        head = str(path).partition(".")[0]
        if head in ("nodes", "edges"):
            raise PatchError(f"'{head}' are changed with the block / arrow operations")
        if head == "room":
            raise PatchError("the name is given with `save_as`")
        if head not in ROOM_FIELDS:
            raise PatchError(f"no room parameter '{head}' (known: {', '.join(ROOM_FIELDS)})")
        _set_path(data, str(path), value)
        done.append(f"{path} = {json.dumps(value, ensure_ascii=False)[:60]}" if value is not None else f"{path} сброшен")
    return "комната: " + "; ".join(done)


def _format_errors(exc) -> list:
    try:
        items = exc.errors()
    except Exception:
        return [str(exc)[:700]]
    out = []
    for it in items[:8]:
        where = ".".join(str(p) for p in it.get("loc", ()))
        out.append(f"{where}: {it.get('msg', '')}".strip(": "))
    return out


def apply_ops(room: Optional[Room], ops: list, name: str) -> PatchResult:
    """Apply `ops` to a copy of `room` (None = a blank room called `name`). All or nothing."""
    result = PatchResult()
    if room is not None:
        data = room.model_dump(by_alias=True, exclude_none=True)
    else:
        data = {"room": name, "description": "", "nodes": [], "edges": [], "budget": dict(BLANK_BUDGET)}
    data["room"] = name
    handlers = {
        "add_node": _add_node, "insert_node": _insert_node, "remove_node": _remove_node,
        "set_param": _set_param, "connect": _connect, "disconnect": _disconnect,
        "set_edge": _set_edge, "set_room": _set_room,
    }
    for index, raw in enumerate(ops, 1):
        op = dict(raw)
        kind = str(op.get("op", "")).strip()
        try:
            if kind == "rename_node":
                line = _rename_node(data, op, result.warnings)
            elif kind in handlers:
                line = handlers[kind](data, op)
            else:
                raise PatchError(f"unknown op '{kind}' (use: {', '.join(list(handlers) + ['rename_node'])})")
        except PatchError as e:
            result.errors.append(f"op #{index} ({kind or '?'}): {e}")
            return result
        result.applied.append(line)
    if len(data["nodes"]) > MAX_PATCH_NODES:
        result.errors.append(f"too many blocks ({len(data['nodes'])} > {MAX_PATCH_NODES}): split the room with room nodes")
        return result
    if not data["nodes"]:
        result.errors.append("the room has no blocks: add some with add_node")
        return result
    try:
        result.room = Room(**data)
    except Exception as e:                     # pydantic ValidationError: say it per field
        result.errors.extend(_format_errors(e))
    return result
