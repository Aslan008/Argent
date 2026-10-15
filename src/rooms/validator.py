"""RoomValidator: checks a room is safe to admit into the graph.

Structural per-node/per-edge invariants are already guaranteed by the pydantic
models (a room that violates them cannot be constructed). This layer adds the
cross-cutting checks that need outside knowledge:
  1. node ids are unique;
  2. every referenced tool exists;
  3. every edge points at a real node or an EXIT;
  4. every condition is in the safe DSL grammar;
  5. every node can reach an EXIT (no dead ends / unescapable loops);
  6. spawn_room appears only in a room that is allowed to spawn (triage);
  7. an agent node's `role` is a known role and its `tools` all exist;
  8. every key of the room's `next` names an exit the room can actually end with.

Returns a list of human-readable problems; empty list == valid.
"""

from src.rooms import condition
from src.rooms import parallel as parallel_mod
from src.rooms import rules as rules_mod
from src.rooms.models import Room

#: Sub-agent roles a node may ask for (agent.ArgentSubAgent knows these personas).
KNOWN_ROLES = frozenset({"Coder", "Researcher", "Reviewer", "DocWriter", "BrowserWorker", "Critic"})

#: Terminals the engine itself produces, besides the EXIT:<name> targets of the edges.
ENGINE_EXITS = frozenset({"budget", "stuck", "human_pause"})

#: Tools a step of a room may not call: they start another run (a room, a sub-agent, ultracode) and
#: so would escape the room's own budget. A room reaches another room through a `room` node.
NOT_IN_ROOMS = frozenset({"rooms", "run_subagent", "subagents", "ultracode"})


def _nodes_that_reach_exit(room: Room, node_ids: set) -> set:
    """Reverse-reachability: the set of nodes from which some path leads to an
    EXIT. Computed to a fixpoint over the edge graph."""
    outgoing = {}
    for e in room.edges:
        outgoing.setdefault(e.from_, []).append(e.to)

    can_exit = set()
    changed = True
    while changed:
        changed = False
        for n in node_ids:
            if n in can_exit:
                continue
            for target in outgoing.get(n, []):
                if target.startswith("EXIT:") or target in can_exit:
                    can_exit.add(n)
                    changed = True
                    break
    return can_exit


def validate_room(room: Room, available_tools: set, allow_spawn: bool = False) -> list:
    """Validate `room` against the tool registry. `allow_spawn` must be True only
    for the triage room (the sole holder of the spawn_room right)."""
    errors = []

    node_ids = [n.id for n in room.nodes]
    unique_ids = set(node_ids)
    if len(unique_ids) != len(node_ids):
        dupes = sorted({i for i in node_ids if node_ids.count(i) > 1})
        errors.append(f"duplicate node ids: {dupes}")

    if not room.nodes:
        errors.append("room has no nodes")

    for n in room.nodes:
        if n.type == "tool" and n.tool not in available_tools:
            errors.append(f"node '{n.id}': tool '{n.tool}' does not exist")
        if n.type == "tool" and n.tool in NOT_IN_ROOMS:
            errors.append(f"node '{n.id}': tool '{n.tool}' starts another run and cannot be a step of a room "
                          f"(use a 'room' node to call another room)")
        if n.type == "spawn_room" and not allow_spawn:
            errors.append(f"node '{n.id}': spawn_room is only allowed in the triage room")
        if n.type == "agent":
            if n.role and n.role not in KNOWN_ROLES:
                errors.append(f"node '{n.id}': unknown role '{n.role}' (use one of {sorted(KNOWN_ROLES)})")
            for t in (n.tools or []):
                if t not in available_tools:
                    errors.append(f"node '{n.id}': tools lists '{t}', which does not exist")

    loops = {n.id: n for n in room.nodes if n.type in ("loop", "foreach")}
    for n in room.nodes:
        if n.type in ("loop", "assert") and n.condition:
            try:
                condition.validate(n.condition)
            except condition.ConditionError as ce:
                errors.append(f"node '{n.id}': bad condition ({ce})")
        if n.type == "set":
            for field, value in n.values.items():
                if isinstance(value, str) and value.lstrip().startswith("="):
                    try:
                        condition.validate(value.lstrip()[1:].strip())
                    except condition.ConditionError as ce:
                        errors.append(f"node '{n.id}': field '{field}': bad expression ({ce})")
        if n.type == "loop":
            for label, value in (("start", n.start), ("stop", n.stop), ("step", n.step)):
                if isinstance(value, str) and value.lstrip().startswith("="):
                    try:
                        condition.validate(value.lstrip()[1:].strip())
                    except condition.ConditionError as ce:
                        errors.append(f"node '{n.id}': {label}: bad expression ({ce})")
        if n.type in ("break", "continue"):
            if n.loop not in loops:
                errors.append(f"node '{n.id}': {n.type} names '{n.loop}', which is not a loop or foreach node of this room")
            elif not any(e.from_ == n.id and e.to == n.loop for e in room.edges):
                errors.append(f"node '{n.id}': a {n.type} must lead back to its loop '{n.loop}' (draw an arrow to it)")
    errors.extend(rules_mod.static_problems(room))
    errors.extend(parallel_mod.problems(room))

    for e in room.edges:
        if e.from_ not in unique_ids:
            errors.append(f"edge from unknown node '{e.from_}'")
        if not e.targets_exit() and e.to not in unique_ids:
            errors.append(f"edge '{e.from_}' -> unknown target '{e.to}'")
        for cond in (e.if_, e.while_):
            if cond:
                try:
                    condition.validate(cond)
                except condition.ConditionError as ce:
                    errors.append(f"edge '{e.from_}'->'{e.to}': bad condition ({ce})")

    if room.entry and room.entry not in unique_ids:
        errors.append(f"entry node '{room.entry}' is not in the room")

    for n in room.nodes:
        if n.into and not n.into.isidentifier():
            errors.append(f"node '{n.id}': 'into' must be a plain field name, got '{n.into}'")
        if n.type == "room" and n.room == room.room:
            errors.append(f"node '{n.id}': a room cannot call itself")
        if sum(1 for e in room.edges if e.from_ == n.id and e.kind == "error") > 1:
            errors.append(f"node '{n.id}': more than one on_error edge")

    exits = {e.to.split(":", 1)[1] for e in room.edges if e.targets_exit()} | ENGINE_EXITS
    for exit_name, target in (room.next or {}).items():
        if exit_name not in exits:
            errors.append(f"next: '{exit_name}' is not an exit of this room (exits: {sorted(exits)})")
        if not target or not isinstance(target, str):
            errors.append(f"next: exit '{exit_name}' has an empty target")

    # Every node must be able to reach an EXIT — no dead ends, no inescapable loop.
    if room.nodes:
        can_exit = _nodes_that_reach_exit(room, unique_ids)
        stuck = sorted(unique_ids - can_exit)
        if stuck:
            errors.append(f"nodes cannot reach any EXIT: {stuck}")

    return errors


def room_warnings(room: Room, catalog=None) -> list:
    """Things that are probably wrong but do not stop a room from being admitted: a server, skill
    or plugin the room names that this machine does not have (rooms travel between machines), a
    list loop that nothing ends, a node that asks Leeth when Leeth is not loaded.

    `catalog` is an Integrations-like object (`mcp_servers()`, `skills()`, `plugins()`,
    `leeth_available()`); with none, only the structural hints are given.
    """
    notes = []
    servers = skills = plugins = None
    if catalog is not None:
        try:
            servers = set(catalog.mcp_servers())
        except Exception:
            servers = None
        try:
            skills = set(catalog.skills())
        except Exception:
            skills = None
        try:
            plugins = catalog.plugins()
        except Exception:
            plugins = None
    leeth_ok = None
    if catalog is not None:
        try:
            leeth_ok = bool(catalog.leeth_available())
        except Exception:
            leeth_ok = None

    shares = (room.rules is not None and room.rules.memory == "shared") or any(
        n.type == "agent" and n.memory == "shared" for n in room.nodes)
    if shares and not any(n.type == "forget" for n in room.nodes):
        notes.append("the agents of this room share memory and nothing in it ever forgets: "
                     "a long run keeps only the most recent part (rules.memory_chars)")

    for n in room.nodes:
        if n.type == "mcp" and servers is not None and n.server not in servers:
            notes.append(f"node '{n.id}': MCP server '{n.server}' is not configured here (add it with /mcp)")
        if n.type == "agent" and servers is not None:
            for s in (n.mcp or []):
                if s not in servers and "*" not in s:
                    notes.append(f"node '{n.id}': MCP server '{s}' is not configured here")
        if n.type == "skill" and skills is not None and n.skill not in skills:
            notes.append(f"node '{n.id}': skill '{n.skill}' is not installed here")
        if n.type == "agent" and skills is not None:
            for s in (n.skills or []):
                if s not in skills:
                    notes.append(f"node '{n.id}': skill '{s}' is not installed here")
        if n.type == "plugin" and plugins is not None:
            if (n.mode or "command") == "command":
                known = {c for p in plugins.values() for c in p.get("commands", [])}
                if n.command not in known:
                    notes.append(f"node '{n.id}': no plugin command '{n.command}' is loaded")
            elif n.plugin and n.plugin not in plugins:
                notes.append(f"node '{n.id}': plugin '{n.plugin}' is not loaded")
        if n.type == "leeth" and leeth_ok is False:
            notes.append(f"node '{n.id}': Leeth is not loaded here, so this step will report that it cannot answer")
        if n.type in ("foreach", "loop"):
            watches = any(
                f"{n.id}_done" in (e.if_ or "") + (e.while_ or "")
                for e in room.edges if e.from_ == n.id)
            if not watches:
                notes.append(f"node '{n.id}': no arrow looks at state.{n.id}_done, so the loop only ends when the room's steps run out")
        if n.type == "power" and (n.mode in ("shutdown", "restart", "sleep", "hibernate", "logout")) and not n.confirm and not n.disabled:
            notes.append(f"node '{n.id}': the computer will {n.mode} without asking (unattended runs need that; "
                         "put a notification before it and keep the delay)")
        if n.type == "hint" and n.targets:
            ids = {x.id for x in room.nodes}
            for t in n.targets:
                if t not in ids:
                    notes.append(f"node '{n.id}': the hint is for '{t}', which is not a step of this room")
        if n.disabled:
            notes.append(f"node '{n.id}' is switched off and is skipped")
    notes.extend(rules_mod.static_warnings(room))
    notes.extend(parallel_mod.warnings(room))
    return notes
