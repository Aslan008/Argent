"""RoomValidator: checks a room is safe to admit into the graph.

Structural per-node/per-edge invariants are already guaranteed by the pydantic
models (a room that violates them cannot be constructed). This layer adds the
cross-cutting checks that need outside knowledge:
  1. node ids are unique;
  2. every referenced tool exists;
  3. every edge points at a real node or an EXIT;
  4. every condition is in the safe DSL grammar;
  5. every node can reach an EXIT (no dead ends / unescapable loops);
  6. spawn_room appears only in a room that is allowed to spawn (triage).

Returns a list of human-readable problems; empty list == valid.
"""

from src.rooms import condition
from src.rooms.models import Room


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
        if n.type == "spawn_room" and not allow_spawn:
            errors.append(f"node '{n.id}': spawn_room is only allowed in the triage room")

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

    # Every node must be able to reach an EXIT — no dead ends, no inescapable loop.
    if room.nodes:
        can_exit = _nodes_that_reach_exit(room, unique_ids)
        stuck = sorted(unique_ids - can_exit)
        if stuck:
            errors.append(f"nodes cannot reach any EXIT: {stuck}")

    return errors
