"""Parallel branches: which steps belong to which branch, and what is wrong with a fork.

A `parallel` node starts every arrow that leaves it at once; the `join` node it names gathers them
again. Everything a branch runs lies between its first step and that join, so a branch is simply the
set of steps reachable from its first step without passing through the join (its *region*).

The engine uses `branch_starts` / `region` to run a fork; the validator uses `problems` / `warnings`
to refuse a fork whose branches would step on each other; `plan_resume` uses `block_nodes` to decide
whether an interrupted fork may simply be run again.
"""

from src.rooms.models import PARALLEL_MAX_BRANCHES, PARALLEL_MAX_DEPTH, Room

#: Tools that drive the one open browser page, and the ones that drive the desktop: two branches doing
#: that at the same time would act on the same page / the same mouse, so the runner takes turns.
PAGE_TOOLS = frozenset({"browser_action", "browser_close", "browser_inspect", "browser_open",
                        "leeth_click_many", "leeth_fill_form", "leeth_map", "leeth_scan_page"})
SCREEN_TOOLS = frozenset({"desktop_act", "desktop_inspect", "capture_desktop_screen",
                          "capture_desktop_window", "list_desktop_windows"})


def resources_of(node) -> tuple:
    """The shared things a step uses, as sorted names: "page" (the open browser page), "screen" (the
    desktop) and "leeth" (the one small model). Steps that share one must not run at the same moment."""
    found = set()
    if node.type == "tool":
        if node.tool in PAGE_TOOLS:
            found.add("page")
        if node.tool in SCREEN_TOOLS:
            found.add("screen")
        if (node.tool or "").startswith("leeth_") or node.tool == "delegate_to_leeth":
            found.add("leeth")
    elif node.type == "leeth":
        found.add("leeth")
        if (node.mode or "delegate") in ("scan", "fill", "click"):
            found.add("page")
    elif node.type == "power":
        found.add("screen")
    return tuple(sorted(found))


def _outgoing(room: Room) -> dict:
    out = {}
    for e in room.edges:
        out.setdefault(e.from_, []).append(e)
    return out


def branch_starts(room: Room, node) -> list:
    """The first steps of the branches of `node`, in the order the arrows were drawn."""
    return [e.to for e in room.edges if e.from_ == node.id and e.kind != "error" and not e.targets_exit()]


def region(room: Room, start: str, join_id: str, outgoing: dict = None) -> set:
    """Ids of the steps reachable from `start` without passing through `join_id` (not included)."""
    outgoing = outgoing if outgoing is not None else _outgoing(room)
    seen, stack = set(), [start]
    while stack:
        current = stack.pop()
        if current == join_id or current in seen:
            continue
        seen.add(current)
        for e in outgoing.get(current, ()):
            if not e.targets_exit():
                stack.append(e.to)
    return seen


def block_nodes(room: Room, node) -> set:
    """Every step that runs inside the fork `node` (all of its branches)."""
    outgoing = _outgoing(room)
    nodes = set()
    for start in branch_starts(room, node):
        nodes |= region(room, start, node.join, outgoing)
    return nodes


def nesting_depth(room: Room, node) -> int:
    """How many forks `node` sits inside (0 when it is on the main path)."""
    outgoing = _outgoing(room)
    depth = 0
    for other in room.nodes:
        if other.type != "parallel" or other.id == node.id:
            continue
        inside = set()
        for start in branch_starts(room, other):
            inside |= region(room, start, other.join, outgoing)
        if node.id in inside:
            depth += 1
    return depth


def problems(room: Room) -> list:
    """What makes a fork unusable, as readable text (empty when every fork is sound)."""
    errors = []
    by_id = {n.id: n for n in room.nodes}
    outgoing = _outgoing(room)
    claimed = {}
    for node in room.nodes:
        if node.type != "parallel":
            continue
        label = f"parallel node '{node.id}'"
        join = by_id.get(node.join)
        if join is None or join.type != "join":
            errors.append(f"{label}: 'join' names '{node.join}', which is not a join node of this room")
            continue
        if node.join in claimed:
            errors.append(f"{label}: the join '{node.join}' already gathers '{claimed[node.join]}' — "
                          "give every parallel node its own join")
        claimed[node.join] = node.id
        edges = [e for e in outgoing.get(node.id, []) if e.kind != "error"]   # an error arrow is not a branch
        bad = [e for e in edges if e.kind != "uncond"]
        if bad:
            errors.append(f"{label}: every arrow that leaves it is a branch and must be unconditional "
                          f"(put the condition inside the branch): {sorted({e.to for e in bad})}")
            continue
        starts = [e.to for e in edges if not e.targets_exit()]
        if len(starts) != len(edges):
            errors.append(f"{label}: a branch cannot lead straight to an exit")
            continue
        if len(set(starts)) != len(starts):
            errors.append(f"{label}: two arrows lead to the same first step")
            continue
        if not 2 <= len(starts) <= PARALLEL_MAX_BRANCHES:
            errors.append(f"{label}: needs 2 to {PARALLEL_MAX_BRANCHES} branches, has {len(starts)}")
            continue
        if nesting_depth(room, node) >= PARALLEL_MAX_DEPTH:
            errors.append(f"{label}: forks nested more than {PARALLEL_MAX_DEPTH} deep are not allowed")
        regions = [region(room, s, node.join, outgoing) for s in starts]
        for start, reg in zip(starts, regions):
            if node.id in reg:
                errors.append(f"{label}: the branch starting at '{start}' leads back to the parallel node "
                              "itself (a break/continue whose loop lies outside the fork does this)")
            if not any(e.to == node.join for sid in reg for e in outgoing.get(sid, ())):
                errors.append(f"{label}: the branch starting at '{start}' never reaches the join '{node.join}'")
            for sid in sorted(reg):
                member = by_id.get(sid)
                if member is not None and member.type == "human_pause":
                    errors.append(f"{label}: '{sid}' asks a person inside a branch — ask before the fork "
                                  "or after the join (questions cannot be asked side by side)")
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                shared = sorted(regions[i] & regions[j])
                if shared:
                    errors.append(f"{label}: branches '{starts[i]}' and '{starts[j]}' share steps {shared} — "
                                  "give every branch its own steps")
    return errors


def warnings(room: Room) -> list:
    """Things that work but are probably not what was meant."""
    notes = []
    by_id = {n.id: n for n in room.nodes}
    outgoing = _outgoing(room)
    used = {n.join for n in room.nodes if n.type == "parallel"}
    for node in room.nodes:
        if node.type == "join" and node.id not in used:
            notes.append(f"join '{node.id}' is not used by any parallel node, so it only passes through")
        if node.type != "parallel" or node.join not in by_id:
            continue
        starts = branch_starts(room, node)
        owners = {}
        for start in starts:
            for sid in region(room, start, node.join, outgoing):
                member = by_id.get(sid)
                if member is None:
                    continue
                for resource in resources_of(member):
                    owners.setdefault(resource, set()).add(start)
        for resource, where in sorted(owners.items()):
            if len(where) > 1:
                what = {"page": "the open browser page", "screen": "the desktop",
                        "leeth": "the Leeth model"}.get(resource, resource)
                notes.append(f"parallel node '{node.id}': several branches use {what}; they take turns, "
                             "so this part is not really side by side")
    return notes
