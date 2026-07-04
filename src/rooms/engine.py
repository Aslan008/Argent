"""The rooms engine: deterministic execution of a validated node graph.

The engine owns MACRO control — which node runs next — while the caller owns the
MICRO side effects (running a tool, an agent's bounded loop, prompting a human)
through a single injected ``run_node`` callback. This keeps the transition logic
pure and fully testable without an LLM or real tools.

Transitions are decided from typed state, not by the model directly:
  * ``if``   edge — taken when its condition holds;
  * ``while`` edge — taken while its condition holds AND its counter (kept in
    ``state.loops[edge.id]``) is below ``max``; the counter increments per hop,
    so a loop is always bounded;
  * ``else`` / unconditional — the fallback.

Terminal outcomes: an ``EXIT:<name>`` target ends the room with that name; a
``human_pause`` node ends with ``human_pause``; exhausting the step budget ends
with ``budget``; a node with no usable outgoing edge ends with ``stuck`` (the
validator makes this unreachable for admitted rooms). The higher room-graph
layer decides how ``budget``/``stuck`` route (typically to human/triage).
"""

from collections import defaultdict
from dataclasses import dataclass

from src.rooms import condition
from src.rooms.models import Edge, Room, State


@dataclass
class Outcome:
    exit: str                 # EXIT name, or "human_pause" | "budget" | "stuck"
    state: State
    steps: int
    reason: str = ""


class Engine:
    def run(self, room: Room, state: State, run_node, max_steps: int = None,
            journal=None, start_node: str = None) -> Outcome:
        """Execute `room` from its entry (or `start_node`) until a terminal outcome.

        `run_node(node, state)` performs the node's side effects (mutating
        `state.data`). It is called for every node, including condition nodes
        (which usually no-op) and human_pause nodes (prompt the user).

        If `journal` is given, each node writes an intent record before its side
        effects and a commit record (with the chosen next target) after — see
        journal.py and plan_resume for crash-safe resume.
        """
        if max_steps is None:
            max_steps = room.budget.max_iterations

        node_by_id = {n.id: n for n in room.nodes}
        outgoing = defaultdict(list)
        for e in room.edges:
            outgoing[e.from_].append(e)

        def _commit(node_id, nxt, exit_name):
            if journal:
                journal.append({"kind": "commit", "room": room.room, "node": node_id,
                                "next": nxt, "exit": exit_name, "state": state.model_dump()})

        current = start_node or room.entry
        steps = 0
        while True:
            if steps >= max_steps:
                return Outcome("budget", state, steps, reason=f"step budget {max_steps} exhausted")

            node = node_by_id[current]
            if journal:
                journal.append({"kind": "intent", "room": room.room, "node": current,
                                "state": state.model_dump()})
            run_node(node, state)
            steps += 1

            if node.type == "human_pause":
                _commit(current, None, "human_pause")
                return Outcome("human_pause", state, steps, reason=f"human_pause node '{node.id}'")

            edge = self._select_edge(outgoing[current], state)
            if edge is None:
                _commit(current, None, "stuck")
                return Outcome("stuck", state, steps, reason=f"node '{current}' has no usable edge")
            if edge.targets_exit():
                exit_name = edge.to.split(":", 1)[1]
                _commit(current, None, exit_name)
                return Outcome(exit_name, state, steps)
            _commit(current, edge.to, None)
            current = edge.to

    @staticmethod
    def _select_edge(edges: list, state: State) -> Edge:
        """First matching conditional edge, else the else/unconditional fallback.
        A taken `while` edge increments its counter in state.loops."""
        fallback = None
        for e in edges:
            kind = e.kind
            if kind == "if":
                if condition.evaluate(e.if_, state.data):
                    return e
            elif kind == "while":
                count = state.loops.get(e.id, 0)
                if count < e.max and condition.evaluate(e.while_, state.data):
                    state.loops[e.id] = count + 1
                    return e
            elif kind == "else":
                fallback = fallback or e
            else:  # unconditional
                fallback = fallback or e
        return fallback


def plan_resume(room: Room, journal):
    """Decide where to resume `room` from its journal.

    Returns ``(start_node, state, status)`` where status is:
      * "fresh"        — nothing to resume; start at the entry with empty state;
      * "resume"       — continue at start_node with the recovered state;
      * "finished"     — the run already reached a terminal (start_node is None);
      * "indeterminate"— a non-idempotent node was interrupted mid-execution; the
        caller must route to a human instead of replaying it.
    """
    last = journal.last()
    if not last or last.get("room") != room.room:
        return room.entry, State(), "fresh"

    state = State(**last["state"])
    if last.get("kind") == "commit":
        nxt = last.get("next")
        if nxt is None:
            return None, state, "finished"
        return nxt, state, "resume"

    # An intent with no following commit: the node was interrupted.
    node = next((n for n in room.nodes if n.id == last.get("node")), None)
    if node is None:
        return room.entry, State(), "fresh"
    safe = node.type in ("condition", "human_pause") or (node.type == "tool" and node.idempotent)
    return last.get("node"), state, ("resume" if safe else "indeterminate")
