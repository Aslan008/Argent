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
    def run(self, room: Room, state: State, run_node, max_steps: int = None) -> Outcome:
        """Execute `room` from its entry until a terminal outcome.

        `run_node(node, state)` performs the node's side effects (mutating
        `state.data`). It is called for every node, including condition nodes
        (which usually no-op) and human_pause nodes (prompt the user).
        """
        if max_steps is None:
            max_steps = room.budget.max_iterations

        node_by_id = {n.id: n for n in room.nodes}
        outgoing = defaultdict(list)
        for e in room.edges:
            outgoing[e.from_].append(e)

        current = room.entry
        steps = 0
        while True:
            if steps >= max_steps:
                return Outcome("budget", state, steps, reason=f"step budget {max_steps} exhausted")

            node = node_by_id[current]
            run_node(node, state)
            steps += 1

            if node.type == "human_pause":
                return Outcome("human_pause", state, steps, reason=f"human_pause node '{node.id}'")

            edge = self._select_edge(outgoing[current], state)
            if edge is None:
                return Outcome("stuck", state, steps, reason=f"node '{current}' has no usable edge")
            if edge.targets_exit():
                return Outcome(edge.to.split(":", 1)[1], state, steps)
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
