"""Rooms engine execution: transitions, loop counters, budgets, terminals.

run_node is a mock — no LLM, no real tools — so the transition logic is tested
in isolation.
"""

from src.rooms.engine import Engine
from src.rooms.models import Room, State


def make_room(nodes, edges, max_iter=50):
    return Room(room="t", budget={"max_iterations": max_iter, "max_tokens": 1000},
                nodes=nodes, edges=edges)


def noop(node, state):
    pass


class TestTransitions:
    def test_unconditional_chain(self):
        room = make_room(
            [{"id": "A", "type": "tool", "tool": "noop"},
             {"id": "B", "type": "tool", "tool": "noop"}],
            [{"from": "A", "to": "B"}, {"from": "B", "to": "EXIT:done"}],
        )
        out = Engine().run(room, State(), noop)
        assert out.exit == "done" and out.steps == 2

    def test_if_true_branch(self):
        room = make_room(
            [{"id": "check", "type": "tool", "tool": "noop"}],
            [{"from": "check", "if": "state.ok", "to": "EXIT:yes"},
             {"from": "check", "else": True, "to": "EXIT:no"}],
        )
        out = Engine().run(room, State(), lambda n, s: s.data.__setitem__("ok", True))
        assert out.exit == "yes"

    def test_if_false_takes_else(self):
        room = make_room(
            [{"id": "check", "type": "tool", "tool": "noop"}],
            [{"from": "check", "if": "state.ok", "to": "EXIT:yes"},
             {"from": "check", "else": True, "to": "EXIT:no"}],
        )
        out = Engine().run(room, State(), lambda n, s: s.data.__setitem__("ok", False))
        assert out.exit == "no"


class TestWhileLoops:
    def test_loop_terminates_on_condition(self):
        room = make_room(
            [{"id": "work", "type": "tool", "tool": "noop"}],
            [{"from": "work", "while": "state.again", "max": 5, "id": "c", "to": "work"},
             {"from": "work", "else": True, "to": "EXIT:done"}],
        )

        def step(n, s):
            i = s.data.get("i", 0) + 1
            s.data["i"] = i
            s.data["again"] = i < 3

        out = Engine().run(room, State(), step)
        assert out.exit == "done"
        assert out.state.data["i"] == 3
        assert out.state.loops["c"] == 2   # looped back twice
        assert out.steps == 3

    def test_loop_capped_by_counter(self):
        room = make_room(
            [{"id": "work", "type": "tool", "tool": "noop"}],
            [{"from": "work", "while": "state.again", "max": 2, "id": "c", "to": "work"},
             {"from": "work", "else": True, "to": "EXIT:capped"}],
        )
        out = Engine().run(room, State(), lambda n, s: s.data.__setitem__("again", True))
        assert out.exit == "capped"
        assert out.state.loops["c"] == 2   # never exceeds max
        assert out.steps == 3


class TestTerminals:
    def test_budget_exhaustion(self):
        room = make_room(
            [{"id": "A", "type": "tool", "tool": "noop"}],
            [{"from": "A", "to": "A"}],   # infinite unconditional loop
            max_iter=5,
        )
        out = Engine().run(room, State(), noop)
        assert out.exit == "budget" and out.steps == 5

    def test_human_pause_node(self):
        room = make_room([{"id": "ask", "type": "human_pause"}], [])
        out = Engine().run(room, State(), noop)
        assert out.exit == "human_pause" and out.steps == 1

    def test_stuck_when_no_edge_matches(self):
        room = make_room(
            [{"id": "x", "type": "tool", "tool": "noop"}],
            [{"from": "x", "if": "state.never", "to": "EXIT:y"}],  # no fallback
        )
        out = Engine().run(room, State(), noop)
        assert out.exit == "stuck" and out.steps == 1

    def test_max_steps_override(self):
        room = make_room(
            [{"id": "A", "type": "tool", "tool": "noop"}],
            [{"from": "A", "to": "A"}],
            max_iter=100,
        )
        out = Engine().run(room, State(), noop, max_steps=3)
        assert out.exit == "budget" and out.steps == 3
