"""Rooms engine — schema (models), safe condition DSL, and RoomValidator."""

import copy

import pytest
from pydantic import ValidationError

from src.rooms import condition
from src.rooms.models import Budget, Edge, Node, Room, State
from src.rooms.validator import validate_room


VALID_ROOM = {
    "room": "tests_failed",
    "description": "After two test failures",
    "budget": {"max_iterations": 5, "max_tokens": 40000},
    "nodes": [
        {"id": "web_search", "type": "tool", "tool": "search",
         "input_from": "state.error_summary", "instruction": "search the error"},
        {"id": "apply", "type": "agent",
         "budget": {"max_iterations": 3, "max_tokens": 20000},
         "context": ["web_search.output"], "writes": ["tests_passed"]},
    ],
    "edges": [
        {"from": "web_search", "to": "apply"},
        {"from": "apply", "if": "state.tests_passed", "to": "EXIT:success"},
        {"from": "apply", "else": True, "to": "EXIT:escalate"},
    ],
}


def _room(**overrides):
    d = copy.deepcopy(VALID_ROOM)
    d.update(overrides)
    return Room(**d)


# ── Models ───────────────────────────────────────────────────────────

class TestModels:
    def test_valid_room_builds(self):
        room = Room(**VALID_ROOM)
        assert room.room == "tests_failed"
        assert room.entry == "web_search"  # defaults to first node

    def test_budget_must_be_positive(self):
        with pytest.raises(ValidationError):
            Budget(max_iterations=0, max_tokens=100)

    def test_tool_node_requires_tool(self):
        with pytest.raises(ValidationError):
            Node(id="n", type="tool")

    def test_agent_node_requires_budget(self):
        with pytest.raises(ValidationError):
            Node(id="n", type="agent")

    def test_condition_node_requires_condition(self):
        with pytest.raises(ValidationError):
            Node(id="n", type="condition")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Node(id="n", type="human_pause", bogus=1)

    def test_edge_aliases_and_kind(self):
        assert Edge(**{"from": "a", "to": "b"}).kind == "uncond"
        assert Edge(**{"from": "a", "to": "b", "if": "state.x"}).kind == "if"
        assert Edge(**{"from": "a", "to": "b", "else": True}).kind == "else"
        assert Edge(**{"from": "a", "to": "b", "while": "state.x", "max": 3, "id": "c"}).kind == "while"

    def test_while_edge_requires_counter(self):
        with pytest.raises(ValidationError):
            Edge(**{"from": "a", "to": "b", "while": "state.x", "max": 3})  # no id
        with pytest.raises(ValidationError):
            Edge(**{"from": "a", "to": "b", "while": "state.x", "id": "c"})  # no max

    def test_edge_cannot_mix_kinds(self):
        with pytest.raises(ValidationError):
            Edge(**{"from": "a", "to": "b", "if": "state.x", "while": "state.y", "max": 2, "id": "c"})

    def test_exit_target_detected(self):
        assert Edge(**{"from": "a", "to": "EXIT:success"}).targets_exit()
        assert not Edge(**{"from": "a", "to": "b"}).targets_exit()

    def test_state_defaults(self):
        s = State()
        assert s.data == {} and s.loops == {}


# ── Condition DSL ────────────────────────────────────────────────────

class TestCondition:
    def test_state_field_truthy(self):
        assert condition.evaluate("state.ok", {"ok": True}) is True
        assert condition.evaluate("state.ok", {"ok": False}) is False

    def test_missing_field_is_none(self):
        assert condition.evaluate("state.nope", {}) is False

    def test_comparisons(self):
        assert condition.evaluate("state.n < 3", {"n": 2}) is True
        assert condition.evaluate("state.n >= 3", {"n": 2}) is False

    def test_membership(self):
        assert condition.evaluate('state.kind in ["a", "b"]', {"kind": "a"}) is True
        assert condition.evaluate('state.kind not in ["a", "b"]', {"kind": "c"}) is True

    def test_boolean_logic(self):
        assert condition.evaluate("state.a and not state.b", {"a": True, "b": False}) is True
        assert condition.evaluate("state.a or state.b", {"a": False, "b": False}) is False

    @pytest.mark.parametrize("expr", [
        "foo()",                 # function call
        "state.x[0]",            # subscript
        "os",                    # bare name
        "state.x.y",             # attribute chain beyond state.<field>
        "__import__('os')",      # call
        "",                      # empty
        "state.x = 1",           # not an expression
    ])
    def test_rejects_forbidden(self, expr):
        with pytest.raises(condition.ConditionError):
            condition.validate(expr)


# ── Validator ────────────────────────────────────────────────────────

class TestValidator:
    def test_valid_room_passes(self):
        assert validate_room(_room(), available_tools={"search"}) == []

    def test_unknown_tool_flagged(self):
        errs = validate_room(_room(), available_tools=set())
        assert any("tool 'search' does not exist" in e for e in errs)

    def test_edge_to_unknown_node(self):
        d = copy.deepcopy(VALID_ROOM)
        d["edges"].append({"from": "apply", "to": "ghost"})
        errs = validate_room(Room(**d), available_tools={"search"})
        assert any("unknown target 'ghost'" in e for e in errs)

    def test_dead_end_node_cannot_reach_exit(self):
        d = copy.deepcopy(VALID_ROOM)
        # A node with no path to an EXIT.
        d["nodes"].append({"id": "orphan", "type": "human_pause"})
        d["edges"].append({"from": "apply", "to": "orphan"})  # orphan has no outgoing edge
        errs = validate_room(Room(**d), available_tools={"search"})
        assert any("cannot reach any EXIT" in e and "orphan" in e for e in errs)

    def test_duplicate_node_ids(self):
        d = copy.deepcopy(VALID_ROOM)
        d["nodes"].append({"id": "apply", "type": "human_pause"})
        errs = validate_room(Room(**d), available_tools={"search"})
        assert any("duplicate node ids" in e for e in errs)

    def test_bad_condition_flagged(self):
        d = copy.deepcopy(VALID_ROOM)
        d["edges"][1] = {"from": "apply", "if": "evil()", "to": "EXIT:success"}
        errs = validate_room(Room(**d), available_tools={"search"})
        assert any("bad condition" in e for e in errs)

    def test_spawn_room_only_in_triage(self):
        d = copy.deepcopy(VALID_ROOM)
        d["nodes"].append({"id": "spawn", "type": "spawn_room"})
        d["edges"].append({"from": "spawn", "to": "EXIT:success"})
        d["edges"].append({"from": "apply", "to": "spawn"})
        room = Room(**d)
        # Disallowed by default…
        assert any("spawn_room is only allowed" in e for e in validate_room(room, {"search"}))
        # …allowed when the room is the triage room.
        assert not any("spawn_room" in e for e in validate_room(room, {"search"}, allow_spawn=True))
