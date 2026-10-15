"""Point operations on a room (src/rooms/patch.py) and the `rooms` tool's patch / create actions."""

import copy
import inspect
import json

import pytest

from src.rooms.models import Room
from src.rooms.patch import MAX_OPS, apply_ops, coerce_ops
from src.rooms.tool import rooms_action
from tests.test_rooms_builder import GOOD
from tests.test_rooms_cli_tool import lib, starter_lib  # noqa: F401  (fixtures)


def base() -> Room:
    return Room(**copy.deepcopy(GOOD))


def apply(*ops, room=None, name="lint_then_tell"):
    return apply_ops(room if room is not None else base(), list(ops), name)


def edge(room: Room, src, dst, kind=None):
    return [e for e in room.edges if e.from_ == src and e.to == dst and (kind is None or e.kind == kind)]


# ── parameters ───────────────────────────────────────────────────────────────

class TestSetParam:
    def test_plain_and_dotted_paths(self):
        r = apply({"op": "set_param", "node": "lint", "field": "args.command", "value": "ruff check src"},
                  {"op": "set_param", "node": "lint", "values": {"retries": 2, "note": "линтер"}})
        assert r.ok, r.errors
        node = next(n for n in r.room.nodes if n.id == "lint")
        assert node.args == {"command": "ruff check src"} and node.retries == 2 and node.note == "линтер"

    def test_null_resets_to_default(self):
        r = apply({"op": "set_param", "node": "lint", "field": "retries", "value": None})
        assert r.ok and next(n for n in r.room.nodes if n.id == "lint").retries == 0

    def test_creates_the_object_for_a_dotted_path(self):
        agent = {"op": "add_node", "node": {"id": "ask", "type": "agent", "role": "Researcher", "writes": ["a"],
                                              "budget": {"max_iterations": 2, "max_tokens": 900}}}
        r = apply(agent, {"op": "set_param", "node": "ask", "field": "sampling.temperature", "value": 0.2},
                  {"op": "set_param", "node": "ask", "field": "budget.max_tokens", "value": 1500})
        assert r.ok, r.errors
        ask = next(n for n in r.room.nodes if n.id == "ask")
        assert ask.sampling == {"temperature": 0.2} and ask.budget.max_tokens == 1500

    def test_unknown_parameter_and_id_are_explained(self):
        r = apply({"op": "set_param", "node": "lint", "field": "colour", "value": "red"})
        assert not r.ok and "no parameter 'colour'" in r.errors[0]
        r = apply({"op": "set_param", "node": "lint", "field": "id", "value": "x"})
        assert "rename_node" in r.errors[0]
        r = apply({"op": "set_param", "node": "ghost", "field": "note", "value": "x"})
        assert "no node 'ghost'" in r.errors[0]

    def test_model_validation_is_reported_per_field(self):
        r = apply({"op": "set_param", "node": "good", "field": "channel", "value": "fax"})
        assert r.room is None and r.errors and "channel" in r.errors[0]

    def test_changing_the_type_recomputes_the_idempotent_default(self):
        r = apply({"op": "set_param", "node": "good", "values": {"type": "power", "mode": "lock", "message": None}})
        assert r.ok, r.errors
        good = next(n for n in r.room.nodes if n.id == "good")
        assert good.type == "power" and good.idempotent is False


# ── blocks and arrows ────────────────────────────────────────────────────────

class TestStructure:
    def test_add_node_and_connect(self):
        r = apply({"op": "add_node", "node": {"id": "tell", "type": "notify", "message": "done", "channel": "desktop"}},
                  {"op": "connect", "from": "tell", "to": "EXIT:clean"})
        assert r.ok, r.errors
        assert any(n.id == "tell" for n in r.room.nodes) and edge(r.room, "tell", "EXIT:clean")

    def test_duplicates_and_dangling_arrows_are_refused(self):
        r = apply({"op": "add_node", "node": {"id": "lint", "type": "notify", "message": "x"}})
        assert "already exists" in r.errors[0]
        r = apply({"op": "connect", "from": "lint", "to": "nowhere"})
        assert "no node 'nowhere'" in r.errors[0]
        r = apply({"op": "connect", "from": "good", "to": "EXIT:clean"})
        assert "already exists" in r.errors[0]

    def test_insert_node_keeps_the_condition_on_the_first_half(self):
        r = apply({"op": "insert_node", "between": ["lint", "good"],
                   "node": {"id": "mid", "type": "notify", "message": "linted", "channel": "console"}})
        assert r.ok, r.errors
        assert edge(r.room, "lint", "mid", "if")[0].if_ == "state.lint_ok"
        assert edge(r.room, "mid", "good", "uncond") and not edge(r.room, "lint", "good")

    def test_insert_into_a_missing_arrow(self):
        r = apply({"op": "insert_node", "between": ["good", "lint"],
                   "node": {"id": "mid", "type": "notify", "message": "x"}})
        assert "no arrow good → lint" in r.errors[0]

    def test_remove_node_closes_the_gap(self):
        r = apply({"op": "remove_node", "node": "good"})
        assert r.ok, r.errors
        assert edge(r.room, "lint", "EXIT:clean", "if") and not any(n.id == "good" for n in r.room.nodes)

    def test_remove_node_with_branches_asks_for_a_decision(self):
        r = apply({"op": "connect", "from": "good", "to": "bad", "if": "state.again"},
                  {"op": "remove_node", "node": "good"})
        assert not r.ok and "2 outgoing arrows" in r.errors[0]

    def test_the_entry_cannot_be_dropped_blindly(self):
        r = apply({"op": "remove_node", "node": "lint", "reconnect": False})
        assert "is the entry" in r.errors[0]

    def test_rename_node_follows_arrows_and_entry_and_warns_about_text(self):
        data = copy.deepcopy(GOOD)
        data["nodes"][2]["message"] = "lint failed: {{lint.output}}"
        r = apply({"op": "rename_node", "node": "lint", "new_id": "ruff"}, room=Room(**data))
        assert r.ok, r.errors
        assert r.room.entry == "ruff" and edge(r.room, "ruff", "good") and not edge(r.room, "lint", "good")
        assert r.warnings and "lint" in r.warnings[0]

    def test_disconnect_needs_a_selector_when_ambiguous(self):
        r = apply({"op": "disconnect", "from": "lint", "to": "bad"})
        assert "say which" in r.errors[0]
        r = apply({"op": "disconnect", "from": "lint", "to": "bad", "else": True})
        assert r.ok and not edge(r.room, "lint", "bad", "else") and edge(r.room, "lint", "bad", "error")

    def test_set_edge_changes_the_condition_and_target(self):
        r = apply({"op": "set_edge", "from": "lint", "to": "good", "set": {"if": "state.lint_ok and not state.dirty"}})
        assert r.ok and edge(r.room, "lint", "good", "if")[0].if_ == "state.lint_ok and not state.dirty"
        r = apply({"op": "set_edge", "from": "lint", "to": "good", "set": {"to": "bad"}})
        assert r.ok and edge(r.room, "lint", "bad", "if")
        r = apply({"op": "set_edge", "from": "lint", "to": "good", "set": {"colour": 1}})
        assert "unknown arrow field" in r.errors[0]

    def test_set_edge_to_another_kind_drops_the_old_one(self):
        r = apply({"op": "set_edge", "from": "lint", "to": "good", "set": {"else": True}})
        assert r.ok
        e = edge(r.room, "lint", "good")[0]
        assert e.kind == "else" and e.if_ is None


class TestRoomFields:
    def test_set_room(self):
        r = apply({"op": "set_room", "values": {"budget.max_tokens": 9000, "defaults.lint_command": "ruff .",
                                                  "next.clean": "END", "rules.read_only": True}})
        assert r.ok, r.errors
        assert r.room.budget.max_tokens == 9000 and r.room.defaults["lint_command"] == "ruff ."
        assert r.room.rules.read_only is True

    def test_nodes_name_and_unknown_fields_are_refused(self):
        assert "block / arrow operations" in apply({"op": "set_room", "field": "nodes", "value": []}).errors[0]
        assert "save_as" in apply({"op": "set_room", "field": "room", "value": "x"}).errors[0]
        assert "no room parameter" in apply({"op": "set_room", "field": "colour", "value": 1}).errors[0]


class TestAllOrNothing:
    def test_a_failure_discards_everything_and_leaves_the_room_alone(self):
        room = base()
        before = room.model_dump_json()
        r = apply({"op": "set_param", "node": "lint", "field": "note", "value": "changed"},
                  {"op": "remove_node", "node": "ghost"}, room=room)
        assert r.room is None and r.errors[0].startswith("op #2 (remove_node)")
        assert room.model_dump_json() == before

    def test_unknown_op_and_limits(self):
        assert "unknown op 'fly'" in apply({"op": "fly"}).errors[0]
        ops, problem = coerce_ops([{"op": "set_room", "field": "description", "value": "x"}] * (MAX_OPS + 1))
        assert problem and "at most" in problem
        assert coerce_ops(None)[1] and coerce_ops("not json")[1] and coerce_ops([1])[1]
        assert coerce_ops(json.dumps({"op": "connect"}))[0] == [{"op": "connect"}]

    def test_a_blank_room_is_built_from_operations(self):
        r = apply_ops(None, [
            {"op": "add_node", "node": {"id": "a", "type": "notify", "message": "hi", "channel": "console"}},
            {"op": "connect", "from": "a", "to": "EXIT:done"},
            {"op": "set_room", "field": "description", "value": "say hi"},
        ], "say_hi")
        assert r.ok, r.errors
        assert r.room.room == "say_hi" and r.room.entry == "a" and r.room.budget.max_iterations == 40

    def test_a_blank_room_without_blocks_is_not_a_room(self):
        r = apply_ops(None, [{"op": "set_room", "field": "description", "value": "x"}], "empty_one")
        assert not r.ok


# ── the tool ─────────────────────────────────────────────────────────────────

def approve(seen=None):
    def ask(flow, name):
        if seen is not None:
            seen.update(flow=flow, name=name)
        return True
    return ask


class TestToolPatch:
    @pytest.fixture
    def mine(self, lib):
        assert lib.add_room(base(), {"run_command", "write_file"}) == []
        return lib

    def test_saved_after_approval_and_the_person_sees_the_changes(self, mine):
        seen = {}
        out = rooms_action("patch", "lint_then_tell", ops=[
            {"op": "set_param", "node": "lint", "field": "retries", "value": 3}], library=mine, ask_save=approve(seen))
        assert "saved" in out
        assert next(n for n in mine.get("lint_then_tell").nodes if n.id == "lint").retries == 3
        assert "Точечные правки модели" in seen["flow"] and "retries" in seen["flow"]
        assert "Что изменилось" in seen["flow"]

    def test_refusal_changes_nothing(self, mine):
        out = rooms_action("patch", "lint_then_tell", ops=[
            {"op": "set_param", "node": "lint", "field": "retries", "value": 3}], library=mine,
            ask_save=lambda f, n: False)
        assert "did NOT approve" in out
        assert next(n for n in mine.get("lint_then_tell").nodes if n.id == "lint").retries == 1

    def test_house_rules_apply_to_what_the_model_adds(self, mine):
        out = rooms_action("patch", "lint_then_tell", ops=[
            {"op": "insert_node", "between": ["good", "EXIT:clean"],
             "node": {"id": "wipe", "type": "tool", "tool": "write_file",
                      "args": {"file_path": "x.txt", "content": "y"}}},
            {"op": "insert_node", "between": ["bad", "EXIT:dirty"],
             "node": {"id": "kill", "type": "power", "mode": "shutdown", "seconds": 60}}],
            library=mine, ask_save=approve())
        assert "saved" in out and "Safety rule" in out
        room = mine.get("lint_then_tell")
        assert next(n for n in room.nodes if n.id == "wipe").confirm is True
        assert next(n for n in room.nodes if n.id == "kill").confirm is True

    def test_confirm_cannot_be_switched_off_for_a_step_that_acts(self, mine):
        rooms_action("patch", "lint_then_tell", ops=[
            {"op": "insert_node", "between": ["good", "EXIT:clean"],
             "node": {"id": "wipe", "type": "tool", "tool": "write_file",
                      "args": {"file_path": "x.txt", "content": "y"}, "confirm": False}}],
            library=mine, ask_save=approve())
        assert next(n for n in mine.get("lint_then_tell").nodes if n.id == "wipe").confirm is True

    def test_an_invalid_result_is_not_even_shown(self, mine):
        asked = []
        out = rooms_action("patch", "lint_then_tell", ops=[
            {"op": "add_node", "node": {"id": "ghost", "type": "tool", "tool": "no_such_tool"}},
            {"op": "connect", "from": "ghost", "to": "EXIT:clean"}],
            library=mine, ask_save=lambda f, n: asked.append(n) or True)
        assert out.startswith("Error: the room would be invalid") and not asked
        assert not any(n.id == "ghost" for n in mine.get("lint_then_tell").nodes)

    def test_packaged_rooms_are_copied_not_overwritten(self, lib):
        before = lib.get("code_review").model_dump_json()
        out = rooms_action("patch", "code_review", ops=[
            {"op": "set_room", "field": "description", "value": "my review"}], library=lib, ask_save=approve())
        assert "saved" in out and "code_review_custom" in out
        assert lib.get("code_review").model_dump_json() == before
        assert lib.get("code_review_custom").description == "my review"
        again = rooms_action("patch", "code_review", ops=[
            {"op": "set_room", "field": "description", "value": "x"}], library=lib, ask_save=approve())
        assert again.startswith("Error:") and "code_review_custom" in again

    def test_save_as_makes_a_copy(self, mine):
        out = rooms_action("patch", "lint_then_tell", save_as="lint_quiet", ops=[
            {"op": "set_param", "node": "good", "field": "channel", "value": "console"}],
            library=mine, ask_save=approve())
        assert "saved" in out
        assert next(n for n in mine.get("lint_quiet").nodes if n.id == "good").channel == "console"
        assert next(n for n in mine.get("lint_then_tell").nodes if n.id == "good").channel == "desktop"

    def test_ops_may_come_as_a_json_string(self, mine):
        out = rooms_action("patch", "lint_then_tell", library=mine, ask_save=approve(),
                           ops=json.dumps([{"op": "set_room", "field": "tags", "value": ["lint", "ruff"]}]))
        assert "saved" in out and mine.get("lint_then_tell").tags == ["lint", "ruff"]

    def test_errors_name_the_operation(self, mine):
        out = rooms_action("patch", "lint_then_tell", library=mine, ops=[{"op": "remove_node", "node": "zzz"}])
        assert out.startswith("Error: nothing was changed") and "op #1 (remove_node)" in out
        assert rooms_action("patch", "ghost", library=mine, ops=[{"op": "set_room", "field": "tags", "value": []}]
                            ).startswith("Error: no room")
        assert rooms_action("patch", "lint_then_tell", library=mine).startswith("Error: `ops`")


class TestToolCreate:
    def test_a_room_made_block_by_block(self, lib):
        seen = {}
        out = rooms_action("create", "hello_room", ops=[
            {"op": "set_room", "values": {"description": "Скажи привет", "budget": {"max_iterations": 5, "max_tokens": 500}}},
            {"op": "add_node", "node": {"id": "hi", "type": "notify", "message": "привет", "channel": "console"}},
            {"op": "connect", "from": "hi", "to": "EXIT:done"}], library=lib, ask_save=approve(seen))
        assert "saved" in out and lib.get("hello_room").entry == "hi"
        assert "Новая" in seen["flow"] or "hi" in seen["flow"]

    def test_existing_and_invalid_names_are_refused(self, lib):
        ops = [{"op": "add_node", "node": {"id": "hi", "type": "notify", "message": "x"}}]
        assert rooms_action("create", "code_review", ops=ops, library=lib).startswith("Error: room 'code_review' already exists")
        assert rooms_action("create", "Bad Name", ops=ops, library=lib).startswith("Error: room name")


class TestWiring:
    def test_plan_mode_blocks_patch_and_create(self):
        from src.agent.plan_mode import PLAN, violation
        PLAN.enter()
        try:
            assert violation("rooms", {"action": "patch"}) and violation("rooms", {"action": "create"})
            assert violation("rooms", {"action": "show"}) is None
        finally:
            PLAN.leave()

    def test_the_tool_function_and_schema_carry_the_new_arguments(self):
        from tools.rooms_tools import rooms
        from tools.schemas import TOOL_SCHEMAS
        params = inspect.signature(rooms).parameters
        assert "ops" in params and "save_as" in params
        entry = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "rooms")
        props = entry["function"]["parameters"]["properties"]
        assert "ops" in props and {"patch", "create"} <= set(props["action"]["enum"])
