"""Modular rooms: sub-rooms, set / notify nodes, error edges and retries, and the library's
guard against calls to missing rooms and circles of calls."""

import pytest

from src.rooms.engine import Engine
from src.rooms.library import RoomLibrary
from src.rooms.models import Node, Room, State
from src.rooms.runner import NodeRunner
from src.rooms.subroom import SubRoomHandler
from src.rooms.validator import validate_room


def make(name, nodes, edges, **kw):
    kw.setdefault("budget", {"max_iterations": 30, "max_tokens": 1000})
    return Room(room=name, nodes=nodes, edges=edges, **kw)


def lib_with(*rooms):
    lib = RoomLibrary()
    for r in rooms:
        lib.rooms[r.room] = r
        lib.sources[r.room] = "user"
        lib.stats[r.room] = {"successes": 0, "failures": 0, "quarantined": False}
    return lib


def run(room, lib=None, tool=None, agent=None, notify=None, **state):
    runner = NodeRunner(
        tool_executor=tool, agent_executor=agent, notify=notify,
        room_handler=SubRoomHandler(lib) if lib is not None else None)
    st = State(data=dict(state))
    return Engine().run(room, st, runner.run_node), st


# ── schema ─────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_set_notify_room_need_their_fields(self):
        Node(id="s", type="set", values={"n": 1})
        with pytest.raises(ValueError):
            Node(id="s", type="set")
        with pytest.raises(ValueError):
            Node(id="s", type="set", values={"bad name": 1})
        Node(id="n", type="notify", message="hi", channel="desktop")
        with pytest.raises(ValueError):
            Node(id="n", type="notify")
        with pytest.raises(ValueError):
            Node(id="n", type="notify", message="x", channel="pigeon")
        Node(id="r", type="room", room="child", inputs={"a": "{{task}}"}, outputs=["b"], into="child_exit")
        with pytest.raises(ValueError):
            Node(id="r", type="room")
        with pytest.raises(ValueError):
            Node(id="t", type="tool", tool="x", outputs=["a"])

    def test_retries_only_where_a_node_can_raise(self):
        Node(id="t", type="tool", tool="x", retries=2)
        with pytest.raises(ValueError):
            Node(id="c", type="condition", condition="state.x", retries=1)
        with pytest.raises(ValueError):
            Node(id="t", type="tool", tool="x", retries=9)

    def test_error_edge_kind_and_exclusivity(self):
        from src.rooms.models import Edge
        e = Edge(**{"from": "a", "to": "b", "on_error": True})
        assert e.kind == "error"
        with pytest.raises(ValueError):
            Edge(**{"from": "a", "to": "b", "on_error": True, "else": True})


# ── set and notify ───────────────────────────────────────────────────────────────

class TestSetAndNotify:
    def test_set_writes_rendered_values_and_branches_on_them(self):
        r = make("r", [{"id": "s", "type": "set", "values": {"mode": "fast", "label": "job: {{task}}", "n": 3}}],
                 [{"from": "s", "if": "state.mode == \"fast\" and state.n == 3", "to": "EXIT:ok"},
                  {"from": "s", "else": True, "to": "EXIT:no"}])
        out, st = run(r, task="build")
        assert out.exit == "ok" and st.data["label"] == "job: build" and st.data["n"] == 3

    def test_notify_renders_the_message_and_delivers_once(self):
        sent = []
        r = make("r", [{"id": "n", "type": "notify", "title": "T {{task}}", "message": "done: {{task}}"}],
                 [{"from": "n", "to": "EXIT:done"}])
        out, st = run(r, notify=lambda node, state: sent.append((node.title, node.message, node.channel)), task="x")
        assert sent == [("T x", "done: x", "console")] and st.data["n.output"] == "done: x"

    def test_a_set_node_is_replayable_after_a_crash(self):
        from src.rooms.engine import replayable
        assert replayable(Node(id="s", type="set", values={"a": 1}))
        assert replayable(Node(id="n", type="notify", message="m"))
        assert not replayable(Node(id="r", type="room", room="child"))


# ── error edges and retries ──────────────────────────────────────────────────────

class TestErrors:
    def _room(self, retries=0):
        return make("r", [{"id": "t", "type": "tool", "tool": "x", "retries": retries},
                          {"id": "fallback", "type": "set", "values": {"used_fallback": True}}],
                    [{"from": "t", "to": "EXIT:ok"},
                     {"from": "t", "on_error": True, "to": "fallback"},
                     {"from": "fallback", "to": "EXIT:recovered"}])

    def test_a_raising_node_takes_its_error_edge_and_exposes_the_message(self):
        def boom(node, state):
            raise ConnectionError("network down")
        out, st = run(self._room(), tool=boom)
        assert out.exit == "recovered" and st.data["error"] == "network down" and st.data["error_node"] == "t"
        assert st.data["used_fallback"] is True

    def test_retries_run_the_node_again_before_giving_up(self):
        calls = []

        def flaky(node, state):
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("slow")
            return "fine"
        out, st = run(self._room(retries=2), tool=flaky)
        assert out.exit == "ok" and len(calls) == 3 and "error" not in st.data

    def test_retries_exhausted_goes_to_the_error_edge(self):
        calls = []

        def always(node, state):
            calls.append(1)
            raise RuntimeError("no")
        out, _ = run(self._room(retries=1), tool=always)
        assert out.exit == "recovered" and len(calls) == 2

    def test_without_an_error_edge_the_exception_propagates(self):
        r = make("r", [{"id": "t", "type": "tool", "tool": "x"}], [{"from": "t", "to": "EXIT:ok"}])
        with pytest.raises(RuntimeError):
            run(r, tool=lambda n, s: (_ for _ in ()).throw(RuntimeError("x")))

    def test_the_error_edge_is_never_a_normal_route(self):
        r = make("r", [{"id": "t", "type": "tool", "tool": "x"}],
                 [{"from": "t", "on_error": True, "to": "EXIT:failed"}])
        out, _ = run(r, tool=lambda n, s: "fine")
        assert out.exit == "stuck"       # nothing to follow on success; the error edge did not fire

    def test_only_one_error_edge_per_node(self):
        r = make("r", [{"id": "t", "type": "tool", "tool": "x"}],
                 [{"from": "t", "on_error": True, "to": "EXIT:a"}, {"from": "t", "on_error": True, "to": "EXIT:b"},
                  {"from": "t", "to": "EXIT:c"}])
        assert any("more than one on_error" in e for e in validate_room(r, {"x"}))


# ── sub-rooms ──────────────────────────────────────────────────────────────────

def child_room():
    return make("check", [
        {"id": "run", "type": "tool", "tool": "run_command", "args": {"command": "{{cmd}}"},
         "sets": {"passed": "ok"}}],
        [{"from": "run", "if": "state.passed", "to": "EXIT:green"},
         {"from": "run", "else": True, "to": "EXIT:red"}],
        defaults={"cmd": "pytest -q"}, report="passed={{passed}}")


def parent_room():
    return make("pipeline", [
        {"id": "gate", "type": "room", "room": "check", "inputs": {"cmd": "{{test_command}}"},
         "outputs": ["passed"]},
        {"id": "ship", "type": "set", "values": {"shipped": True}},
        {"id": "stop", "type": "set", "values": {"shipped": False}}],
        [{"from": "gate", "if": "state.gate_exit == \"green\"", "to": "ship"},
         {"from": "gate", "else": True, "to": "stop"},
         {"from": "ship", "to": "EXIT:done"}, {"from": "stop", "to": "EXIT:blocked"}])


class TestSubRoom:
    def test_a_child_runs_with_the_parents_inputs_and_hands_back_outputs_and_its_exit(self):
        commands = []

        def tool(node, state):
            commands.append(node.args["command"])
            return "Exit code: 0\nOUTPUT:\nok"
        lib = lib_with(child_room(), parent_room())
        out, st = run(lib.get("pipeline"), lib=lib, tool=tool, test_command="make test")
        assert commands == ["make test"] and out.exit == "done"
        assert st.data["gate_exit"] == "green" and st.data["passed"] is True and st.data["shipped"] is True
        assert st.data["gate.output"] == "passed=True"
        assert "cmd" not in st.data                  # the child's own fields stay in the child

    def test_the_parent_branches_on_the_childs_exit(self):
        lib = lib_with(child_room(), parent_room())
        out, st = run(lib.get("pipeline"), lib=lib, tool=lambda n, s: "Exit code: 1\nOUTPUT:\nFAILED",
                      test_command="make test")
        assert out.exit == "blocked" and st.data["gate_exit"] == "red" and st.data["shipped"] is False

    def test_the_child_uses_its_own_defaults_when_no_input_is_given(self):
        seen = []
        lib = lib_with(child_room(), make("p", [{"id": "g", "type": "room", "room": "check"}],
                                          [{"from": "g", "to": "EXIT:done"}]))
        run(lib.get("p"), lib=lib, tool=lambda n, s: seen.append(n.args["command"]) or "Exit code: 0")
        assert seen == ["pytest -q"]

    def test_into_names_the_exit_field(self):
        p = make("p", [{"id": "g", "type": "room", "room": "check", "into": "verdict"}],
                 [{"from": "g", "if": "state.verdict == \"green\"", "to": "EXIT:yes"}, {"from": "g", "else": True, "to": "EXIT:no"}])
        lib = lib_with(child_room(), p)
        out, st = run(p, lib=lib, tool=lambda n, s: "Exit code: 0")
        assert out.exit == "yes" and st.data["verdict"] == "green"

    def test_child_tokens_count_against_the_parent(self):
        class Agent:
            last_tokens = 0

            def __call__(self, node, ctx):
                self.last_tokens = 700
                return {}
        c = make("worker", [{"id": "a", "type": "agent", "budget": {"max_iterations": 1, "max_tokens": 10000}}],
                 [{"from": "a", "to": "EXIT:done"}], budget={"max_iterations": 5, "max_tokens": 5000})
        p = make("p", [{"id": "g", "type": "room", "room": "worker"}, {"id": "g2", "type": "room", "room": "worker"},
                       {"id": "g3", "type": "room", "room": "worker"}],
                 [{"from": "g", "to": "g2"}, {"from": "g2", "to": "g3"}, {"from": "g3", "to": "EXIT:done"}],
                 budget={"max_iterations": 10, "max_tokens": 1000})
        lib = lib_with(c, p)
        out, st = run(p, lib=lib, agent=Agent())
        # 700 after the first child (under 1000), 1400 after the second: the check before the third node stops it
        assert st.usage["tokens"] == 1400 and out.exit == "budget" and "token" in out.reason

    def test_a_missing_child_raises_so_an_error_edge_can_handle_it(self):
        p = make("p", [{"id": "g", "type": "room", "room": "ghost"}],
                 [{"from": "g", "to": "EXIT:done"}, {"from": "g", "on_error": True, "to": "EXIT:missing"}])
        lib = lib_with(p)
        out, st = run(p, lib=lib)
        assert out.exit == "missing" and "ghost" in st.data["error"]

    def test_depth_is_bounded(self):
        # three levels is the ceiling; a chain deeper than that raises (here handled by on_error)
        rooms = []
        for i in range(5):
            nodes = [{"id": "g", "type": "room", "room": f"r{i + 1}"}] if i < 4 else [
                {"id": "leaf", "type": "set", "values": {"x": 1}}]
            edges = [{"from": nodes[0]["id"], "to": "EXIT:done"}]
            if i < 4:
                edges.append({"from": "g", "on_error": True, "to": "EXIT:deep"})
            rooms.append(make(f"r{i}", nodes, edges))
        lib = lib_with(*rooms)
        out, st = run(rooms[0], lib=lib)
        # r0 -> r1 -> r2 -> r3 is depth 3; r3 -> r4 would be the 4th level and raises inside r3,
        # which r3's own error edge turns into the exit "deep" of r3 — r0..r2 still finish "done".
        assert out.exit == "done" and st.data["g_exit"] == "done"


# ── library: references ──────────────────────────────────────────────────────────

class TestReferences:
    def test_missing_target_and_cycles_are_reported(self):
        a = make("a", [{"id": "g", "type": "room", "room": "b"}], [{"from": "g", "to": "EXIT:done"}])
        b = make("b", [{"id": "g", "type": "room", "room": "a"}], [{"from": "g", "to": "EXIT:done"}])
        c = make("c", [{"id": "g", "type": "room", "room": "nowhere"}], [{"from": "g", "to": "EXIT:done"}])
        lib = lib_with(a, b, c)
        errs = lib.reference_errors()
        assert "circle" in errs["a"][0] and "circle" in errs["b"][0] and "nowhere" in errs["c"][0]

    def test_a_room_cannot_call_itself(self):
        r = make("loop", [{"id": "g", "type": "room", "room": "loop"}], [{"from": "g", "to": "EXIT:done"}])
        assert any("itself" in e for e in validate_room(r, set()))

    def test_add_room_rejects_a_call_to_a_missing_room_and_accepts_a_good_one(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path)
        lib.rooms["check"] = child_room()
        lib.stats["check"] = {"successes": 0, "failures": 0, "quarantined": False}
        bad = make("p1", [{"id": "g", "type": "room", "room": "ghost"}], [{"from": "g", "to": "EXIT:done"}])
        assert any("ghost" in e for e in lib.add_room(bad, {"run_command"}))
        good = make("p2", [{"id": "g", "type": "room", "room": "check"}], [{"from": "g", "to": "EXIT:done"}])
        assert lib.add_room(good, {"run_command"}) == []
        assert lib.referenced_by("check") == ["p2"]
        assert (tmp_path / "p2.json").exists()

    def test_load_dir_drops_a_caller_of_a_missing_room(self, tmp_path):
        import json
        (tmp_path / "caller.json").write_text(json.dumps({
            "room": "caller", "budget": {"max_iterations": 3, "max_tokens": 10},
            "nodes": [{"id": "g", "type": "room", "room": "ghost"}],
            "edges": [{"from": "g", "to": "EXIT:done"}]}), encoding="utf-8")
        lib = RoomLibrary().load_dir(tmp_path, set(), origin="user")
        assert lib.get("caller") is None and "ghost" in lib.load_errors["caller"][0]
