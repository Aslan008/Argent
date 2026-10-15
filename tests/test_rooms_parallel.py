"""Parallel branches: a `parallel` node starts every arrow at once, a `join` node merges the branches."""

import threading
import time

import pytest

import run_scope
from src.rooms import parallel as par
from src.rooms.engine import Engine, MergeConflict, plan_resume
from src.rooms.journal import Journal
from src.rooms.models import Room, State
from src.rooms.runner import AgentExecutor, NodeRunner
from src.rooms.validator import room_warnings, validate_room

TOOLS = {"run_command", "read_file", "browser_open", "browser_action", "write_file"}


def make_room(branches, join=None, edges_extra=(), nodes_extra=(), budget=None, workers=None):
    """A room `fork -> <branch steps> -> merge -> EXIT:done`. `branches` maps a branch's first step id to the
    node dicts of the branch (the first one is the entry; they are chained in order)."""
    join = dict({"id": "merge", "type": "join"}, **(join or {}))
    nodes = [dict({"id": "fork", "type": "parallel", "join": "merge"}, **({"workers": workers} if workers else {})), join]
    edges = []
    for steps in branches:
        nodes.extend(steps)
        edges.append({"from": "fork", "to": steps[0]["id"]})
        for a, b in zip(steps, steps[1:]):
            edges.append({"from": a["id"], "to": b["id"]})
        edges.append({"from": steps[-1]["id"], "to": "merge"})
    edges.append({"from": "merge", "to": "EXIT:done"})
    nodes.extend(nodes_extra)
    edges.extend(edges_extra)
    return Room(room="p", entry="fork", nodes=nodes, edges=edges,
                budget=budget or {"max_iterations": 50, "max_tokens": 100000})


def setter(node_id, **values):
    return {"id": node_id, "type": "set", "values": values}


def run(room, runner=None, **kw):
    runner = runner or NodeRunner()
    return Engine().run(room, State(), runner.run_node, **kw)


class TestMerge:
    def test_disjoint_fields_are_all_kept(self):
        out = run(make_room([[setter("a", x=1)], [setter("b", y=2)]]))
        assert out.exit == "done"
        assert out.state.data["x"] == 1 and out.state.data["y"] == 2
        assert out.state.data["merge_arrived"] == ["a", "b"]
        assert out.state.data["merge_failed"] == []

    def test_a_field_set_before_the_fork_is_seen_by_every_branch(self):
        room = make_room([[setter("a", seen="{{start}}")], [setter("b", seen_too="{{start}}")]])
        out = Engine().run(room, State(data={"start": "S"}), NodeRunner().run_node)
        assert out.state.data["seen"] == "S" and out.state.data["seen_too"] == "S"

    def test_branches_do_not_see_each_others_writes(self):
        calls = []

        def tool(node, state):
            calls.append((node.id, dict(state.data)))
            return "ok"
        room = make_room([[setter("a", x=1), {"id": "ta", "type": "tool", "tool": "read_file"}],
                          [setter("b", y=2), {"id": "tb", "type": "tool", "tool": "read_file"}]], workers=1)
        run(room, NodeRunner(tool_executor=tool))
        seen = dict(calls)
        assert "y" not in seen["ta"] and "x" not in seen["tb"]

    def test_conflict_is_an_error_by_default(self):
        room = make_room([[setter("a", x=1)], [setter("b", x=2)]])
        with pytest.raises(MergeConflict):
            run(room)

    def test_equal_values_are_not_a_conflict(self):
        out = run(make_room([[setter("a", x=1)], [setter("b", x=1)]]))
        assert out.state.data["x"] == 1

    @pytest.mark.parametrize("mode, expected", [("first", 1), ("last", 2), ("list", [1, 2])])
    def test_merge_policies_follow_the_order_of_the_arrows(self, mode, expected):
        out = run(make_room([[setter("a", x=1)], [setter("b", x=2)]], join={"merge": mode}))
        assert out.state.data["x"] == expected

    def test_arrow_order_wins_even_when_the_first_branch_finishes_last(self):
        def tool(node, state):
            if node.id == "slow":
                time.sleep(0.15)
            return "from slow" if node.id == "slow" else "from fast"
        room = make_room([[{"id": "slow", "type": "tool", "tool": "read_file", "sets": {"x": "first_line"}}],
                          [{"id": "fast", "type": "tool", "tool": "read_file", "sets": {"x": "first_line"}}]],
                         join={"merge": "last"})
        out = run(room, NodeRunner(tool_executor=tool))
        assert out.state.data["x"] == "from fast"          # the last arrow, although it finished first
        first = run(make_room([[{"id": "slow", "type": "tool", "tool": "read_file", "sets": {"x": "first_line"}}],
                               [{"id": "fast", "type": "tool", "tool": "read_file", "sets": {"x": "first_line"}}]],
                              join={"merge": "first"}), NodeRunner(tool_executor=tool))
        assert first.state.data["x"] == "from slow"        # the first arrow, although it finished last

    def test_visit_counters_are_summed(self):
        room = make_room([[setter("a", x=1)], [setter("b", y=1)]])
        out = run(room)
        assert out.state.data["a_visits"] == 1 and out.state.data["b_visits"] == 1
        assert out.state.data["merge_visits"] == 1

    def test_memory_and_hints_of_every_branch_are_kept(self):
        room = make_room([[{"id": "a", "type": "agent", "role": "Researcher", "tools": [],
                            "budget": {"max_iterations": 1, "max_tokens": 100}, "memory": "shared"}],
                          [{"id": "b", "type": "agent", "role": "Researcher", "tools": [],
                            "budget": {"max_iterations": 1, "max_tokens": 100}, "memory": "shared"}]])

        class Said(AgentExecutor):
            def __init__(self):
                super().__init__(lambda prompt, **kw: "done " + prompt[-5:])
        out = run(room, NodeRunner(agent_executor=Said()))
        assert len(out.state.memory) == 2
        assert {m.split("]")[0] for m in out.state.memory} == {"[a", "[b"}

    def test_token_usage_of_all_branches_is_added(self):
        class Costly:
            accepts_brief = False

            def __init__(self):
                self._l = threading.local()

            @property
            def last_tokens(self):
                return getattr(self._l, "t", 0)

            @last_tokens.setter
            def last_tokens(self, v):
                self._l.t = v

            def __call__(self, node, ctx):
                self.last_tokens = 700
                return {}
        ag = lambda i: {"id": i, "type": "agent", "role": "Researcher", "tools": [],
                        "budget": {"max_iterations": 1, "max_tokens": 100}}
        out = run(make_room([[ag("a")], [ag("b")]]), NodeRunner(agent_executor=Costly()))
        assert out.state.usage["tokens"] == 1400


class TestConcurrency:
    def test_branches_really_run_at_the_same_time(self):
        barrier = threading.Barrier(2, timeout=5)

        def tool(node, state):
            barrier.wait()                 # both branches must be inside this step at once
            return "ok"
        room = make_room([[{"id": "a", "type": "tool", "tool": "read_file"}],
                          [{"id": "b", "type": "tool", "tool": "read_file"}]])
        out = run(room, NodeRunner(tool_executor=tool))
        assert out.exit == "done"

    def test_workers_one_runs_them_one_after_another(self):
        active, peak = [0], [0]
        lock = threading.Lock()

        def tool(node, state):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return "ok"
        room = make_room([[{"id": "a", "type": "tool", "tool": "read_file"}],
                          [{"id": "b", "type": "tool", "tool": "read_file"}]], workers=1)
        run(room, NodeRunner(tool_executor=tool))
        assert peak[0] == 1

    def test_agent_executor_keeps_its_cost_per_thread(self):
        ex = AgentExecutor(lambda prompt, **kw: "x")
        seen = {}

        def worker(name, tokens):
            ex.last_tokens = tokens
            time.sleep(0.05)
            seen[name] = ex.last_tokens
        threads = [threading.Thread(target=worker, args=("a", 11)), threading.Thread(target=worker, args=("b", 22))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert seen == {"a": 11, "b": 22}

    def test_steps_in_the_page_are_taken_in_turns(self):
        active, peak = [0], [0]
        lock = threading.Lock()

        def tool(node, state):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return "ok"
        room = make_room([[{"id": "a", "type": "tool", "tool": "browser_open"}],
                          [{"id": "b", "type": "tool", "tool": "browser_action"}]])
        run(room, NodeRunner(tool_executor=tool))
        assert peak[0] == 1

    def test_sampling_of_one_branch_is_not_read_by_the_other(self):
        from config import _get, generation_overrides
        seen = {}
        both_in = threading.Barrier(2, timeout=5)

        def branch(name, temperature):
            with run_scope.branch():
                with generation_overrides({"temperature": temperature}):
                    both_in.wait()
                    seen[name] = _get("temperature")
                    both_in.wait()
        import contextvars
        threads = [threading.Thread(target=contextvars.copy_context().run, args=(branch, "a", 0.1)),
                   threading.Thread(target=contextvars.copy_context().run, args=(branch, "b", 0.9))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert seen == {"a": 0.1, "b": 0.9}                 # each branch read its own value

    def test_a_nested_scope_sees_the_layer_of_its_parent(self):
        from config import _get, generation_overrides
        with run_scope.branch():
            with generation_overrides({"temperature": 0.3}):
                with run_scope.branch():
                    assert _get("temperature") == 0.3


class TestFailures:
    def failing(self):
        def tool(node, state):
            if node.id == "bad":
                raise RuntimeError("boom")
            return "ok"
        return NodeRunner(tool_executor=tool)

    def room(self, **join):
        return make_room([[setter("a", x=1)], [{"id": "bad", "type": "tool", "tool": "read_file"}]], join=join)

    def test_a_raising_branch_stops_the_run_by_default(self):
        with pytest.raises(RuntimeError, match="boom"):
            run(self.room(), self.failing())

    def test_an_error_arrow_on_the_fork_catches_it(self):
        room = make_room([[setter("a", x=1)], [{"id": "bad", "type": "tool", "tool": "read_file"}]],
                         nodes_extra=[setter("rescue", rescued=True)],
                         edges_extra=[{"from": "fork", "to": "rescue", "on_error": True}, {"from": "rescue", "to": "EXIT:rescued"}])
        # the fork's own arrows must stay unconditional branches; an error arrow is not a branch
        out = run(room, self.failing())
        assert out.exit == "rescued" and out.state.data["rescued"] is True

    def test_continue_drops_the_failed_branch(self):
        out = run(self.room(on_branch_fail="continue"), self.failing())
        assert out.exit == "done" and out.state.data["x"] == 1
        assert out.state.data["merge_arrived"] == ["a"]
        assert out.state.data["merge_failed"][0]["branch"] == "bad"

    def test_every_branch_failing_is_still_a_failure(self):
        room = make_room([[{"id": "bad", "type": "tool", "tool": "read_file"}],
                          [{"id": "bad2", "type": "tool", "tool": "run_command"}]], join={"on_branch_fail": "continue"})
        with pytest.raises(RuntimeError):
            run(room, NodeRunner(tool_executor=lambda n, s: (_ for _ in ()).throw(RuntimeError("down"))))

    def test_an_exit_inside_a_branch_ends_the_room_with_that_exit(self):
        room = make_room([[setter("a", x=1)], [setter("b", y=2), {"id": "chk", "type": "condition", "condition": "state.y == 2"}]],
                         edges_extra=[{"from": "chk", "if": "state.y == 2", "to": "EXIT:abort"}])
        out = run(room)
        assert out.exit == "abort"
        assert out.state.data["x"] == 1                      # what the other branch did is kept for the report

    def test_any_takes_the_first_branch_that_arrives(self):
        ran = []

        def tool(node, state):
            ran.append(node.id)
            return "ok"
        room = make_room([[{"id": "a", "type": "tool", "tool": "read_file", "sets": {"who": "first_line"}}],
                          [{"id": "b", "type": "tool", "tool": "read_file", "sets": {"who": "first_line"}}]],
                         join={"wait_for": "any"}, workers=1)
        out = run(room, NodeRunner(tool_executor=tool))
        assert ran == ["a"]                                   # the second branch was never started
        assert out.state.data["merge_arrived"] == ["a"]


class TestBudgets:
    def test_the_step_budget_is_shared_by_all_branches(self):
        steps = lambda p: [setter(p + str(i), **{p + str(i): i}) for i in range(5)]
        room = make_room([steps("a"), steps("b")], budget={"max_iterations": 8, "max_tokens": 1000})
        out = run(room)
        assert out.exit == "budget"

    def test_enough_budget_for_everything(self):
        steps = lambda p: [setter(p + str(i), **{p + str(i): i}) for i in range(3)]
        out = run(make_room([steps("a"), steps("b")], budget={"max_iterations": 12, "max_tokens": 1000}))
        assert out.exit == "done"

    def test_events_name_the_branch(self):
        events = []
        Engine().run(make_room([[setter("a", x=1)], [setter("b", y=2)]]), State(), NodeRunner().run_node,
                     on_event=events.append)
        kinds = [e["type"] for e in events]
        assert "fork" in kinds and "join" in kinds
        starts = {e["node"]: e.get("branch") for e in events if e["type"] == "node_start"}
        assert starts["a"] == "a" and starts["b"] == "b" and starts["fork"] is None


class TestNested:
    def test_a_fork_inside_a_branch(self):
        room = Room(room="n", entry="f1", budget={"max_iterations": 60, "max_tokens": 1000}, nodes=[
            {"id": "f1", "type": "parallel", "join": "j1"},
            setter("x", x=1),
            {"id": "f2", "type": "parallel", "join": "j2"},
            setter("p", p=1), setter("q", q=2),
            {"id": "j2", "type": "join"},
            setter("y", y=1),
            {"id": "j1", "type": "join"}],
            edges=[{"from": "f1", "to": "x"}, {"from": "f1", "to": "f2"}, {"from": "x", "to": "j1"},
                   {"from": "f2", "to": "p"}, {"from": "f2", "to": "q"}, {"from": "p", "to": "j2"},
                   {"from": "q", "to": "j2"}, {"from": "j2", "to": "y"}, {"from": "y", "to": "j1"},
                   {"from": "j1", "to": "EXIT:done"}])
        assert validate_room(room, TOOLS) == []
        out = run(room)
        assert out.exit == "done"
        assert {k: out.state.data[k] for k in ("x", "p", "q", "y")} == {"x": 1, "p": 1, "q": 2, "y": 1}
        assert out.state.data["j2_arrived"] == ["p", "q"] and out.state.data["j1_arrived"] == ["x", "f2"]


class TestValidation:
    def test_a_sound_fork_is_valid(self):
        assert validate_room(make_room([[setter("a", x=1)], [setter("b", y=2)]]), TOOLS) == []

    def test_branches_must_not_share_steps(self):
        room = make_room([[setter("a", x=1)], [setter("b", y=2)]],
                         nodes_extra=[setter("shared", z=1)],
                         edges_extra=[{"from": "a", "to": "shared"}, {"from": "b", "to": "shared"},
                                      {"from": "shared", "to": "merge"}])
        text = " ".join(validate_room(room, TOOLS))
        assert "share steps" in text and "shared" in text

    def test_a_question_inside_a_branch_is_refused(self):
        room = make_room([[{"id": "ask", "type": "human_pause", "prompt": "ok?"}], [setter("b", y=2)]])
        assert any("asks a person inside a branch" in p for p in validate_room(room, TOOLS))

    def test_a_branch_must_reach_the_join(self):
        room = Room(room="r", entry="f", budget={"max_iterations": 9, "max_tokens": 99}, nodes=[
            {"id": "f", "type": "parallel", "join": "j"}, setter("a", x=1), setter("b", y=1), {"id": "j", "type": "join"}],
            edges=[{"from": "f", "to": "a"}, {"from": "f", "to": "b"}, {"from": "a", "to": "j"},
                   {"from": "b", "to": "EXIT:early"}, {"from": "j", "to": "EXIT:done"}])
        assert any("never reaches the join" in p for p in validate_room(room, TOOLS))

    def test_branch_arrows_must_be_unconditional(self):
        room = make_room([[setter("a", x=1)], [setter("b", y=2)]])
        room.edges[0].if_ = "state.go"
        assert any("unconditional" in p for p in validate_room(room, TOOLS))

    def test_the_join_must_exist_and_be_a_join(self):
        room = make_room([[setter("a", x=1)], [setter("b", y=2)]])
        room.nodes[0].join = "nowhere"
        assert any("not a join node" in p for p in validate_room(room, TOOLS))

    def test_two_forks_cannot_share_a_join(self):
        room = Room(room="r", entry="f1", budget={"max_iterations": 9, "max_tokens": 99}, nodes=[
            {"id": "f1", "type": "parallel", "join": "j"}, {"id": "f2", "type": "parallel", "join": "j"},
            setter("a", a=1), setter("b", b=1), setter("c", c=1), setter("d", d=1), {"id": "j", "type": "join"}],
            edges=[{"from": "f1", "to": "a"}, {"from": "f1", "to": "f2"}, {"from": "f2", "to": "c"},
                   {"from": "f2", "to": "d"}, {"from": "a", "to": "j"}, {"from": "c", "to": "j"},
                   {"from": "d", "to": "j"}, {"from": "j", "to": "EXIT:done"}])
        assert any("already gathers" in p for p in validate_room(room, TOOLS))

    def test_a_browser_in_two_branches_is_a_warning_not_an_error(self):
        room = make_room([[{"id": "a", "type": "tool", "tool": "browser_open"}],
                          [{"id": "b", "type": "tool", "tool": "browser_action"}]])
        assert validate_room(room, TOOLS) == []
        assert any("browser page" in w for w in room_warnings(room))

    def test_model_rejects_a_parallel_without_a_join(self):
        with pytest.raises(ValueError):
            Room(room="r", budget={"max_iterations": 1, "max_tokens": 1}, nodes=[{"id": "f", "type": "parallel"}])

    def test_model_rejects_unknown_merge_mode(self):
        with pytest.raises(ValueError):
            Room(room="r", budget={"max_iterations": 1, "max_tokens": 1}, nodes=[{"id": "j", "type": "join", "merge": "average"}])

    def test_depth_is_limited(self):
        nodes, edges = [], []
        for i in range(4):
            nodes += [{"id": f"f{i}", "type": "parallel", "join": f"j{i}"}, {"id": f"j{i}", "type": "join"},
                      setter(f"s{i}", **{f"s{i}": 1})]
        # f0 -> (f1 -> (f2 -> (f3 -> s3, s3b) ...)) is deeper than allowed
        nodes += [setter("leaf", leaf=1)]
        edges += [{"from": "f0", "to": "f1"}, {"from": "f0", "to": "s0"}, {"from": "s0", "to": "j0"}, {"from": "j0", "to": "EXIT:done"},
                  {"from": "f1", "to": "f2"}, {"from": "f1", "to": "s1"}, {"from": "s1", "to": "j1"}, {"from": "j1", "to": "j0"},
                  {"from": "f2", "to": "f3"}, {"from": "f2", "to": "s2"}, {"from": "s2", "to": "j2"}, {"from": "j2", "to": "j1"},
                  {"from": "f3", "to": "s3"}, {"from": "f3", "to": "leaf"}, {"from": "s3", "to": "j3"},
                  {"from": "leaf", "to": "j3"}, {"from": "j3", "to": "j2"}]
        room = Room(room="d", entry="f0", budget={"max_iterations": 99, "max_tokens": 99}, nodes=nodes, edges=edges)
        assert any("nested more than" in p for p in validate_room(room, TOOLS))


class TestResume:
    def room(self, tool="read_file", idempotent=True):
        return make_room([[{"id": "a", "type": "tool", "tool": tool, "idempotent": idempotent}], [setter("b", y=2)]])

    def interrupted(self, tmp_path, room):
        journal = Journal(tmp_path / "j.jsonl")
        journal.append({"kind": "intent", "room": room.room, "node": "fork", "state": State().model_dump()})
        return journal

    def test_a_fork_of_safe_steps_is_run_again(self, tmp_path):
        room = self.room()
        start, _, status = plan_resume(room, self.interrupted(tmp_path, room))
        assert (start, status) == ("fork", "resume")

    def test_a_fork_with_an_action_asks_a_person(self, tmp_path):
        room = self.room(tool="write_file", idempotent=False)
        _, _, status = plan_resume(room, self.interrupted(tmp_path, room))
        assert status == "indeterminate"

    def test_the_whole_fork_is_one_step_in_the_journal(self, tmp_path):
        room = self.room()
        journal = Journal(tmp_path / "j2.jsonl")
        Engine().run(room, State(), NodeRunner(tool_executor=lambda n, s: "ok").run_node, journal=journal)
        nodes = [r["node"] for r in journal.records() if r["kind"] == "intent"]
        assert nodes == ["fork", "merge"]                       # branch steps are not journaled one by one
        commits = [r for r in journal.records() if r["kind"] == "commit"]
        assert commits[0]["node"] == "fork" and commits[0]["next"] == "merge"
        assert commits[0]["state"]["data"]["y"] == 2            # the commit carries the merged state
        assert par.block_nodes(room, room.nodes[0]) == {"a", "b"}
