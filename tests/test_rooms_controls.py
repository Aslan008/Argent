"""Rooms: room-wide rules, memory and hints, power actions, model parameters, loops with
break / continue, arithmetic and assertions."""

import json

import pytest

import config
from src.rooms import condition
from src.rooms.diagram import render_flow
from src.rooms.engine import Engine, replayable
from src.rooms.integrations import Integrations
from src.rooms.library import describe_room
from src.rooms.models import Node, Room, State
from src.rooms.rules import Rules, merge, static_problems, static_warnings
from src.rooms.runner import AgentExecutor, NodeRunner
from src.rooms.validator import room_warnings, validate_room

ALL_TOOLS = {"run_command", "read_file", "write_file", "list_directory", "browser_open", "noop"}


def room(nodes, edges, **kw):
    return Room(room="t", budget={"max_iterations": 200, "max_tokens": 100000}, nodes=nodes, edges=edges, **kw)


def run(r, runner, data=None, state=None):
    st = state or State(data=dict(data or {}))
    return Engine().run(r, st, runner.run_node)


# ── room rules ───────────────────────────────────────────────────────────────────

class TestRules:
    def test_merge_adds_up(self):
        base = merge({}, room([{"id": "a", "type": "wait", "seconds": 1}], [{"from": "a", "to": "EXIT:d"}],
                              rules={"read_only": True, "deny_tools": ["x"], "sampling": {"top_p": 0.9}}).rules)
        child = merge(base, Room(room="c", budget={"max_iterations": 1, "max_tokens": 1},
                                 nodes=[{"id": "a", "type": "wait", "seconds": 1}],
                                 rules={"no_tools": True, "deny_tools": ["y"], "sampling": {"temperature": 0.1}}).rules)
        assert child["read_only"] and child["no_tools"]
        assert child["deny_tools"] == ["x", "y"]
        assert child["sampling"] == {"top_p": 0.9, "temperature": 0.1}

    def test_tool_block_reasons(self):
        r = Rules({"read_only": True, "deny_tools": ["browser_open"]})
        assert r.tool_block("read_file") is None
        assert "read-only" in r.tool_block("write_file")
        assert "run_command" and r.tool_block("run_command")           # a shell can write
        assert "browser_open" in r.tool_block("browser_open")
        assert Rules({"no_tools": True}).tool_block("read_file")

    def test_node_block_by_type(self):
        ro = Rules({"read_only": True})
        assert ro.node_block(Node(id="a", type="mcp", server="s", tool="t"))
        assert ro.node_block(Node(id="a", type="mcp", server="s", mode="list")) is None
        assert ro.node_block(Node(id="a", type="plugin", command="c"))
        assert ro.node_block(Node(id="a", type="leeth", mode="click", input_from="x"))
        assert ro.node_block(Node(id="a", type="leeth", mode="scan", instruction="q")) is None
        assert ro.node_block(Node(id="a", type="notify", message="m")) is None
        assert ro.node_block(Node(id="a", type="power", mode="shutdown")) is None      # power is not a tool
        assert Rules({"no_tools": True}).node_block(Node(id="a", type="leeth", mode="scan", instruction="q"))

    def test_agent_tools(self):
        assert Rules({}).agent_tools(None) is None
        assert Rules({"no_tools": True}).agent_tools(["read_file"]) == []
        assert Rules({"read_only": True}).agent_tools(["read_file", "write_file", "run_command"]) == ["read_file"]
        assert "write_file" not in Rules({"read_only": True}).agent_tools(None)

    def test_saving_refuses_a_step_the_rules_forbid(self):
        r = room([{"id": "w", "type": "tool", "tool": "write_file"}, {"id": "r", "type": "tool", "tool": "read_file"}],
                 [{"from": "w", "to": "r"}, {"from": "r", "to": "EXIT:d"}], rules={"read_only": True})
        problems = validate_room(r, ALL_TOOLS)
        assert any("'w'" in p and "read-only" in p for p in problems) and not any("'r'" in p for p in problems)

    def test_a_disabled_step_is_not_an_error(self):
        r = room([{"id": "w", "type": "tool", "tool": "write_file", "disabled": True}],
                 [{"from": "w", "to": "EXIT:d"}], rules={"read_only": True})
        assert static_problems(r) == []

    def test_agent_tool_list_gets_a_warning(self):
        r = room([{"id": "a", "type": "agent", "budget": {"max_iterations": 1, "max_tokens": 10},
                   "tools": ["read_file", "write_file"]}], [{"from": "a", "to": "EXIT:d"}], rules={"read_only": True})
        assert "write_file" in " ".join(static_warnings(r))
        assert "write_file" in " ".join(room_warnings(r))

    def test_runtime_block_leaves_an_explanation(self):
        r = room([{"id": "w", "type": "tool", "tool": "write_file", "sets": {"ok": "ok"}},
                  {"id": "after", "type": "set", "values": {"reached": True}}],
                 [{"from": "w", "to": "after"}, {"from": "after", "to": "EXIT:d"}], rules={"read_only": True})
        called = []
        out = run(r, NodeRunner(tool_executor=lambda n, s: called.append(n.tool) or "x"))
        assert called == [] and out.exit == "d"
        assert out.state.data["w.blocked"] is True and "read-only" in out.state.data["w.output"]
        assert out.state.data["ok"] is False and out.state.data["reached"] is True

    def test_rules_reach_the_agent(self):
        seen = {}

        def agent(node, ctx):
            seen["tools"], seen["mcp"], seen["sampling"] = node.tools, node.mcp, node.sampling
            return {}
        r = room([{"id": "a", "type": "agent", "budget": {"max_iterations": 1, "max_tokens": 10},
                   "tools": ["read_file", "write_file"], "sampling": {"temperature": 0.5}}],
                 [{"from": "a", "to": "EXIT:d"}],
                 rules={"read_only": True, "sampling": {"temperature": 0.9, "top_p": 0.8}})
        run(r, NodeRunner(agent_executor=agent))
        assert seen["tools"] == ["read_file"]
        assert seen["sampling"] == {"temperature": 0.5, "top_p": 0.8}      # the step's own value wins

    def test_no_tools_gives_an_agent_nothing(self):
        seen = {}
        r = room([{"id": "a", "type": "agent", "budget": {"max_iterations": 1, "max_tokens": 10}, "mcp": ["x"]}],
                 [{"from": "a", "to": "EXIT:d"}], rules={"no_tools": True})
        run(r, NodeRunner(agent_executor=lambda n, c: seen.update(tools=n.tools, mcp=n.mcp) or {}))
        assert seen == {"tools": [], "mcp": []}

    def test_ignore_rules_join_the_nodes(self):
        got = {}

        class Fake:
            def skill(self, node, state):
                got["ignore"] = list(node.ignore)
                return "body"
        r = room([{"id": "s", "type": "skill", "skill": "docs", "ignore": ["skill:docs#A"]}],
                 [{"from": "s", "to": "EXIT:d"}], rules={"ignore": ["skill:docs#B"]})
        run(r, NodeRunner(integrations=Fake()))
        assert got["ignore"] == ["skill:docs#A", "skill:docs#B"]

    def test_a_called_room_works_under_the_callers_rules(self):
        from src.rooms.subroom import SubRoomHandler
        child = Room(room="child", budget={"max_iterations": 5, "max_tokens": 100},
                     nodes=[{"id": "w", "type": "tool", "tool": "write_file"}],
                     edges=[{"from": "w", "to": "EXIT:done"}])

        class Lib:
            def get(self, name):
                return child if name == "child" else None

            def record_run(self, *a, **k):
                pass
        called = []
        runner = NodeRunner(tool_executor=lambda n, s: called.append(n.tool) or "x", room_handler=SubRoomHandler(Lib()))
        parent = room([{"id": "sub", "type": "room", "room": "child"}], [{"from": "sub", "to": "EXIT:d"}],
                      rules={"read_only": True})
        out = run(parent, runner)
        assert called == [] and out.exit == "d"

    def test_diagram_and_description_mention_rules(self):
        r = room([{"id": "a", "type": "wait", "seconds": 1}], [{"from": "a", "to": "EXIT:d"}],
                 rules={"read_only": True, "deny_tools": ["browser_open"]})
        assert "только чтение" in render_flow(r) and "browser_open" in render_flow(r)
        assert "rules:" in describe_room(r)

    def test_bad_rules_are_rejected(self):
        with pytest.raises(ValueError):
            room([{"id": "a", "type": "wait", "seconds": 1}], [], rules={"memory": "forever"})
        with pytest.raises(ValueError):
            room([{"id": "a", "type": "wait", "seconds": 1}], [], rules={"ignore": ["nonsense"]})
        with pytest.raises(ValueError):
            room([{"id": "a", "type": "wait", "seconds": 1}], [], rules={"sampling": {"temperature": 9}})


# ── memory and hints ─────────────────────────────────────────────────────────────

def _agent(i, **kw):
    return {"id": i, "type": "agent", "budget": {"max_iterations": 1, "max_tokens": 100}, **kw}


class RecordingAgent:
    """An executor that records the brief each call got and 'says' something."""
    accepts_brief = True

    def __init__(self):
        self.calls = []
        self.last_tokens = 0
        self.last_text = ""

    def __call__(self, node, context, brief=None):
        self.calls.append((node.id, brief))
        self.last_text = f"{node.id} did its part"
        return {}


class TestMemoryAndHints:
    def test_shared_memory_is_carried_forward(self):
        agent = RecordingAgent()
        r = room([_agent("a"), _agent("b")], [{"from": "a", "to": "b"}, {"from": "b", "to": "EXIT:d"}],
                 rules={"memory": "shared"})
        out = run(r, NodeRunner(agent_executor=agent))
        assert agent.calls[0][1]["memory"] == ""                           # nothing happened yet
        assert "[a] a did its part" in agent.calls[1][1]["memory"]
        assert len(out.state.memory) == 2

    def test_isolated_agents_neither_read_nor_write(self):
        agent = RecordingAgent()
        r = room([_agent("a"), _agent("b")], [{"from": "a", "to": "b"}, {"from": "b", "to": "EXIT:d"}])
        out = run(r, NodeRunner(agent_executor=agent))
        assert agent.calls[1][1]["memory"] == "" and out.state.memory == []

    def test_memory_is_bounded(self):
        agent = RecordingAgent()
        agent.last_text = ""
        long = "x" * 5000

        class Chatty(RecordingAgent):
            def __call__(self, node, context, brief=None):
                super().__call__(node, context, brief)
                self.last_text = long
                return {}
        nodes = [_agent(f"a{i}") for i in range(4)]
        edges = [{"from": f"a{i}", "to": f"a{i + 1}"} for i in range(3)] + [{"from": "a3", "to": "EXIT:d"}]
        out = run(room(nodes, edges, rules={"memory": "shared", "memory_chars": 1500}), NodeRunner(agent_executor=Chatty()))
        assert sum(len(m) for m in out.state.memory) <= 1500 + 50

    def test_forget_wipes_memory_hints_and_fields(self):
        r = room([_agent("a"), {"id": "f", "type": "forget", "clear": ["scratch"]}, _agent("b")],
                 [{"from": "a", "to": "f"}, {"from": "f", "to": "b"}, {"from": "b", "to": "EXIT:d"}],
                 rules={"memory": "shared", "hint": None})
        agent = RecordingAgent()
        out = run(r, NodeRunner(agent_executor=agent), {"scratch": "junk"})
        assert agent.calls[1][1]["memory"].startswith("[b]") is False
        assert "scratch" not in out.state.data and "forgot" in out.state.data["f.output"]

    def test_hint_for_the_next_agent_only(self):
        agent = RecordingAgent()
        r = room([{"id": "h", "type": "hint", "message": "Do not repeat {{topic}}"}, _agent("a"), _agent("b")],
                 [{"from": "h", "to": "a"}, {"from": "a", "to": "b"}, {"from": "b", "to": "EXIT:d"}])
        run(r, NodeRunner(agent_executor=agent), {"topic": "cats"})
        assert agent.calls[0][1]["hints"] == ["Do not repeat cats"]
        assert agent.calls[1][1]["hints"] == []

    def test_sticky_hint_until_cleared(self):
        agent = RecordingAgent()
        r = room([{"id": "h", "type": "hint", "mode": "all", "message": "stay on goal"}, _agent("a"),
                  {"id": "c", "type": "hint", "mode": "clear", "message": "x"}, _agent("b")],
                 [{"from": "h", "to": "a"}, {"from": "a", "to": "c"}, {"from": "c", "to": "b"}, {"from": "b", "to": "EXIT:d"}])
        run(r, NodeRunner(agent_executor=agent))
        assert agent.calls[0][1]["hints"] == ["stay on goal"]
        assert agent.calls[1][1]["hints"] == []

    def test_hint_with_targets(self):
        agent = RecordingAgent()
        r = room([{"id": "h", "type": "hint", "message": "only b", "targets": ["b"]}, _agent("a"), _agent("b")],
                 [{"from": "h", "to": "a"}, {"from": "a", "to": "b"}, {"from": "b", "to": "EXIT:d"}])
        run(r, NodeRunner(agent_executor=agent))
        assert agent.calls[0][1]["hints"] == [] and agent.calls[1][1]["hints"] == ["only b"]

    def test_room_and_agent_hints(self):
        agent = RecordingAgent()
        r = room([_agent("a", hint="mine {{x}}")], [{"from": "a", "to": "EXIT:d"}], rules={"hint": "for all"})
        run(r, NodeRunner(agent_executor=agent), {"x": "1"})
        assert agent.calls[0][1]["hints"] == ["for all", "mine 1"]

    def test_hint_reaches_the_prompt(self):
        prompts = []
        ex = AgentExecutor(lambda prompt, **kw: prompts.append(prompt) or "{}")
        node = Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10}, instruction="Do it")
        ex(node, {}, brief={"hints": ["be careful"], "memory": "[x] did y", "shared": True})
        assert "HINTS" in prompts[0] and "be careful" in prompts[0] and "[x] did y" in prompts[0]

    def test_composed_hint(self):
        r = room([{"id": "h", "type": "hint", "message": "Remind the goal", "compose": True}, _agent("a")],
                 [{"from": "h", "to": "a"}, {"from": "a", "to": "EXIT:d"}])
        agent = RecordingAgent()
        run(r, NodeRunner(agent_executor=agent, compose=lambda ins, data: f"WRITTEN({ins})"))
        assert agent.calls[0][1]["hints"] == ["WRITTEN(Remind the goal)"]

    def test_visits_counters_catch_a_loop(self):
        r = room([{"id": "work", "type": "tool", "tool": "noop"},
                  {"id": "nudge", "type": "hint", "message": "you are going in circles"}],
                 [{"from": "work", "if": "state.work_visits >= 3", "to": "nudge"},
                  {"from": "work", "else": True, "to": "work"},
                  {"from": "nudge", "to": "EXIT:d"}])
        out = run(r, NodeRunner(tool_executor=lambda n, s: "ok"))
        assert out.state.data["work_visits"] == 3 and out.state.data["_steps"] == 4
        assert out.state.hints and out.state.hints[0]["text"] == "you are going in circles"


# ── notify ───────────────────────────────────────────────────────────────────────

class TestNotify:
    def test_compose_and_delivery_report(self):
        sent = []
        r = room([{"id": "n", "type": "notify", "channel": "telegram", "compose": True,
                   "message": "Tell me how the work went"}], [{"from": "n", "to": "EXIT:d"}])
        runner = NodeRunner(compose=lambda ins, data: f"Done: {data['count']} items",
                            notify=lambda node, st: sent.append((node.channel, node.message)) or True)
        out = run(r, runner, {"count": 4})
        assert sent == [("telegram", "Done: 4 items")]
        assert out.state.data["n.output"] == "Done: 4 items" and out.state.data["n.delivered"] is True

    def test_failed_delivery_is_recorded_not_raised(self):
        r = room([{"id": "n", "type": "notify", "message": "hi", "channel": "desktop"}], [{"from": "n", "to": "EXIT:d"}])
        out = run(r, NodeRunner(notify=lambda n, s: False))
        assert out.state.data["n.delivered"] is False

    def test_compose_only_for_notify_and_hint(self):
        with pytest.raises(ValueError):
            Node(id="a", type="tool", tool="x", compose=True)


# ── power ────────────────────────────────────────────────────────────────────────

class TestPower:
    def test_shutdown_is_never_immediate(self):
        with pytest.raises(ValueError):
            Node(id="p", type="power", mode="shutdown", seconds=2)
        assert Node(id="p", type="power", mode="sleep", seconds=0).seconds == 0

    def test_commands_per_system(self):
        win, _ = Integrations.power_command("shutdown", 30, "bye", platform="win32")
        assert win[:4] == ["shutdown", "/s", "/t", "30"]
        assert Integrations.power_command("shutdown", 1, "x", platform="win32")[0][3] == "10"     # the floor
        assert Integrations.power_command("restart", 90, "x", platform="linux")[0][:3] == ["shutdown", "-r", "+2"]
        assert Integrations.power_command("sleep", 0, "", platform="darwin")[0] == ["pmset", "sleepnow"]
        argv, why = Integrations.power_command("hibernate", 0, "", platform="darwin")
        assert argv is None and "not available" in why

    def test_dry_run_runs_nothing(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
        node = Node(id="p", type="power", mode="shutdown", seconds=60, args={"dry_run": True})
        out = Integrations().power(node, State())
        assert out.startswith("Dry run") and "shutdown" in out

    def test_real_call_reports_how_to_cancel(self, monkeypatch):
        import subprocess
        got = {}

        class Done:
            returncode, stdout, stderr = 0, "", ""
        monkeypatch.setattr(subprocess, "run", lambda argv, **k: got.update(argv=argv) or Done())
        monkeypatch.setattr(Integrations, "power_command",
                            staticmethod(lambda mode, seconds=60, message="", platform=None: (["shutdown", "/s"], "shutdown /a")))
        out = Integrations().power(Node(id="p", type="power", mode="shutdown", seconds=60), State())
        assert got["argv"] == ["shutdown", "/s"] and out.startswith("OK") and "shutdown /a" in out

    def test_failure_is_an_error_text(self, monkeypatch):
        import subprocess

        class Bad:
            returncode, stdout, stderr = 1, "", "access denied"
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Bad())
        out = Integrations().power(Node(id="p", type="power", mode="lock"), State())
        assert out.startswith("Error") and "access denied" in out

    def test_confirm_gates_it_and_it_is_not_replayed(self):
        class Fake:
            def __init__(self):
                self.ran = 0

            def power(self, node, state):
                self.ran += 1
                return "OK"
        fake = Fake()
        node = Node(id="p", type="power", mode="shutdown", seconds=60, confirm=True)
        NodeRunner(integrations=fake).run_node(node, State())          # nobody to ask
        assert fake.ran == 0
        NodeRunner(integrations=fake, confirm=lambda n, s: True).run_node(node, State())
        assert fake.ran == 1
        assert replayable(node) is False

    def test_the_builder_makes_it_ask(self):
        from src.rooms.builder import harden
        r = room([{"id": "p", "type": "power", "mode": "shutdown", "seconds": 60}], [{"from": "p", "to": "EXIT:d"}])
        notes = harden(r, lambda t: False)
        assert r.nodes[0].confirm is True and notes


# ── model parameters ─────────────────────────────────────────────────────────────

class TestSampling:
    def test_executor_passes_it(self):
        seen = {}
        ex = AgentExecutor(lambda prompt, **kw: seen.update(kw) or "{}")
        ex(Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10},
                sampling={"temperature": 0.1, "top_k": 20}), {})
        assert seen["sampling"] == {"temperature": 0.1, "top_k": 20}

    def test_bounds(self):
        for bad in ({"temperature": 3}, {"top_p": -1}, {"top_k": 0}, {"reasoning_effort": "max"}, {"nope": 1}):
            with pytest.raises(ValueError):
                Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10}, sampling=bad)
        Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 10},
             sampling={"temperature": 0, "top_p": 1, "top_k": 1, "min_p": 0.05, "reasoning_effort": "high"})

    def test_overrides_reach_the_config_getters_and_unwind(self):
        before = (config.get_temperature(), config.get_top_p(), config.get_min_p())
        with config.generation_overrides({"temperature": 0.01, "top_p": 0.42, "min_p": 0.07, "reasoning_effort": "low"}):
            assert (config.get_temperature(), config.get_top_p(), config.get_min_p()) == (0.01, 0.42, 0.07)
            assert config.get_reasoning_effort() == "low"
            with config.generation_overrides({"top_p": 0.9}):
                assert config.get_top_p() == 0.9 and config.get_temperature() == 0.01
            assert config.get_top_p() == 0.42
        assert (config.get_temperature(), config.get_top_p(), config.get_min_p()) == before

    def test_overrides_unwind_after_an_error(self):
        before = config.get_temperature()
        with pytest.raises(RuntimeError):
            with config.generation_overrides({"temperature": 1.5}):
                raise RuntimeError("boom")
        assert config.get_temperature() == before


# ── loops ────────────────────────────────────────────────────────────────────────

def loop_room(header: dict, body_extra=None, before=None):
    """header -> body(work) -> header ...; header's else leaves."""
    nodes = [{"id": "loop", "type": "loop", **header}, {"id": "work", "type": "tool", "tool": "noop"},
             {"id": "end", "type": "set", "values": {"finished": True}}]
    edges = [{"from": "loop", "if": "not state.loop_done", "to": "work"}, {"from": "loop", "else": True, "to": "end"},
             {"from": "work", "to": "loop"}, {"from": "end", "to": "EXIT:done"}]
    return room(nodes, edges)


class TestLoops:
    def _run(self, header, data=None):
        seen = []
        runner = NodeRunner(tool_executor=lambda n, s: seen.append(
            {k: v for k, v in s.data.items() if k in ("i", "key", "val", "n")}) or "ok")
        out = run(loop_room(header), runner, data)
        return out, seen

    def test_for_range(self):
        out, seen = self._run({"mode": "for", "start": 2, "stop": 11, "step": 3})
        assert [s["i"] for s in seen] == [2, 5, 8] and out.state.data["loop_count"] == 3 and out.state.data["finished"]

    def test_for_counting_down(self):
        _, seen = self._run({"mode": "for", "start": 5, "stop": 0, "step": -2})
        assert [s["i"] for s in seen] == [5, 3, 1]

    def test_for_with_templates_and_expressions(self):
        _, seen = self._run({"mode": "for", "start": "{{lo}}", "stop": "= state.hi + 1"}, {"lo": 1, "hi": 3})
        assert [s["i"] for s in seen] == [1, 2, 3]

    def test_times(self):
        _, seen = self._run({"mode": "times", "stop": 3, "item": "n"})
        assert [s["n"] for s in seen] == [0, 1, 2]

    def test_while_may_not_run_at_all(self):
        out, seen = self._run({"mode": "while", "condition": "state.go"}, {"go": False})
        assert seen == [] and out.state.data["loop_done"] is True

    def test_while_counts_with_the_body(self):
        room_ = room(
            [{"id": "loop", "type": "loop", "mode": "while", "condition": "state.n < 3"},
             {"id": "inc", "type": "set", "values": {"n": "= state.n + 1"}}],
            [{"from": "loop", "if": "not state.loop_done", "to": "inc"}, {"from": "loop", "else": True, "to": "EXIT:done"},
             {"from": "inc", "to": "loop"}])
        out = run(room_, NodeRunner(), {"n": 0})
        assert out.exit == "done" and out.state.data["n"] == 3

    def test_do_while_runs_once_first(self):
        _, seen = self._run({"mode": "do_while", "condition": "state.go"}, {"go": False})
        assert len(seen) == 1

    def test_repeat_until(self):
        room_ = room(
            [{"id": "loop", "type": "loop", "mode": "until", "condition": "state.n >= 2"},
             {"id": "inc", "type": "set", "values": {"n": "= state.n + 1"}}],
            [{"from": "loop", "if": "not state.loop_done", "to": "inc"}, {"from": "loop", "else": True, "to": "EXIT:done"},
             {"from": "inc", "to": "loop"}])
        out = run(room_, NodeRunner(), {"n": 0})
        assert out.state.data["n"] == 2

    def test_for_in_a_dict(self):
        _, seen = self._run({"mode": "for_in", "input_from": "table", "item": "key", "item_value": "val"},
                            {"table": {"a": 1, "b": 2}})
        assert [(s["key"], s["val"]) for s in seen] == [("a", 1), ("b", 2)]

    def test_for_in_a_list_and_json_text(self):
        _, seen = self._run({"mode": "for_in", "input_from": "xs", "item": "key", "item_value": "val"}, {"xs": '["p", "q"]'})
        assert [(s["key"], s["val"]) for s in seen] == [(0, "p"), (1, "q")]

    def test_the_cap_ends_a_runaway(self):
        out, seen = self._run({"mode": "while", "condition": "state.always", "max_items": 5}, {"always": True})
        assert len(seen) == 5 and out.state.data["loop_capped"] is True and out.exit == "done"

    def test_default_cap_is_bounded(self):
        r = loop_room({"mode": "times", "stop": 100000})
        r.budget.max_iterations = 1000            # the loop's own cap must be what stops it, not the room's
        seen = []
        out = run(r, NodeRunner(tool_executor=lambda n, s: seen.append(1) or "ok"))
        assert len(seen) == 100 and out.state.data["loop_capped"] is True and out.exit == "done"

    def test_loop_can_be_entered_again(self):
        nodes = [{"id": "loop", "type": "loop", "mode": "times", "stop": 2}, {"id": "work", "type": "tool", "tool": "noop"},
                 {"id": "again", "type": "set", "values": {"round": "= state.round + 1"}}]
        edges = [{"from": "loop", "if": "not state.loop_done", "to": "work"}, {"from": "loop", "else": True, "to": "again"},
                 {"from": "work", "to": "loop"},
                 {"from": "again", "if": "state.round < 2", "to": "loop"}, {"from": "again", "else": True, "to": "EXIT:done"}]
        seen = []
        out = run(room(nodes, edges), NodeRunner(tool_executor=lambda n, s: seen.append(1) or "ok"), {"round": 0})
        assert len(seen) == 4 and out.state.data["round"] == 2

    def _break_room(self, kind):
        nodes = [{"id": "loop", "type": "loop", "mode": "times", "stop": 10},
                 {"id": "work", "type": "tool", "tool": "noop"},
                 {"id": kind, "type": kind, "loop": "loop"},
                 {"id": "end", "type": "set", "values": {"finished": True}}]
        edges = [{"from": "loop", "if": "not state.loop_done", "to": "work"}, {"from": "loop", "else": True, "to": "end"},
                 {"from": "work", "if": "state.i >= 2", "to": kind}, {"from": "work", "else": True, "to": "loop"},
                 {"from": kind, "to": "loop"}, {"from": "end", "to": "EXIT:done"}]
        return room(nodes, edges)

    def test_break_leaves_the_loop(self):
        seen = []
        out = run(self._break_room("break"), NodeRunner(tool_executor=lambda n, s: seen.append(s.data["i"]) or "ok"))
        assert seen == [0, 1, 2] and out.state.data["loop_broken"] is True and out.state.data["finished"]

    def test_continue_goes_to_the_next_round(self):
        seen = []
        out = run(self._break_room("continue"), NodeRunner(tool_executor=lambda n, s: seen.append(s.data["i"]) or "ok"))
        assert len(seen) == 10 and out.state.data["loop_broken"] is False

    def test_break_in_foreach(self):
        nodes = [{"id": "each", "type": "foreach", "input_from": "xs", "item": "x"}, {"id": "work", "type": "tool", "tool": "noop"},
                 {"id": "stop", "type": "break", "loop": "each"}]
        edges = [{"from": "each", "if": "not state.each_done", "to": "work"}, {"from": "each", "else": True, "to": "EXIT:done"},
                 {"from": "work", "if": "state.x == 2", "to": "stop"}, {"from": "work", "else": True, "to": "each"},
                 {"from": "stop", "to": "each"}]
        seen = []
        out = run(room(nodes, edges), NodeRunner(tool_executor=lambda n, s: seen.append(s.data["x"]) or "ok"), {"xs": [1, 2, 3]})
        assert seen == [1, 2] and out.exit == "done"

    def test_validator_wants_the_way_back_and_a_real_loop(self):
        nodes = [{"id": "loop", "type": "loop", "mode": "times", "stop": 2}, {"id": "b", "type": "break", "loop": "loop"},
                 {"id": "c", "type": "break", "loop": "ghost"}]
        edges = [{"from": "loop", "if": "not state.loop_done", "to": "b"}, {"from": "loop", "else": True, "to": "EXIT:d"},
                 {"from": "b", "to": "EXIT:d"}, {"from": "c", "to": "EXIT:d"}]
        problems = validate_room(room(nodes, edges), ALL_TOOLS)
        assert any("'b'" in p and "lead back" in p for p in problems)
        assert any("'c'" in p and "ghost" in p for p in problems)

    def test_validator_checks_loop_conditions(self):
        nodes = [{"id": "loop", "type": "loop", "mode": "while", "condition": "state.x ** 2"}]
        edges = [{"from": "loop", "to": "EXIT:d"}]
        assert any("bad condition" in p for p in validate_room(room(nodes, edges), ALL_TOOLS))

    def test_loop_model_rules(self):
        for bad in (dict(mode="for"), dict(mode="while"), dict(mode="for_in"), dict(mode="for", stop=3, step=0),
                    dict(mode="wat", stop=1), dict(mode="times", stop=2, max_items=5000)):
            with pytest.raises(ValueError):
                Node(id="l", type="loop", **bad)


# ── arithmetic, set, assert ──────────────────────────────────────────────────────

class TestExpressions:
    def test_arithmetic(self):
        d = {"i": 7, "name": "a"}
        assert condition.evaluate_value("state.i * 2 + 1", d) == 15
        assert condition.evaluate_value("state.i // 2", d) == 3 and condition.evaluate_value("state.i % 4", d) == 3
        assert condition.evaluate("state.i + 1 == 8", d)
        assert condition.evaluate_value("-state.i", d) == -7

    def test_no_power_no_calls_no_runaway(self):
        for bad in ("state.i ** 9", "abs(1)", "state.i << 3", "1 if state.i else 2"):
            with pytest.raises(condition.ConditionError):
                condition.validate(bad)
        with pytest.raises(condition.ConditionError):
            condition.validate("1" + " + 1" * 400)

    def test_division_by_zero_is_a_condition_error(self):
        with pytest.raises(condition.ConditionError):
            condition.evaluate_value("state.i / 0", {"i": 1})

    def test_missing_field_compares_false(self):
        assert condition.evaluate("state.nothing > 3", {}) is False

    def test_set_computes(self):
        r = room([{"id": "s", "type": "set", "values": {"n": "= state.n * 2", "label": "n is {{n}}"}}],
                 [{"from": "s", "to": "EXIT:d"}])
        out = run(r, NodeRunner(), {"n": 4})
        assert out.state.data["n"] == 8 and out.state.data["label"] == "n is 4"

    def test_bad_set_expression_is_found_when_saving(self):
        r = room([{"id": "s", "type": "set", "values": {"n": "= state.n ** 2"}}], [{"from": "s", "to": "EXIT:d"}])
        assert any("bad expression" in p for p in validate_room(r, ALL_TOOLS))

    def test_assert_passes_and_fails(self):
        r = room([{"id": "chk", "type": "assert", "condition": "state.n > 0", "message": "n must be positive"}],
                 [{"from": "chk", "to": "EXIT:ok"}])
        assert run(r, NodeRunner(), {"n": 1}).exit == "ok"
        with pytest.raises(AssertionError, match="n must be positive"):
            run(r, NodeRunner(), {"n": 0})

    def test_assert_takes_the_error_edge(self):
        r = room([{"id": "chk", "type": "assert", "condition": "state.n > 0", "message": "n must be positive"}],
                 [{"from": "chk", "to": "EXIT:ok"}, {"from": "chk", "on_error": True, "to": "EXIT:bad"}])
        out = run(r, NodeRunner(), {"n": 0})
        assert out.exit == "bad" and "positive" in out.state.data["error"]


# ── every new type is known to the views ─────────────────────────────────────────

def test_views_know_the_new_types():
    r = room(
        [{"id": "lp", "type": "loop", "mode": "for", "stop": 3}, {"id": "br", "type": "break", "loop": "lp"},
         {"id": "ct", "type": "continue", "loop": "lp"}, {"id": "pw", "type": "power", "mode": "sleep"},
         {"id": "hn", "type": "hint", "message": "careful", "mode": "all"}, {"id": "fg", "type": "forget"},
         {"id": "as", "type": "assert", "condition": "state.x"}],
        [{"from": "lp", "to": "br"}, {"from": "br", "to": "ct"}, {"from": "ct", "to": "pw"}, {"from": "pw", "to": "hn"},
         {"from": "hn", "to": "fg"}, {"from": "fg", "to": "as"}, {"from": "as", "to": "EXIT:d"}])
    flow, desc = render_flow(r), describe_room(r)
    for word in ("цикл for", "выйти из цикла", "к следующему шагу", "спящий режим", "подсказка всем", "забыть", "проверить"):
        assert word in flow, word
    for word in ("mode=for", "loop=lp", "mode=sleep", "mode=all", "cond=state.x"):
        assert word in desc, word
