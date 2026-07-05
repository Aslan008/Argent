"""NodeRunner: writes-whitelist enforcement, output plumbing, executor dispatch,
and an integration run where the agent lies about greenness but the whitelist +
verifier neutralise it.
"""

import tools.schemas as schemas

from src.rooms.engine import Engine
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import Node, State
from src.rooms.runner import NodeRunner, argent_tool_executor


class TestWhitelistAndPlumbing:
    def test_tool_output_and_declared_writes_applied(self):
        node = Node(id="run_tests", type="tool", tool="run_command",
                    writes=["tests_passed", "error_summary"])
        runner = NodeRunner(tool_executor=lambda n, s: {
            "tests_passed": True, "error_summary": "none", "output": "ran"})
        state = State()
        runner.run_node(node, state)
        assert state.data["run_tests.output"] == "ran"
        assert state.data["tests_passed"] is True
        assert state.data["error_summary"] == "none"

    def test_agent_cannot_write_outside_whitelist(self):
        # apply_fix may write error_summary but NOT tests_passed — the verifier
        # owns greenness. A proposed tests_passed must be dropped.
        node = Node(id="apply_fix", type="agent",
                    budget={"max_iterations": 3, "max_tokens": 100},
                    writes=["error_summary"])
        runner = NodeRunner(agent_executor=lambda n, ctx: {
            "error_summary": "applied", "tests_passed": True})
        state = State()
        runner.run_node(node, state)
        assert state.data["error_summary"] == "applied"
        assert "tests_passed" not in state.data   # dropped

    def test_agent_context_is_resolved(self):
        seen = {}
        node = Node(id="apply", type="agent",
                    budget={"max_iterations": 3, "max_tokens": 100},
                    context=["web_search.output", "state.error_summary"], writes=[])
        runner = NodeRunner(agent_executor=lambda n, ctx: seen.update(ctx) or {})
        state = State(data={"web_search.output": "found", "error_summary": "boom"})
        runner.run_node(node, state)
        assert seen == {"web_search.output": "found", "state.error_summary": "boom"}

    def test_human_pause_invokes_prompt(self):
        calls = []
        node = Node(id="ask", type="human_pause")
        runner = NodeRunner(human_prompt=lambda n, s: calls.append(n.id))
        runner.run_node(node, State())
        assert calls == ["ask"]

    def test_bare_string_stored_as_output(self):
        node = Node(id="web_search", type="tool", tool="search_web")
        runner = NodeRunner(tool_executor=lambda n, s: "raw text")
        state = State()
        runner.run_node(node, state)
        assert state.data["web_search.output"] == "raw text"


class TestArgentToolExecutor:
    def test_unknown_tool(self):
        node = Node(id="n", type="tool", tool="ghost")
        assert "not available" in argent_tool_executor(node, State())

    def test_binds_input_to_first_required_param(self, monkeypatch):
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"echo": lambda query: f"got:{query}"})
        node = Node(id="n", type="tool", tool="echo", input_from="state.q")
        assert argent_tool_executor(node, State(data={"q": "hi"})) == "got:hi"

    def test_tool_error_is_caught(self, monkeypatch):
        def boom():
            raise RuntimeError("nope")
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"boom": boom})
        node = Node(id="n", type="tool", tool="boom")
        assert "Error running tool" in argent_tool_executor(node, State())

    def test_literal_args_passed(self, monkeypatch):
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"cmd": lambda command: f"ran:{command}"})
        node = Node(id="n", type="tool", tool="cmd", args={"command": "pytest -q"})
        assert argent_tool_executor(node, State()) == "ran:pytest -q"

    def test_args_and_input_from_combine(self, monkeypatch):
        monkeypatch.setattr(schemas, "AVAILABLE_TOOLS", {"f": lambda a, b: f"{a}:{b}"})
        # args gives `a`; input_from fills the first required param not already set (`b`).
        node = Node(id="n", type="tool", tool="f", args={"a": "X"}, input_from="state.q")
        assert argent_tool_executor(node, State(data={"q": "Y"})) == "X:Y"


class Fakes:
    def __init__(self, pass_on):
        self.pass_on = pass_on
        self.test_runs = self.searches = self.fixes = self.commits = 0

    def tool(self, node, state):
        if node.id == "web_search":
            self.searches += 1
            return {"output": "found"}
        if node.id == "commit":
            self.commits += 1
            return {"output": "committed"}
        return {}

    def agent(self, node, context):
        if node.id == "run_tests":   # the verifier agent — owns tests_passed
            self.test_runs += 1
            passed = self.test_runs >= self.pass_on
            return {"tests_passed": passed,
                    "error_summary": None if passed else f"fail{self.test_runs}"}
        if node.id == "apply_fix":
            self.fixes += 1
            # The agent LIES: claims the tests pass. writes=["error_summary"]
            # drops tests_passed, so the verifier still decides.
            return {"error_summary": "applied", "tests_passed": True}
        return {}


class TestIntegration:
    def test_agent_lie_does_not_short_circuit_verifier(self):
        room = RoomLibrary().load_dir(default_starter_dir(), {"run_command", "search_web"}).get("debug_loop")
        f = Fakes(pass_on=3)
        runner = NodeRunner(tool_executor=f.tool, agent_executor=f.agent)
        out = Engine().run(room, State(), runner.run_node)
        assert out.exit == "success"
        # If the agent's tests_passed lie had been applied, the loop would have
        # exited after the first apply_fix. It ran the verifier all 3 times.
        assert f.test_runs == 3
        assert f.commits == 1
        assert out.state.loops["fix_attempts"] == 2
