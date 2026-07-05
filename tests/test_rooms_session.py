"""Live-wiring pieces: JSON-write parsing, AgentExecutor, and run_rooms assembly.

The LLM is never called: AgentExecutor's model call is injected, and run_rooms is
driven with a fake NodeRunner over the real starter library.
"""

from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import Node
from src.rooms.runner import AgentExecutor, NodeRunner, _consume_bounded, _extract_json_object
from src.rooms.session import default_routes, run_rooms


class TestExtractJson:
    def test_pulls_object_from_prose(self):
        raw = 'Done.\nResult:\n{"tests_passed": true, "error_summary": "none"}\nBye.'
        assert _extract_json_object(raw) == {"tests_passed": True, "error_summary": "none"}

    def test_returns_last_object(self):
        raw = '{"a": 1} then reconsidered {"b": 2}'
        assert _extract_json_object(raw) == {"b": 2}

    def test_none_when_no_object(self):
        assert _extract_json_object("no json here") is None
        assert _extract_json_object("") is None


class TestAgentExecutor:
    def test_prompt_includes_instruction_context_and_writes(self):
        node = Node(id="apply", type="agent",
                    budget={"max_iterations": 3, "max_tokens": 100},
                    instruction="Fix the bug", context=["web_search.output"],
                    writes=["error_summary"])
        prompt = AgentExecutor._build_prompt(node, {"web_search.output": "clue"})
        assert "Fix the bug" in prompt
        assert "web_search.output: clue" in prompt
        assert "error_summary" in prompt and "JSON" in prompt

    def test_call_parses_writes_from_model_text(self):
        node = Node(id="apply", type="agent",
                    budget={"max_iterations": 3, "max_tokens": 100},
                    writes=["error_summary"])
        # The model reports two fields; AgentExecutor returns them as-is, the
        # NodeRunner is what enforces the whitelist downstream.
        ex = AgentExecutor(lambda prompt, max_iterations=None: 'ok {"error_summary": "fixed", "tests_passed": true}')
        assert ex(node, {}) == {"error_summary": "fixed", "tests_passed": True}

    def test_call_empty_on_no_json(self):
        ex = AgentExecutor(lambda prompt, max_iterations=None: "I could not produce structured output")
        assert ex(Node(id="a", type="agent",
                       budget={"max_iterations": 1, "max_tokens": 10}, writes=[]), {}) == {}

    def test_passes_node_iteration_budget(self):
        seen = {}

        def run_agent(prompt, max_iterations=None):
            seen["max"] = max_iterations
            return "{}"
        node = Node(id="a", type="agent",
                    budget={"max_iterations": 5, "max_tokens": 100}, writes=[])
        AgentExecutor(run_agent)(node, {})
        assert seen["max"] == 5


def _agent_chunks(n_tool_rounds):
    for i in range(n_tool_rounds):
        yield {"type": "content_stream", "content": f"c{i}"}
        yield {"type": "tool_start", "name": "t"}
        yield {"type": "tool_end", "name": "t", "result": "r"}
    yield {"type": "content_stream", "content": "final"}


class TestConsumeBounded:
    def test_no_cap_consumes_all(self):
        out = _consume_bounded(_agent_chunks(3), max_iterations=None)
        assert out == "c0c1c2final"

    def test_stops_after_iteration_budget(self):
        out = _consume_bounded(_agent_chunks(5), max_iterations=2)
        assert "c0" in out and "c1" in out
        assert "c2" not in out and "final" not in out   # nothing after the cap
        assert "budget (2)" in out

    def test_natural_finish_before_cap(self):
        out = _consume_bounded(_agent_chunks(1), max_iterations=5)
        assert out == "c0final"   # finished on its own, no budget note


class _FakeRunner:
    """A NodeRunner with fake executors — drives the debug_loop without an LLM."""

    def __init__(self, pass_on):
        self.pass_on = pass_on
        self.test_runs = 0

        def tool(node, state):
            return {"output": "ok"}

        def agent(node, context):
            if node.id == "run_tests":   # verifier agent
                self.test_runs += 1
                passed = self.test_runs >= self.pass_on
                return {"tests_passed": passed, "error_summary": None if passed else "boom"}
            return {"error_summary": "applied"}

        self._runner = NodeRunner(tool_executor=tool, agent_executor=agent)

    def run_node(self, node, state):
        return self._runner.run_node(node, state)


class TestRunRoomsAssembly:
    def test_debug_loop_greens_through_full_wiring(self):
        lib = RoomLibrary().load_dir(default_starter_dir(), {"run_command", "search_web"})
        fake = _FakeRunner(pass_on=2)
        res = run_rooms("fix tests", start="debug_loop", library=lib, runner=fake)
        assert res.outcome == "done"
        assert ("debug_loop", "success") in res.history

    def test_default_routes_cover_starter_rooms(self):
        routes = default_routes()
        assert routes[("analyze", "planned")] == "debug_loop"
        assert routes[("debug_loop", "success")] == "END:done"
