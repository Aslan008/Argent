"""Node runners: turn declarative nodes into real side effects.

The Engine decides WHICH node runs; the NodeRunner decides what running it means
— call a tool, run a bounded agent, prompt a human. Tool and agent execution are
injected, so engine+runner stay testable without an LLM; production wiring is a
thin adapter (argent_tool_executor).

Both executors return a dict of *proposed* state writes. The runner applies them
through the node's ``writes`` whitelist — a field the executor proposes but the
node did not declare is dropped. This is the verifier-first invariant in the
plumbing: an agent that self-reports ``tests_passed`` it wasn't granted simply
can't, so greenness stays owned by the verifier node that declares it. A plain
string return is treated as the node's ``.output`` only.
"""

from src.rooms.models import Node, State


class NodeRunner:
    def __init__(self, tool_executor=None, agent_executor=None, human_prompt=None):
        self._tool = tool_executor        # (node, state) -> dict | str
        self._agent = agent_executor      # (node, context: dict) -> dict
        self._human = human_prompt        # (node, state) -> None

    def run_node(self, node: Node, state: State) -> None:
        if node.type == "tool":
            if self._tool:
                self._apply(node, state, self._tool(node, state))
        elif node.type == "agent":
            if self._agent:
                context = {ref: self._resolve(ref, state) for ref in node.context}
                self._apply(node, state, self._agent(node, context) or {})
        elif node.type == "human_pause":
            if self._human:
                self._human(node, state)
        # condition / spawn_room: no side effect at this layer

    @staticmethod
    def _resolve(ref: str, state: State):
        key = ref[len("state."):] if ref.startswith("state.") else ref
        return state.data.get(key)

    @staticmethod
    def _apply(node: Node, state: State, result) -> None:
        if not isinstance(result, dict):
            # bare output
            if result is not None:
                state.data[f"{node.id}.output"] = result
            return
        if "output" in result and result["output"] is not None:
            state.data[f"{node.id}.output"] = result["output"]
        allowed = set(node.writes)
        for key, value in result.items():
            if key == "output":
                continue
            if key in allowed:
                state.data[key] = value
            # keys outside the whitelist are silently dropped — bounded influence


def argent_tool_executor(node: Node, state: State):
    """Production adapter: run a real Argent tool for a tool node.

    `input_from` supplies the primary argument value, bound to the tool's first
    required parameter. Real tools return a string, which the runner stores as
    the node's `.output`; deriving structured flags from that output is the job
    of a following agent/condition node. Never raises — errors come back as text.
    """
    import inspect
    from tools.schemas import AVAILABLE_TOOLS

    fn = AVAILABLE_TOOLS.get(node.tool)
    if fn is None:
        return f"Error: tool '{node.tool}' is not available"

    kwargs = {}
    if node.input_from:
        key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
        value = state.data.get(key)
        if value is not None:
            required = [
                p for p in inspect.signature(fn).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            ]
            if required:
                kwargs[required[0].name] = value
    try:
        return str(fn(**kwargs))
    except Exception as e:
        return f"Error running tool '{node.tool}': {e}"
