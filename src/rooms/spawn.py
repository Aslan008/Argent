"""spawn_room: the AI proposes a NEW room as data.

The novel-but-risky bit of the design, gated three ways so it can't hurt:
  1. capability is off by default (config.get_rooms_spawn);
  2. a proposal is only DATA — it must parse the schema and pass the validator
     (tools exist, EXIT reachable, budgets, and — crucially — the proposed room
     may not itself spawn: allow_spawn=False);
  3. a human approves it before it is added and routed to.

On success the handler adds the room to the library and sets ``state.route_to``
so the triage room exits "routed" into the freshly-authored room. Proposal and
approval are injected, so the gate logic is tested without an LLM or prompts.
"""

from logger import get_logger
from src.rooms.models import Room
from src.rooms.validator import validate_room

log = get_logger("rooms")


class SpawnHandler:
    def __init__(self, library, tools, propose, approve):
        self.library = library
        self.tools = set(tools)
        self.propose = propose      # (node, state) -> dict | None
        self.approve = approve      # (room) -> bool

    def __call__(self, node, state) -> None:
        proposed = self.propose(node, state)
        if not proposed:
            return
        try:
            room = Room(**proposed)
        except Exception as e:
            log.info("spawn_room: proposal doesn't fit the schema (%s)", e)
            return
        if self.library.get(room.room):
            log.info("spawn_room: '%s' already exists — not clobbering", room.room)
            return
        # A spawned room may not itself hold spawn_room (only triage may spawn).
        errors = validate_room(room, self.tools, allow_spawn=False)
        if errors:
            log.info("spawn_room: proposed room invalid: %s", errors)
            return
        if not self.approve(room):
            log.info("spawn_room: human rejected '%s'", room.room)
            return
        if self.library.add_room(room, self.tools):
            return   # add_room re-validates; empty list == added
        state.data["route_to"] = room.room
        log.info("spawn_room: added and routing to new room '%s'", room.room)


class RoomProposer:
    """Production proposal: ask a sub-agent to author a room as JSON."""

    def __init__(self, run_agent, library):
        self._run_agent = run_agent
        self._library = library

    def __call__(self, node, state):
        from src.rooms.runner import _extract_json_object
        max_iter = node.budget.max_iterations if node.budget else 3
        raw = self._run_agent(self._build_prompt(state), max_iterations=max_iter)
        return _extract_json_object(raw)

    def _build_prompt(self, state) -> str:
        existing = ", ".join(self._library.names())
        task = state.data.get("task", "")
        failure = state.data.get("error_summary") or state.data.get("failure") or ""
        return (
            "You are the triage room and NO existing room fits. Propose ONE new room "
            "as a single JSON object matching this schema: "
            '{"room": name, "description": ..., "budget": {"max_iterations": N, "max_tokens": N}, '
            '"nodes": [...], "edges": [...]}. Node types: tool (needs \"tool\"), agent '
            "(needs a \"budget\" and a \"writes\" list), condition, human_pause. Edges: "
            "unconditional, or {\"if\": expr}, {\"else\": true}, {\"while\": expr, \"max\": N, \"id\": ...}. "
            "Every node must be able to reach an EXIT:<name>. Do NOT use spawn_room. "
            f"Existing rooms: {existing}. Task: {task}. Failure: {failure}. "
            "Output ONLY the JSON object."
        )


def questionary_approve(room) -> bool:
    """Production approval: show the proposed room and ask the human."""
    import questionary
    from src.rooms.library import describe_room
    from ui import print_system
    print_system("[bold cyan]Предложена новая комната:[/bold cyan]")
    print_system(describe_room(room))
    return bool(questionary.confirm(
        f"Принять и добавить комнату '{room.room}'?", default=False).ask())
