"""Wiring + entry point for a live rooms run.

Assembles the production pieces — the starter library validated against the real
tool registry, a NodeRunner backed by real tools and a real sub-agent, and a
Supervisor with a default route table — and runs a task through them. The
library and runner are injectable so the assembly path is testable without an
LLM.
"""

from pathlib import Path

from src.rooms.journal import Journal
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import State
from src.rooms.runner import (
    AgentExecutor, NodeRunner, argent_run_agent, argent_tool_executor,
)
from src.rooms.supervisor import build_supervisor, resume_run


def default_journal() -> Journal:
    return Journal(Path(".argent") / "rooms_journal.jsonl")


def default_routes() -> dict:
    """Meta-graph edges for the starter rooms. Unlisted failure exits fall
    through to triage automatically (see the Supervisor)."""
    return {
        ("analyze", "planned"): "debug_loop",
        ("analyze", "incomplete"): "TRIAGE",
        ("debug_loop", "success"): "END:done",
        ("debug_loop", "escalate"): "TRIAGE",
    }


def available_tool_names() -> set:
    from tools.schemas import AVAILABLE_TOOLS
    return set(AVAILABLE_TOOLS)


def build_library(available_tools=None) -> RoomLibrary:
    tools = available_tool_names() if available_tools is None else set(available_tools)
    return RoomLibrary().load_dir(default_starter_dir(), tools)


def build_runner(human_prompt=None) -> NodeRunner:
    return NodeRunner(
        tool_executor=argent_tool_executor,
        agent_executor=AgentExecutor(argent_run_agent),
        human_prompt=human_prompt,
    )


def run_rooms(task: str, start: str = "analyze", *, library=None, runner=None,
              routes=None, human_prompt=None, max_rooms: int = 50,
              journal=None, resume: bool = False):
    """Run `task` through the rooms engine. Returns the Supervisor RunResult.

    `library`/`runner` are injectable for tests; by default they are the real
    starter library and the real (tool + sub-agent) runner. When `journal` is
    given, room steps are journaled for crash-safe resume; `resume=True` picks up
    an interrupted run from that journal instead of starting `task` fresh.
    """
    library = library if library is not None else build_library()
    runner = runner if runner is not None else build_runner(human_prompt)
    supervisor = build_supervisor(
        library, runner, routes=routes if routes is not None else default_routes(),
        max_rooms=max_rooms, journal=journal,
    )

    if resume and journal is not None:
        resumed = resume_run(supervisor, library, journal, runner)
        if resumed is not None:
            return resumed

    if journal is not None:
        journal.clear()   # a fresh run starts a clean journal
    state = State(data={"task": task})
    return supervisor.run(start, state)
