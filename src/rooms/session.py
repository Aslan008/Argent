"""Wiring + entry point for a live rooms run.

Assembles the production pieces — the starter library validated against the real
tool registry, a NodeRunner backed by real tools and a real sub-agent, and a
Supervisor — and runs a task through them. The library and runner are injectable
so the assembly path is testable without an LLM.

Two entry points: `run_rooms` (the bare run, returns the Supervisor's RunResult)
and `execute_room_run` (the whole experience: progress on the screen, a run
record on disk, the final report) which the `/rooms` command and the `rooms` tool
both use.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

from src.rooms.journal import Journal
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import State
from src.rooms.runner import (
    AgentExecutor, NodeRunner, argent_compose, argent_run_agent, argent_tool_executor,
)
from src.rooms.supervisor import build_supervisor, resume_run


def default_journal() -> Journal:
    return Journal(Path(".argent") / "rooms_journal.jsonl")


def default_routes() -> dict:
    """The meta-graph edges the starter rooms used before rooms carried their own
    `next`. Kept for callers that pass routes explicitly; a plain run now relies on
    each room's own `next` (unlisted failure exits fall through to triage)."""
    return {
        ("analyze", "planned"): "debug_loop",
        ("analyze", "incomplete"): "TRIAGE",
        ("debug_loop", "success"): "END:done",
        ("debug_loop", "escalate"): "TRIAGE",
    }


def available_tool_names() -> set:
    from tools.schemas import AVAILABLE_TOOLS
    return set(AVAILABLE_TOOLS)


def user_rooms_dir() -> Path:
    return Path(".argent") / "rooms"


def stats_path() -> Path:
    return Path(".argent") / "rooms_stats.json"


def build_library(available_tools=None, user_dir=None, stats_file=None) -> RoomLibrary:
    """Starter rooms (packaged, read-only) plus any user/AI-authored rooms in
    .argent/rooms/ (writable). Run counters and quarantines persist in
    .argent/rooms_stats.json. `user_dir`/`stats_file` pin the locations (the editor
    server does, so a later `/cd` cannot move what it edits)."""
    tools = available_tool_names() if available_tools is None else set(available_tools)
    udir = Path(user_dir) if user_dir else user_rooms_dir()
    lib = RoomLibrary(user_dir=udir, stats_path=stats_file or stats_path())
    lib.load_dir(default_starter_dir(), tools, origin="starter")
    if udir.exists():
        lib.load_dir(udir, tools, origin="user")
    return lib


def describe_call(node) -> str:
    """What a node that reaches outside the run is about to do, in one line (for a question)."""
    args = ", ".join(f"{k}={v!r}" for k, v in (node.args or {}).items())
    if node.type == "mcp":
        what = f"MCP {node.server}.{node.tool or '(список инструментов)'}"
    elif node.type == "plugin":
        what = f"плагин {node.plugin + '.' if node.plugin else ''}{node.command}"
    elif node.type == "leeth":
        what = f"Leeth ({node.mode or 'delegate'}): {(node.instruction or '')[:80]}"
    elif node.type == "power":
        delay = f" через {node.seconds:g} с" if node.seconds else ""
        what = f"ПК: {node.mode}{delay}"
    else:
        what = str(node.tool)
    return f"{what}({args})" if args else what


def questionary_confirm(node, state) -> bool:
    """Production confirmation for a `confirm: true` node: show what is about to run."""
    import questionary
    return bool(questionary.confirm(
        f"Комната просит выполнить шаг «{node.id}»: {describe_call(node)}. Выполнить?", default=False).ask())


def build_runner(human_prompt=None, library=None, confirm=None, allow_spawn: bool = True) -> NodeRunner:
    spawn_handler = None
    from config import get_rooms_spawn
    # `allow_spawn` is False for a run with nobody at the keyboard: proposing a room needs an approval
    if allow_spawn and library is not None and get_rooms_spawn():
        from src.rooms.spawn import RoomProposer, SpawnHandler, questionary_approve
        tools = available_tool_names()
        spawn_handler = SpawnHandler(
            library, tools, RoomProposer(argent_run_agent, library), questionary_approve)
    from src.rooms.integrations import Integrations
    from src.rooms.subroom import SubRoomHandler, default_notify
    return NodeRunner(
        integrations=Integrations(),
        tool_executor=argent_tool_executor,
        agent_executor=AgentExecutor(argent_run_agent),
        human_prompt=human_prompt,
        spawn_handler=spawn_handler,
        confirm=confirm if confirm is not None else questionary_confirm,
        notify=default_notify,
        compose=argent_compose,
        room_handler=SubRoomHandler(library) if library is not None else None,
    )


def run_rooms(task: str, start: str = "analyze", *, library=None, runner=None,
              routes=None, human_prompt=None, max_rooms: int = 50,
              journal=None, resume: bool = False, params=None, on_event=None):
    """Run `task` through the rooms engine. Returns the Supervisor RunResult.

    `library`/`runner` are injectable for tests; by default they are the real
    starter library and the real (tool + sub-agent) runner. When `journal` is
    given, room steps are journaled for crash-safe resume; `resume=True` picks up
    an interrupted run from that journal instead of starting `task` fresh.
    `params` are extra state fields for the run (`test_command`, ...) that
    override the rooms' own defaults.
    """
    library = library if library is not None else build_library()
    runner = runner if runner is not None else build_runner(human_prompt, library=library)
    if on_event is not None and hasattr(runner, "set_on_event"):
        runner.set_on_event(on_event)       # a sub-room's nodes join the same progress and record
    supervisor = build_supervisor(
        library, runner, routes=routes if routes is not None else {},
        max_rooms=max_rooms, journal=journal, on_event=on_event,
    )

    if resume and journal is not None:
        resumed = resume_run(supervisor, library, journal, runner, on_event=on_event)
        if resumed is not None:
            return resumed

    if journal is not None:
        journal.clear()   # a fresh run starts a clean journal
    data = dict(params or {})
    if task or "task" not in data:
        data["task"] = task
    # what triage needs to route a failure: the rooms there are, in the person's words
    data.setdefault("rooms_catalog", rooms_catalog(library))
    return supervisor.run(start, State(data=data))


def rooms_catalog(library) -> str:
    """One line per usable room (triage itself excluded): `name: what it does`."""
    active = library.active_rooms() if hasattr(library, "active_rooms") else library.rooms
    lines = []
    for name in sorted(active):
        if name == "triage":
            continue
        from src.rooms.library import short_description
        lines.append(f"- {name}: {short_description(active[name], 140)}")
    return "\n".join(lines)


@dataclass
class RunSummary:
    result: object            # the Supervisor's RunResult
    report: str               # the text to show / return
    record_path: object       # Path of the run record, or None
    seconds: float = 0.0


def execute_room_run(task: str, start: str = "analyze", *, params=None, library=None, runner=None,
                     journal=None, resume: bool = False, human_prompt=None, confirm=None,
                     show_progress: bool = True, printer=None, max_rooms: int = 50,
                     record_dir=None) -> RunSummary:
    """A run the way a person should meet it: progress while it goes, a record on
    disk, a report at the end."""
    from src.rooms.board import RoomBoard
    from src.rooms.history import RunRecorder, format_report

    library = library if library is not None else build_library()
    if runner is None:
        runner = build_runner(human_prompt, library=library, confirm=confirm)
    recorder = RunRecorder(task, start, directory=record_dir, params=params)
    board = RoomBoard(library, printer=printer) if show_progress else None

    def on_event(event):
        recorder.on_event(event)
        if board is not None:
            board.on_event(event)

    t0 = time.monotonic()
    try:
        result = run_rooms(task, start, library=library, runner=runner, journal=journal,
                           resume=resume, max_rooms=max_rooms, params=params, on_event=on_event)
    finally:
        if board is not None:
            board.close()
    seconds = time.monotonic() - t0
    path = recorder.finish(result, resumed=resume)
    return RunSummary(result, format_report(result, library, task, seconds), path, seconds)


_PARAM = re.compile(r"""(?<!\S)([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|'[^']*'|[^\s=]\S*)""")


def parse_params(text: str) -> tuple:
    """Split `key=value` words off a run command: ('fix it', {'test_command': 'pytest -x'}).
    A value in quotes may hold spaces; everything that is not `key=value` stays the task
    exactly as it was typed (paths and quotes in it are not touched)."""
    params = {}

    def take(m):
        value = m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        params[m.group(1)] = value
        return " "

    rest = _PARAM.sub(take, text or "")
    return " ".join(rest.split()), params
