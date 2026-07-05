"""Supervisor: the meta-graph where rooms are the nodes.

The same idea as the intra-room engine, one level up: run a room, then route to
the next room based on its terminal exit. Normal exits follow the routing table
(or END the run); failure terminals (budget / stuck / EXIT:escalate) fall through
to the triage room, which is the sole meta-level decision point. Triage sets
``state.route_to`` (through its whitelisted agent) to name the next room.

Two hard bounds keep the meta-loop from running away:
  * a total-rooms budget across the whole run;
  * consecutive triage passes — more than ``max_triage_passes`` without a
    non-triage room making progress ends the run with an escalation to a human.

Room execution is injected (``run_room``) so the routing logic is testable
without the engine, runner or an LLM; ``build_supervisor`` wires the production
path to Engine + NodeRunner.
"""

from dataclasses import dataclass, field

from src.rooms.engine import Engine
from src.rooms.models import State

# Terminal exits that mean "this room did not succeed".
FAILURE_EXITS = {"budget", "stuck", "escalate", "human_pause"}


@dataclass
class RunResult:
    outcome: str                       # "done" | "escalated" | "budget"
    history: list = field(default_factory=list)   # [(room_name, exit), ...]
    state: State = None


class Supervisor:
    def __init__(self, library, run_room, routes=None,
                 max_rooms: int = 50, max_triage_passes: int = 2,
                 triage_room: str = "triage"):
        self.library = library
        self.run_room = run_room       # (room, state) -> Outcome
        self.routes = routes or {}     # (room_name, exit) -> next | "END:x" | "TRIAGE"
        self.max_rooms = max_rooms
        self.max_triage_passes = max_triage_passes
        self.triage_room = triage_room

    def run(self, start_room: str, state: State = None) -> RunResult:
        state = state or State()
        history = []
        current = start_room
        rooms_run = 0
        triage_passes = 0

        while True:
            if rooms_run >= self.max_rooms:
                return RunResult("budget", history, state)

            room = self.library.get(current)
            if room is None:
                # A route named a room that isn't in the library — treat as a
                # failure to route: hand off to triage (or escalate if we're
                # already stuck at triage).
                if current == self.triage_room:
                    return RunResult("escalated", history, state)
                current = self.triage_room
                continue

            outcome = self.run_room(room, state)
            rooms_run += 1
            history.append((current, outcome.exit))
            self.library.record_run(current, success=outcome.exit not in FAILURE_EXITS)

            # Progress = a non-triage room reaching a success-like exit. Only that
            # resets the triage budget — triage merely routing back is NOT progress,
            # otherwise a failing room + rerouting triage would loop forever.
            if current != self.triage_room and outcome.exit not in FAILURE_EXITS:
                triage_passes = 0

            nxt = self._route(current, outcome, state)
            if nxt is None:
                return RunResult("done", history, state)
            if nxt == "__ESCALATE__":
                return RunResult("escalated", history, state)

            if nxt == self.triage_room:
                triage_passes += 1
                if triage_passes > self.max_triage_passes:
                    return RunResult("escalated", history, state)

            current = nxt

    def _route(self, current, outcome, state):
        """Return the next room name, None to END, or "__ESCALATE__"."""
        if current == self.triage_room:
            route_to = state.data.get("route_to")
            if outcome.exit == "routed" and route_to:
                return route_to
            return "__ESCALATE__"   # triage proposed / paused / gave no route

        key = (current, outcome.exit)
        if key in self.routes:
            target = self.routes[key]
            if target.startswith("END:"):
                return None
            if target == "TRIAGE":
                return self.triage_room
            return target

        if outcome.exit == "human_pause":
            return "__ESCALATE__"
        if outcome.exit in FAILURE_EXITS:
            return self.triage_room
        return None   # a success-like exit with no explicit route ends the run


def build_supervisor(library, runner, routes=None, journal=None, **kwargs) -> Supervisor:
    """Production wiring: rooms are executed by the Engine driving a NodeRunner.
    When `journal` is given, each room's node steps are journaled for crash-safe
    resume (see resume_run)."""
    engine = Engine()

    def run_room(room, state):
        return engine.run(room, state, runner.run_node, journal=journal)

    return Supervisor(library, run_room, routes=routes, **kwargs)


def resume_run(supervisor, library, journal, runner) -> "RunResult":
    """Resume an interrupted run from `journal`, then hand back to the supervisor.

    Reads where the last room was, recovers its node-level position via
    plan_resume, and continues the meta-loop from that room's exit. Meta-counters
    (history, triage passes) start fresh on resume — the guarantee is "don't redo
    completed work and don't double-apply a side effect", not a byte-perfect
    replay. Returns None when there is nothing to resume.
    """
    from src.rooms.engine import Engine, Outcome, plan_resume

    last = journal.last()
    if not last:
        return None
    room = library.get(last.get("room"))
    if room is None:
        return None

    start_node, state, status = plan_resume(room, journal)
    if status == "indeterminate":
        return RunResult("escalated", [(room.room, "indeterminate")], state)

    history = []
    if status == "finished":
        exit_name = last.get("exit")
    else:  # "resume" — finish the interrupted room first
        outcome = Engine().run(room, state, runner.run_node,
                               journal=journal, start_node=start_node)
        exit_name = outcome.exit
        history.append((room.room, exit_name))

    nxt = supervisor._route(room.room, Outcome(exit_name, state, 0), state)
    if nxt is None:
        return RunResult("done", history, state)
    if nxt == "__ESCALATE__":
        return RunResult("escalated", history, state)

    result = supervisor.run(nxt, state)
    result.history = history + result.history
    return result
