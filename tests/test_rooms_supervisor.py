"""Supervisor: meta-graph routing between rooms, triage handoff, and bounds.

Room execution is scripted (FakeRun returns canned Outcomes), so only the
supervisor's routing logic is under test — no engine, runner or LLM.
"""

from collections import defaultdict

from src.rooms.engine import Outcome
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.supervisor import Supervisor


def _lib():
    return RoomLibrary().load_dir(default_starter_dir(), {"run_command", "search_web"})


class FakeRun:
    """Scripts each room's successive outcomes; the last entry repeats."""

    def __init__(self):
        self.script = defaultdict(list)   # room -> [(exit, state_updates)]
        self.calls = defaultdict(int)

    def add(self, room, exit, **updates):
        self.script[room].append((exit, updates))
        return self

    def __call__(self, room, state):
        seq = self.script[room.room]
        idx = min(self.calls[room.room], len(seq) - 1)
        self.calls[room.room] += 1
        exit, updates = seq[idx]
        state.data.update(updates)
        return Outcome(exit, state, 1)


class TestRouting:
    def test_linear_pipeline(self):
        f = FakeRun().add("analyze", "planned").add("debug_loop", "success")
        routes = {("analyze", "planned"): "debug_loop", ("debug_loop", "success"): "END:done"}
        res = Supervisor(_lib(), f, routes).run("analyze")
        assert res.outcome == "done"
        assert res.history == [("analyze", "planned"), ("debug_loop", "success")]

    def test_success_without_route_ends_run(self):
        f = FakeRun().add("analyze", "planned")
        res = Supervisor(_lib(), f, routes={}).run("analyze")
        assert res.outcome == "done"
        assert res.history == [("analyze", "planned")]

    def test_failure_falls_through_to_triage_then_reroutes(self):
        f = (FakeRun()
             .add("debug_loop", "escalate").add("debug_loop", "success")
             .add("triage", "routed", route_to="debug_loop"))
        routes = {("debug_loop", "success"): "END:done"}
        res = Supervisor(_lib(), f, routes).run("debug_loop")
        assert res.outcome == "done"
        assert res.history == [
            ("debug_loop", "escalate"), ("triage", "routed"), ("debug_loop", "success")]

    def test_triage_without_route_escalates(self):
        f = FakeRun().add("debug_loop", "escalate").add("triage", "proposed")
        res = Supervisor(_lib(), f, routes={}).run("debug_loop")
        assert res.outcome == "escalated"
        assert res.history == [("debug_loop", "escalate"), ("triage", "proposed")]

    def test_two_triage_passes_without_progress_escalate(self):
        # debug_loop always escalates; triage always routes back — no progress.
        f = (FakeRun()
             .add("debug_loop", "escalate")
             .add("triage", "routed", route_to="debug_loop"))
        res = Supervisor(_lib(), f, routes={}, max_triage_passes=2).run("debug_loop")
        assert res.outcome == "escalated"
        assert sum(1 for r, _ in res.history if r == "triage") == 2   # ran twice, then gave up
        assert sum(1 for r, _ in res.history if r == "debug_loop") == 3

    def test_progress_resets_triage_counter(self):
        # triage recovers once, debug_loop succeeds -> counter resets, so a later
        # failure gets a fresh triage budget rather than escalating immediately.
        f = (FakeRun()
             .add("debug_loop", "escalate").add("debug_loop", "success")
             .add("triage", "routed", route_to="debug_loop"))
        routes = {("debug_loop", "success"): "END:done"}
        res = Supervisor(_lib(), f, routes, max_triage_passes=2).run("debug_loop")
        assert res.outcome == "done"

    def test_max_rooms_budget(self):
        f = FakeRun().add("debug_loop", "success")
        routes = {("debug_loop", "success"): "debug_loop"}   # self-loop forever
        res = Supervisor(_lib(), f, routes, max_rooms=5).run("debug_loop")
        assert res.outcome == "budget"
        assert len(res.history) == 5

    def test_route_to_unknown_room_goes_to_triage(self):
        f = FakeRun().add("analyze", "planned").add("triage", "proposed")
        routes = {("analyze", "planned"): "ghost_room"}
        res = Supervisor(_lib(), f, routes).run("analyze")
        # ghost_room isn't in the library -> triage -> (no route) escalate
        assert res.outcome == "escalated"
        assert ("triage", "proposed") in res.history

    def test_record_run_updates_stats(self):
        lib = _lib()
        f = FakeRun().add("analyze", "planned")
        Supervisor(lib, f, routes={}).run("analyze")
        assert lib.stats["analyze"]["successes"] == 1
