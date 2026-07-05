"""Crash-safe resume wired through the supervisor + run_rooms."""

from src.rooms.engine import Engine
from src.rooms.journal import Journal
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import State
from src.rooms.runner import NodeRunner
from src.rooms.session import default_routes, run_rooms
from src.rooms.supervisor import build_supervisor, resume_run


def _lib():
    return RoomLibrary().load_dir(default_starter_dir(), {"run_command", "search_web"})


class FakeRunner:
    def __init__(self, pass_on=1):
        self.pass_on = pass_on
        self.test_runs = 0
        self.commits = 0

        def tool(node, state):
            if node.id == "commit":
                self.commits += 1
            return {"output": "ok"}

        def agent(node, context):
            if node.id == "run_tests":   # verifier agent
                self.test_runs += 1
                passed = self.test_runs >= self.pass_on
                return {"tests_passed": passed, "error_summary": None if passed else "boom"}
            return {"error_summary": "applied"}

        self._nr = NodeRunner(tool_executor=tool, agent_executor=agent)

    def run_node(self, node, state):
        return self._nr.run_node(node, state)


class TestResumeWiring:
    def test_resume_mid_room_completes(self, tmp_path):
        lib = _lib()
        journal = Journal(tmp_path / "j.jsonl")
        room = lib.get("debug_loop")

        def crash(node, state):
            if node.id == "run_tests":
                state.data["tests_passed"] = False
                state.data["error_summary"] = "boom"
            elif node.id == "web_search":
                raise RuntimeError("process died")

        try:
            Engine().run(room, State(data={"task": "x"}), crash, journal=journal)
        except RuntimeError:
            pass
        assert journal.last()["kind"] == "intent" and journal.last()["node"] == "web_search"

        fake = FakeRunner(pass_on=1)   # greens on the first verifier run after resume
        res = run_rooms("", library=lib, runner=fake, journal=journal, resume=True)
        assert res.outcome == "done"
        assert fake.test_runs >= 1     # the verifier ran during resume

    def test_resume_finished_room_routes_without_rerun(self, tmp_path):
        lib = _lib()
        journal = Journal(tmp_path / "j.jsonl")
        journal.append({"kind": "commit", "room": "debug_loop", "node": "commit",
                        "next": None, "exit": "success", "state": {"data": {}, "loops": {}}})
        fake = FakeRunner()
        res = run_rooms("", library=lib, runner=fake, journal=journal, resume=True)
        assert res.outcome == "done"
        assert fake.test_runs == 0     # the already-finished room was not re-run

    def test_resume_indeterminate_escalates(self, tmp_path):
        lib = _lib()
        journal = Journal(tmp_path / "j.jsonl")
        # An interrupted agent node is non-idempotent -> must not be replayed.
        journal.append({"kind": "intent", "room": "debug_loop", "node": "apply_fix",
                        "state": {"data": {}, "loops": {}}})
        res = run_rooms("", library=lib, runner=FakeRunner(), journal=journal, resume=True)
        assert res.outcome == "escalated"

    def test_nothing_to_resume_returns_none(self, tmp_path):
        lib = _lib()
        empty = Journal(tmp_path / "empty.jsonl")
        sup = build_supervisor(lib, FakeRunner(), routes=default_routes(), journal=empty)
        assert resume_run(sup, lib, empty, FakeRunner()) is None

    def test_fresh_run_clears_stale_journal(self, tmp_path):
        lib = _lib()
        journal = Journal(tmp_path / "j.jsonl")
        journal.append({"kind": "intent", "room": "stale", "node": "x",
                        "state": {"data": {}, "loops": {}}})
        # Fresh run (resume=False) must wipe the stale journal before starting.
        run_rooms("task", start="debug_loop", library=lib, runner=FakeRunner(pass_on=1),
                  journal=journal)
        rooms_in_journal = {r.get("room") for r in journal.records()}
        assert "stale" not in rooms_in_journal
