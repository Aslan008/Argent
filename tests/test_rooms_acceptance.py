"""End-to-end acceptance of the debug_loop room on mock tools.

Drives the spec's core scenario — tests fail, search the error, apply a fix,
re-verify, loop (bounded), commit on green — through the real Engine + the real
validated starter room, with only the tool/agent side effects mocked. This is
the "rails hold a realistic debug loop" proof. The cross-room triage layer is a
later step; here the loop lives inside one room.
"""

from src.rooms.engine import Engine
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import State


TOOLS = {"run_command", "search_web"}


class MockTools:
    """run_tests is the authoritative verifier: it flips tests_passed on the
    `pass_on`-th run. apply_fix (the agent) does NOT set tests_passed — the
    room's `writes` whitelist wouldn't allow it — so greenness is verifier-owned."""

    def __init__(self, pass_on):
        self.pass_on = pass_on
        self.test_runs = 0
        self.searches = 0
        self.fixes = 0
        self.commits = 0

    def run_node(self, node, state):
        if node.id == "run_tests":
            self.test_runs += 1
            passed = self.test_runs >= self.pass_on
            state.data["tests_passed"] = passed
            if not passed:
                state.data["error_summary"] = f"failure #{self.test_runs}"
        elif node.id == "web_search":
            self.searches += 1
            state.data["search_output"] = "candidate fix"
        elif node.id == "apply_fix":
            self.fixes += 1
        elif node.id == "commit":
            self.commits += 1


def _debug_room():
    return RoomLibrary().load_dir(default_starter_dir(), TOOLS).get("debug_loop")


def test_fails_twice_then_greens_and_commits():
    room = _debug_room()
    mock = MockTools(pass_on=3)          # green on the 3rd test run
    out = Engine().run(room, State(), mock.run_node)
    assert out.exit == "success"
    assert mock.test_runs == 3           # verified three times
    assert mock.searches == 2 and mock.fixes == 2
    assert mock.commits == 1             # committed exactly once, on green
    assert out.state.loops["fix_attempts"] == 2   # looped back twice, bounded


def test_never_greens_escalates_without_commit():
    room = _debug_room()
    mock = MockTools(pass_on=99)         # never passes
    out = Engine().run(room, State(), mock.run_node)
    assert out.exit == "escalate"        # gave up after the fix-attempt budget
    assert mock.commits == 0             # never committed a red state
    assert out.state.loops["fix_attempts"] == 3   # exactly max attempts
    assert mock.test_runs == 4           # initial run + 3 bounded retries
