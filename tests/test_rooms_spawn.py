"""spawn_room gate: an AI-proposed room enters only if it parses, validates, and
a human approves — otherwise nothing is added and no route is set.
"""

from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import Node, Room, State
from src.rooms.runner import NodeRunner
from src.rooms.spawn import SpawnHandler


NEW = {
    "room": "translate",
    "description": "translate a file",
    "budget": {"max_iterations": 3, "max_tokens": 100},
    "nodes": [{"id": "t", "type": "agent",
               "budget": {"max_iterations": 3, "max_tokens": 100}, "writes": ["done"]}],
    "edges": [{"from": "t", "if": "state.done", "to": "EXIT:done"},
              {"from": "t", "else": True, "to": "EXIT:incomplete"}],
}

BAD_TOOL = {
    "room": "broken",
    "budget": {"max_iterations": 2, "max_tokens": 50},
    "nodes": [{"id": "n", "type": "tool", "tool": "ghost"}],
    "edges": [{"from": "n", "to": "EXIT:done"}],
}

NODE = Node(id="propose", type="spawn_room")


def _handler(lib, propose, approve):
    return SpawnHandler(lib, {"run_command", "search_web"}, propose=propose, approve=approve)


class TestSpawnGate:
    def test_valid_proposal_added_and_routed(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        approved = []
        st = State(data={"task": "x"})
        _handler(lib, lambda n, s: dict(NEW),
                 lambda r: approved.append(r.room) or True)(NODE, st)
        assert lib.get("translate") is not None       # added
        assert st.data["route_to"] == "translate"      # supervisor will route here
        assert approved == ["translate"]               # human was asked
        assert (tmp_path / "rooms" / "translate.json").exists()  # persisted

    def test_bad_schema_no_add(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        st = State()
        _handler(lib, lambda n, s: {"room": "x"}, lambda r: True)(NODE, st)  # missing budget/nodes
        assert "route_to" not in st.data and not lib.rooms

    def test_invalid_room_rejected(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        st = State()
        _handler(lib, lambda n, s: dict(BAD_TOOL), lambda r: True)(NODE, st)  # tool 'ghost'
        assert "route_to" not in st.data and not lib.rooms

    def test_name_collision_not_clobbered(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        lib.add_room(Room(**NEW), {"run_command", "search_web"})
        approvals = []
        _handler(lib, lambda n, s: dict(NEW), lambda r: approvals.append(1) or True)(NODE, State())
        assert approvals == []   # collision short-circuits before approval

    def test_human_rejection_not_added(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        st = State()
        _handler(lib, lambda n, s: dict(NEW), lambda r: False)(NODE, st)
        assert "route_to" not in st.data and not lib.rooms

    def test_no_proposal(self, tmp_path):
        lib = RoomLibrary(user_dir=tmp_path / "rooms")
        st = State()
        _handler(lib, lambda n, s: None, lambda r: True)(NODE, st)
        assert "route_to" not in st.data


class TestRunnerDispatch:
    def test_spawn_room_dispatched(self):
        calls = []
        NodeRunner(spawn_handler=lambda n, s: calls.append(n.id)).run_node(NODE, State())
        assert calls == ["propose"]

    def test_spawn_room_noop_without_handler(self):
        NodeRunner().run_node(NODE, State())   # must not raise


class TestTriageStillValid:
    def test_triage_with_spawn_edges_valid(self):
        lib = RoomLibrary().load_dir(default_starter_dir(), {"run_command", "search_web"})
        assert lib.load_errors == {} and lib.get("triage") is not None
