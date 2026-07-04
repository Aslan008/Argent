"""Write-ahead journal + crash-safe resume."""

from src.rooms.engine import Engine, plan_resume
from src.rooms.journal import Journal
from src.rooms.models import Room, State


def make_room(nodes, edges):
    return Room(room="t", budget={"max_iterations": 50, "max_tokens": 1000},
                nodes=nodes, edges=edges)


CHAIN = make_room(
    [{"id": "A", "type": "tool", "tool": "noop"},
     {"id": "B", "type": "tool", "tool": "noop"}],
    [{"from": "A", "to": "B"}, {"from": "B", "to": "EXIT:done"}],
)


class TestJournal:
    def test_append_and_read(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent", "node": "A"})
        j.append({"kind": "commit", "node": "A"})
        recs = j.records()
        assert [r["kind"] for r in recs] == ["intent", "commit"]
        assert j.last()["node"] == "A"

    def test_missing_file_is_empty(self, tmp_path):
        j = Journal(tmp_path / "nope.jsonl")
        assert j.records() == [] and j.last() is None

    def test_clear(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent"})
        j.clear()
        assert j.records() == []

    def test_tolerates_torn_line(self, tmp_path):
        p = tmp_path / "j.jsonl"
        p.write_text('{"kind": "commit", "node": "A"}\n{"kind": "inten', encoding="utf-8")
        j = Journal(p)
        assert len(j.records()) == 1 and j.last()["node"] == "A"


class TestEngineJournaling:
    def test_records_intent_and_commit_per_node(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")
        out = Engine().run(CHAIN, State(), lambda n, s: None, journal=j)
        assert out.exit == "done"
        recs = j.records()
        assert [r["kind"] for r in recs] == ["intent", "commit", "intent", "commit"]
        assert recs[-1]["exit"] == "done" and recs[-1]["next"] is None


class TestPlanResume:
    def test_fresh_when_no_journal(self, tmp_path):
        start, state, status = plan_resume(CHAIN, Journal(tmp_path / "j.jsonl"))
        assert status == "fresh" and start == "A" and state.data == {}

    def test_finished_when_terminal_committed(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")
        Engine().run(CHAIN, State(), lambda n, s: None, journal=j)
        start, _, status = plan_resume(CHAIN, j)
        assert status == "finished" and start is None

    def test_resume_after_mid_commit(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent", "room": "t", "node": "A", "state": {"data": {}, "loops": {}}})
        j.append({"kind": "commit", "room": "t", "node": "A", "next": "B", "exit": None,
                  "state": {"data": {"A": "ran"}, "loops": {}}})
        start, state, status = plan_resume(CHAIN, j)
        assert status == "resume" and start == "B" and state.data["A"] == "ran"

    def test_indeterminate_for_interrupted_non_idempotent_tool(self, tmp_path):
        room = make_room(
            [{"id": "A", "type": "tool", "tool": "noop", "idempotent": False}],
            [{"from": "A", "to": "EXIT:done"}],
        )
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent", "room": "t", "node": "A", "state": {"data": {}, "loops": {}}})
        start, _, status = plan_resume(room, j)
        assert status == "indeterminate" and start == "A"

    def test_interrupted_idempotent_tool_resumes(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent", "room": "t", "node": "B", "state": {"data": {"A": "ran"}, "loops": {}}})
        start, state, status = plan_resume(CHAIN, j)
        assert status == "resume" and start == "B"


class TestCrashResume:
    def test_crash_then_resume_completes(self, tmp_path):
        j = Journal(tmp_path / "j.jsonl")

        def crashing(node, state):
            if node.id == "B":
                raise RuntimeError("process died mid-node")
            state.data[node.id] = "ran"

        try:
            Engine().run(CHAIN, State(), crashing, journal=j)
        except RuntimeError:
            pass

        # The interrupted B (idempotent) is safe to replay from A's committed state.
        start, state, status = plan_resume(CHAIN, j)
        assert status == "resume" and start == "B"
        assert state.data["A"] == "ran"   # A's effect survived the crash

        out = Engine().run(CHAIN, state, lambda n, s: s.data.__setitem__(n.id, "ran"),
                           journal=j, start_node=start)
        assert out.exit == "done"
        assert plan_resume(CHAIN, j)[2] == "finished"
