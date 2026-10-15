"""Phase 2 tests: trajectory-derived verdict enrichment (RLCD signal)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("LEETH_TRAJECTORY_PATH", str(tmp_path / "trajectories.jsonl"))
    monkeypatch.setenv("LEETH_TRAJECTORY_FORCE", "1")
    monkeypatch.setenv("LEETH_FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setenv("LEETH_FEEDBACK_FORCE", "1")
    # conftest's autouse _leeth_feedback_off sets DISABLED=1; it outranks FORCE
    # in feedback._is_enabled, so remove it for these tests.
    monkeypatch.delenv("LEETH_FEEDBACK_DISABLED", raising=False)
    from src.leeth import trajectory, feedback, enrich
    trajectory._local.stack = []  # reset thread-local state between tests
    yield trajectory, feedback, enrich
    trajectory._local.stack = []


def _make_turn(trajectory, feedback, query, skip_reason=None,
               llm_tools=(), exit_code=None, denied=False):
    """Builds one trajectory + one decision carrying the given evidence."""
    tid = trajectory.start_trajectory(query)
    did = feedback.log_decision(
        query, "state text",
        {"tool": {"choice": "read_file", "confidence": 0.5}}, trajectory_id=tid)
    trajectory.attach_decision(did)
    if skip_reason:
        feedback.mark_skipped(did, skip_reason)
    for t in llm_tools or ():
        trajectory.record_event("llm_tool_exec", tool=t, ok=True,
                                args={"file_path": "x.py"})
    if denied:
        trajectory.record_event("approval", action="выполнить команду: x",
                                approved=False, destructive=True)
    if exit_code is not None:
        trajectory.record_event("command_exit", command="x", code=exit_code)
    trajectory.end_trajectory("completed")
    return did


def test_skip_with_llm_tool_becomes_partial(stores):
    trajectory, feedback, enrich = stores
    did = _make_turn(trajectory, feedback, "покажи лог",
                     skip_reason="low_tool_confidence",
                     llm_tools=["wait_heartbeat", "read_file"])
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] == "partial"
    assert rec["verdict_source"] == "trajectory"
    assert rec["correction"]["tool"] == "read_file"
    assert stats["partial_llm_tool"] == 1


def test_skip_confirmed_when_llm_passive(stores):
    trajectory, feedback, enrich = stores
    did = _make_turn(trajectory, feedback, "как дела",
                     skip_reason="need_tool_false")
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] == "accept"
    assert rec["verdict_source"] == "trajectory"
    assert stats["accept_skip_confirmed"] == 1


def test_executed_exit0_becomes_accept(stores):
    trajectory, feedback, enrich = stores
    did = _make_turn(trajectory, feedback, "прочитай main.py", exit_code=0)
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] == "accept"
    assert rec["outcome"]["type"] == "executed_ok"
    assert stats["accept_exit0"] == 1


def test_executed_denied_becomes_reject(stores):
    trajectory, feedback, enrich = stores
    did = _make_turn(trajectory, feedback, "удали build", denied=True)
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] == "reject"
    assert stats["reject_denied"] == 1


def test_executed_error_records_outcome_only(stores):
    trajectory, feedback, enrich = stores
    did = _make_turn(trajectory, feedback, "запусти тесты", exit_code=2)
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] is None
    assert rec["outcome"]["type"] == "executed_error"
    assert stats["outcome_error"] == 1


def test_heartbeat_queries_untouched(stores):
    trajectory, feedback, enrich = stores
    did = _make_turn(trajectory, feedback, "[Heartbeat прерван] что-то",
                     skip_reason="need_tool_false", llm_tools=["read_file"])
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] is None
    assert stats["heartbeat_skipped"] == 1


def test_explicit_verdict_never_overwritten(stores):
    trajectory, feedback, enrich = stores
    tid = trajectory.start_trajectory("прочитай config.py")
    did = feedback.log_decision("прочитай config.py", "state text",
                                {"tool": {"choice": "read_file", "confidence": 0.9}},
                                trajectory_id=tid)
    trajectory.attach_decision(did)
    feedback.submit_verdict(did, "reject", source="human", reason="no")
    trajectory.record_event("command_exit", command="x", code=0)
    trajectory.end_trajectory("completed")
    stats = enrich.derive_verdicts()
    rec = feedback.get_record(did)
    assert rec["verdict"] == "reject"
    assert rec["verdict_source"] == "human"
    assert stats["already_labeled"] == 1