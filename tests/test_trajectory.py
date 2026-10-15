"""Phase 4 tests: trajectory store + feedback linkage."""
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
    from src.leeth import trajectory, feedback
    trajectory._local.stack = []  # reset thread-local state between tests
    yield trajectory, feedback
    trajectory._local.stack = []


def test_lifecycle_with_decision_link(stores):
    trajectory, feedback = stores
    traj_id = trajectory.start_trajectory("прочитай config.py")
    assert traj_id and traj_id.startswith("tr_")
    assert trajectory.current_trajectory_id() == traj_id

    decision_id = feedback.log_decision("прочитай config.py", "state", {}, trajectory_id=traj_id)
    assert decision_id
    assert trajectory.attach_decision(decision_id)

    assert trajectory.record_event("tool_exec", tool="read_file", ok=True, detail="ok")
    trajectory.end_trajectory("completed")
    assert trajectory.current_trajectory_id() is None

    rec = trajectory.get_trajectory(traj_id)
    assert rec["status"] == "completed"
    assert rec["decision_ids"] == [decision_id]
    assert rec["events"][0]["type"] == "tool_exec"
    assert rec["events"][0]["ok"] is True
    assert rec["end_reason"] is None or rec["status"] == "completed"

    fb = feedback.get_record(decision_id)
    assert fb["trajectory_id"] == traj_id


def test_outcome_recorded_on_decision(stores):
    trajectory, feedback = stores
    trajectory.start_trajectory("q")
    did = feedback.log_decision("q", "state", {})
    assert feedback.record_outcome(did, "executed_ok", {"tool": "read_file"})
    rec = feedback.get_record(did)
    assert rec["outcome"]["type"] == "executed_ok"
    assert rec["outcome"]["tool"] == "read_file"
    assert "ts" in rec["outcome"]


def test_nesting_stack(stores):
    trajectory, _ = stores
    outer = trajectory.start_trajectory("outer")
    inner = trajectory.start_trajectory("inner")
    assert trajectory.current_trajectory_id() == inner
    trajectory.end_trajectory("completed")
    assert trajectory.current_trajectory_id() == outer
    trajectory.end_trajectory("completed")
    assert trajectory.current_trajectory_id() is None
    assert trajectory.get_trajectory(outer)["status"] == "completed"
    assert trajectory.get_trajectory(inner)["status"] == "completed"


def test_end_without_start_is_noop(stores):
    trajectory, _ = stores
    trajectory.end_trajectory("completed")  # must not raise
    assert trajectory.current_trajectory_id() is None


def test_disabled_without_force(monkeypatch, tmp_path):
    monkeypatch.setenv("LEETH_TRAJECTORY_PATH", str(tmp_path / "t.jsonl"))
    monkeypatch.delenv("LEETH_TRAJECTORY_FORCE", raising=False)
    from src.leeth import trajectory
    assert trajectory.start_trajectory("q") is None
    assert trajectory.current_trajectory_id() is None


def test_stats_counts(stores):
    trajectory, _ = stores
    t1 = trajectory.start_trajectory("a")
    trajectory.record_event("tool_exec", tool="read_file", ok=True)
    trajectory.end_trajectory("completed")
    trajectory.start_trajectory("b")
    trajectory.end_trajectory("error")
    s = trajectory.stats()
    assert s["total"] >= 2
    assert s["by_status"].get("completed", 0) >= 1
    assert s["events"] >= 1