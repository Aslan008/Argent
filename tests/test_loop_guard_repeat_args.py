"""The same call with a shifting answer is still a loop.

The result-based signature catches a call that returns the identical thing forever.
It cannot see the other spin: the model rewrites a file, re-reads it, rewrites it
again — every answer is slightly different, so nothing looks repeated, and an
unattended run keeps going until the context or the budget is gone.
"""

import pytest

from src.agent import loop_guard as lg


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr("argent.config.get_loop_guard_mode", lambda: "auto")
    return lg.LoopGuard()


def test_the_same_call_with_a_different_answer_earns_a_warning(guard):
    levels = [guard.record("read_file", {"path": "a.py"}, f"version {i}") for i in range(3)]
    assert levels[:2] == [None, None]
    assert levels[2] == "warn"


def test_and_then_the_turn_is_stopped(guard):
    levels = [guard.record("run_command", {"command": "pytest -q"}, f"run {i}") for i in range(5)]
    assert levels[-1] == "stop"
    assert lg.REPEAT_ARGS_WARN <= lg.REPEAT_ARGS_STOP


def test_a_different_argument_resets_the_streak(guard):
    guard.record("read_file", {"path": "a.py"}, "x")
    guard.record("read_file", {"path": "a.py"}, "x2")
    assert guard.record("read_file", {"path": "b.py"}, "y") is None
    # ... and two more of the NEW call are needed again, not one.
    assert guard.record("read_file", {"path": "b.py"}, "y2") is None


def test_a_different_tool_resets_it_too(guard):
    guard.record("read_file", {"path": "a.py"}, "x")
    guard.record("read_file", {"path": "a.py"}, "x2")
    assert guard.record("grep_search", {"pattern": "x"}, "y") is None


def test_waiting_tools_may_repeat_forever(guard):
    # Asking the same question again is how you wait for a background job.
    for i in range(10):
        assert guard.record("read_background_command", {"pid": "7"}, f"out {i}") is None
    for i in range(10):
        assert guard.record("subagents", {"action": "result", "id": 3}, f"r {i}") is None


def test_a_genuine_progress_sequence_is_left_alone(guard):
    # Different files, different commands: ordinary work, never stopped.
    for i in range(12):
        assert guard.record("read_file", {"path": f"src/f{i}.py"}, f"content {i}") is None


def test_the_old_identical_result_rule_still_works(guard):
    levels = [guard.record("read_file", {"path": "a.py"}, "same") for _ in range(4)]
    assert "stop" in levels or "warn" in levels


def test_reset_clears_the_streak(guard):
    guard.record("read_file", {"path": "a.py"}, "x")
    guard.record("read_file", {"path": "a.py"}, "x2")
    guard.reset()
    assert guard.record("read_file", {"path": "a.py"}, "x3") is None


def test_an_excluded_tool_never_counts(guard):
    for i in range(10):
        assert guard.record("wait_heartbeat", {}, f"wake {i}") is None
