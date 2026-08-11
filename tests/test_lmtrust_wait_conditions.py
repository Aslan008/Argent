"""LMTrust deep tests for src/agent/wait_conditions.py.

Targets blind spots in AST condition evaluation, predicate dispatch,
timeout/poll loop logic, and file/process predicates.
"""
import ast
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.wait_conditions import (
    ConditionError,
    DEFAULT_POLL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    PREDICATES,
    _eval_node,
    _file_contains,
    _file_exists,
    _file_missing,
    _process_finished,
    _process_running,
    describe_predicates,
    evaluate,
    wait_for,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _node(expr: str):
    """Parse an expression string into an AST node."""
    return ast.parse(expr, mode="eval").body


# ─── _eval_node: BoolOp ─────────────────────────────────────────────────────

class TestEvalBoolOp:
    def test_and_all_true(self):
        assert _eval_node(_node("True and True")) is True

    def test_and_one_false(self):
        assert _eval_node(_node("True and False")) is False

    def test_and_all_false(self):
        assert _eval_node(_node("False and False")) is False

    def test_or_one_true(self):
        assert _eval_node(_node("False or True")) is True

    def test_or_all_false(self):
        assert _eval_node(_node("False or False")) is False

    def test_and_three_values(self):
        assert _eval_node(_node("True and True and True")) is True

    def test_and_three_one_false(self):
        assert _eval_node(_node("True and False and True")) is False

    def test_or_three_one_true(self):
        assert _eval_node(_node("False or True or False")) is True

    def test_nested_and_or(self):
        assert _eval_node(_node("True and (False or True)")) is True

    def test_nested_or_and(self):
        assert _eval_node(_node("False or (True and False)")) is False


# ─── _eval_node: UnaryOp (not) ──────────────────────────────────────────────

class TestEvalUnaryNot:
    def test_not_true(self):
        assert _eval_node(_node("not True")) is False

    def test_not_false(self):
        assert _eval_node(_node("not False")) is True

    def test_double_negation(self):
        assert _eval_node(_node("not not True")) is True

    def test_triple_negation(self):
        assert _eval_node(_node("not not not True")) is False

    def test_not_with_and(self):
        assert _eval_node(_node("not (True and False)")) is True

    def test_not_with_or(self):
        assert _eval_node(_node("not (False or False)")) is True


# ─── _eval_node: Constants ──────────────────────────────────────────────────

class TestEvalConstants:
    def test_true_constant(self):
        assert _eval_node(_node("True")) is True

    def test_false_constant(self):
        assert _eval_node(_node("False")) is False


# ─── _eval_node: Call ───────────────────────────────────────────────────────

class TestEvalCall:
    def test_call_non_name_func_raises(self):
        with pytest.raises(ConditionError, match="only direct calls"):
            _eval_node(_node("(lambda: True)()"))

    def test_unknown_predicate_raises(self):
        with pytest.raises(ConditionError, match="unknown check"):
            _eval_node(_node("nonexistent('x')"))

    def test_keyword_args_raises(self):
        with pytest.raises(ConditionError, match="keyword arguments"):
            _eval_node(_node("file_exists(path='x')"))

    def test_non_string_arg_raises(self):
        with pytest.raises(ConditionError, match="plain strings or numbers"):
            _eval_node(_node("file_exists([1, 2])"))

    def test_list_arg_raises(self):
        with pytest.raises(ConditionError, match="plain strings or numbers"):
            _eval_node(_node("file_exists([])"))

    def test_attribute_arg_raises(self):
        with pytest.raises(ConditionError, match="plain strings or numbers"):
            _eval_node(_node("file_exists(obj.attr)"))

    def test_float_arg_raises(self):
        with pytest.raises(ConditionError, match="plain strings or numbers"):
            _eval_node(_node("file_exists(3.14)"))


# ─── _eval_node: unsupported elements ───────────────────────────────────────

class TestEvalUnsupported:
    def test_binop_raises(self):
        with pytest.raises(ConditionError, match="unsupported"):
            _eval_node(_node("1 + 2"))

    def test_name_raises(self):
        with pytest.raises(ConditionError, match="unsupported"):
            _eval_node(_node("some_variable"))

    def test_compare_raises(self):
        with pytest.raises(ConditionError, match="unsupported"):
            _eval_node(_node("1 < 2"))

    def test_int_constant_raises(self):
        with pytest.raises(ConditionError, match="unsupported"):
            _eval_node(_node("42"))

    def test_string_constant_raises(self):
        with pytest.raises(ConditionError, match="unsupported"):
            _eval_node(_node("'hello'"))


# ─── evaluate ───────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_empty_string_raises(self):
        with pytest.raises(ConditionError, match="empty"):
            evaluate("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ConditionError, match="empty"):
            evaluate("   ")

    def test_syntax_error_raises(self):
        with pytest.raises(ConditionError, match="could not parse"):
            evaluate("file_exists(")

    def test_valid_true(self):
        assert evaluate("True") is True

    def test_valid_false(self):
        assert evaluate("False") is False

    def test_strips_whitespace(self):
        assert evaluate("  True  ") is True

    def test_unknown_predicate_raises(self):
        with pytest.raises(ConditionError, match="unknown check"):
            evaluate("bogus('x')")


# ─── file predicates ────────────────────────────────────────────────────────

class TestFilePredicates:
    def test_file_exists_true(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        assert _file_exists(str(f)) is True

    def test_file_exists_false_missing(self, tmp_path):
        assert _file_exists(str(tmp_path / "nope.txt")) is False

    def test_file_exists_false_for_directory(self, tmp_path):
        assert _file_exists(str(tmp_path)) is False

    def test_file_missing_true(self, tmp_path):
        assert _file_missing(str(tmp_path / "nope.txt")) is True

    def test_file_missing_false(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        assert _file_missing(str(f)) is False

    def test_file_contains_true(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        assert _file_contains(str(f), "world") is True

    def test_file_contains_false(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        assert _file_contains(str(f), "xyz") is False

    def test_file_contains_missing_file(self, tmp_path):
        assert _file_contains(str(tmp_path / "nope.txt"), "x") is False

    def test_file_contains_unicode(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("привет мир", encoding="utf-8")
        assert _file_contains(str(f), "мир") is True


# ─── process predicates ─────────────────────────────────────────────────────

class TestProcessPredicates:
    def test_process_finished_missing_pid(self):
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
        key = "999999"
        with ACTIVE_PROCESSES_LOCK:
            saved = ACTIVE_PROCESSES.pop(key, None)
        try:
            assert _process_finished(999999) is True
        finally:
            if saved is not None:
                with ACTIVE_PROCESSES_LOCK:
                    ACTIVE_PROCESSES[key] = saved

    def test_process_finished_none_process(self):
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
        key = "test_pf_none"
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[key] = {}
        try:
            assert _process_finished("test_pf_none") is True
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.pop(key, None)

    def test_process_finished_running(self):
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
        key = "test_pf_running"
        proc = MagicMock()
        proc.poll.return_value = None
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[key] = {"process": proc}
        try:
            assert _process_finished("test_pf_running") is False
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.pop(key, None)

    def test_process_finished_done(self):
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
        key = "test_pf_done"
        proc = MagicMock()
        proc.poll.return_value = 0
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[key] = {"process": proc}
        try:
            assert _process_finished("test_pf_done") is True
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.pop(key, None)

    def test_process_running_missing_pid(self):
        # missing PID = finished = not running
        assert _process_running(999999) is False

    def test_process_running_active(self):
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
        key = "test_pr_active"
        proc = MagicMock()
        proc.poll.return_value = None
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES[key] = {"process": proc}
        try:
            assert _process_running("test_pr_active") is True
        finally:
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.pop(key, None)


# ─── PREDICATES dict ────────────────────────────────────────────────────────

class TestPredicatesDict:
    def test_has_file_exists(self):
        assert "file_exists" in PREDICATES

    def test_has_file_missing(self):
        assert "file_missing" in PREDICATES

    def test_has_file_contains(self):
        assert "file_contains" in PREDICATES

    def test_has_process_finished(self):
        assert "process_finished" in PREDICATES

    def test_has_process_running(self):
        assert "process_running" in PREDICATES

    def test_all_callables(self):
        for name, fn in PREDICATES.items():
            assert callable(fn), f"{name} is not callable"


# ─── describe_predicates ────────────────────────────────────────────────────

class TestDescribePredicates:
    def test_contains_all_predicate_names(self):
        desc = describe_predicates()
        for name in PREDICATES:
            assert name in desc

    def test_mentions_and_or_not(self):
        desc = describe_predicates()
        assert "and" in desc
        assert "or" in desc
        assert "not" in desc

    def test_mentions_combine(self):
        desc = describe_predicates()
        assert "combine" in desc.lower()


# ─── wait_for ───────────────────────────────────────────────────────────────

class TestWaitFor:
    def test_condition_met_immediately(self):
        sleeps = []
        clocks = iter([0.0, 0.0])
        result = wait_for(
            "True",
            sleep=lambda s: sleeps.append(s),
            clock=lambda: next(clocks),
        )
        assert result["met"] is True
        assert result["reason"] == "condition met"
        assert result["waited"] == 0.0
        assert sleeps == []

    def test_timeout(self):
        t = [0.0]
        def clock():
            t[0] += 10.0
            return t[0]
        result = wait_for(
            "False",
            timeout=100,
            poll_seconds=5,
            sleep=lambda s: None,
            clock=clock,
        )
        assert result["met"] is False
        assert "timed out" in result["reason"]

    def test_invalid_condition_returns_immediately(self):
        sleeps = []
        result = wait_for(
            "bogus_predicate('x')",
            sleep=lambda s: sleeps.append(s),
            clock=lambda: 0.0,
        )
        assert result["met"] is False
        assert "invalid condition" in result["reason"]
        assert sleeps == []

    def test_timeout_zero_clamped(self):
        t = [0.0]
        def clock():
            t[0] += 2.0
            return t[0]
        result = wait_for(
            "False",
            timeout=0,
            poll_seconds=1,
            sleep=lambda s: None,
            clock=clock,
        )
        assert result["met"] is False
        # timeout=0 → clamped to 1.0, first eval at t=0, second at t=2 → timeout
        assert "timed out" in result["reason"]

    def test_max_timeout_clamping(self):
        t = [0.0]
        def clock():
            t[0] += MAX_TIMEOUT_SECONDS + 1
            return t[0]
        result = wait_for(
            "False",
            timeout=MAX_TIMEOUT_SECONDS + 100,
            poll_seconds=1,
            sleep=lambda s: None,
            clock=clock,
        )
        assert result["met"] is False

    def test_poll_greater_than_timeout(self):
        t = [0.0]
        def clock():
            t[0] += 0.5
            return t[0]
        result = wait_for(
            "False",
            timeout=1.0,
            poll_seconds=10.0,
            sleep=lambda s: None,
            clock=clock,
        )
        assert result["met"] is False
        assert "timed out" in result["reason"]

    def test_waited_value_in_result(self):
        clocks = iter([0.0, 5.0])
        result = wait_for(
            "True",
            sleep=lambda s: None,
            clock=lambda: next(clocks),
        )
        assert result["waited"] == 5.0

    def test_condition_met_after_polls(self):
        state = [False]
        t = [0.0]
        def clock():
            return t[0]
        def sleep(s):
            t[0] += s
            state[0] = True
        # First eval: False, sleep, then True
        results = []
        def my_evaluate(condition):
            results.append(len(results))
            return len(results) >= 2
        # Monkeypatch evaluate via module
        import src.agent.wait_conditions as wc
        original = wc.evaluate
        wc.evaluate = my_evaluate
        try:
            result = wait_for("dummy", timeout=100, poll_seconds=1,
                              sleep=sleep, clock=clock)
        finally:
            wc.evaluate = original
        assert result["met"] is True

    def test_default_timeout_is_600(self):
        assert DEFAULT_TIMEOUT_SECONDS == 600

    def test_max_timeout_is_6_hours(self):
        assert MAX_TIMEOUT_SECONDS == 6 * 3600

    def test_default_poll_is_5(self):
        assert DEFAULT_POLL_SECONDS == 5