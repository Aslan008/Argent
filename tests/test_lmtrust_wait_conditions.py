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
# ─── error handlers (OSError in _resolve) ────────────────────────────────────

class TestErrorHandlers:
    """Tests that OSError handlers in _file_exists/_file_contains return False.

    Kills FALSE_TO_TRUE mutations that flip ``return False`` to ``return True``
    in the except-OSError branches (lines 48 and 64).
    """

    def test_file_exists_oserror_returns_false(self, monkeypatch):
        """_file_exists must return False (not True) when _resolve raises OSError."""
        def raise_oserror(path):
            raise OSError("permission denied")
        monkeypatch.setattr("src.agent.wait_conditions._resolve", raise_oserror)
        assert _file_exists("/some/path") is False

    def test_file_contains_oserror_returns_false(self, monkeypatch):
        """_file_contains must return False (not True) when _resolve raises OSError."""
        def raise_oserror(path):
            raise OSError("permission denied")
        monkeypatch.setattr("src.agent.wait_conditions._resolve", raise_oserror)
        assert _file_contains("/some/path", "text") is False


# ─── waited precision (non-zero clock) ────────────────────────────────────────

class TestWaitedPrecision:
    """Tests that ``waited`` is computed as ``clock() - started`` (not ``+``)
    and rounded to 1 decimal (not 0).

    Kills:
    * SUB_TO_ADD @ lines 172, 175, 177, 181 (``clock()-started`` → ``clock()+started``,
      ``timeout-waited`` → ``timeout+waited``).
    * ONE_TO_ZERO @ round(x, 1) → round(x, 0) on the same lines.

    Strategy: use a clock that starts at a NON-ZERO value so that
    ``clock() + started`` differs from ``clock() - started``.
    """

    def test_waited_nonzero_clock_met(self):
        """Condition met: waited = round(105.3 - 100.0, 1) == 5.3.

        SUB_TO_ADD mutant: round(105.3 + 100.0, 1) == 205.3.
        ONE_TO_ZERO mutant: round(5.3, 0) == 5.0.
        """
        clocks = iter([100.0, 105.3])
        result = wait_for(
            "True",
            sleep=lambda s: None,
            clock=lambda: next(clocks),
        )
        assert result["met"] is True
        assert result["waited"] == 5.3

    def test_waited_nonzero_clock_timeout(self):
        """Timeout: waited = round(105.3 - 100.0, 1) == 5.3.

        SUB_TO_ADD mutant at line 177: waited = 105.3 + 100.0 = 205.3.
        ONE_TO_ZERO mutant at line 180: round(5.3, 0) == 5.0.
        """
        clocks = iter([100.0, 105.3])
        result = wait_for(
            "False",
            timeout=1.0,
            poll_seconds=5,
            sleep=lambda s: None,
            clock=lambda: next(clocks),
        )
        assert result["met"] is False
        assert "timed out" in result["reason"]
        assert result["waited"] == 5.3

    def test_waited_nonzero_clock_invalid_condition(self):
        """Invalid condition: waited = round(105.3 - 100.0, 1) == 5.3.

        Kills SUB_TO_ADD @ line 175 and ONE_TO_ZERO on that line.
        """
        clocks = iter([100.0, 105.3])
        result = wait_for(
            "bogus_predicate('x')",
            sleep=lambda s: None,
            clock=lambda: next(clocks),
        )
        assert result["met"] is False
        assert "invalid condition" in result["reason"]
        assert result["waited"] == 5.3

    def test_sleep_uses_timeout_minus_waited(self):
        """sleep(min(poll, timeout - waited)) must subtract, not add.

        Kills SUB_TO_ADD @ line 181 (``timeout - waited`` → ``timeout + waited``).

        started=100.0, first eval: waited=2.0 < 5.0 → sleep(min(10, 3)) = 3.0.
        Mutant would sleep(min(10, 7)) = 7.0.
        """
        clocks = iter([100.0, 102.0, 105.0])
        sleeps = []
        result = wait_for(
            "False",
            timeout=5.0,
            poll_seconds=10.0,
            sleep=lambda s: sleeps.append(s),
            clock=lambda: next(clocks),
        )
        assert result["met"] is False
        assert sleeps == [3.0]


# ─── timeout clamping ─────────────────────────────────────────────────────────

class TestTimeoutClamping:
    """Tests that ``timeout`` is clamped with ``max(1.0, ...)``.

    Kills:
    * ONE_TO_ZERO @ pos 5940: ``max(1.0, ...)`` → ``max(0.0, ...)``.
    * ZERO_TO_ONE @ pos 5942: ``max(1.0, ...)`` → ``max(1.1, ...)``.
    """

    def test_timeout_clamp_1_0_vs_0_0(self):
        """timeout=0.5 clamped to 1.0 (not 0.0).

        Original: clamped to 1.0 → first eval waited=0.6 < 1.0 → sleep,
        second eval waited=1.2 >= 1.0 → timeout. len(sleeps)==1.
        Mutant max(0.0,...): clamped to 0.5 → waited=0.6 >= 0.5 → timeout
        immediately. len(sleeps)==0.
        """
        t = [0.0]

        def clock():
            t[0] += 0.6
            return t[0]

        sleeps = []
        result = wait_for(
            "False",
            timeout=0.5,
            poll_seconds=5,
            sleep=lambda s: sleeps.append(s),
            clock=clock,
        )
        assert result["met"] is False
        assert "timed out" in result["reason"]
        assert len(sleeps) == 1

    def test_timeout_clamp_1_0_vs_1_1(self):
        """timeout=1.05 clamped to 1.05 (not 1.1).

        Original: max(1.0, 1.05) = 1.05 → waited=1.07 >= 1.05 → timeout
        immediately, 0 sleeps, waited=round(1.07, 1)=1.1.
        Mutant max(1.1,...): max(1.1, 1.05) = 1.1 → waited=1.07 < 1.1 → sleep,
        then waited=1.2 >= 1.1 → timeout, 1 sleep, waited=1.2.
        """
        clocks = iter([0.0, 1.07, 1.2])
        sleeps = []
        result = wait_for(
            "False",
            timeout=1.05,
            poll_seconds=5,
            sleep=lambda s: sleeps.append(s),
            clock=lambda: next(clocks),
        )
        assert result["met"] is False
        assert "timed out" in result["reason"]
        assert len(sleeps) == 0
        assert result["waited"] == 1.1


# ─── poll clamping ────────────────────────────────────────────────────────────

class TestPollClamping:
    """Tests that ``poll_seconds`` is clamped with ``max(0.1, ...)``.

    Kills:
    * ZERO_TO_ONE @ pos 6037: ``max(0.1, ...)`` → ``max(1.1, ...)``.
    """

    def test_poll_clamp_0_1_vs_1_1(self, monkeypatch):
        """poll_seconds=0.5 should sleep 0.5, not 1.1.

        Original: max(0.1, 0.5) = 0.5 → sleeps [0.5].
        Mutant max(1.1,...): max(1.1, 0.5) = 1.1 → sleeps [1.1].
        """
        t = [0.0]

        def clock():
            return t[0]

        sleeps = []

        def sleep(s):
            sleeps.append(s)
            t[0] += s

        call_count = [0]

        def my_evaluate(condition):
            call_count[0] += 1
            return call_count[0] >= 2

        monkeypatch.setattr("src.agent.wait_conditions.evaluate", my_evaluate)
        result = wait_for(
            "dummy",
            timeout=10,
            poll_seconds=0.5,
            sleep=sleep,
            clock=clock,
        )
        assert result["met"] is True
        assert sleeps == [0.5]