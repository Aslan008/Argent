"""Waiting ON a condition instead of on a guessed delay.

Each heartbeat wake costs a full model turn (re-prefill + generation), so a
"sleep 60, look, sleep again" loop spends ten turns to notice one event. A
condition is evaluated locally for free, so the agent wakes once — when the
thing actually happened.
"""

import pytest

from src.agent.wait_conditions import (
    ConditionError, MAX_TIMEOUT_SECONDS, evaluate, wait_for,
)
from tools.misc_tools import wait_heartbeat


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestFilePredicates:
    def test_file_exists(self, project):
        (project / "a.txt").write_text("x", encoding="utf-8")
        assert evaluate('file_exists("a.txt")') is True
        assert evaluate('file_exists("nope.txt")') is False

    def test_file_missing_is_the_inverse(self, project):
        assert evaluate('file_missing("nope.txt")') is True
        (project / "a.txt").write_text("x", encoding="utf-8")
        assert evaluate('file_missing("a.txt")') is False

    def test_file_contains(self, project):
        (project / "build.log").write_text("compiling...\nBUILD SUCCESSFUL\n", encoding="utf-8")
        assert evaluate('file_contains("build.log", "BUILD SUCCESSFUL")') is True
        assert evaluate('file_contains("build.log", "FAILED")') is False

    def test_missing_file_never_contains(self, project):
        assert evaluate('file_contains("absent.log", "anything")') is False

    def test_directory_is_not_a_file(self, project):
        (project / "sub").mkdir()
        assert evaluate('file_exists("sub")') is False


class TestProcessPredicates:
    def _register(self, monkeypatch, poll_value):
        import tools

        class _Proc:
            def poll(self):
                return poll_value

        monkeypatch.setitem(tools.ACTIVE_PROCESSES, "7", {"process": _Proc()})

    def test_running_process(self, monkeypatch):
        self._register(monkeypatch, None)
        assert evaluate('process_finished("7")') is False
        assert evaluate('process_running("7")') is True

    def test_exited_process(self, monkeypatch):
        self._register(monkeypatch, 0)
        assert evaluate('process_finished("7")') is True

    def test_unknown_pid_counts_as_finished(self):
        """read_background_command removes a process once it completes; treating
        'gone' as 'running' would hang the wait forever."""
        assert evaluate('process_finished("does-not-exist")') is True

    def test_numeric_pid_is_accepted(self, monkeypatch):
        self._register(monkeypatch, 0)
        assert evaluate("process_finished(7)") is True


class TestBooleanCombination:
    def test_and_or_not(self, project):
        (project / "a").write_text("x", encoding="utf-8")
        assert evaluate('file_exists("a") and file_missing("b")') is True
        assert evaluate('file_exists("b") or file_exists("a")') is True
        assert evaluate('not file_exists("b")') is True
        assert evaluate('file_exists("a") and file_exists("b")') is False


class TestSafety:
    @pytest.mark.parametrize("expr", [
        "__import__('os').system('echo pwned')",
        "open('secret.txt').read()",
        "command_succeeds('rm -rf /')",     # deliberately not a predicate
        "file_exists.__class__",
        "[x for x in range(3)]",
        "lambda: 1",
        "1 + 1",
    ])
    def test_dangerous_or_unknown_input_is_rejected(self, expr):
        with pytest.raises(ConditionError):
            evaluate(expr)

    def test_no_command_execution_predicate_exists(self):
        """Running a command in a poll loop with nobody watching is exactly the
        authority unattended runs refuse elsewhere, so no predicate may execute
        anything or reach the network."""
        from src.agent.wait_conditions import PREDICATES
        forbidden = {"command_succeeds", "run_command", "shell", "http_ok",
                     "url_responds", "eval"}
        assert not (forbidden & set(PREDICATES))
        # And the allowed set stays small enough to audit at a glance.
        assert set(PREDICATES) == {
            "file_exists", "file_missing", "file_contains",
            "process_finished", "process_running",
        }

    def test_non_literal_arguments_rejected(self, project):
        with pytest.raises(ConditionError):
            evaluate('file_exists(file_missing("x"))')

    def test_empty_condition(self):
        with pytest.raises(ConditionError):
            evaluate("   ")


class _FakeClock:
    """Virtual time: the fake sleep advances it, so a test never actually waits.
    Stubbing sleep alone would leave the deadline on the real clock and turn the
    loop into a busy-wait for the full timeout."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        self.now += seconds


class TestWaitLoop:
    def test_returns_as_soon_as_the_condition_holds(self, project):
        clock = _FakeClock()
        target = project / "done.flag"

        def fake_sleep(seconds):
            clock.sleep(seconds)
            if clock.sleeps == 3:
                target.write_text("ok", encoding="utf-8")

        out = wait_for('file_exists("done.flag")', timeout=100, poll_seconds=1,
                       sleep=fake_sleep, clock=clock)
        assert out["met"] is True and out["reason"] == "condition met"
        assert clock.sleeps == 3          # stopped at the poll that flipped it

    def test_true_immediately_does_not_sleep(self, project):
        (project / "a").write_text("x", encoding="utf-8")
        out = wait_for('file_exists("a")', timeout=100,
                       sleep=lambda s: pytest.fail("must not sleep"))
        assert out["met"] is True

    def test_timeout_is_reported_as_failure(self, project):
        clock = _FakeClock()
        out = wait_for('file_exists("never")', timeout=20, poll_seconds=5,
                       sleep=clock.sleep, clock=clock)
        assert out["met"] is False and "timed out" in out["reason"]
        assert clock.sleeps == 4                      # 20s of virtual waiting

    def test_broken_condition_stops_immediately(self, project):
        """Burning the whole timeout on an expression that can never work — and
        then reporting a plain timeout — would hide the real problem."""
        out = wait_for("nonsense_check('x')", timeout=600,
                       sleep=lambda s: pytest.fail("must not sleep on a bad condition"))
        assert out["met"] is False and "invalid condition" in out["reason"]

    def test_timeout_is_capped(self, project):
        clock = _FakeClock()
        out = wait_for('file_exists("never")', timeout=10 ** 9, poll_seconds=3600,
                       sleep=clock.sleep, clock=clock)
        assert out["met"] is False
        assert clock.now <= MAX_TIMEOUT_SECONDS       # not the requested billion


class TestToolSignal:
    def test_until_emits_a_condition_heartbeat(self, project):
        (project / "build.log").write_text("x", encoding="utf-8")
        out = wait_heartbeat(until='file_contains("build.log", "DONE")',
                             timeout_seconds=120, condition_to_check="build to finish")
        assert "[HEARTBEAT_REQUEST: 0|" in out
        assert 'until=file_contains("build.log", "DONE")' in out
        assert "timeout=120" in out

    def test_plain_delay_still_works(self):
        out = wait_heartbeat(60, "check the server")
        assert out == "Heartbeat scheduled. [HEARTBEAT_REQUEST: 60|check the server]"

    def test_invalid_condition_is_refused_up_front(self):
        out = wait_heartbeat(until="command_succeeds('rm -rf /')")
        assert out.startswith("Error:")
        assert "file_contains" in out          # tells the model what IS available

    def test_neither_delay_nor_condition(self):
        assert wait_heartbeat().startswith("Error:")

    def test_timeout_is_clamped(self, project):
        (project / "a").write_text("x", encoding="utf-8")
        out = wait_heartbeat(until='file_exists("a")', timeout_seconds=10 ** 9)
        assert f"timeout={MAX_TIMEOUT_SECONDS}" in out


class TestSignalParsing:
    def _parse(self, result):
        from src.cli.cli_ui import _check_auto_signals
        return _check_auto_signals(result, True, 0, "")

    def test_condition_form_yields_a_spec(self):
        _, sleep_t, ctx = self._parse(
            'Waiting. [HEARTBEAT_REQUEST: 0|build|until=file_exists("a.exe")|timeout=300]')
        assert sleep_t == 0
        assert ctx == {"reason": "build", "until": 'file_exists("a.exe")', "timeout": 300}

    def test_plain_form_still_yields_text(self):
        _, sleep_t, ctx = self._parse("[HEARTBEAT_REQUEST: 45|check server]")
        assert sleep_t == 45
        assert isinstance(ctx, str) and "check server" in ctx
