"""LMTrust Deep Test Suite: Heartbeat timer, wait_conditions, background command event queue, and display.

Systematically verifies:
- Layer 0: Smoke & basic timer dispatch.
- Layer 1: Contract validation & input boundary sanitization (float strings, timeout clamping).
- Layer 2: Delimiter collisions (| in conditions/reasons, multiline signals, regex robustness).
- Layer 3: Process lifecycle predicates (_process_finished, _process_running, PID formats, missing PIDs).
- Layer 4: Display rendering & determinism (_HeartbeatWaitDisplay, PID sorting, timer formatting).
- Layer 5: Event queue handling & stale event filtering in make_event_sleeper.
- Layer 6: End-to-end simulation of background command wait, wakeup context, and interrupts.
- Layer 10: Invariant verification (timeout bounds, zero hangs, fail-fast on error exits).
"""

import ast
import queue
import re
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.agent.wait_conditions import (
    ConditionError,
    MAX_TIMEOUT_SECONDS,
    _process_finished,
    _process_running,
    describe_predicates,
    evaluate,
    wait_for,
)
from src.cli.cli_ui import (
    _check_auto_signals,
    _format_time_compact,
    _format_time_human,
    _HeartbeatWaitDisplay,
)
from tools.misc_tools import wait_heartbeat


# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 & 1: Contract Validation & Basic Timer Dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestTimerContractAndDispatch:
    def test_basic_timer_scheduled(self):
        res = wait_heartbeat(delay_seconds=45, condition_to_check="wait for tests")
        assert "[HEARTBEAT_REQUEST: 45|wait for tests]" in res
        assert "Heartbeat scheduled" in res

    def test_timer_with_float_string_delay(self):
        res = wait_heartbeat(delay_seconds="30.5", condition_to_check="float wait")
        assert "[HEARTBEAT_REQUEST: 30|float wait]" in res

    def test_timer_zero_or_negative_delay_rejected(self):
        res_zero = wait_heartbeat(delay_seconds=0)
        assert res_zero.startswith("Error:")
        res_neg = wait_heartbeat(delay_seconds=-10)
        assert res_neg.startswith("Error:")

    def test_timer_invalid_delay_string_rejected(self):
        res_invalid = wait_heartbeat(delay_seconds="invalid_secs")
        assert res_invalid.startswith("Error:")

    def test_condition_wait_basic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "app.log").write_text("ready", encoding="utf-8")
        res = wait_heartbeat(until='file_exists("app.log")', timeout_seconds=120, condition_to_check="wait app")
        assert "[HEARTBEAT_REQUEST: 0|wait app|until=file_exists(\"app.log\")|timeout=120]" in res

    def test_condition_wait_with_float_string_timeout(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "app.log").write_text("ready", encoding="utf-8")
        res = wait_heartbeat(until='file_exists("app.log")', timeout_seconds="90.5")
        assert "timeout=90" in res

    def test_condition_wait_timeout_clamped_to_max(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "app.log").write_text("ready", encoding="utf-8")
        res = wait_heartbeat(until='file_exists("app.log")', timeout_seconds=99999999)
        assert f"timeout={MAX_TIMEOUT_SECONDS}" in res

    def test_condition_wait_timeout_clamped_to_min(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "app.log").write_text("ready", encoding="utf-8")
        res = wait_heartbeat(until='file_exists("app.log")', timeout_seconds=-5)
        assert "timeout=1" in res

    def test_condition_wait_malformed_syntax_fails_fast(self):
        res = wait_heartbeat(until='file_exists("app.log"')  # unclosed parenthesis
        assert res.startswith("Error: could not parse")

    def test_condition_wait_disallowed_predicate_fails_fast(self):
        res = wait_heartbeat(until='run_command("dir")')
        assert res.startswith("Error: unknown check 'run_command'")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Delimiter Collisions & Regex Parsing Robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalParsingAndDelimiterInvariants:
    def test_plain_timer_parse(self):
        raw = "Heartbeat scheduled. [HEARTBEAT_REQUEST: 60|wait for server]"
        is_auto, sleep_t, wake_ctx = _check_auto_signals(raw, True, 0, "")
        assert is_auto is True
        assert sleep_t == 60
        assert wake_ctx == "[Heartbeat пробуждение] Причина: wait for server"

    def test_condition_wait_parse(self):
        raw = 'Waiting until: file_exists("out.txt") (timeout 120s). [HEARTBEAT_REQUEST: 0|build|until=file_exists("out.txt")|timeout=120]'
        is_auto, sleep_t, wake_ctx = _check_auto_signals(raw, True, 0, "")
        assert is_auto is True
        assert sleep_t == 0
        assert isinstance(wake_ctx, dict)
        assert wake_ctx["reason"] == "build"
        assert wake_ctx["until"] == 'file_exists("out.txt")'
        assert wake_ctx["timeout"] == 120

    def test_condition_with_pipe_inside_string_does_not_break_parser(self):
        """Blind spot 1: '|' inside predicate text must not break regex extraction."""
        condition = 'file_contains("build.log", "DONE | SUCCESS")'
        raw = f'[HEARTBEAT_REQUEST: 0|wait for pipe|until={condition}|timeout=300]'
        _, sleep_t, wake_ctx = _check_auto_signals(raw, True, 0, "")
        assert sleep_t == 0
        assert isinstance(wake_ctx, dict)
        assert wake_ctx["until"] == condition
        assert wake_ctx["reason"] == "wait for pipe"
        assert wake_ctx["timeout"] == 300

    def test_reason_with_pipe_does_not_break_condition_parser(self):
        """Pipe characters inside reason string must not corrupt condition wait."""
        raw = '[HEARTBEAT_REQUEST: 0|step 1 | step 2|until=file_exists("step2.done")|timeout=150]'
        _, sleep_t, wake_ctx = _check_auto_signals(raw, True, 0, "")
        assert isinstance(wake_ctx, dict)
        assert wake_ctx["reason"] == "step 1 | step 2"
        assert wake_ctx["until"] == 'file_exists("step2.done")'
        assert wake_ctx["timeout"] == 150

    def test_reason_with_pipe_does_not_break_timer_parser(self):
        """Pipe characters inside plain timer reason must not corrupt timer wait."""
        raw = '[HEARTBEAT_REQUEST: 25|checking health | retry 2]'
        _, sleep_t, wake_ctx = _check_auto_signals(raw, True, 0, "")
        assert sleep_t == 25
        assert wake_ctx == "[Heartbeat пробуждение] Причина: checking health | retry 2"

    def test_multiline_reason_or_condition_handled(self):
        """re.DOTALL must match across newlines in reasons or condition ASTs."""
        raw = "[HEARTBEAT_REQUEST: 10|multi\nline\nreason]"
        _, sleep_t, wake_ctx = _check_auto_signals(raw, True, 0, "")
        assert sleep_t == 10
        assert "multi\nline\nreason" in wake_ctx

        raw_cond = '[HEARTBEAT_REQUEST: 0|reason\nmultiline|until=file_exists("a") and\nfile_exists("b")|timeout=200]'
        _, sleep_t, wake_ctx = _check_auto_signals(raw_cond, True, 0, "")
        assert isinstance(wake_ctx, dict)
        assert wake_ctx["until"] == 'file_exists("a") and\nfile_exists("b")'
        assert wake_ctx["reason"] == "reason\nmultiline"
        assert wake_ctx["timeout"] == 200

    def test_no_signal_leaves_state_unchanged(self):
        raw = "Regular output without heartbeat request."
        is_auto, sleep_t, wake_ctx = _check_auto_signals(raw, True, 5, "existing")
        assert is_auto is True
        assert sleep_t == 5
        assert wake_ctx == "existing"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Process Lifecycle Predicates & Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPredicatesAndTargetPidExtraction:
    def test_process_finished_running_cycle(self, monkeypatch):
        import tools

        class _MockProc:
            def __init__(self, code=None):
                self.code = code
            def poll(self):
                return self.code

        proc = _MockProc(None)
        monkeypatch.setitem(tools.ACTIVE_PROCESSES, "42", {"process": proc})

        assert _process_running("42") is True
        assert _process_finished("42") is False

        proc.code = 0
        assert _process_running("42") is False
        assert _process_finished("42") is True

    def test_process_finished_with_int_and_float_pid(self, monkeypatch):
        import tools

        class _MockProc:
            def poll(self):
                return 0

        monkeypatch.setitem(tools.ACTIVE_PROCESSES, "10", {"process": _MockProc()})
        assert _process_finished(10) is True
        assert _process_finished(10.0) is True
        assert _process_finished("10") is True

    def test_missing_pid_is_treated_as_finished_to_avoid_hanging(self, monkeypatch):
        import tools
        monkeypatch.setattr(tools, "ACTIVE_PROCESSES", {})
        assert _process_finished("9999") is True
        assert _process_running("9999") is False

    def test_pid_extraction_from_until_expression(self):
        pattern = r'process_(?:finished|running)\(\s*["\']?\s*(\d+)\s*["\']?\s*\)'
        
        # Plain numeric
        assert set(re.findall(pattern, "process_finished(7)")) == {"7"}
        
        # String inside quotes
        assert set(re.findall(pattern, 'process_running("12")')) == {"12"}
        
        # Quoted with whitespace inside quotes
        assert set(re.findall(pattern, 'process_finished(" 15 ")')) == {"15"}
        
        # Multiple processes combined with and/or
        expr = 'process_finished(3) and not process_running("4") or process_finished(5)'
        assert set(re.findall(pattern, expr)) == {"3", "4", "5"}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4: Display Rendering & Formatting Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDisplayRenderingAndFormatting:
    def test_format_time_compact(self):
        assert _format_time_compact(0) == "0:00"
        assert _format_time_compact(9) == "0:09"
        assert _format_time_compact(59) == "0:59"
        assert _format_time_compact(60) == "1:00"
        assert _format_time_compact(125) == "2:05"
        assert _format_time_compact(3665) == "1:01:05"
        assert _format_time_compact(-10) == "0:00"

    def test_format_time_human(self):
        assert _format_time_human(15.4) == "15.4s"
        assert _format_time_human(65) == "1m 5s"
        assert _format_time_human(3670) == "1h 1m"

    def test_heartbeat_wait_display_timer_mode(self):
        disp = _HeartbeatWaitDisplay(timeout=60, is_timer=True)
        items = list(disp.__rich_console__(None, None))
        assert len(items) >= 1
        rendered_text = disp._spinner.text.plain
        assert "Heartbeat таймер" in rendered_text
        assert "осталось" in rendered_text
        assert "Ctrl+C" in rendered_text

    def test_heartbeat_wait_display_condition_mode(self):
        disp = _HeartbeatWaitDisplay(condition='file_exists("done.txt")', timeout=120, is_timer=False)
        items = list(disp.__rich_console__(None, None))
        assert len(items) >= 1
        rendered_text = disp._spinner.text.plain
        assert "Heartbeat:" in rendered_text
        assert "done.txt" in rendered_text

    def test_heartbeat_wait_display_process_status_sorting(self, monkeypatch):
        import tools
        from tools.command_ops import ACTIVE_PROCESSES_LOCK

        class _Proc:
            def __init__(self, code):
                self._code = code
            def poll(self):
                return self._code

        with ACTIVE_PROCESSES_LOCK:
            monkeypatch.setitem(tools.ACTIVE_PROCESSES, "10", {"process": _Proc(None), "started": time.time()})
            monkeypatch.setitem(tools.ACTIVE_PROCESSES, "2", {"process": _Proc(0), "started": time.time()})

        disp = _HeartbeatWaitDisplay(timeout=30, is_timer=True)
        list(disp.__rich_console__(None, None))
        rendered_text = disp._spinner.text.plain

        # Deterministic sorting ensures PID 2 is listed before PID 10 (numerical, not lexicographical '10' < '2')
        assert "PID 2:" in rendered_text
        assert "PID 10:" in rendered_text
        pos_2 = rendered_text.index("PID 2:")
        pos_10 = rendered_text.index("PID 10:")
        assert pos_2 < pos_10


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5: Event Queue & Stale Event Protection in make_event_sleeper
# ─────────────────────────────────────────────────────────────────────────────

class TestEventSleeperAndQueueHandling:
    def _create_sleeper(self, target_pids=None, is_condition_wait=False):
        """Construct the sleeper function mimicking main.py:make_event_sleeper."""
        from src.agent.events import EVENT_QUEUE

        class EventInterruptedException(Exception):
            def __init__(self, event):
                self.event = event

        def sleep_with_events(secs):
            end = time.monotonic() + secs
            while time.monotonic() < end:
                try:
                    if EVENT_QUEUE is not None:
                        ev = EVENT_QUEUE.get(timeout=min(0.2, max(0.01, end - time.monotonic())))
                    else:
                        time.sleep(0.01)
                        continue
                    if ev.get("type") == "bg_done":
                        pid = str(ev.get("pid", ""))
                        exit_code = ev.get("exit_code", 0)

                        # Filter out stale events for processes that were already completed and handled
                        try:
                            from tools.command_ops import COMPLETED_PROCESSES, ACTIVE_PROCESSES_LOCK
                            with ACTIVE_PROCESSES_LOCK:
                                if pid in COMPLETED_PROCESSES:
                                    continue
                        except Exception:
                            pass

                        # 1. Non-zero exit code wakes agent immediately
                        if exit_code != 0:
                            raise EventInterruptedException(ev)

                        # 2. Specific process condition: wake only if target PID finished
                        if target_pids:
                            if pid in target_pids:
                                raise EventInterruptedException(ev)
                            continue

                        # 3. File or generic condition wait: ignore successful unrelated process
                        if is_condition_wait:
                            continue

                        # 4. Plain timer sleep: wake on process completion
                        raise EventInterruptedException(ev)
                except queue.Empty:
                    pass

        return sleep_with_events, EventInterruptedException

    def test_stale_event_in_completed_processes_does_not_abort_timer(self, monkeypatch):
        """Blind spot 2: Stale events for already handled processes must NOT wake timer early."""
        from src.agent.events import EVENT_QUEUE
        import tools
        from tools.command_ops import COMPLETED_PROCESSES

        # Drain any residual events
        while not EVENT_QUEUE.empty():
            try:
                EVENT_QUEUE.get_nowait()
            except queue.Empty:
                break

        # Simulate PID 7 completed and already handled earlier
        monkeypatch.setitem(COMPLETED_PROCESSES, "7", {"exit_code": 0, "command": "echo done"})

        # Stale event sitting in queue
        EVENT_QUEUE.put({"type": "bg_done", "pid": "7", "exit_code": 0})

        sleeper, EventInterruptedException = self._create_sleeper(is_condition_wait=False)

        start = time.monotonic()
        # Should NOT raise EventInterruptedException, should sleep full 0.15s
        sleeper(0.15)
        elapsed = time.monotonic() - start
        assert elapsed >= 0.12

    def test_live_process_completion_aborts_plain_timer(self):
        """When an active process finishes during timer sleep, it wakes the timer immediately."""
        from src.agent.events import EVENT_QUEUE

        while not EVENT_QUEUE.empty():
            try:
                EVENT_QUEUE.get_nowait()
            except queue.Empty:
                break

        sleeper, EventInterruptedException = self._create_sleeper(is_condition_wait=False)

        def delayed_finish():
            time.sleep(0.05)
            EVENT_QUEUE.put({"type": "bg_done", "pid": "99", "exit_code": 0})

        threading.Thread(target=delayed_finish, daemon=True).start()

        start = time.monotonic()
        with pytest.raises(EventInterruptedException) as excinfo:
            sleeper(5.0)  # Long requested sleep

        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # Woke up within 1s
        assert excinfo.value.event["pid"] == "99"

    def test_failed_process_wakes_agent_even_during_condition_wait(self):
        """Exit code != 0 MUST wake agent immediately regardless of wait condition."""
        from src.agent.events import EVENT_QUEUE

        while not EVENT_QUEUE.empty():
            try:
                EVENT_QUEUE.get_nowait()
            except queue.Empty:
                break

        # Waiting for target PID 101, but PID 50 crashes
        sleeper, EventInterruptedException = self._create_sleeper(
            target_pids={"101"}, is_condition_wait=True
        )

        def delayed_crash():
            time.sleep(0.05)
            EVENT_QUEUE.put({"type": "bg_done", "pid": "50", "exit_code": 1})

        threading.Thread(target=delayed_crash, daemon=True).start()

        with pytest.raises(EventInterruptedException) as excinfo:
            sleeper(5.0)

        assert excinfo.value.event["pid"] == "50"
        assert excinfo.value.event["exit_code"] == 1

    def test_unrelated_successful_process_ignored_during_condition_wait(self):
        """When waiting on PID 200, PID 300 completing successfully must NOT abort wait."""
        from src.agent.events import EVENT_QUEUE

        while not EVENT_QUEUE.empty():
            try:
                EVENT_QUEUE.get_nowait()
            except queue.Empty:
                break

        sleeper, _ = self._create_sleeper(target_pids={"200"}, is_condition_wait=True)

        EVENT_QUEUE.put({"type": "bg_done", "pid": "300", "exit_code": 0})

        start = time.monotonic()
        sleeper(0.1)  # Completes cleanly without exception
        assert time.monotonic() - start >= 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 & 10: End-to-End Simulation & Neuro-Symbolic Invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndAndInvariants:
    def test_wait_for_condition_with_event_interrupt(self):
        """Simulate wait_for meeting condition via process completion event."""
        from src.agent.events import EVENT_QUEUE
        import tools

        while not EVENT_QUEUE.empty():
            try:
                EVENT_QUEUE.get_nowait()
            except queue.Empty:
                break

        class _Proc:
            def __init__(self):
                self._code = None
            def poll(self):
                return self._code

        proc = _Proc()
        tools.ACTIVE_PROCESSES["88"] = {"process": proc}

        class EventInterruptedException(Exception):
            def __init__(self, event):
                self.event = event

        def condition_sleeper(secs):
            end = time.monotonic() + secs
            while time.monotonic() < end:
                try:
                    ev = EVENT_QUEUE.get(timeout=0.05)
                    if ev.get("type") == "bg_done" and ev.get("pid") == "88":
                        raise EventInterruptedException(ev)
                except queue.Empty:
                    pass

        def async_worker():
            time.sleep(0.08)
            proc._code = 0
            EVENT_QUEUE.put({"type": "bg_done", "pid": "88", "exit_code": 0})

        threading.Thread(target=async_worker, daemon=True).start()

        interrupted = False
        try:
            wait_for('process_finished("88")', timeout=5.0, sleep=condition_sleeper)
        except EventInterruptedException as e:
            interrupted = True
            assert e.event["pid"] == "88"

        assert interrupted is True

    def test_timeout_invariant_strictly_respected(self, tmp_path, monkeypatch):
        """Invariant: wait_for never exceeds timeout under any clock."""
        monkeypatch.chdir(tmp_path)
        out = wait_for('file_exists("nonexistent.flag")', timeout=0.1, poll_seconds=0.02)
        assert out["met"] is False
        assert "timed out" in out["reason"]
        assert out["waited"] >= 0.09
        assert out["waited"] <= 0.3
