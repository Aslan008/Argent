"""Tests for the idle-poll hint in read_background_command and the
loop-guard normalisation of the 'RUNNING for NNs' status line.
"""

import queue
import re

import tools.command_ops as command_ops
from src.agent.loop_guard import LoopGuard, _normalize_volatile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, code=None):
        self._code = code
        self.stdout = None
        self.stderr = None
        self.stdin = None

    def poll(self):
        return self._code


def _register(pid, code=None, started=100.0):
    command_ops.ACTIVE_PROCESSES[pid] = {
        "process": _FakeProc(code),
        "out_queue": queue.Queue(),
        "err_queue": queue.Queue(),
        "command": "sleep 999",
        "started": started,
    }


# ---------------------------------------------------------------------------
# 1. Idle-poll hint appears after IDLE_POLL_HINT_THRESHOLD empty reads
# ---------------------------------------------------------------------------

class TestIdlePollHint:
    def test_hint_appears_after_threshold(self, monkeypatch):
        pid = "hint-1"
        _register(pid, code=None, started=100.0)
        monkeypatch.setattr(command_ops.time, "time", lambda: 110.0)
        try:
            for _ in range(command_ops.IDLE_POLL_HINT_THRESHOLD - 1):
                out = command_ops.read_background_command(pid)
                assert "[Hint]" not in out, "hint should not appear before threshold"

            out = command_ops.read_background_command(pid)
            assert "[Hint]" in out
            assert "wait_heartbeat" in out
            assert "stop_background_command" in out
        finally:
            command_ops.ACTIVE_PROCESSES.pop(pid, None)

    def test_hint_resets_on_new_output(self, monkeypatch):
        pid = "hint-2"
        _register(pid, code=None, started=100.0)
        monkeypatch.setattr(command_ops.time, "time", lambda: 110.0)
        try:
            # Two empty reads (below threshold of 3)
            command_ops.read_background_command(pid)
            command_ops.read_background_command(pid)

            # Inject stdout data — should reset counter
            command_ops.ACTIVE_PROCESSES[pid]["out_queue"].put(b"hello\n")
            out = command_ops.read_background_command(pid)
            assert "hello" in out
            assert "[Hint]" not in out

            # Now the counter is reset — need another 3 empty reads for hint
            command_ops.read_background_command(pid)
            out2 = command_ops.read_background_command(pid)
            assert "[Hint]" not in out2, "counter was reset, should need 3 more"
        finally:
            command_ops.ACTIVE_PROCESSES.pop(pid, None)

    def test_no_hint_for_exited_process(self, monkeypatch):
        """When a process has already exited, the hint is pointless."""
        pid = "hint-3"
        _register(pid, code=0, started=100.0)
        monkeypatch.setattr(command_ops.time, "time", lambda: 110.0)
        # Even though there's no output, the process is done — no hint needed.
        out = command_ops.read_background_command(pid)
        assert "[Hint]" not in out
        assert "EXITED" in out

    def test_hint_includes_pid(self, monkeypatch):
        pid = "hint-4"
        _register(pid, code=None, started=100.0)
        monkeypatch.setattr(command_ops.time, "time", lambda: 110.0)
        try:
            for _ in range(command_ops.IDLE_POLL_HINT_THRESHOLD):
                out = command_ops.read_background_command(pid)
            assert pid in out
        finally:
            command_ops.ACTIVE_PROCESSES.pop(pid, None)

    def test_consecutive_count_grows(self, monkeypatch):
        """The hint should say 'for N consecutive reads' with increasing N."""
        pid = "hint-5"
        _register(pid, code=None, started=100.0)
        monkeypatch.setattr(command_ops.time, "time", lambda: 110.0)
        try:
            for _ in range(command_ops.IDLE_POLL_HINT_THRESHOLD):
                command_ops.read_background_command(pid)

            out4 = command_ops.read_background_command(pid)
            # idle_polls is now threshold+1
            expected_count = command_ops.IDLE_POLL_HINT_THRESHOLD + 1
            assert str(expected_count) in out4
        finally:
            command_ops.ACTIVE_PROCESSES.pop(pid, None)


# ---------------------------------------------------------------------------
# 2. Loop guard now normalises 'RUNNING for NNs'
# ---------------------------------------------------------------------------

class TestLoopGuardElapsedNormalization:
    def test_normalize_collapses_elapsed(self):
        """'RUNNING for 42s' and 'RUNNING for 99s' should normalise identically."""
        a = _normalize_volatile("Process 1 is RUNNING for 42s.")
        b = _normalize_volatile("Process 1 is RUNNING for 99s.")
        assert a == b
        assert "RUNNING for <elapsed>" in a

    def test_normalize_preserves_other_numbers(self):
        """Short numbers outside the 'RUNNING for' pattern are left alone."""
        text = _normalize_volatile("exit code 1, PID 42")
        assert "1" in text
        assert "42" in text

    def test_loop_guard_catches_idle_read_loop(self):
        """Repeated read_background_command calls with 'RUNNING for NNs / No
        new output' should trigger the loop guard despite different elapsed."""
        g = LoopGuard()
        args = {"pid": "7"}

        for elapsed in [10, 15, 22, 30]:
            result = (
                f"--- Process 7 is RUNNING for {elapsed}s. ---\n"
                "No new output.\n"
            )
            level = g.record("read_background_command", args, result)

        assert level == "stop", (
            "loop guard should stop after STOP_REPEATS identical-when-normalised results"
        )

    def test_loop_guard_does_not_catch_different_output(self):
        """When there IS new output, results differ genuinely — no trigger."""
        g = LoopGuard()
        args = {"pid": "8"}

        outputs = ["line 1\n", "line 2\n", "line 3\n", "line 4\n"]
        for i, out_text in enumerate(outputs):
            result = (
                f"--- Process 8 is RUNNING for {i * 5}s. ---\n"
                f"STDOUT:\n{out_text}\n"
            )
            level = g.record("read_background_command", args, result)
            assert level is None, f"false positive at iteration {i}"
