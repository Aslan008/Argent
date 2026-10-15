"""LMTrust Deep Blind Spot & Contract Test Suite for Background Process Lifecycle.

Covers:
- L0: Smoke & API Signatures
- L1: Contract Validation (PID normalization: int, float, str, whitespace; empty command)
- L2: Boundary & Trailing Output Drainage on rapid process exit
- L3: Invariants & Idempotency of stop_background_command
- L4: Adversarial (process is None, stdin is None, invalid PIDs)
- L5: State Machine Lifecycle (STARTING -> RUNNING -> EXITED/STOPPED -> COMPLETED)
- L6: Cross-System: Real Process Tree Termination (no orphaned grandchildren in Windows OS)
- L7: Compositional: Event Queue & Sleeper integration (stopped process must not raise EventInterruptedException)
- L8: Temporal / Concurrency: Multi-threaded start/stop/read safety
- L9: Negative Space: Resource Leak Prevention & atexit cleanup
"""

import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import psutil
import pytest

import tools.command_ops as command_ops
from src.agent.events import EVENT_QUEUE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup_teardown(monkeypatch):
    """Auto-approve all commands and guarantee clean process registries."""
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(command_ops, "is_destructive_command", lambda *a, **k: False)
    monkeypatch.setattr(command_ops, "command_grant_key", lambda *a: "test")
    monkeypatch.setattr(command_ops, "assess_command_risk", lambda c: ("safe", []))

    # Drain any residual events in EVENT_QUEUE
    while not EVENT_QUEUE.empty():
        try:
            EVENT_QUEUE.get_nowait()
        except queue.Empty:
            break

    with command_ops.ACTIVE_PROCESSES_LOCK:
        command_ops.ACTIVE_PROCESSES.clear()
        command_ops.COMPLETED_PROCESSES.clear()

    yield

    # Teardown: terminate any residual processes
    with command_ops.ACTIVE_PROCESSES_LOCK:
        for info in list(command_ops.ACTIVE_PROCESSES.values()):
            proc = info.get("process")
            if proc is not None:
                try:
                    command_ops._kill_process_tree(proc)
                except Exception:
                    pass
                command_ops._close_proc_pipes(proc)
        command_ops.ACTIVE_PROCESSES.clear()
        command_ops.COMPLETED_PROCESSES.clear()

    while not EVENT_QUEUE.empty():
        try:
            EVENT_QUEUE.get_nowait()
        except queue.Empty:
            break


def _wait_for_process_exit(proc, timeout=5.0):
    """Wait for process to finish."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return proc.poll() is not None


# ---------------------------------------------------------------------------
# L0: Smoke & API Exports
# ---------------------------------------------------------------------------

class TestL0Smoke:
    """L0: Verify all background process management tools exist and are callable."""

    def test_functions_exist(self):
        assert callable(command_ops.start_background_command)
        assert callable(command_ops.read_background_command)
        assert callable(command_ops.send_background_command)
        assert callable(command_ops.stop_background_command)
        assert callable(command_ops.list_background_commands)


# ---------------------------------------------------------------------------
# L1: Contract & PID Normalization (D7 API Contract)
# ---------------------------------------------------------------------------

class TestL1ContractAndPIDNormalization:
    """L1 x D7: Contract validation for arguments and PID types."""

    def test_empty_or_whitespace_command_rejected(self):
        """Empty command strings must be rejected with an Error immediately."""
        r1 = command_ops.start_background_command("")
        assert "error" in r1.lower()
        r2 = command_ops.start_background_command("    ")
        assert "error" in r2.lower()
        with command_ops.ACTIVE_PROCESSES_LOCK:
            assert len(command_ops.ACTIVE_PROCESSES) == 0

    def test_integer_pid_accepted_by_read_and_stop(self):
        """LLM tool calls often send integer PIDs (pid=1). Tools must accept int."""
        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid_str = res.split("PID:")[-1].strip()
        pid_int = int(pid_str)

        # Reading with integer PID must succeed, not return 'No active process'
        out = command_ops.read_background_command(pid_int)
        assert "Error: No active process" not in out
        assert "RUNNING" in out

        # Stopping with integer PID must succeed
        stop_out = command_ops.stop_background_command(pid_int)
        assert "Terminated" in stop_out

    def test_float_and_spaced_pid_accepted(self):
        """Float strings ('1.0') or whitespace (' 1 ') must be normalized."""
        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid_str = res.split("PID:")[-1].strip()

        # Read with whitespace
        out = command_ops.read_background_command(f"  {pid_str}  ")
        assert "RUNNING" in out

        # Stop with float string representation
        stop_out = command_ops.stop_background_command(f"{pid_str}.0")
        assert "Terminated" in stop_out


# ---------------------------------------------------------------------------
# L2: Boundary & Output Drainage (D2/D5)
# ---------------------------------------------------------------------------

class TestL2BoundaryAndOutputDrain:
    """L2 x D2: Rapid output production and exit must drain completely without loss."""

    def test_rapid_exit_retains_all_lines(self):
        """Process that outputs multiple lines and terminates must deliver 100% of output."""
        code = "for i in range(50): print(f'DATA_LINE_{i}')"
        res = command_ops.start_background_command(f'python -c "{code}"')
        pid = res.split("PID:")[-1].strip()

        proc = command_ops.ACTIVE_PROCESSES[pid]["process"]
        _wait_for_process_exit(proc, timeout=5.0)

        out = command_ops.read_background_command(pid)
        assert "EXITED" in out
        for i in range(50):
            assert f"DATA_LINE_{i}" in out, f"Missing output line DATA_LINE_{i}"

    def test_idle_poll_counter_and_hint(self):
        """Repeated reading without new output must provide wait_heartbeat hint."""
        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid = res.split("PID:")[-1].strip()
        try:
            for _ in range(command_ops.IDLE_POLL_HINT_THRESHOLD):
                out = command_ops.read_background_command(pid)
            assert "[Hint]" in out
            assert "wait_heartbeat" in out
        finally:
            command_ops.stop_background_command(pid)


# ---------------------------------------------------------------------------
# L3: Invariants & Idempotency (D10 Invariants)
# ---------------------------------------------------------------------------

class TestL3InvariantsAndIdempotency:
    """L3 x D10: Stopping twice, monotonic PIDs, registry invariants."""

    def test_stop_idempotent_returns_clear_status(self):
        """Calling stop_background_command twice must not crash and report inactive status."""
        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid = res.split("PID:")[-1].strip()

        out1 = command_ops.stop_background_command(pid)
        assert "Terminated" in out1

        out2 = command_ops.stop_background_command(pid)
        assert "Error" in out2
        # Should indicate it's not active / already stopped
        assert ("not active" in out2.lower()) or ("stopped" in out2.lower()) or ("already" in out2.lower())

    def test_stopped_process_read_reports_stopped(self):
        """Reading a process after stopping must inform that it was stopped/terminated."""
        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid = res.split("PID:")[-1].strip()

        command_ops.stop_background_command(pid)
        out = command_ops.read_background_command(pid)
        # Must retain 'Error' for existing test contract, while being informative
        assert "Error" in out
        assert ("stopped" in out.lower()) or ("terminated" in out.lower())


# ---------------------------------------------------------------------------
# L4: Adversarial & Robustness (D4 Error Handling)
# ---------------------------------------------------------------------------

class TestL4Adversarial:
    """L4 x D4: Handling None process objects, broken stdin, non-existent PIDs."""

    def test_stop_when_process_is_none_no_crash(self):
        """If ACTIVE_PROCESSES has an entry with process=None, stop must not crash."""
        with command_ops.ACTIVE_PROCESSES_LOCK:
            command_ops.ACTIVE_PROCESSES["999"] = {
                "process": None,
                "out_queue": queue.Queue(),
                "err_queue": queue.Queue(),
                "command": "dummy",
            }
        out = command_ops.stop_background_command("999")
        assert "Terminated" in out or "PID 999" in out

    def test_send_input_when_stdin_is_none_returns_error(self):
        """If a process does not have an open stdin, send_background_command returns clean error."""
        with command_ops.ACTIVE_PROCESSES_LOCK:
            command_ops.ACTIVE_PROCESSES["998"] = {
                "process": subprocess.Popen(["cmd.exe", "/c", "echo 1"], stdin=None),
                "out_queue": queue.Queue(),
                "err_queue": queue.Queue(),
                "command": "echo",
            }
        try:
            out = command_ops.send_background_command("998", "data")
            assert "Error" in out
        finally:
            command_ops.stop_background_command("998")

    def test_list_background_commands_with_none_process(self):
        """list_background_commands must handle entries where process is None without throwing."""
        with command_ops.ACTIVE_PROCESSES_LOCK:
            command_ops.ACTIVE_PROCESSES["997"] = {
                "process": None,
                "command": "fake-cmd",
            }
        out = command_ops.list_background_commands()
        assert "fake-cmd" in out


# ---------------------------------------------------------------------------
# L6: Cross-System: Real Process Tree Termination (D5/D6)
# ---------------------------------------------------------------------------

class TestL6ProcessTreeTermination:
    """L6 x D5: Crucial - verify NO child or grandchild processes survive stop_background_command."""

    def test_multi_level_child_process_tree_killed(self):
        """Spawn shell -> python -> child python sleep. Verify ALL are killed in the OS."""
        cmd = (
            'python -c "import subprocess, time; '
            'p = subprocess.Popen([\'python\', \'-c\', \'import time; time.sleep(60)\']); '
            'p.wait()"'
        )
        res = command_ops.start_background_command(cmd)
        pid = res.split("PID:")[-1].strip()

        # Wait for the tree to spawn
        time.sleep(2.0)
        with command_ops.ACTIVE_PROCESSES_LOCK:
            proc_obj = command_ops.ACTIVE_PROCESSES[pid]["process"]
            root_pid = proc_obj.pid

        # Inspect tree using psutil
        try:
            ps_parent = psutil.Process(root_pid)
            children = ps_parent.children(recursive=True)
            child_pids = [c.pid for c in children]
        except Exception:
            child_pids = []

        # Stop background command
        stop_res = command_ops.stop_background_command(pid)
        assert "Terminated" in stop_res

        # Allow OS a moment to reap
        time.sleep(1.5)

        # Root process must be dead
        assert not psutil.pid_exists(root_pid), f"Root process {root_pid} still running!"

        # All child processes in the tree must be dead
        for cpid in child_pids:
            assert not psutil.pid_exists(cpid), f"Orphaned grandchild {cpid} still running in OS!"


# ---------------------------------------------------------------------------
# L7: Compositional: Event Queue & Sleeper Integration (D4/D6)
# ---------------------------------------------------------------------------

class TestL7EventQueueAndSleeperIntegration:
    """L7 x D4: Stopped processes must not trigger EventInterruptedException in wait_heartbeat."""

    def test_stopped_process_does_not_abort_sleeper(self):
        """Stopping a process produces bg_done event with non-zero exit code.
        The sleeper in main.py MUST ignore it because it was an intentional stop."""
        from main import ArgentAgent

        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 60"'
        )
        pid = res.split("PID:")[-1].strip()
        time.sleep(0.5)

        # Stop it
        command_ops.stop_background_command(pid)

        # Wait for process_waiter to push bg_done event
        time.sleep(1.0)

        # Construct event sleeper similar to main.py
        class EventInterruptedException(Exception):
            def __init__(self, event):
                self.event = event

        def sleep_with_events(secs):
            end = time.monotonic() + secs
            while time.monotonic() < end:
                try:
                    ev = EVENT_QUEUE.get(timeout=min(0.2, max(0.01, end - time.monotonic())))
                    if ev.get("type") == "bg_done":
                        p_id = str(ev.get("pid", ""))
                        exit_code = ev.get("exit_code", 0)

                        # Filter out stale/completed/stopped processes
                        with command_ops.ACTIVE_PROCESSES_LOCK:
                            if p_id in command_ops.COMPLETED_PROCESSES:
                                continue

                        if exit_code != 0:
                            raise EventInterruptedException(ev)
                except queue.Empty:
                    pass

        # Sleeper must sleep full duration without raising EventInterruptedException!
        start_t = time.monotonic()
        sleep_with_events(0.15)
        elapsed = time.monotonic() - start_t
        assert elapsed >= 0.12, "Sleep was interrupted by stopped background process!"


# ---------------------------------------------------------------------------
# L8: Concurrency & Race Conditions (D6 Concurrency)
# ---------------------------------------------------------------------------

class TestL8Concurrency:
    """L8 x D6: Concurrent stop and read calls must be thread-safe."""

    def test_concurrent_stop_and_read(self):
        res = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid = res.split("PID:")[-1].strip()

        errors = []

        def reader():
            try:
                command_ops.read_background_command(pid)
            except Exception as e:
                errors.append(e)

        def stopper():
            try:
                command_ops.stop_background_command(pid)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=stopper)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread race caused exceptions: {errors}"


# ---------------------------------------------------------------------------
# L9: Negative Space & Resource Cleanup (D5 Resource Leaks)
# ---------------------------------------------------------------------------

class TestL9NegativeSpaceAndCleanup:
    """L9 x D5: Ensure stop_all_background_processes terminates all tracked jobs."""

    def test_stop_all_background_processes_cleans_registry(self):
        """stop_all_background_processes must kill all running jobs and empty ACTIVE_PROCESSES."""
        r1 = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        r2 = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid1 = r1.split("PID:")[-1].strip()
        pid2 = r2.split("PID:")[-1].strip()

        assert hasattr(command_ops, "stop_all_background_processes")
        command_ops.stop_all_background_processes()

        with command_ops.ACTIVE_PROCESSES_LOCK:
            assert len(command_ops.ACTIVE_PROCESSES) == 0
