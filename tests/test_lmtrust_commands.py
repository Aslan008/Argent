"""LMTrust blind-spot tests for tools/command_ops.py background command tools.

Covers start_background_command, read_background_command,
stop_background_command, list_background_commands, and send_background_command.

Each test class is named per the LMTrust layer convention.  The approval gate
is monkeypatched so the tests run fully unattended.
"""

import queue
import subprocess
import threading
import time

import pytest

import tools.command_ops as command_ops


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _auto_approve(monkeypatch):
    """Bypass the interactive approval gate for every test in this module."""
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(command_ops, "is_destructive_command", lambda *a, **k: False)
    monkeypatch.setattr(command_ops, "command_grant_key", lambda *a: "test")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure the global process registry starts and ends empty."""
    with command_ops.ACTIVE_PROCESSES_LOCK:
        command_ops.ACTIVE_PROCESSES.clear()
    yield
    # Close any leftover process pipes before clearing the registry.
    with command_ops.ACTIVE_PROCESSES_LOCK:
        for info in command_ops.ACTIVE_PROCESSES.values():
            proc = info.get("process")
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                command_ops._close_proc_pipes(proc)
        command_ops.ACTIVE_PROCESSES.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_exit(proc, timeout=5):
    """Block until *proc* exits or *timeout* seconds elapse."""
    try:
        proc.wait(timeout=timeout)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# L1 — start_background_command
# ---------------------------------------------------------------------------

class TestStartBackgroundCommand:
    """L1×D1 — launching a command returns a PID string."""

    def test_simple_echo_returns_pid(self):
        result = command_ops.start_background_command('echo hello')
        assert "PID" in result
        # The PID is a numeric string at the end of the message.
        pid = result.split("PID:")[-1].strip()
        assert pid.isdigit()

    def test_pid_increments(self):
        r1 = command_ops.start_background_command('echo one')
        r2 = command_ops.start_background_command('echo two')
        pid1 = int(r1.split("PID:")[-1].strip())
        pid2 = int(r2.split("PID:")[-1].strip())
        assert pid2 > pid1

    def test_process_registered_in_active_processes(self):
        result = command_ops.start_background_command('echo registered')
        pid = result.split("PID:")[-1].strip()
        with command_ops.ACTIVE_PROCESSES_LOCK:
            assert pid in command_ops.ACTIVE_PROCESSES
            assert command_ops.ACTIVE_PROCESSES[pid]["command"] == 'echo registered'

    def test_aborted_when_not_approved(self, monkeypatch):
        monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: False)
        result = command_ops.start_background_command('echo blocked')
        assert "aborted" in result.lower()
        with command_ops.ACTIVE_PROCESSES_LOCK:
            assert len(command_ops.ACTIVE_PROCESSES) == 0


# ---------------------------------------------------------------------------
# L2 — read_background_command
# ---------------------------------------------------------------------------

class TestReadBackgroundCommand:
    """L2×D2 — reading output from a background process."""

    def test_read_valid_pid_returns_output(self):
        result = command_ops.start_background_command('echo hello-output')
        pid = result.split("PID:")[-1].strip()
        # Give the process a moment to produce output.
        proc = command_ops.ACTIVE_PROCESSES[pid]["process"]
        _wait_for_exit(proc, timeout=5)
        # Small delay for the reader threads to drain the pipe into the queue.
        time.sleep(0.3)
        out = command_ops.read_background_command(pid)
        assert "hello-output" in out

    def test_read_invalid_pid_errors(self):
        out = command_ops.read_background_command("99999")
        assert "Error" in out
        assert "99999" in out

    def test_read_running_process_no_output_says_no_new_output(self):
        # A long-running process that has not printed anything yet.
        result = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid = result.split("PID:")[-1].strip()
        try:
            out = command_ops.read_background_command(pid)
            assert "RUNNING" in out
            assert "No new output" in out
        finally:
            command_ops.stop_background_command(pid)

    def test_read_after_exit_reports_exit_code_and_cleans_up(self):
        result = command_ops.start_background_command('echo done')
        pid = result.split("PID:")[-1].strip()
        proc = command_ops.ACTIVE_PROCESSES[pid]["process"]
        _wait_for_exit(proc, timeout=5)
        time.sleep(0.3)
        out = command_ops.read_background_command(pid)
        assert "EXITED" in out
        # After reading an exited process it must be removed from the registry.
        with command_ops.ACTIVE_PROCESSES_LOCK:
            assert pid not in command_ops.ACTIVE_PROCESSES


# ---------------------------------------------------------------------------
# L3 — stop_background_command
# ---------------------------------------------------------------------------

class TestStopBackgroundCommand:
    """L3×D2 — terminating background processes."""

    def test_stop_invalid_pid_errors(self):
        out = command_ops.stop_background_command("99999")
        assert "Error" in out
        assert "99999" in out

    def test_stop_valid_pid_terminates(self):
        result = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 60"'
        )
        pid = result.split("PID:")[-1].strip()
        out = command_ops.stop_background_command(pid)
        assert "Terminated" in out
        with command_ops.ACTIVE_PROCESSES_LOCK:
            assert pid not in command_ops.ACTIVE_PROCESSES


# ---------------------------------------------------------------------------
# L4 — list_background_commands
# ---------------------------------------------------------------------------

class TestListBackgroundCommands:
    """L4×D2 — listing tracked processes."""

    def test_no_processes_says_none(self):
        out = command_ops.list_background_commands()
        assert "No background processes" in out

    def test_active_processes_listed(self):
        r1 = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        r2 = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid1 = r1.split("PID:")[-1].strip()
        pid2 = r2.split("PID:")[-1].strip()
        try:
            out = command_ops.list_background_commands()
            assert "Background processes" in out
            assert f"PID {pid1}" in out
            assert f"PID {pid2}" in out
            assert "RUNNING" in out
        finally:
            command_ops.stop_background_command(pid1)
            command_ops.stop_background_command(pid2)

    def test_list_mentions_companion_tools(self):
        r = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 10"'
        )
        pid = r.split("PID:")[-1].strip()
        try:
            out = command_ops.list_background_commands()
            assert "read_background_command" in out
            assert "send_background_command" in out
            assert "stop_background_command" in out
        finally:
            command_ops.stop_background_command(pid)


# ---------------------------------------------------------------------------
# L5 — send_background_command
# ---------------------------------------------------------------------------

class TestSendBackgroundCommand:
    """L5×D2 — sending input to a background process stdin."""

    def test_send_to_invalid_pid_errors(self):
        out = command_ops.send_background_command("99999", "hello")
        assert "Error" in out
        assert "99999" in out

    def test_send_to_exited_process_errors(self):
        result = command_ops.start_background_command('echo quick')
        pid = result.split("PID:")[-1].strip()
        proc = command_ops.ACTIVE_PROCESSES[pid]["process"]
        _wait_for_exit(proc, timeout=5)
        out = command_ops.send_background_command(pid, "late input")
        # The process has exited — either it was already reaped by read, or
        # send reports the "already exited" error.
        assert ("Error" in out) or ("exited" in out.lower())


# ---------------------------------------------------------------------------
# L6 — multi-process & lifecycle
# ---------------------------------------------------------------------------

class TestMultiProcessAndLifecycle:
    """L6×D3 — starting multiple commands and full lifecycle transitions."""

    def test_start_multiple_list_shows_all(self):
        starts = [
            command_ops.start_background_command(
                'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
            )
            for _ in range(3)
        ]
        pids = [r.split("PID:")[-1].strip() for r in starts]
        try:
            out = command_ops.list_background_commands()
            for pid in pids:
                assert f"PID {pid}" in out
        finally:
            for pid in pids:
                command_ops.stop_background_command(pid)

    def test_lifecycle_start_read_stop_read(self):
        # start
        result = command_ops.start_background_command(
            'powershell -NoProfile -Command "Start-Sleep -Seconds 30"'
        )
        pid = result.split("PID:")[-1].strip()

        # read while running
        out1 = command_ops.read_background_command(pid)
        assert "RUNNING" in out1

        # stop
        out_stop = command_ops.stop_background_command(pid)
        assert "Terminated" in out_stop

        # read again — process is gone from the registry
        out2 = command_ops.read_background_command(pid)
        assert "Error" in out2

    def test_max_processes_reached_errors(self, monkeypatch):
        """When ACTIVE_PROCESSES is full, start must refuse with an error."""
        # Fill the registry with fake entries up to MAX_BACKGROUND_PROCESSES.
        with command_ops.ACTIVE_PROCESSES_LOCK:
            for i in range(command_ops.MAX_BACKGROUND_PROCESSES):
                command_ops.ACTIVE_PROCESSES[str(i)] = {
                    "process": None,
                    "out_queue": queue.Queue(),
                    "err_queue": queue.Queue(),
                    "command": f"fake-{i}",
                    "started": time.time(),
                }
        try:
            result = command_ops.start_background_command('echo overflow')
            assert "Error" in result
            assert "Maximum" in result or "maximum" in result.lower()
        finally:
            with command_ops.ACTIVE_PROCESSES_LOCK:
                command_ops.ACTIVE_PROCESSES.clear()