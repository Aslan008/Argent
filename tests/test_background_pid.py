"""The handle given for a background job is the process's own OS PID.

It used to be `str(_pid_counter)` — 1, 2, 3 — so the number in "Started background
process with PID: 2" existed only inside Argent: neither `Get-Process`, nor Task
Manager, nor psutil could find it, and nothing outside could stop the job. These
tests start a real, live process and check the number against the OS.
"""

import re
import time

import psutil
import pytest

from argent.tools.command_ops import (
    ACTIVE_PROCESSES,
    ACTIVE_PROCESSES_LOCK,
    list_background_commands,
    read_background_command,
    start_background_command,
    stop_background_command,
)

# Long enough that it is still running when the assertions look at it.
SLEEP = 'python -c "import time; time.sleep(30)"'


def _start(command: str = SLEEP) -> str:
    started = start_background_command(command)
    assert "Started background process with PID:" in started, started
    match = re.search(r"PID:\s*(\d+)", started)
    assert match, f"no pid in the reply: {started!r}"
    return match.group(1)


def _wait_gone(pid: int, seconds: float = 8.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not psutil.pid_exists(pid):
            return True
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.Error:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture(autouse=True)
def auto_approve(monkeypatch):
    """Nobody is at the keyboard to answer the gate: the same device the other
    command tests use, so the working code keeps its approval step intact."""
    monkeypatch.setattr("argent.tools.command_ops.request_approval", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _no_leftovers():
    yield
    with ACTIVE_PROCESSES_LOCK:
        pids = list(ACTIVE_PROCESSES.keys())
    for pid in pids:
        try:
            stop_background_command(pid)
        except Exception:
            pass


def test_the_handle_is_a_pid_the_operating_system_knows():
    pid = _start()

    assert pid.isdigit(), f"the handle must be a number, got {pid!r}"
    assert psutil.pid_exists(int(pid)), f"no process {pid} in the OS process table"
    assert psutil.Process(int(pid)).is_running()


def test_stopping_it_by_that_pid_really_ends_the_process():
    pid = _start()

    said = stop_background_command(pid)

    assert "Terminated" in said, said
    assert _wait_gone(int(pid)), f"process {pid} is still alive after being stopped"


def test_a_pid_given_as_an_int_finds_the_same_job():
    pid = _start()

    # Callers read the number out of a message or a panel row and may pass it as
    # an int; the registry has to answer either way.
    output = read_background_command(int(pid))

    assert "No active process" not in output, output
    assert f"PID {pid}" in output or "no output" in output.lower() or output.strip()


def test_the_job_list_shows_the_real_pid():
    pid = _start()

    listed = list_background_commands()

    assert f"PID {pid}" in listed, listed


def test_two_jobs_get_two_different_pids():
    first = _start()
    second = _start()

    assert first != second
    assert psutil.pid_exists(int(first)) and psutil.pid_exists(int(second))
