"""Releasing a finished process's pipes must not block the thread that reads it.

`_close_proc_pipes` calls `close()` on the child's pipes. On Windows that waits
while somebody is still inside `read()` on them — exactly the case when a grandchild
briefly outlives the process we just killed. `read_background_command` and
`_drain_briefly` called it inline, so a step of the agent (and the auto-reap path,
which held `ACTIVE_PROCESSES_LOCK`) could hang there. Only
`stop_background_command` did it right.
"""

import queue
import subprocess
import sys
import threading
import time

import pytest

import argent.tools.command_ops as co


@pytest.fixture(autouse=True)
def clean_registry():
    with co.ACTIVE_PROCESSES_LOCK:
        co.ACTIVE_PROCESSES.clear()
        co.COMPLETED_PROCESSES.clear()
    yield
    with co.ACTIVE_PROCESSES_LOCK:
        co.ACTIVE_PROCESSES.clear()
        co.COMPLETED_PROCESSES.clear()


def exited_process():
    proc = subprocess.Popen([sys.executable, "-c", "print(1)"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE)
    proc.wait(timeout=30)
    return proc


class TestClosingDoesNotBlockTheCaller:
    def test_the_helper_returns_before_close_finishes(self, monkeypatch):
        started, release = threading.Event(), threading.Event()

        def stuck_close(process):
            started.set()
            release.wait(10)

        monkeypatch.setattr(co, "_close_proc_pipes", stuck_close)
        begin = time.monotonic()
        co._close_proc_pipes_async(object())
        assert time.monotonic() - begin < 0.5, "the caller waited for close()"
        assert started.wait(5), "close() never ran"
        release.set()

    def test_reading_an_exited_process_closes_off_the_calling_thread(self, monkeypatch):
        seen = []
        real = co._close_proc_pipes

        def record(process):
            seen.append(threading.current_thread())
            real(process)

        monkeypatch.setattr(co, "_close_proc_pipes", record)
        with co.ACTIVE_PROCESSES_LOCK:
            co.ACTIVE_PROCESSES["4242"] = {
                "process": exited_process(),
                "out_queue": queue.Queue(),
                "err_queue": queue.Queue(),
                "command": "python -c print(1)",
            }

        co.read_background_command("4242")

        assert seen, "the pipes were never closed"
        assert all(t is not threading.main_thread() for t in seen), \
            "close() ran on the caller's thread"
