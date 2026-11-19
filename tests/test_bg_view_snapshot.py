"""The window on a background task: a copy of the tail, never a withdrawal.

The TUI polls a snapshot several times a second while its window is open, and the
agent may be reading the same process at the same moment. If the window drained the
output queue, opening it would quietly eat what the agent was waiting for.
"""

import queue
import time

from argent.tools import command_ops as ops

PID = "4242"


def _running(monkeypatch, text: bytes, elapsed: float = 95.0):
    out = queue.Queue()
    out.put(text)                      # what the agent has not read yet
    err = queue.Queue()
    tail = ops._OutputTail()
    tail.add(text)
    info = {
        "process": None,
        "out_queue": out,
        "err_queue": err,
        "command": "npm run dev",
        "started": time.time() - elapsed,
        "pty": False,
        "tail": tail,
    }
    monkeypatch.setitem(ops.ACTIVE_PROCESSES, PID, info)
    return out, tail


def _finished(monkeypatch, code=1, stopped=False, output="build failed\n"):
    monkeypatch.setitem(ops.COMPLETED_PROCESSES, PID, {
        "exit_code": code,
        "command": "cargo build",
        "started": time.time() - 40,
        "finished": time.time(),
        "pty": False,
        "output": output,
        **({"stopped": True} if stopped else {}),
    })


def test_the_window_sees_a_running_task(monkeypatch):
    _running(monkeypatch, b"vite ready\nwatching\n")

    view = ops.background_view(PID)

    assert view["pid"] == PID
    assert view["status"] == "running"
    assert view["command"] == "npm run dev"
    assert view["elapsed"] >= 90
    assert view["code"] is None
    assert "vite ready" in view["output"]
    assert view["error"] == ""


def test_reading_the_window_never_takes_the_output_from_the_agent(monkeypatch):
    out, _ = _running(monkeypatch, b"line one\n")

    ops.background_view(PID)
    ops.background_view(PID)

    assert not out.empty(), "очередь агента не должна вычерпываться окном"


def test_a_finished_task_reports_its_code(monkeypatch):
    _finished(monkeypatch, code=1)

    view = ops.background_view(PID)

    assert view["status"] == "done"
    assert view["code"] == 1
    assert view["output"].startswith("build failed")
    assert view["elapsed"] == 40


def test_a_stopped_task_says_it_was_stopped(monkeypatch):
    _finished(monkeypatch, code=-1, stopped=True)

    assert ops.background_view(PID)["status"] == "stopped"


def test_an_unknown_pid_is_an_error_not_a_blank_window(monkeypatch):
    monkeypatch.setitem(ops.ACTIVE_PROCESSES, PID, None)
    monkeypatch.delitem(ops.ACTIVE_PROCESSES, PID, raising=False)

    view = ops.background_view("999999")

    assert view["status"] == "unknown"
    assert "не найдена" in view["error"]


def test_colour_codes_do_not_reach_the_window(monkeypatch):
    _running(monkeypatch, b"\x1b[31mred line\x1b[0m\n")

    assert "\x1b[" not in ops.background_view(PID)["output"]
    assert "red line" in ops.background_view(PID)["output"]


def test_the_window_keeps_the_end_of_a_long_output(monkeypatch):
    _running(monkeypatch, b"x" * 10 + b"\nthe last line\n")

    view = ops.background_view(PID)

    assert view["output"].rstrip().endswith("the last line")
