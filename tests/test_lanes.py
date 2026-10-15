"""Lanes: one queue per local model server, shared by every Argent process on the machine."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.agent import lanes
from src.agent.lanes import (Lane, LaneCancelled, P_ADVISOR, P_BACKGROUND, P_INTERACTIVE, P_WORKER)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lane(tmp_path):
    return Lane("test-lane", capacity=1, root=tmp_path)


def _in_thread(fn):
    out = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as e:           # noqa: BLE001 — surfaced to the test
            out["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_a_free_lane_is_taken_at_once(lane):
    lease = lane.acquire(P_INTERACTIVE, "me")
    assert lease.holds()
    assert not lease.preempted()
    lease.release()
    assert lane.snapshot() == []


def test_the_second_request_waits_for_the_first(lane):
    first = lane.acquire(P_WORKER, "first")
    t, out = _in_thread(lambda: lane.acquire(P_WORKER, "second"))
    time.sleep(0.4)
    assert t.is_alive(), "the second request must not get the lane while the first holds it"
    first.release()
    t.join(5)
    assert out["value"].holds()
    out["value"].release()


def test_better_priority_goes_first_among_the_waiting(lane):
    holder = lane.acquire(P_INTERACTIVE, "holder")
    order = []

    def want(prio, name):
        with lane.acquire(prio, name):
            order.append(name)

    background, _ = _in_thread(lambda: want(P_BACKGROUND, "background"))
    time.sleep(0.15)
    person, _ = _in_thread(lambda: want(P_INTERACTIVE, "person"))
    time.sleep(0.3)
    holder.release()
    background.join(5)
    person.join(5)
    assert order == ["person", "background"]


def test_a_running_worker_learns_that_the_person_arrived(lane):
    worker = lane.acquire(P_WORKER, "worker")
    assert not worker.preempted()
    t, out = _in_thread(lambda: lane.acquire(P_INTERACTIVE, "person"))
    time.sleep(0.3)
    assert worker.preempted(), "a better request sorts ahead of the running worker"
    worker.release()
    t.join(5)
    assert out["value"].holds()
    out["value"].release()


def test_a_pinned_lease_is_not_preempted(lane):
    worker = lane.acquire(P_WORKER, "worker")
    worker.pin()
    assert worker.pinned
    t, out = _in_thread(lambda: lane.acquire(P_INTERACTIVE, "person"))
    time.sleep(0.3)
    assert worker.holds() and not worker.preempted()
    assert t.is_alive()
    worker.release()
    t.join(5)
    out["value"].release()


def test_capacity_two_lets_two_through(tmp_path):
    lane = Lane("cap2", capacity=2, root=tmp_path)
    a = lane.acquire(P_WORKER, "a")
    b = lane.acquire(P_WORKER, "b")
    assert a.holds() and b.holds()
    t, out = _in_thread(lambda: lane.acquire(P_WORKER, "c"))
    time.sleep(0.3)
    assert t.is_alive()
    a.release()
    t.join(5)
    assert out["value"].holds()
    for lease in (b, out["value"]):
        lease.release()


def test_a_cancelled_wait_leaves_no_ticket(lane):
    holder = lane.acquire(P_INTERACTIVE, "holder")
    cancel = threading.Event()
    t, out = _in_thread(lambda: lane.acquire(P_WORKER, "waiter", cancel=cancel))
    time.sleep(0.25)
    cancel.set()
    t.join(5)
    assert isinstance(out.get("error"), LaneCancelled)
    assert [x["label"] for x in lane.snapshot()] == ["holder"]
    holder.release()


def test_timeout_raises(lane):
    holder = lane.acquire(P_INTERACTIVE, "holder")
    with pytest.raises(LaneCancelled):
        lane.acquire(P_WORKER, "late", timeout=0.3)
    holder.release()


def test_a_ticket_of_a_dead_process_is_swept(tmp_path):
    folder = tmp_path / "dead"
    folder.mkdir()
    ghost = folder / f"12-{time.time_ns():020d}-4242-deadbeef.ticket"
    ghost.write_text("ghost")
    lane = Lane("dead", capacity=1, root=tmp_path, pid_alive=lambda pid: pid != 4242)
    lease = lane.acquire(P_WORKER, "me", timeout=2)
    assert lease.holds()
    assert not ghost.exists()
    lease.release()


def test_a_ticket_nobody_touches_any_more_is_swept(tmp_path):
    folder = tmp_path / "stale"
    folder.mkdir()
    old = folder / f"12-{time.time_ns():020d}-4243-deadbeef.ticket"
    old.write_text("old")
    long_ago = time.time() - 600
    os.utime(old, (long_ago, long_ago))
    lane = Lane("stale", capacity=1, root=tmp_path, pid_alive=lambda pid: True, stale_after=20)
    with lane.acquire(P_WORKER, "me", timeout=2) as lease:
        assert lease.holds()
    assert not old.exists()


def test_snapshot_lists_holder_and_waiters_in_order(lane):
    holder = lane.acquire(P_INTERACTIVE, "main model")
    t, _ = _in_thread(lambda: lane.acquire(P_WORKER, "worker 1"))
    time.sleep(0.3)
    snap = lane.snapshot()
    assert [s["label"] for s in snap] == ["main model", "worker 1"]
    assert snap[0]["holding"] and not snap[1]["holding"]
    assert snap[0]["name"] == "interactive" and snap[1]["name"] == "worker"
    holder.release()
    t.join(5)


def test_the_priority_numbers_are_ordered():
    assert P_INTERACTIVE < P_ADVISOR < P_WORKER < P_BACKGROUND


def test_two_processes_share_a_lane(tmp_path, monkeypatch):
    """The point of file tickets: a sub-agent PROCESS queues behind the main agent's process."""
    monkeypatch.setenv("ARGENT_LANES_DIR", str(tmp_path))
    script = (
        "import sys, time\n"
        "from src.agent.lanes import Lane\n"
        "l = Lane('shared', capacity=1)\n"
        "lease = l.acquire(2, 'child')\n"
        "print('held', flush=True)\n"
        "time.sleep(1.2)\n"
        "lease.release()\n"
    )
    env = dict(os.environ, ARGENT_LANES_DIR=str(tmp_path), PYTHONPATH=str(ROOT))
    child = subprocess.Popen([sys.executable, "-c", script], cwd=str(ROOT), env=env,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        started = time.monotonic()
        with Lane("shared", capacity=1).acquire(P_WORKER, "parent", timeout=15) as lease:
            waited = time.monotonic() - started
            assert lease.holds()
        assert waited > 0.5, "the parent must have waited for the child's ticket"
    finally:
        child.wait(10)


@pytest.mark.parametrize("clock", [lambda: 1_000_000_000, lambda c=iter(range(10**12, 0, -1)): next(c)],
                         ids=["frozen", "steps back"])
def test_a_newcomer_never_sorts_ahead_of_a_request_that_is_running(tmp_path, monkeypatch, clock):
    """A coarse wall clock gives two arrivals one number (Windows ticks every ~15 ms) and a clock may step back;
    neither may put a newcomer in front of the holder: the request would run beside it."""
    monkeypatch.setattr(lanes, "_wall_ns", clock)
    lane = Lane("clock-lane", capacity=1, root=tmp_path)
    for _ in range(15):                      # a random tie-break would fail one of these almost surely
        holder = lane.acquire(P_INTERACTIVE, "holder")
        with pytest.raises(LaneCancelled):
            lane.acquire(P_INTERACTIVE, "newcomer", timeout=0.15)
        assert holder.holds()
        holder.release()
