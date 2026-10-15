"""Non-blocking waits: wait_heartbeat registers a watch and returns; a thread wakes the frontend."""

import os
import subprocess
import sys
import threading
import time

import pytest

from src.agent import watches as W
from src.agent.watches import WatchRegistry


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def reg():
    clock = Clock()
    r = WatchRegistry(poll_seconds=0.01, clock=clock)
    r.clock = clock
    fired = []
    r.set_handler(lambda message, watch: fired.append((message, watch)))
    r.fired = fired
    yield r
    r.set_handler(None)


def test_condition_fires_once_when_met(reg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg._thread = object()                 # drive it by hand, no thread
    reg.add("build", until='file_exists("done.txt")', timeout=60)
    assert reg.check_once() == 0 and reg.fired == []
    (tmp_path / "done.txt").write_text("x")
    reg.clock.now += 3
    assert reg.check_once() == 1
    (message, watch), = reg.fired
    assert "Условие выполнено" in message and 'file_exists("done.txt")' in message and watch.reason == "build"
    assert reg.check_once() == 0 and reg.pending() == []


def test_condition_timeout_is_not_reported_as_success(reg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg._thread = object()
    reg.add("x", until='file_exists("never")', timeout=30)
    reg.clock.now += 31
    assert reg.check_once() == 1
    message = reg.fired[0][0]
    assert "НЕ выполнено" in message and "timed out" in message


def test_timer_wakes_after_its_delay(reg):
    reg._thread = object()
    reg.add("check the server", delay=60)
    reg.clock.now += 59
    assert reg.check_once() == 0
    reg.clock.now += 2
    assert reg.check_once() == 1
    assert reg.fired[0][0] == "[Heartbeat пробуждение] Причина: check the server"


def test_timer_is_clamped_so_it_cannot_be_polling(reg):
    reg._thread = object()
    w = reg.add("x", delay=1)
    assert w.timeout == W.MIN_TIMER_SECONDS


def test_timer_wakes_early_when_a_background_command_finishes(reg):
    from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES["77"] = {"process": proc}
        reg._thread = object()
        reg.add("wait", delay=600)
        assert reg.check_once() == 0
        proc.kill()
        proc.wait()
        assert reg.check_once() == 1
        assert "Фоновая команда 77 завершилась" in reg.fired[0][0]
    finally:
        proc.kill()
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES.pop("77", None)


def test_malformed_condition_is_rejected_at_registration(reg):
    with pytest.raises(Exception):
        reg.add("x", until="rm_rf('/')")
    assert reg.pending() == []


def test_too_many_watches(reg):
    reg._thread = object()
    for _ in range(W.MAX_WATCHES):
        reg.add("x", delay=60)
    with pytest.raises(ValueError):
        reg.add("x", delay=60)


def test_cancel(reg):
    reg._thread = object()
    a = reg.add("a", delay=60)
    reg.add("b", delay=60)
    assert reg.cancel(a.id) == 1 and len(reg.pending()) == 1
    assert reg.cancel() == 1 and reg.pending() == []


def test_a_watch_cancelled_while_deciding_does_not_fire(reg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg._thread = object()
    (tmp_path / "f").write_text("x")
    reg.add("x", until='file_exists("f")')
    original = reg._verdict
    def verdict_then_cancel(watch, now):
        message = original(watch, now)
        reg.cancel()
        return message
    reg._verdict = verdict_then_cancel
    reg.check_once()
    assert reg.fired == []


def test_a_handler_that_raises_does_not_kill_the_watcher(reg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg._thread = object()
    (tmp_path / "f").write_text("x")
    reg.set_handler(lambda m, w: 1 / 0)
    reg.add("a", until='file_exists("f")')
    reg.add("b", until='file_exists("f")')
    assert reg.check_once() == 2


def test_the_real_thread_wakes_the_handler(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = WatchRegistry(poll_seconds=0.05)
    got = threading.Event()
    r.set_handler(lambda m, w: got.set())
    r.add("x", until='file_exists("go")', timeout=30)
    (tmp_path / "go").write_text("1")
    assert got.wait(5)
    time.sleep(0.2)
    assert r._thread is None               # the thread retires when nothing is left to watch
    got.clear()
    r.add("y", until='file_exists("go")')  # ...and a later watch starts a new one
    assert got.wait(5)
    r.set_handler(None)


# -- the tool -------------------------------------------------------------------

def test_tool_keeps_the_blocking_protocol_without_a_frontend():
    from tools.misc_tools import wait_heartbeat
    assert "[HEARTBEAT_REQUEST:" in wait_heartbeat(until='file_exists("a")')
    assert "[HEARTBEAT_REQUEST: 30|" in wait_heartbeat(delay_seconds=30, condition_to_check="x")


def test_tool_registers_a_watch_when_a_frontend_can_wake_the_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tools.misc_tools import wait_heartbeat
    W.WATCHES.set_handler(lambda m, w: None)
    try:
        out = wait_heartbeat(until='file_exists("a")', condition_to_check="build", timeout_seconds=90)
        assert "[HEARTBEAT_REQUEST" not in out            # the agent loop must NOT end the turn
        assert "end your turn" in out and "[watch #" in out
        (watch,) = W.WATCHES.pending()
        assert watch.reason == "build" and watch.timeout == 90
        out = wait_heartbeat(delay_seconds=120, condition_to_check="later")
        assert "[HEARTBEAT_REQUEST" not in out and len(W.WATCHES.pending()) == 2
        assert wait_heartbeat(until="nonsense(").startswith("Error")      # still validated
        assert wait_heartbeat().startswith("Error")
    finally:
        W.WATCHES.set_handler(None)


# -- the bridge ------------------------------------------------------------------

class TestBridgeWakes:
    @pytest.fixture
    def bridge(self):
        from src.bridge.repl_bridge import ReplBridge
        b = ReplBridge()
        b.events, b.started = [], []
        b._event_sink = b.events.append
        b.start_turn = lambda text, _wake=False: b.started.append((text, _wake))
        return b

    def fire(self, bridge, message="[Heartbeat пробуждение] x", watch_id=1):
        bridge._on_watch_fired(message, W.Watch(id=watch_id, reason="r", until='file_exists("a")'))

    def test_idle_agent_is_woken_at_once(self, bridge):
        self.fire(bridge)
        assert bridge.started == [("[Heartbeat пробуждение] x", True)]
        assert any("Ожидание #1" in e.get("text", "") for e in bridge.events)

    def test_busy_agent_is_woken_when_its_turn_ends(self, bridge):
        bridge.session._turn_running = True
        self.fire(bridge, "first", 1)
        self.fire(bridge, "second", 2)
        assert bridge.started == []
        bridge._drain_wakes()                          # still running: nothing happens
        assert bridge.started == []
        bridge.session._turn_running = False
        bridge._drain_wakes()
        assert bridge.started == [("first\n\nsecond", True)]
        bridge._drain_wakes()
        assert len(bridge.started) == 1                # delivered once

    def test_a_turn_a_person_started_first_gets_the_wake_back_in_the_queue(self, bridge):
        def rejecting(text, _wake=False):
            bridge.turn_rejected = True
        bridge.start_turn = rejecting
        bridge.turn_rejected = False
        bridge._deliver_wakes(["m"])
        assert bridge._pending_wakes == ["m"]

    def test_chained_wakes_are_capped_until_a_person_speaks(self, bridge):
        from src.bridge import repl_bridge
        bridge._wakes_in_a_row = repl_bridge.AUTO_MAX_WAKES
        self.fire(bridge)
        assert bridge.started == []
        assert any("остановлено" in e.get("text", "") for e in bridge.events)

    def test_a_superseded_or_closed_bridge_stays_quiet(self, bridge):
        from src.bridge.repl_bridge import ReplBridge
        bridge._shutting_down = True
        self.fire(bridge)
        bridge._shutting_down = False
        ReplBridge()                                   # a newer bridge took over
        self.fire(bridge)
        assert bridge.started == []

    def test_attaching_a_frontend_enables_non_blocking_waits_and_cleanup_disables_them(self):
        from src.bridge.repl_bridge import ReplBridge
        b = ReplBridge()
        assert not W.async_enabled()
        b.set_event_sink(lambda e: None)
        assert W.async_enabled()
        b.cleanup()
        assert not W.async_enabled()

    def test_state_reports_pending_waits(self, bridge):
        W.WATCHES.set_handler(bridge._on_watch_fired)
        W.WATCHES.add("x", delay=60)
        assert bridge.get_state()["waits"] == 1
