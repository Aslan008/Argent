"""The bridge's background-completion monitor must not outlive its bridge.

EVENT_QUEUE is process-global and has one consumer at a time: the bridge's
monitor in the TUI, the event sleeper in the REPL. A monitor that could not be
stopped kept taking ``bg_done`` events after its bridge was gone, so whoever
waited on the queue next (a heartbeat sleeper) never woke up. It showed as
three heartbeat tests that failed only in a full run, after enough bridges had
been built to leave a monitor behind.
"""

import queue
import time

import pytest

from src.agent.events import EVENT_QUEUE
from src.bridge.repl_bridge import ReplBridge


def _drain():
    while True:
        try:
            EVENT_QUEUE.get_nowait()
        except queue.Empty:
            return


@pytest.fixture(autouse=True)
def clean_queue():
    _drain()
    yield
    _drain()


def _bridge():
    bridge = ReplBridge()
    bridge.set_event_sink(lambda event: None)
    return bridge


def _monitors():
    import threading
    return [t for t in threading.enumerate() if t.name == "ArgentTuiEventQueueMonitor" and t.is_alive()]


class TestTheMonitorStops:
    def test_stopping_it_ends_the_thread(self):
        bridge = _bridge()
        assert bridge._monitor_thread is not None and bridge._monitor_thread.is_alive()
        bridge.stop_event_monitor(join=True)
        assert not bridge._monitor_thread.is_alive()

    def test_a_stopped_monitor_leaves_the_queue_alone(self):
        bridge = _bridge()
        bridge.stop_event_monitor(join=True)
        EVENT_QUEUE.put({"type": "bg_done", "pid": "77", "exit_code": 0})
        time.sleep(0.8)                      # longer than the monitor's poll
        assert EVENT_QUEUE.get_nowait()["pid"] == "77"

    def test_an_event_caught_while_stopping_goes_back(self):
        """The thread may already be blocked in get() when stop arrives; whatever it
        pulls out then belongs to the next waiter."""
        bridge = _bridge()
        bridge._monitor_stop.set()           # stop WITHOUT joining: the thread is mid-get
        EVENT_QUEUE.put({"type": "bg_done", "pid": "78", "exit_code": 0})
        bridge._monitor_thread.join(timeout=2.0)
        assert not bridge._monitor_thread.is_alive()
        assert EVENT_QUEUE.get(timeout=1)["pid"] == "78"

    def test_cleanup_stops_it(self):
        bridge = _bridge()
        thread = bridge._monitor_thread
        bridge.stop_event_monitor()
        thread.join(timeout=2.0)
        assert not thread.is_alive()


class TestOnlyTheActiveBridgeListens:
    def test_a_superseded_bridge_stops_listening(self):
        first = _bridge()
        second = _bridge()                   # registers itself as the active bridge
        first._monitor_thread.join(timeout=2.0)
        assert not first._monitor_thread.is_alive()
        assert second._monitor_thread.is_alive()
        second.stop_event_monitor(join=True)

    def test_events_reach_the_active_bridge(self):
        seen = []
        bridge = ReplBridge()
        bridge.set_event_sink(seen.append)
        EVENT_QUEUE.put({"type": "bg_done", "pid": "5", "exit_code": 0})
        deadline = time.time() + 3
        while time.time() < deadline and not any("Фоновая команда 5" in e.get("text", "") for e in seen):
            time.sleep(0.05)
        assert any("Фоновая команда 5" in e.get("text", "") for e in seen)
        bridge.stop_event_monitor(join=True)
