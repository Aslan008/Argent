"""The governor: provider calls wait in the model server's lane, workers yield to the person."""

import threading
import time

import pytest

from src.agent import governor, lanes, pressure
from src.agent.lanes import P_ADVISOR, P_BACKGROUND, P_INTERACTIVE, P_WORKER
from src.agent.local_backend import CLOUD, Backend

LOCAL = Backend(True, "ollama", "localhost", 11434)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGENT_LANES_DIR", str(tmp_path / "lanes"))
    monkeypatch.delenv("ARGENT_LANE_PRIORITY", raising=False)
    monkeypatch.delenv("ARGENT_GOVERNOR", raising=False)
    lanes.forget_lanes()
    monkeypatch.setattr(governor, "classify", lambda provider, model="": LOCAL)
    # a calm PC; tests that want pressure install their own monitor
    calm = type("Calm", (), {"wait_until_calm": staticmethod(lambda **kw: True)})()
    monkeypatch.setattr(pressure, "monitor", lambda: calm)
    yield
    lanes.forget_lanes()


class FakeProvider:
    name = "ollama"

    def __init__(self, chunks=5, delay=0.0):
        self.chunks, self.delay = chunks, delay
        self.calls = 0
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None, format_schema=None):
        with self._lock:
            self.calls += 1
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            for i in range(self.chunks):
                if self.delay:
                    time.sleep(self.delay)
                yield {"content": str(i)}
        finally:
            with self._lock:
                self.running -= 1

    def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None):
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            time.sleep(self.delay)
            return "answer"
        finally:
            with self._lock:
                self.running -= 1


def tickets():
    return [t for lane in lanes.lanes_root().glob("*") for t in lane.glob("*.ticket")]


def test_priority_defaults_to_the_person(monkeypatch):
    assert governor.current_priority() == P_INTERACTIVE
    monkeypatch.setenv("ARGENT_LANE_PRIORITY", "2")
    assert governor.current_priority() == P_WORKER


def test_a_context_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("ARGENT_LANE_PRIORITY", "2")
    with governor.priority(P_BACKGROUND):
        assert governor.current_priority() == P_BACKGROUND
    assert governor.current_priority() == P_WORKER


def test_capped_never_improves_a_priority(monkeypatch):
    with governor.capped(P_ADVISOR):
        assert governor.current_priority() == P_ADVISOR
    monkeypatch.setenv("ARGENT_LANE_PRIORITY", "2")
    with governor.capped(P_ADVISOR):
        assert governor.current_priority() == P_WORKER, "a worker's own critic is still a worker"


def test_cloud_requests_do_not_touch_the_lanes(monkeypatch):
    monkeypatch.setattr(governor, "classify", lambda provider, model="": CLOUD)
    p = governor.govern(FakeProvider())
    seen = []
    for chunk in p.stream_chat("m", []):
        seen.append(len(tickets()))
    assert seen == [0] * 5
    assert p.sync_chat("m", []) == "answer"


def test_the_ticket_is_held_while_streaming_and_released_after():
    p = governor.govern(FakeProvider())
    seen = []
    for _ in p.stream_chat("m", []):
        seen.append(len(tickets()))
    assert seen == [1] * 5
    assert tickets() == []


def test_the_ticket_is_released_when_the_consumer_stops_early():
    p = governor.govern(FakeProvider())
    gen = p.stream_chat("m", [])
    next(gen)
    assert len(tickets()) == 1
    gen.close()
    assert tickets() == []


def test_two_requests_never_run_on_the_server_together():
    provider = FakeProvider(chunks=4, delay=0.05)
    governor.govern(provider)

    def one():
        list(provider.stream_chat("m", []))

    threads = [threading.Thread(target=one) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert provider.calls == 3
    assert provider.max_running == 1


def test_sync_calls_share_the_lane_with_streams():
    provider = FakeProvider(chunks=4, delay=0.05)
    governor.govern(provider)
    t = threading.Thread(target=lambda: list(provider.stream_chat("m", [])))
    t.start()
    time.sleep(0.08)
    assert provider.sync_chat("m", []) == "answer"
    t.join(10)
    assert provider.max_running == 1


def test_a_call_made_from_inside_a_call_does_not_wait_for_itself():
    class Nesting(FakeProvider):
        def stream_chat(self, *a, **kw):
            self.sync_chat("m", [])                      # goes through the governed instance attribute
            yield from super().stream_chat(*a, **kw)

    p = governor.govern(Nesting(chunks=2))
    result = []
    t = threading.Thread(target=lambda: result.extend(p.stream_chat("m", [])))
    t.start()
    t.join(5)
    assert not t.is_alive(), "deadlock: the nested call waited for the lane its own request holds"
    assert len(result) == 2


def test_govern_is_idempotent():
    p = FakeProvider()
    governor.govern(p)
    first = p.stream_chat
    governor.govern(p)
    assert p.stream_chat is first


def test_the_switch_turns_it_all_off(monkeypatch):
    monkeypatch.setenv("ARGENT_GOVERNOR", "0")
    p = governor.govern(FakeProvider())
    seen = [len(tickets()) for _ in p.stream_chat("m", [])]
    assert seen == [0] * 5


def test_a_worker_gets_its_answer_whole():
    p = governor.govern(FakeProvider(chunks=4))
    with governor.priority(P_WORKER):
        out = list(p.stream_chat("m", []))
    assert [c["content"] for c in out] == ["0", "1", "2", "3"]
    assert tickets() == []


def test_a_worker_waits_for_a_calm_pc_but_the_person_does_not(monkeypatch):
    waited = []
    monkeypatch.setattr(governor, "_wait_for_calm", lambda prio: waited.append(prio))
    p = governor.govern(FakeProvider(chunks=1))
    list(p.stream_chat("m", []))
    with governor.priority(P_WORKER):
        list(p.stream_chat("m", []))
        p.sync_chat("m", [])
    assert waited == [P_WORKER, P_WORKER], waited


def test_the_pressure_gate_ignores_the_person(monkeypatch):
    calls = []
    busy = type("Busy", (), {"wait_until_calm": staticmethod(lambda **kw: calls.append(1) or True)})()
    monkeypatch.setattr(pressure, "monitor", lambda: busy)
    governor._wait_for_calm(P_INTERACTIVE)
    governor._wait_for_calm(P_ADVISOR)
    assert calls == []
    governor._wait_for_calm(P_WORKER)
    assert calls == [1]


def test_a_worker_is_dropped_for_the_person_and_makes_its_request_again():
    provider = FakeProvider(chunks=30, delay=0.03)
    governor.govern(provider)
    result = {}

    def worker():
        with governor.priority(P_WORKER):
            result["chunks"] = list(provider.stream_chat("m", []))

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.3)                                       # the worker is streaming
    lane = governor.lane_for(LOCAL)
    started = time.monotonic()
    with lane.acquire(P_INTERACTIVE, "person"):
        waited = time.monotonic() - started
        time.sleep(0.2)
    t.join(20)

    assert waited < 0.6, f"the person waited {waited:.2f}s for a worker's generation to end"
    assert provider.calls == 2, "the dropped request is made again"
    assert [c["content"] for c in result["chunks"]] == [str(i) for i in range(30)], \
        "nothing of the dropped attempt reaches the caller"


def test_a_pinned_worker_is_allowed_to_finish(monkeypatch):
    monkeypatch.setattr(governor, "MAX_PREEMPTS", 0)
    provider = FakeProvider(chunks=12, delay=0.03)
    governor.govern(provider)
    result = {}

    def worker():
        with governor.priority(P_WORKER):
            result["chunks"] = list(provider.stream_chat("m", []))

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.15)
    with governor.lane_for(LOCAL).acquire(P_INTERACTIVE, "person"):
        pass
    t.join(20)
    assert provider.calls == 1
    assert len(result["chunks"]) == 12


def test_a_worker_process_is_run_below_normal_priority():
    import subprocess
    import sys
    psutil = pytest.importorskip("psutil")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        before = psutil.Process(child.pid).nice()
        assert governor.lower_process_priority(child.pid) is True
        assert psutil.Process(child.pid).nice() != before
    finally:
        child.kill()
        child.wait(5)
    assert governor.lower_process_priority(child.pid) is False, "a process that is gone is not an error"


def test_server_options_are_off_unless_asked(monkeypatch):
    import config
    monkeypatch.setattr(governor, "classify", lambda provider, model="": LOCAL)
    assert governor.server_options("ollama", "m") == {}
    monkeypatch.setattr(config, "is_limit_server_threads_enabled", lambda: True)
    opts = governor.server_options("ollama", "m")
    assert opts.get("num_thread", 0) >= 1
    monkeypatch.setattr(governor, "classify", lambda provider, model="": CLOUD)
    assert governor.server_options("ollama", "m:cloud") == {}


def test_slot_serialises_hand_made_requests():
    order = []

    def use(name, hold):
        with governor.slot(LOCAL, name, P_WORKER):
            order.append(f"{name}+")
            time.sleep(hold)
            order.append(f"{name}-")

    a = threading.Thread(target=use, args=("a", 0.3))
    a.start()
    time.sleep(0.1)
    b = threading.Thread(target=use, args=("b", 0.0))
    b.start()
    a.join(10)
    b.join(10)
    assert order == ["a+", "a-", "b+", "b-"]


def test_a_broken_lane_never_fails_the_request(monkeypatch):
    """The queue is a courtesy to the PC; an unwritable folder must not turn into a failed model call."""
    def broken(self, *a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(lanes.Lane, "acquire", broken)
    p = governor.govern(FakeProvider(chunks=3))
    assert [c["content"] for c in p.stream_chat("m", [])] == ["0", "1", "2"]
    assert p.sync_chat("m", []) == "answer"
    with governor.priority(P_WORKER):
        assert len(list(p.stream_chat("m", []))) == 3
    with governor.slot(LOCAL, "x"):
        pass
