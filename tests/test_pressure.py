"""The pressure monitor: is the PC busy with something other than Argent?"""

import threading

from src.agent.pressure import PressureMonitor, Reading, Thresholds


class Script:
    """A reader that plays back a list of readings (the last one repeats)."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.calls = 0

    def __call__(self):
        i = min(self.calls, len(self.readings) - 1)
        self.calls += 1
        return self.readings[i]


def cpu(x):
    return Reading(cpu_external=x)


def test_one_busy_sample_is_not_pressure():
    mon = PressureMonitor(reader=Script(cpu(95), cpu(10)))
    assert mon.sample() is False
    assert mon.sample() is False


def test_sustained_cpu_trips_the_verdict():
    mon = PressureMonitor(Thresholds(hold_samples=3), reader=Script(cpu(95)))
    assert [mon.sample() for _ in range(3)] == [False, False, True]
    assert "процессор" in mon.reason


def test_the_verdict_has_hysteresis():
    mon = PressureMonitor(Thresholds(hold_samples=1, cpu_high=85, cpu_low=60),
                          reader=Script(cpu(90), cpu(75), cpu(70), cpu(40)))
    assert mon.sample() is True            # 90 trips
    assert mon.sample() is True            # 75: below the trip level, above the release level
    assert mon.sample() is True            # 70
    assert mon.sample() is False           # 40: well below


def test_low_ram_trips_at_once():
    mon = PressureMonitor(reader=Script(Reading(ram_free_gb=0.8)))
    assert mon.sample() is True
    assert "памяти" in mon.reason


def test_low_vram_trips_at_once():
    mon = PressureMonitor(reader=Script(Reading(vram_free_mb=100)))
    assert mon.sample() is True
    assert "видеопамяти" in mon.reason


def test_unknown_vram_is_not_pressure():
    mon = PressureMonitor(reader=Script(Reading(vram_free_mb=None)))
    assert mon.sample() is False


def test_wait_until_calm_returns_when_the_load_goes_away():
    mon = PressureMonitor(Thresholds(hold_samples=1),
                          reader=Script(cpu(99), cpu(99), cpu(10)))
    told = []
    slept = []
    assert mon.wait_until_calm(on_wait=told.append, sleep=slept.append) is True
    assert len(told) == 1, "the person is told once, not on every poll"
    assert len(slept) >= 1


def test_wait_until_calm_gives_up_on_cancel():
    mon = PressureMonitor(Thresholds(hold_samples=1), reader=Script(cpu(99)))
    cancel = threading.Event()
    cancel.set()
    assert mon.wait_until_calm(cancel=cancel, sleep=lambda s: None) is False


def test_wait_until_calm_gives_up_on_timeout():
    now = [0.0]

    def tick(_):
        now[0] += 5.0

    mon = PressureMonitor(Thresholds(hold_samples=1), reader=Script(cpu(99)), clock=lambda: now[0])
    assert mon.wait_until_calm(timeout=12, sleep=tick) is False
