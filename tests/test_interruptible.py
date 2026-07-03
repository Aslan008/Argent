"""interruptible_sleep: chunked sleep that honours Ctrl+C within one tick."""

import pytest

from src.cli.interruptible import interruptible_sleep


def test_sleeps_in_bounded_chunks():
    calls = []
    interruptible_sleep(2.0, tick=0.5, _sleep=calls.append)
    assert calls == [0.5, 0.5, 0.5, 0.5]      # four full ticks
    assert all(c <= 0.5 for c in calls)


def test_partial_final_chunk():
    calls = []
    interruptible_sleep(1.2, tick=0.5, _sleep=calls.append)
    assert calls == [0.5, 0.5, pytest.approx(0.2)]
    assert sum(calls) == pytest.approx(1.2)


def test_total_duration_respected():
    calls = []
    interruptible_sleep(3.0, tick=0.7, _sleep=calls.append)
    assert sum(calls) == pytest.approx(3.0)
    assert all(c <= 0.7 for c in calls)


@pytest.mark.parametrize("total", [0, -1, 0.0])
def test_non_positive_does_not_sleep(total):
    calls = []
    interruptible_sleep(total, tick=0.5, _sleep=calls.append)
    assert calls == []


def test_keyboardinterrupt_propagates_promptly():
    # A Ctrl+C on the 2nd tick must propagate immediately — the caller must NOT
    # keep sleeping through the remaining (here: enormous) duration.
    calls = []

    def fake_sleep(d):
        calls.append(d)
        if len(calls) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interruptible_sleep(10_000, tick=0.5, _sleep=fake_sleep)
    assert len(calls) == 2  # stopped at the interrupt, not after 20000 ticks
