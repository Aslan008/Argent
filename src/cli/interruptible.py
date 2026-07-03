"""Chunked, promptly-interruptible sleep.

A single long ``time.sleep`` can swallow a Ctrl+C for its whole duration on
Windows, where delivering SIGINT into a blocking C-level wait isn't guaranteed
to be prompt. Sleeping in short ticks bounds the worst-case interrupt latency to
one tick: KeyboardInterrupt is raised at the next tick boundary instead of after
the entire duration. Used for the auto-mode heartbeat sleep, which can be very
long (minutes/hours) yet must stay abortable.
"""

import time


def interruptible_sleep(total_seconds: float, tick: float = 0.5,
                        _sleep=time.sleep) -> None:
    """Sleep ``total_seconds`` in ``tick``-sized chunks so a Ctrl+C is honoured
    within ~one tick rather than after the whole duration.

    KeyboardInterrupt propagates to the caller unchanged (same semantics as a
    plain ``time.sleep``), just sooner. ``_sleep`` is injectable for testing.
    """
    if total_seconds <= 0:
        return
    remaining = float(total_seconds)
    while remaining > 0:
        _sleep(min(tick, remaining))
        remaining -= tick
