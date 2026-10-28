"""How recently a person touched the interface — the one thing vibe asks.

Vibe used to refuse a dangerous action outright. That is right for an unattended
run and wrong for one being watched, and the difference is not a mode: it is
presence. Every keystroke, click and wheel notch re-arms a window; while the
window is alive vibe asks instead of refusing, and when it lapses vibe goes back
to refusing silently, exactly as it did before.

Why a window and not "first time only": a person who types at minute 40 is still
there. A rule that switched to refusal forever after one missed question would
stop asking precisely while somebody is working.

Nothing here writes to disk: presence is a property of this session, and after a
restart the launch itself counts as presence.
"""

from __future__ import annotations

import os
import threading
import time

#: Default window: five minutes, the same order as the approval answer timeout.
DEFAULT_WINDOW = 300.0

_lock = threading.Lock()
_last_touch = time.monotonic()


def window() -> float:
    """The presence window in seconds (`ARGENT_PRESENCE_WINDOW`, 0 disables the ask)."""
    try:
        raw = float(os.environ.get("ARGENT_PRESENCE_WINDOW", DEFAULT_WINDOW))
    except (TypeError, ValueError):
        return DEFAULT_WINDOW
    return max(0.0, raw)


def touch() -> None:
    """The person did something in the interface. Called from the input path only."""
    global _last_touch
    with _lock:
        _last_touch = time.monotonic()


def seconds_since() -> float:
    with _lock:
        return time.monotonic() - _last_touch


def remaining() -> float:
    """Seconds left in the window; 0.0 when it has lapsed."""
    return max(0.0, window() - seconds_since())


def is_attended() -> bool:
    """True while a human has touched the interface within the window.

    The TUI renews the window on every keystroke and click. The plain REPL has no
    such signal, so it counts as attended only when its stdin really is a terminal:
    a script, a test run or a scheduled job is not a person, and must keep the old
    behaviour of refusing without asking.
    """
    if os.environ.get("ARGENT_TUI") == "1":
        return remaining() > 0.0
    try:
        import sys

        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:
        return False


def reset() -> None:
    """Fresh window (used on start, and by tests)."""
    touch()


def describe() -> str:
    """Short state for the vibe indicator: waiting, or refusing by itself."""
    left = remaining()
    if left > 0:
        return f"жду решения {int(left // 60)}:{int(left % 60):02d}"
    return f"авто-отказ ({int(seconds_since() // 60)} мин без действий)"
