"""Whether the code running on this thread may stop and ask the person at the keyboard.

A question to the user (pick another model, confirm) is only sensible when a person is
looking at the frontend that started the work. The Telegram bridge, automations and
background workers also run agent code, and a question asked there waits forever for
nobody. So asking is opt-in per thread: the terminal frontends mark the thread that serves
their turns, and the REPL's main thread counts when it is attached to a console.
"""

from __future__ import annotations

import os
import sys
import threading

_tls = threading.local()


def mark_interactive(flag: bool = True) -> None:
    """Called by a frontend on the thread that runs turns for a person."""
    _tls.interactive = bool(flag)


def can_ask_user() -> bool:
    if getattr(_tls, "interactive", False):
        return True
    # The classic REPL runs the agent on the main thread, on a real console.
    if os.environ.get("ARGENT_TUI") == "1":
        return False
    try:
        return threading.current_thread() is threading.main_thread() and bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False
