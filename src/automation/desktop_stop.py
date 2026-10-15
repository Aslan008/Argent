"""The emergency stop of desktop control: a key chord held anywhere ends it.

A thread watches the chord (Ctrl+Alt+Shift+F12 by default) for as long as desktop control is on. It does not
register a system hot key — that would need a message loop and would swallow the chord for every other program —
it only asks the keyboard state a few times a second. When the chord is down it marks the series stopped: every
later action is refused until the person switches control on again, and the one in progress is cut at its
next step.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from logger import get_logger

log = get_logger("desktop_stop")

POLL = 0.05

_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()
_lock = threading.Lock()


def _announce() -> None:
    try:
        from ui import console
        console.print("\n[bold red]Управление рабочим столом остановлено (аварийная клавиша). "
                      "Включить снова: /desktop on[/bold red]")
    except Exception:
        pass


def _watch(series, chord_provider: Callable[[], str], is_down: Callable[[str], bool]) -> None:
    held = False
    while not _stop_flag.is_set():
        try:
            down = is_down(chord_provider())
        except Exception:
            down = False
        if down and not held:
            series.stop("аварийная клавиша")
            try:
                from src.automation import desktop_guard
                desktop_guard.audit("emergency_stop")
            except Exception:
                pass
            log.warning("desktop control: emergency stop")
            _announce()
        held = down
        time.sleep(POLL)


def start(series, chord_provider: Optional[Callable[[], str]] = None,
          is_down: Optional[Callable[[str], bool]] = None) -> None:
    global _thread
    from src.automation import desktop_guard, uia
    chord_provider = chord_provider or desktop_guard.stop_chord
    is_down = is_down or uia.chord_is_down
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop_flag.clear()
        _thread = threading.Thread(target=_watch, args=(series, chord_provider, is_down), daemon=True,
                                   name="ArgentDesktopStop")
        _thread.start()


def stop() -> None:
    global _thread
    with _lock:
        _stop_flag.set()
        thread, _thread = _thread, None
    if thread is not None:
        thread.join(timeout=1.0)
