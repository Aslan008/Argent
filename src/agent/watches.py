"""Waits that do not hold the conversation hostage.

`wait_heartbeat` used to END the agent's turn and park the frontend in a sleep:
no closing words to the user, no way to type until it woke up. A frontend that
can be woken from outside does not need that. It registers a *watch* here and
the tool returns at once; the model finishes its turn with an ordinary answer
("started the build, I'll continue when it is done"), the user stays free, and
this module's thread wakes the frontend when the thing actually happened.

Only frontends that can start a turn on their own register a handler (the TUI
bridge does). Without one `wait_heartbeat` keeps its blocking protocol, so the
classic REPL and headless runs behave exactly as before.

Everything a watch observes is a local read — the same whitelist
`wait_conditions` evaluates — so a watcher running with nobody looking gains no
authority a polling model did not already have.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from logger import get_logger
from src.agent.wait_conditions import (
    ConditionError, MAX_TIMEOUT_SECONDS, condition_wake_message, evaluate,
)

log = get_logger("agent")

POLL_SECONDS = 2.0
MAX_WATCHES = 8
#: A timer shorter than this is polling dressed up as waiting: every wake costs a model turn.
MIN_TIMER_SECONDS = 5
MAX_TIMER_SECONDS = 6 * 3600


@dataclass
class Watch:
    id: int
    reason: str
    until: str = ""                 # empty: a plain timer
    deadline: float = 0.0
    created: float = 0.0
    timeout: int = 0
    pids: frozenset = field(default_factory=frozenset)   # timer only: background commands that end it early

    @property
    def is_condition(self) -> bool:
        return bool(self.until)

    def describe(self) -> str:
        return self.until if self.is_condition else (self.reason or "таймер")


def _registry():
    try:
        from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
    except ImportError:
        return None, None
    return ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK


def running_background_pids() -> frozenset:
    procs, lock = _registry()
    if procs is None:
        return frozenset()
    with lock:
        items = list(procs.items())
    return frozenset(str(pid) for pid, info in items
                     if (info.get("process") is not None and info["process"].poll() is None))


def _exit_code(pid: str):
    """(finished, exit code) of one background command; a vanished one counts as finished."""
    procs, lock = _registry()
    if procs is None:
        return True, "?"
    with lock:
        info = procs.get(pid)
    process = info.get("process") if info else None
    if process is None:
        return True, "?"
    code = process.poll()
    return code is not None, code


class WatchRegistry:
    def __init__(self, poll_seconds: float = POLL_SECONDS, clock: Callable[[], float] = time.monotonic):
        self._poll = poll_seconds
        self._clock = clock
        self._lock = threading.RLock()
        self._watches = []
        self._next_id = 1
        self._handler: Optional[Callable[[str, Watch], None]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- who listens ------------------------------------------------------------

    def set_handler(self, handler: Optional[Callable[[str, Watch], None]]) -> None:
        """The frontend that can start a turn when a watch fires; None turns the feature off."""
        with self._lock:
            self._handler = handler
            if handler is None:
                self._watches.clear()
                self._stop.set()

    def has_handler(self) -> bool:
        return self._handler is not None

    # -- registering ------------------------------------------------------------

    def add(self, reason: str, until: str = "", timeout: int = 600, delay: int = 0) -> Watch:
        """Start watching. `until` waits on a condition (with `timeout`); otherwise it is a timer of `delay` s."""
        if until:
            evaluate(until)                       # a malformed condition fails here, not minutes later
        with self._lock:
            if len(self._watches) >= MAX_WATCHES:
                raise ValueError(f"already waiting on {MAX_WATCHES} things — cancel one or wait for it")
            now = self._clock()
            if until:
                span = max(1, min(int(timeout or 600), MAX_TIMEOUT_SECONDS))
                pids = frozenset()
            else:
                span = max(MIN_TIMER_SECONDS, min(int(delay), MAX_TIMER_SECONDS))
                pids = running_background_pids()
            watch = Watch(id=self._next_id, reason=reason or "", until=until or "",
                          deadline=now + span, created=now, timeout=span, pids=pids)
            self._next_id += 1
            self._watches.append(watch)
            self._ensure_thread()
        return watch

    def cancel(self, watch_id: Optional[int] = None) -> int:
        """Drop one watch (or all); how many were dropped."""
        with self._lock:
            before = len(self._watches)
            self._watches = [w for w in self._watches if watch_id is not None and w.id != watch_id]
            return before - len(self._watches)

    def pending(self) -> list:
        with self._lock:
            return list(self._watches)

    # -- checking ---------------------------------------------------------------

    def _verdict(self, watch: Watch, now: float) -> Optional[str]:
        """The wake-up message if this watch is over, else None."""
        waited = round(now - watch.created, 1)
        if watch.is_condition:
            try:
                met = evaluate(watch.until)
            except ConditionError as e:
                return condition_wake_message(watch.until, watch.reason, False, f"invalid condition: {e}", waited)
            if met:
                return condition_wake_message(watch.until, watch.reason, True, "condition met", waited)
            if now >= watch.deadline:
                return condition_wake_message(watch.until, watch.reason, False,
                                              f"timed out after {int(waited)}s", waited)
            return None
        for pid in sorted(watch.pids):
            finished, code = _exit_code(pid)
            if finished:
                return (f"[Heartbeat прерван] Фоновая команда {pid} завершилась с кодом {code}. "
                        f"Проверьте её вывод с помощью read_background_command.")
        if now >= watch.deadline:
            return f"[Heartbeat пробуждение] Причина: {watch.reason}" if watch.reason else "[Heartbeat пробуждение]"
        return None

    def check_once(self) -> int:
        """Fire every finished watch; how many fired. The thread calls this, and so can a test."""
        now = self._clock()
        fired = []
        for watch in self.pending():
            message = self._verdict(watch, now)
            if message is not None:
                fired.append((watch, message))
        if not fired:
            return 0
        with self._lock:
            ids = {w.id for w, _ in fired}
            alive = {w.id for w in self._watches}
            self._watches = [w for w in self._watches if w.id not in ids]
            handler = self._handler
        for watch, message in fired:
            if watch.id not in alive or handler is None:    # cancelled while we were deciding
                continue
            try:
                handler(message, watch)
            except Exception as e:
                log.warning("watch handler failed: %s", e)
        return len(fired)

    # -- the thread -------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and not self._stop.is_set():
            return                      # _run clears this under the lock before it exits
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop,), daemon=True,
                                        name="ArgentWatches")
        self._thread.start()

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.check_once()
            except Exception as e:
                log.warning("watch check failed: %s", e)
            with self._lock:
                if not self._watches:
                    if self._stop is stop:
                        self._thread = None
                    return
            stop.wait(self._poll)


WATCHES = WatchRegistry()


def async_enabled() -> bool:
    return WATCHES.has_handler()
