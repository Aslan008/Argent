"""Lanes: one local model server, one queue — for every Argent process on this machine.

A local model is one physical thing. Four sub-agents asking it at the same time do not get four answers at four
times the speed; they split the same cores and the same VRAM four ways, and the person's own request — the one they
are waiting for — gets a quarter. On a laptop that is the freeze.

A lane is the queue in front of one server. Whoever wants the model takes a ticket; the tickets are ordered by
(priority, arrival) and the first `capacity` of them hold the lane. Priorities:

    P_INTERACTIVE 0  the person's own turn — never waits behind anything but another P0
    P_ADVISOR     1  critic, advisor: part of a turn the person is waiting for
    P_WORKER      2  a sub-agent's request (Ultracode workers)
    P_BACKGROUND  3  compaction pre-computation, the night memory pass, anything nobody is waiting for

A ticket is a file in a folder shared by all processes, so a sub-agent (a separate process) queues behind the main
agent and the other way round, with no server and no sockets. A ticket whose process is gone, or whose heartbeat
stopped, is swept by whoever notices it — a crashed worker cannot hold the lane forever.

A lease can be PREEMPTED: a request that arrives with a better priority sorts ahead of a running one, and the
running one finds out by asking `lease.preempted()`. A holder that was bumped too often is pinned: its ticket
sorts first, and it is allowed to finish. Nobody is starved by being thrown out forever.

Nothing here knows about models, providers or HTTP — `governor.py` puts the lanes around provider calls.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from logger import get_logger

log = get_logger("lanes")

P_INTERACTIVE = 0
P_ADVISOR = 1
P_WORKER = 2
P_BACKGROUND = 3

PRIORITY_NAMES = {P_INTERACTIVE: "interactive", P_ADVISOR: "advisor", P_WORKER: "worker",
                  P_BACKGROUND: "background"}

#: A ticket not touched for this long (and not obviously alive) belongs to nobody.
STALE_AFTER = 20.0
HEARTBEAT_EVERY = 2.0
POLL = 0.1


def lanes_root() -> Path:
    """Shared by every project and every process of this user (the model server is not per-project)."""
    override = os.environ.get("ARGENT_LANES_DIR")
    return Path(override) if override else Path.home() / ".argent" / "run" / "lanes"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


_ISSUE = threading.Lock()


def _wall_ns() -> int:
    return time.time_ns()

_NAME = re.compile(r"^(?P<pin>[01])(?P<prio>\d)-(?P<ns>\d+)-(?P<pid>\d+)-(?P<rand>[0-9a-f]+)\.ticket$")


class LaneCancelled(Exception):
    """The wait for the lane was abandoned (the caller's cancel event was set, or the timeout ran out)."""


class Lease:
    """A held place in a lane. Use as a context manager; release is idempotent."""

    def __init__(self, lane: "Lane", path: Path, priority: int, label: str):
        self.lane = lane
        self.path = path
        self.priority = priority
        self.label = label
        self.pinned = False
        self._released = False
        self.waited = 0.0
        self.acquired_at = time.monotonic()

    def holds(self) -> bool:
        return not self._released and self.lane._holds(self.path.name)

    def preempted(self) -> bool:
        """A better-priority request sorted ahead of this one while it was running."""
        return not self._released and not self.holds()

    def pin(self) -> None:
        """Make this lease sort first: it may finish, whoever arrives."""
        if self.pinned or self._released:
            return
        m = _NAME.match(self.path.name)
        if not m:
            return
        new = self.path.with_name(f"0{m['prio']}-{m['ns']}-{m['pid']}-{m['rand']}.ticket")
        try:
            os.replace(self.path, new)
            self.lane._rename(self.path, new)
            self.path = new
            self.pinned = True
        except OSError:
            pass

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.lane._forget(self.path)
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class Lane:
    def __init__(self, lane_id: str, capacity: int = 1, root: Optional[Path] = None,
                 stale_after: float = STALE_AFTER, pid_alive: Callable[[int], bool] = _pid_alive,
                 clock: Callable[[], float] = time.time):
        self.lane_id = lane_id
        self.capacity = max(1, int(capacity))
        self._root = Path(root) if root is not None else None
        self._stale_after = stale_after
        self._pid_alive = pid_alive
        self._clock = clock
        self._mine: Dict[str, Path] = {}
        self._lock = threading.Lock()

    @property
    def folder(self) -> Path:
        return (self._root or lanes_root()) / self.lane_id

    # -- tickets ------------------------------------------------------------------------------

    def _tickets(self) -> List[str]:
        """Names of the live tickets, in the order they are served. Dead ones are swept on the way."""
        folder = self.folder
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return []
        now = self._clock()
        names = []
        for entry in entries:
            m = _NAME.match(entry.name)
            if not m:
                continue
            with self._lock:
                ours = entry.name in {p.name for p in self._mine.values()}
            alive = True
            if not ours:
                try:
                    age = now - entry.stat().st_mtime
                except OSError:
                    continue
                alive = self._pid_alive(int(m["pid"])) and age < self._stale_after
            if alive:
                names.append(entry.name)
            else:
                try:
                    os.unlink(entry.path)
                    log.info("lane %s: swept a dead ticket %s", self.lane_id, entry.name)
                except OSError:
                    pass
        names.sort()
        return names

    def _newest_stamp(self) -> int:
        """The largest arrival number on any ticket in the folder (0 for an empty or unreadable one)."""
        try:
            names = os.listdir(self.folder)
        except OSError:
            return 0
        newest = 0
        for name in names:
            m = _NAME.match(name)
            if m:
                newest = max(newest, int(m["ns"]))
        return newest

    def _holds(self, name: str) -> bool:
        return name in self._tickets()[: self.capacity]

    def _rename(self, old: Path, new: Path) -> None:
        with self._lock:
            for key, path in list(self._mine.items()):
                if path == old:
                    self._mine[key] = new
        _HEART.track(self)

    def _forget(self, path: Path) -> None:
        with self._lock:
            for key, p in list(self._mine.items()):
                if p == path:
                    self._mine.pop(key, None)

    def touch_mine(self) -> None:
        with self._lock:
            paths = list(self._mine.values())
        for p in paths:
            try:
                os.utime(p, None)
            except OSError:
                pass

    @property
    def has_tickets_of_ours(self) -> bool:
        with self._lock:
            return bool(self._mine)

    # -- using the lane -----------------------------------------------------------------------

    def acquire(self, priority: int = P_INTERACTIVE, label: str = "", cancel: Optional[threading.Event] = None,
                timeout: Optional[float] = None, on_wait: Optional[Callable[[int], None]] = None,
                pinned: bool = False) -> Lease:
        """Take a ticket and wait until it is among the first `capacity`. Raises LaneCancelled."""
        priority = min(max(int(priority), 0), 9)
        folder = self.folder
        folder.mkdir(parents=True, exist_ok=True)
        # The number and the file come into being together: a thread that took a smaller number but wrote its file
        # later would sort ahead of a request that had already seen itself first and started (two runs at once).
        # Other processes are not covered by this lock; their window is the microseconds of one file write.
        # The number is also above every ticket already in the folder: the wall clock is coarse (about 15 ms on
        # Windows) and can step back, and a newcomer must never sort in front of a request that is running.
        with _ISSUE:
            stamp = max(_wall_ns(), self._newest_stamp() + 1)
            name = (f"{0 if pinned else 1}{priority}-{stamp:020d}-{os.getpid()}-"
                    f"{random.getrandbits(32):08x}.ticket")
            path = folder / name
            path.write_text(label[:200], encoding="utf-8")
            with self._lock:
                self._mine[name] = path
        _HEART.track(self)
        lease = Lease(self, path, priority, label)
        lease.pinned = pinned
        started = time.monotonic()
        told = False
        try:
            while True:
                order = self._tickets()
                if path.name not in order:           # swept (a heartbeat gap): take a place again
                    path.write_text(label[:200], encoding="utf-8")
                    continue
                position = order.index(path.name)
                if position < self.capacity:
                    lease.waited = time.monotonic() - started
                    lease.acquired_at = time.monotonic()
                    return lease
                if on_wait is not None and not told:
                    told = True
                    try:
                        on_wait(position - self.capacity + 1)
                    except Exception:
                        pass
                if cancel is not None and cancel.is_set():
                    raise LaneCancelled("cancelled while waiting for the model")
                if timeout is not None and time.monotonic() - started > timeout:
                    raise LaneCancelled(f"the model stayed busy for {timeout:.0f}s")
                time.sleep(POLL)
        except BaseException:
            lease.release()
            raise

    def snapshot(self) -> List[dict]:
        """For `/lanes`: who holds the lane and who waits, in service order."""
        out = []
        for index, name in enumerate(self._tickets()):
            m = _NAME.match(name)
            if not m:
                continue
            label = ""
            try:
                label = (self.folder / name).read_text(encoding="utf-8")[:80]
            except OSError:
                pass
            out.append({"position": index, "holding": index < self.capacity, "pid": int(m["pid"]),
                        "priority": int(m["prio"]), "name": PRIORITY_NAMES.get(int(m["prio"]), "?"),
                        "pinned": m["pin"] == "0", "label": label})
        return out


class _Heartbeat:
    """One thread per process keeps the tickets of every lane this process waits on or holds fresh."""

    def __init__(self):
        self._lanes: "set[Lane]" = set()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def track(self, lane: Lane) -> None:
        with self._lock:
            self._lanes.add(lane)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="ArgentLaneHeartbeat")
                self._thread.start()

    def _run(self) -> None:
        while True:
            time.sleep(HEARTBEAT_EVERY)
            with self._lock:
                lanes = list(self._lanes)
            busy = False
            for lane in lanes:
                lane.touch_mine()
                busy = busy or lane.has_tickets_of_ours
            if not busy:
                with self._lock:
                    if not any(l.has_tickets_of_ours for l in self._lanes):
                        self._lanes.clear()
                        self._thread = None
                        return


_HEART = _Heartbeat()

_LANES: Dict[str, Lane] = {}
_LANES_LOCK = threading.Lock()


def get_lane(lane_id: str, capacity: int = 1) -> Lane:
    """The process-wide handle of a lane (the same object for the same id; capacity follows the caller)."""
    with _LANES_LOCK:
        lane = _LANES.get(lane_id)
        if lane is None:
            lane = _LANES[lane_id] = Lane(lane_id, capacity)
        else:
            lane.capacity = max(1, int(capacity))
        return lane


def forget_lanes() -> None:
    """Tests: drop the cached handles."""
    with _LANES_LOCK:
        _LANES.clear()
