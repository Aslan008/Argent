"""Is the PC busy with something other than us?

Background work (an Ultracode worker, the pre-computed summary, the night memory pass) may only use what the
person is not using. The lanes keep our own requests from piling onto one model server; this module watches the
rest: a build, a game, a video call, the browser with forty tabs.

What is measured:

* CPU — the machine's load WITHOUT the model server and our own processes. The server working hard on the
  person's own request is not "pressure"; a compiler is.
* free RAM, and free VRAM when `nvidia-smi` answers.

The verdict has hysteresis: it turns to "overloaded" quickly and back to "calm" only when the numbers are well
below the trip level, so a worker does not start, stall and start again on every sample.

`wait_until_calm` is what background priorities call before they take a lane. Interactive requests never wait here.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from logger import get_logger

log = get_logger("pressure")


@dataclass
class Thresholds:
    cpu_high: float = 85.0           # external CPU % that trips the verdict
    cpu_low: float = 60.0            # ... and that releases it
    ram_low_gb: float = 1.5          # free RAM that trips it
    ram_ok_gb: float = 2.5           # ... and releases it
    vram_low_mb: float = 512.0
    vram_ok_mb: float = 1024.0
    hold_samples: int = 3            # consecutive high CPU samples needed to trip


@dataclass
class Reading:
    cpu_external: float = 0.0
    cpu_total: float = 0.0
    ram_free_gb: float = 99.0
    vram_free_mb: Optional[float] = None
    at: float = field(default_factory=time.time)


def _vram_free_mb() -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        values = [float(x) for x in out.decode().split() if x.strip().replace(".", "", 1).isdigit()]
        return min(values) if values else None
    except Exception:
        return None


class PressureMonitor:
    def __init__(self, thresholds: Optional[Thresholds] = None,
                 reader: Optional[Callable[[], Reading]] = None,
                 own_pids: Optional[Callable[[], List[int]]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.t = thresholds or Thresholds()
        self._reader = reader or self._read
        self._own_pids = own_pids or (lambda: [])
        self._clock = clock
        self._overloaded = False
        self._reason = ""
        self._streak = 0
        self._vram: Optional[float] = None
        self._vram_at = -1e9
        self.last: Optional[Reading] = None
        self._lock = threading.Lock()

    # -- reading ------------------------------------------------------------------------------

    def _read(self) -> Reading:
        try:
            import psutil
        except Exception:
            return Reading()
        total = psutil.cpu_percent(interval=None)
        ours = 0.0
        for pid in set(self._own_pids()):
            try:
                ours += psutil.Process(pid).cpu_percent(interval=None) / max(1, psutil.cpu_count() or 1)
            except Exception:
                continue
        now = self._clock()
        if now - self._vram_at > 5.0:                    # nvidia-smi is a process spawn: not on every sample
            self._vram = _vram_free_mb()
            self._vram_at = now
        return Reading(cpu_external=max(0.0, total - ours), cpu_total=total,
                       ram_free_gb=psutil.virtual_memory().available / (1024 ** 3), vram_free_mb=self._vram)

    def sample(self) -> bool:
        """Take a reading, update the verdict. True when the machine is overloaded."""
        reading = self._reader()
        t = self.t
        with self._lock:
            self.last = reading
            if reading.cpu_external >= t.cpu_high:
                self._streak += 1
            else:
                self._streak = 0
            low_ram = reading.ram_free_gb < t.ram_low_gb
            low_vram = reading.vram_free_mb is not None and reading.vram_free_mb < t.vram_low_mb
            if not self._overloaded:
                if self._streak >= t.hold_samples:
                    self._overloaded, self._reason = True, f"процессор занят другими программами ({reading.cpu_external:.0f}%)"
                elif low_ram:
                    self._overloaded, self._reason = True, f"мало свободной памяти ({reading.ram_free_gb:.1f} ГБ)"
                elif low_vram:
                    self._overloaded, self._reason = True, f"мало видеопамяти ({reading.vram_free_mb:.0f} МБ)"
            else:
                calm = (reading.cpu_external <= t.cpu_low and reading.ram_free_gb >= t.ram_ok_gb
                        and (reading.vram_free_mb is None or reading.vram_free_mb >= t.vram_ok_mb))
                if calm:
                    self._overloaded, self._reason, self._streak = False, "", 0
            return self._overloaded

    @property
    def overloaded(self) -> bool:
        return self._overloaded

    @property
    def reason(self) -> str:
        return self._reason

    # -- waiting ------------------------------------------------------------------------------

    def wait_until_calm(self, cancel: Optional[threading.Event] = None, timeout: Optional[float] = None,
                        interval: float = 1.0, on_wait: Optional[Callable[[str], None]] = None,
                        sleep: Callable[[float], None] = time.sleep) -> bool:
        """Block while the machine is overloaded. False when cancelled or out of time (the caller decides)."""
        started = self._clock()
        told = False
        while self.sample():
            if cancel is not None and cancel.is_set():
                return False
            if timeout is not None and self._clock() - started > timeout:
                return False
            if on_wait is not None and not told:
                told = True
                try:
                    on_wait(self._reason)
                except Exception:
                    pass
            sleep(interval)
        return True


_MONITOR: Optional[PressureMonitor] = None
_MONITOR_LOCK = threading.Lock()


def monitor() -> PressureMonitor:
    """The process-wide monitor; the server's processes are excluded from the "external" CPU."""
    global _MONITOR
    with _MONITOR_LOCK:
        if _MONITOR is None:
            def own() -> List[int]:
                import os
                pids = [os.getpid()]
                try:
                    from src.agent.local_backend import listener_pids
                    import psutil
                    from src.agent import lanes
                    for lane_dir in (lanes.lanes_root().glob("*") if lanes.lanes_root().exists() else []):
                        port = lane_dir.name.rsplit("-", 1)[-1]
                        if port.isdigit():
                            for pid in listener_pids(int(port)):
                                pids.append(pid)
                                try:
                                    pids += [c.pid for c in psutil.Process(pid).children(recursive=True)]
                                except Exception:
                                    pass
                    pids += [c.pid for c in psutil.Process().children(recursive=True)]
                except Exception:
                    pass
                return pids

            _MONITOR = PressureMonitor(own_pids=own)
        return _MONITOR


def reset_monitor() -> None:
    global _MONITOR
    with _MONITOR_LOCK:
        _MONITOR = None
