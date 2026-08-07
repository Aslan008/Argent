"""The loop that fires automations while Argent is up.

Runs inside the existing server process rather than as a separate service:
automations are scoped to "while Argent is open", so a second process to
supervise would be machinery without a job.

Two properties matter more than throughput here:

* **No overlap.** A run that takes longer than its interval must not start a
  second copy of itself — an automation that opens a file, appends and saves
  would corrupt it, and a slow one would pile up copies until the box dies.
* **No blocking.** Runs happen on a worker thread; the tick itself only decides
  what is due, so the server keeps serving the GUI while a job is running.
"""

import threading
from datetime import datetime

from logger import get_logger
from src.automation.schedule import is_due
from src.automation.store import load_automations

log = get_logger("automation")

DEFAULT_TICK_SECONDS = 30


class AutomationScheduler:
    """Polls the definitions and runs whatever is due.

    ``runner`` is injectable so the loop is testable without an LLM; ``on_event``
    receives {"type": "automation", ...} dicts for the GUI stream.
    """

    def __init__(self, runner=None, on_event=None, tick_seconds: int = DEFAULT_TICK_SECONDS):
        self._runner = runner
        self._on_event = on_event
        self._tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._thread = None
        self._running_names = set()
        self._lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="argent-automations")
        self._thread.start()
        log.info("automation scheduler started (tick %ss)", self._tick_seconds)

    def stop(self):
        self._stop.set()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:      # a scheduler must outlive its jobs
                log.warning("automation tick failed: %s", e)
            self._stop.wait(self._tick_seconds)

    # --- one pass ----------------------------------------------------------
    def due_automations(self, now=None) -> list:
        """Enabled automations whose time has come and that aren't already
        running."""
        now = now or datetime.now()
        out = []
        for a in load_automations():
            if not a.enabled:
                continue
            with self._lock:
                if a.name in self._running_names:
                    # Still working from the previous tick; skipping is the
                    # whole point — see the module docstring.
                    log.info("automation %r still running; skipping this tick", a.name)
                    continue
            if is_due(a.schedule, a.last_run_dt(), now):
                out.append(a)
        return out

    def tick(self, now=None) -> list:
        """Start every due automation. Returns the ones started."""
        started = []
        for automation in self.due_automations(now):
            self._start_run(automation)
            started.append(automation)
        return started

    def _start_run(self, automation):
        with self._lock:
            self._running_names.add(automation.name)
        self._emit({"type": "automation", "event": "started", "name": automation.name})

        def _work():
            try:
                runner = self._runner
                if runner is None:
                    from src.automation.runner import run_and_record
                    runner = run_and_record
                result = runner(automation)
                self._emit({
                    "type": "automation", "event": "finished",
                    "name": automation.name,
                    "status": (result or {}).get("status", "ok"),
                    "summary": ((result or {}).get("summary") or "")[:500],
                    "denied": len((result or {}).get("denied_actions") or []),
                    "new_items": (result or {}).get("new_items", 0),
                })
            except Exception as e:
                log.warning("automation %r crashed: %s", automation.name, e)
                self._emit({"type": "automation", "event": "finished",
                            "name": automation.name, "status": "error",
                            "summary": str(e), "denied": 0})
            finally:
                with self._lock:
                    self._running_names.discard(automation.name)

        threading.Thread(target=_work, daemon=True,
                         name=f"automation-{automation.name}").start()

    def _emit(self, event: dict):
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            pass          # a broken listener must not take down the scheduler
