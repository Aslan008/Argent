"""The loop that fires automations while Argent is up.

Runs inside the interactive host (the REPL, or the TUI's Python bridge) rather
than as a separate service: automations are scoped to "while Argent is open", so
a second process to supervise would be machinery without a job. The jobs
themselves run in child processes (see runner.SubprocessRunner). For jobs that
must run with Argent closed there is `run_due_once` (the OS scheduler calls it,
see service.py): the same decisions, made once, by whoever happens to be awake.

It used to live in a GUI server process. When that server was removed nothing
started the scheduler any more: `/tasks add` accepted a definition and the
schedule never fired. `start_background_scheduler` is what the hosts call now.

Properties that matter more than throughput here:

* **No overlap.** A run that takes longer than its interval must not start a
  second copy of itself — an automation that opens a file, appends and saves
  would corrupt it, and a slow one would pile up copies until the box dies.
* **No blocking.** Runs happen on a worker thread; the tick itself only decides
  what is due, so the host stays responsive while a job is running.
* **No double run across processes.** Open twice (terminal + TUI) and each host
  has a scheduler; a per-automation lock file decides who runs it, and the
  decision is re-checked after the lock is won (the other process may have
  finished the very same run a moment ago).
* **A pause is a pause.** `/tasks pause` stops every scheduled run, in every
  process, until it is lifted or expires.
"""

import atexit
import inspect
import os
import threading
from datetime import datetime

from logger import get_logger
from src.automation.schedule import ScheduleError, is_due, parse_schedule
from src.automation.store import (
    claim_run, get_automation, load_automations, pause_until, release_run, running_elsewhere,
)

log = get_logger("automation")

DEFAULT_TICK_SECONDS = 30

# A failed daily/monthly/one-shot run is tried again after this long (times the
# number of failures so far), and only for a couple of hours: a retry the next
# morning is not a retry, it is a second, surprising run.
RETRY_BASE_SECONDS = 5 * 60
RETRY_MAX_DELAY_SECONDS = 30 * 60
RETRY_WINDOW_SECONDS = 2 * 60 * 60

SUMMARY_IN_EVENT = 1500


def _schedule_kind(schedule: str, now: datetime):
    try:
        return parse_schedule(schedule, now=now)["kind"]
    except ScheduleError:
        return None


def _retry_due(automation, now: datetime) -> bool:
    """A run that failed gets another chance soon — unless the schedule itself
    will try again shortly (an interval), or it failed too many times already."""
    if automation.last_status != "error":
        return False
    streak = int(automation.fail_streak or 0)
    if not 1 <= streak <= int(automation.retries or 0):
        return False
    if _schedule_kind(automation.schedule, now) in (None, "interval", "startup"):
        return False
    last = automation.last_run_dt()
    if last is None:
        return False
    waited = (now - last).total_seconds()
    delay = min(RETRY_BASE_SECONDS * streak, RETRY_MAX_DELAY_SECONDS)
    return delay <= waited <= RETRY_WINDOW_SECONDS


def due_trigger(automation, now: datetime, starting: bool = False):
    """Why this automation should run now ("schedule" | "startup" | "retry"), or None."""
    catchup = max(0, int(automation.catchup_minutes or 0)) * 60
    if is_due(automation.schedule, automation.last_run_dt(), now,
              starting=starting, catchup_seconds=catchup):
        return "startup" if _schedule_kind(automation.schedule, now) == "startup" else "schedule"
    if _retry_due(automation, now):
        return "retry"
    return None


def _accepts_trigger(runner) -> bool:
    try:
        parameters = inspect.signature(runner).parameters
    except (TypeError, ValueError):
        return False
    return "trigger" in parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


class AutomationScheduler:
    """Polls the definitions and runs whatever is due.

    ``runner`` is injectable so the loop is testable without an LLM; ``on_event``
    receives {"type": "automation", ...} dicts for the host's UI.
    """

    def __init__(self, runner=None, on_event=None, tick_seconds: int = DEFAULT_TICK_SECONDS):
        self._runner = runner
        self._on_event = on_event
        self._tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._thread = None
        self._running: dict = {}            # name -> {"since": datetime, "trigger": str}
        self._workers: list = []
        self._lock = threading.Lock()
        self._starting = True               # the next tick is the first of this launch

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
        # A job must not outlive the program whose lifetime it is scoped to.
        terminate = getattr(self._runner, "terminate_all", None)
        if callable(terminate):
            try:
                terminate()
            except Exception as e:
                log.warning("could not stop running automations: %s", e)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:      # a scheduler must outlive its jobs
                log.warning("automation tick failed: %s", e)
            self._stop.wait(self._tick_seconds)

    def join(self, timeout: float = None) -> None:
        """Wait for the runs started so far (for the one-shot headless tick)."""
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout)

    # --- one pass ----------------------------------------------------------
    def _is_running(self, name: str) -> bool:
        with self._lock:
            return name in self._running

    def _decisions(self, now: datetime, starting: bool) -> list:
        if pause_until(now) is not None:
            return []
        out = []
        for a in load_automations():
            if not a.enabled:
                continue
            if self._is_running(a.name):
                # Still working from the previous tick; skipping is the
                # whole point — see the module docstring.
                log.info("automation %r still running; skipping this tick", a.name)
                continue
            trigger = due_trigger(a, now, starting)
            if trigger:
                out.append((a, trigger))
        return out

    def due_automations(self, now=None) -> list:
        """Enabled automations whose time has come and that aren't already
        running."""
        return [a for a, _ in self._decisions(now or datetime.now(), False)]

    def tick(self, now=None) -> list:
        """Start every due automation. Returns the ones started."""
        now = now or datetime.now()
        starting, self._starting = self._starting, False
        started = []
        for automation, trigger in self._decisions(now, starting):
            if self._start_run(automation, trigger, now):
                started.append(automation)
        return started

    # --- starting and stopping runs ----------------------------------------
    def run_now(self, name: str, trigger: str = "manual"):
        """Start a run immediately, in the background. Returns (ok, reason)."""
        automation = get_automation(name)
        if automation is None:
            return False, f"Автоматизация '{name}' не найдена."
        if self._is_running(name):
            return False, f"'{name}' уже выполняется."
        if not self._start_run(automation, trigger, datetime.now()):
            return False, f"'{name}' уже выполняется в другом окне Argent."
        return True, ""

    def stop_run(self, name: str) -> bool:
        stop = getattr(self._runner, "stop", None)
        if callable(stop):
            try:
                return bool(stop(name))
            except Exception as e:
                log.warning("could not stop %r: %s", name, e)
        return False

    def status(self) -> dict:
        """What is running right now, here and in other Argent processes."""
        with self._lock:
            here = {name: dict(info) for name, info in self._running.items()}
        elsewhere = [n for n in running_elsewhere() if n not in here]
        return {"running": here, "elsewhere": elsewhere, "paused_until": pause_until()}

    def _start_run(self, automation, trigger: str, now: datetime):
        if not claim_run(automation.name):
            # Another Argent process is running it right now.
            log.info("automation %r is running in another process; skipping", automation.name)
            return False
        if trigger != "manual" and not self._still_due(automation.name, trigger, now):
            # The other process finished the very same run between our decision
            # and our claim.
            release_run(automation.name)
            log.info("automation %r was already run by another process; skipping", automation.name)
            return False
        with self._lock:
            self._running[automation.name] = {"since": datetime.now(), "trigger": trigger}
        self._emit({"type": "automation", "event": "started",
                    "name": automation.name, "trigger": trigger})

        def _work():
            result, crash = None, None
            try:
                result = self._invoke(automation, trigger)
            except Exception as e:
                crash = e
                log.warning("automation %r crashed: %s", automation.name, e)
            finally:
                # Release BEFORE announcing: whoever hears "finished" may well
                # tick or run it again at once, and must not find it still held.
                with self._lock:
                    self._running.pop(automation.name, None)
                release_run(automation.name)
            self._emit(self._finished_event(automation, trigger, result, crash))

        worker = threading.Thread(target=_work, daemon=True, name=f"automation-{automation.name}")
        with self._lock:
            self._workers = [w for w in self._workers if w.is_alive()] + [worker]
        worker.start()
        return True

    def _still_due(self, name: str, trigger: str, now: datetime) -> bool:
        fresh = get_automation(name)
        if fresh is None or not fresh.enabled:
            return False
        return due_trigger(fresh, now, starting=(trigger == "startup")) is not None

    def _invoke(self, automation, trigger: str):
        runner = self._runner
        if runner is None:
            from src.automation.runner import run_and_record

            def runner(a, trigger="schedule"):
                return run_and_record(a, notify=(trigger != "manual"), trigger=trigger)
        if _accepts_trigger(runner):
            return runner(automation, trigger=trigger)
        return runner(automation)

    @staticmethod
    def _finished_event(automation, trigger: str, result, crash) -> dict:
        if crash is not None:
            return {"type": "automation", "event": "finished", "name": automation.name,
                    "status": "error", "summary": str(crash), "denied": 0, "new_items": 0,
                    "alert": True, "silent": False, "trigger": trigger,
                    "mode": automation.mode}
        result = result or {}
        return {
            "type": "automation", "event": "finished", "name": automation.name,
            "status": result.get("status", "ok"),
            "summary": (result.get("summary") or "")[:SUMMARY_IN_EVENT],
            "denied": len(result.get("denied_actions") or []),
            "new_items": result.get("new_items", 0),
            "alert": result.get("alert"),
            "silent": bool(result.get("silent")),
            "trigger": trigger,
            "mode": result.get("mode", automation.mode),
        }

    def _emit(self, event: dict):
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            pass          # a broken listener must not take down the scheduler


# --- the host's side ------------------------------------------------------------
_instance = None
_instance_lock = threading.Lock()

# Set by the test suite (and usable by a user who wants Argent never to fire a
# schedule): a test that builds a bridge must not run the developer's real jobs.
DISABLE_ENV = "ARGENT_DISABLE_AUTOMATIONS"


def automations_disabled() -> bool:
    return os.environ.get(DISABLE_ENV) == "1"


def get_scheduler():
    """This process's running scheduler, or None (never started, or disabled)."""
    return _instance


def start_background_scheduler(on_event=None):
    """Start this process's scheduler, once, and stop it (and its jobs) at exit.

    Returns the scheduler, or None when automations are disabled. Calling it a
    second time returns the running instance, so a host that initialises twice
    does not double-fire everything.
    """
    global _instance
    if automations_disabled():
        return None
    with _instance_lock:
        if _instance is None:
            from src.automation.runner import SubprocessRunner
            _instance = AutomationScheduler(runner=SubprocessRunner(), on_event=on_event)
            _instance.start()
            atexit.register(_instance.stop)
            try:                       # rooms' webhooks live and die with the same host
                from src.rooms.webhooks import start_if_configured
                start_if_configured()
            except Exception as e:
                log.info("webhooks not started: %s", e)
        return _instance


def run_due_once(now: datetime = None, runner=None) -> int:
    """Run whatever is due right now, wait for it, and return an exit code.

    This is what the OS scheduler runs while Argent is closed. It makes the same
    decisions as the live scheduler (pause, locks, retries) but never starts a
    loop; 'on start' jobs are not its business.
    """
    if automations_disabled():
        return 0
    if runner is None:
        from src.automation.runner import SubprocessRunner
        runner = SubprocessRunner()
    scheduler = AutomationScheduler(runner=runner)
    scheduler._starting = False
    started = scheduler.tick(now)
    scheduler.join()
    log.info("headless tick finished: %d automation(s) run", len(started))
    return 0
