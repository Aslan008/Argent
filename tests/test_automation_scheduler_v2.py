"""The scheduler beyond "is it due": pause, start-up jobs, retries, manual runs,
and the two races that used to make a finished job look still-running.
"""

import sys
import threading
import time
from datetime import datetime, timedelta

import pytest

from src.automation import scheduler as scheduler_module
from src.automation import store
from src.automation.runner import RESULT_MARKER, SubprocessRunner
from src.automation.scheduler import AutomationScheduler, due_trigger, run_due_once
from src.automation.store import Automation

NOW = datetime(2026, 10, 8, 14, 30)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


def _a(name="job", **kw):
    kw.setdefault("task", "t")
    kw.setdefault("schedule", "every 1m")
    return Automation(name=name, **kw)


def _sched(runner=None, on_event=None):
    return AutomationScheduler(runner=runner, on_event=on_event, tick_seconds=1)


def _ok(automation, **kw):
    return {"status": "ok", "summary": ""}


class TestPause:
    def test_nothing_is_scheduled_while_paused(self, project):
        store.upsert_automation(_a())
        store.set_paused()
        sched = _sched(_ok)
        assert sched.tick(NOW) == []
        store.resume()
        assert [a.name for a in sched.tick(NOW)] == ["job"]

    def test_a_timed_pause_ends_by_itself(self, project):
        store.upsert_automation(_a())
        store.set_paused(until=NOW - timedelta(seconds=5))     # relative to the tick's clock, not the wall clock
        assert [a.name for a in _sched(_ok).tick(NOW)] == ["job"]

    def test_a_pause_does_not_stop_a_manual_run(self, project):
        store.upsert_automation(_a(schedule="daily at 03:00"))
        store.set_paused()
        done = threading.Event()
        sched = _sched(lambda a, trigger: done.set() or _ok(a))
        assert sched.run_now("job") == (True, "")
        assert done.wait(5)


class TestStartupJobs:
    def test_fire_on_the_first_tick_of_a_launch_only(self, project):
        store.upsert_automation(_a(schedule="on start"))
        sched = _sched(_ok)
        assert [a.name for a in sched.tick(NOW)] == ["job"]
        sched.join()
        assert sched.tick(NOW + timedelta(minutes=1)) == []

    def test_a_new_launch_fires_again(self, project):
        store.upsert_automation(_a(schedule="on start"))
        first = _sched(_ok)
        first.tick(NOW)
        first.join()
        assert [a.name for a in _sched(_ok).tick(NOW + timedelta(hours=1))] == ["job"]

    def test_the_trigger_is_startup(self):
        assert due_trigger(_a(schedule="on start"), NOW, starting=True) == "startup"
        assert due_trigger(_a(schedule="on start"), NOW, starting=False) is None


class TestRetry:
    def _failed(self, minutes_ago=10, streak=1, **kw):
        last = (NOW - timedelta(minutes=minutes_ago)).isoformat()
        kw.setdefault("schedule", "daily at 09:00")
        kw.setdefault("retries", 1)
        return _a(last_run=last, last_status="error", fail_streak=streak, **kw)

    def test_a_failed_daily_job_is_tried_again_soon(self):
        assert due_trigger(self._failed(), NOW) == "retry"

    def test_not_before_the_pause_has_passed(self):
        assert due_trigger(self._failed(minutes_ago=2), NOW) is None

    def test_the_wait_grows_with_each_failure(self):
        assert due_trigger(self._failed(minutes_ago=7, streak=2, retries=2), NOW) is None
        assert due_trigger(self._failed(minutes_ago=11, streak=2, retries=2), NOW) == "retry"

    def test_not_hours_later(self):
        assert due_trigger(self._failed(minutes_ago=200), NOW) is None

    def test_only_as_many_times_as_asked(self):
        assert due_trigger(self._failed(streak=2), NOW) is None
        assert due_trigger(self._failed(retries=0), NOW) is None

    def test_an_interval_job_just_waits_for_its_next_slot(self):
        assert due_trigger(self._failed(minutes_ago=10, schedule="every 1h"), NOW) is None

    def test_a_job_that_last_succeeded_is_not_retried(self):
        job = self._failed()
        job.last_status = "ok"
        assert due_trigger(job, NOW) is None

    def test_a_failed_one_shot_is_retried(self):
        job = self._failed(schedule="once at 2026-10-08 14:00")
        assert due_trigger(job, NOW) == "retry"

    def test_the_scheduler_passes_the_trigger_on(self, project):
        job = self._failed()
        job.last_run = (datetime.now() - timedelta(minutes=10)).isoformat()
        job.name = "daily-job"
        store.upsert_automation(job)
        seen = []
        sched = _sched(lambda a, trigger: seen.append(trigger) or _ok(a))
        sched.tick(datetime.now())
        sched.join()
        assert seen == ["retry"]


class TestRunners:
    def test_a_runner_that_takes_a_trigger_gets_it(self, project):
        store.upsert_automation(_a())
        seen = []
        sched = _sched(lambda a, trigger="x": seen.append(trigger) or _ok(a))
        sched.tick(NOW)
        sched.join()
        assert seen == ["schedule"]

    def test_the_old_one_argument_runner_still_works(self, project):
        store.upsert_automation(_a())
        seen = []
        sched = _sched(lambda a: seen.append(a.name) or _ok(a))
        sched.tick(NOW)
        sched.join()
        assert seen == ["job"]


class TestManualRuns:
    def test_runs_even_when_disabled_and_not_due(self, project):
        store.upsert_automation(_a(enabled=False, schedule="daily at 03:00"))
        triggers = []
        sched = _sched(lambda a, trigger: triggers.append(trigger) or _ok(a))
        assert sched.run_now("job") == (True, "")
        sched.join()
        assert triggers == ["manual"]

    def test_unknown_name(self, project):
        ok, why = _sched(_ok).run_now("ghost")
        assert ok is False and "не найдена" in why

    def test_already_running_here(self, project):
        store.upsert_automation(_a())
        release = threading.Event()
        sched = _sched(lambda a, trigger: release.wait(5) and _ok(a))
        assert sched.run_now("job")[0] is True
        ok, why = sched.run_now("job")
        release.set()
        assert ok is False and "уже выполняется" in why

    def test_already_running_in_another_window(self, project):
        store.upsert_automation(_a())
        store.claim_run("job")
        ok, why = _sched(_ok).run_now("job")
        assert ok is False and "другом окне" in why

    def test_stop_goes_to_the_runner(self, project):
        class R:
            stopped = []
            def __call__(self, a): return _ok(a)
            def stop(self, name): self.stopped.append(name); return True

        runner = R()
        assert _sched(runner).stop_run("x") is True and runner.stopped == ["x"]
        assert _sched(_ok).stop_run("x") is False


class TestRaces:
    def test_the_lock_is_free_by_the_time_finished_is_announced(self, project):
        """Whoever hears 'finished' may tick or run the job again straight away."""
        store.upsert_automation(_a())
        seen = []
        done = threading.Event()

        def listener(event):
            if event.get("event") == "finished":
                seen.append(store.claim_run("job"))
                done.set()

        sched = _sched(_ok, on_event=listener)
        sched.tick(NOW)
        assert done.wait(5)
        assert seen == [True]
        assert sched.status()["running"] == {}

    def test_a_run_another_process_just_finished_is_not_repeated(self, project, monkeypatch):
        """Two Argent windows decide 'due' at the same instant; one runs it, and the
        other must notice after winning the lock that the job is no longer due."""
        store.upsert_automation(_a())
        real_claim = store.claim_run

        def claim_after_the_other_finished(name, *a, **k):
            store.update_automation(name, lambda x: setattr(x, "last_run", NOW.isoformat()))
            return real_claim(name, *a, **k)

        monkeypatch.setattr(scheduler_module, "claim_run", claim_after_the_other_finished)
        ran = []
        sched = _sched(lambda a: ran.append(1) or _ok(a))
        assert sched.tick(NOW + timedelta(seconds=5)) == []
        sched.join()
        assert ran == [] and real_claim("job") is True        # and the lock was given back

    def test_the_finished_event_describes_the_run(self, project):
        store.upsert_automation(_a(mode="remind"))
        events = []
        done = threading.Event()

        def listener(event):
            events.append(event)
            done.set() if event["event"] == "finished" else None

        sched = _sched(lambda a, trigger: {"status": "ok", "summary": "hi", "alert": True,
                                           "silent": False, "new_items": 0, "denied_actions": [],
                                           "mode": "remind"}, on_event=listener)
        sched.tick(NOW)
        assert done.wait(5)
        started, finished = events
        assert started["trigger"] == "schedule"
        assert (finished["alert"], finished["mode"], finished["summary"]) == (True, "remind", "hi")

    def test_a_crash_is_reported_as_an_alert(self, project):
        store.upsert_automation(_a())
        events = []
        done = threading.Event()

        def boom(a):
            raise RuntimeError("exploded")

        sched = _sched(boom, on_event=lambda e: (events.append(e), done.set() if e["event"] == "finished" else None))
        sched.tick(NOW)
        assert done.wait(5)
        assert events[-1]["status"] == "error" and events[-1]["alert"] is True

    def test_status_shows_what_is_running(self, project):
        store.upsert_automation(_a())
        release = threading.Event()
        sched = _sched(lambda a: release.wait(5) and _ok(a))
        sched.tick(NOW)
        running = sched.status()["running"]
        release.set()
        assert list(running) == ["job"] and running["job"]["trigger"] == "schedule"


class TestHeadlessTick:
    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch):
        monkeypatch.delenv(scheduler_module.DISABLE_ENV, raising=False)

    def test_runs_what_is_due_and_waits_for_it(self, project):
        store.upsert_automation(_a("due"))
        store.upsert_automation(_a("later", schedule="daily at 03:00"))
        ran = []

        def slow(automation):
            time.sleep(0.2)
            ran.append(automation.name)
            return _ok(automation)

        assert run_due_once(NOW, runner=slow) == 0
        assert ran == ["due"]                            # and it did not return early

    def test_respects_the_pause(self, project):
        store.upsert_automation(_a())
        store.set_paused()
        ran = []
        run_due_once(NOW, runner=lambda a: ran.append(1) or _ok(a))
        assert ran == []

    def test_startup_jobs_are_not_its_business(self, project):
        store.upsert_automation(_a(schedule="on start"))
        ran = []
        run_due_once(NOW, runner=lambda a: ran.append(1) or _ok(a))
        assert ran == []

    def test_the_kill_switch_still_applies(self, project, monkeypatch):
        monkeypatch.setenv(scheduler_module.DISABLE_ENV, "1")
        store.upsert_automation(_a())
        ran = []
        assert run_due_once(NOW, runner=lambda a: ran.append(1) or _ok(a)) == 0 and ran == []


class TestSubprocessRunnerControls:
    def _runner(self, code="import time; time.sleep(60)"):
        return SubprocessRunner(timeout=60, argv=lambda a: [sys.executable, "-c", code])

    def test_a_run_can_be_stopped(self, project):
        store.upsert_automation(_a())
        runner = self._runner()
        box = {}
        worker = threading.Thread(target=lambda: box.update(runner(_a())))
        worker.start()
        deadline = time.time() + 10
        while not runner.is_running("job") and time.time() < deadline:
            time.sleep(0.05)
        assert runner.running() == ["job"]
        assert runner.stop("job") is True
        worker.join(15)
        assert not worker.is_alive()
        assert box["status"] == "error" and "stopped by the user" in box["summary"]
        assert runner.running() == [] and runner.stop("job") is False

    def test_the_per_automation_timeout_beats_the_default(self):
        runner = SubprocessRunner(timeout=1800)
        assert runner._limit_for(_a(timeout_minutes=2)) == 120
        assert runner._limit_for(_a(timeout_minutes=0)) == 1800

    def test_a_failure_the_child_could_not_report_is_an_alert_in_the_log(self, project):
        store.upsert_automation(_a())
        runner = SubprocessRunner(timeout=60, argv=lambda a: [sys.executable, "-c", "raise SystemExit(5)"])
        result = runner(_a(), trigger="retry")
        assert result["alert"] is True
        entry = store.load_runs()[-1]
        assert entry["trigger"] == "retry" and entry["alert"] is True

    def test_the_trigger_reaches_the_child(self):
        runner = SubprocessRunner()
        argv = runner._build_argv(_a(name="x y"), "manual")
        assert argv[-3:] == ["x y", "--trigger", "manual"]


class TestRealChildProcess:
    """No stand-ins: the real `python -m src.automation.runner`, end to end, for
    the modes that need no model."""

    def test_a_reminder(self, project):
        store.upsert_automation(_a("ping", mode="remind", task="hello from the child",
                                   notify="never", schedule="every 1h"))
        result = SubprocessRunner(timeout=120)(store.get_automation("ping"), trigger="manual")
        assert result["status"] == "ok" and result["summary"] == "hello from the child"
        entry = store.load_runs()[-1]
        assert entry["name"] == "ping" and entry["trigger"] == "manual"

    def test_a_command(self, project):
        script = project / "job.py"
        script.write_text("print('disk 41% full')", encoding="utf-8")
        store.upsert_automation(_a("disk", mode="command", notify="never",
                                   task=f'"{sys.executable}" "{script}"'))
        result = SubprocessRunner(timeout=120)(store.get_automation("disk"), trigger="schedule")
        assert result["status"] == "ok" and "disk 41% full" in result["summary"]
        assert store.get_automation("disk").run_count == 1

    def test_a_failing_command_is_recorded_as_an_error(self, project):
        store.upsert_automation(_a("bad", mode="command", notify="never",
                                   task=f'"{sys.executable}" -c "raise SystemExit(2)"'))
        result = SubprocessRunner(timeout=120)(store.get_automation("bad"))
        assert result["status"] == "error"
        assert store.get_automation("bad").fail_streak == 1
