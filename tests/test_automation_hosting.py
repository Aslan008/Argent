"""Something has to run the schedule.

`/tasks add report | daily at 09:00 | ...` stores a definition; the thing that
fires it is AutomationScheduler. It used to be started by the GUI server, and
when that server was removed nothing started it any more — definitions were
accepted, the README promised "scheduled tasks that run while Argent is open",
and no schedule ever fired. A component with no caller fails silently, so the
callers are pinned here, together with the two things that make hosting it in an
interactive program safe: jobs run in their own process, and two Argent windows
do not both run the same job.
"""

import json
import os
import sys
import textwrap
import threading
import time
from datetime import datetime, timedelta

import pytest

from src.automation import runner as runner_module
from src.automation import scheduler as scheduler_module
from src.automation import store
from src.automation.runner import RESULT_MARKER, SubprocessRunner
from src.automation.scheduler import AutomationScheduler
from src.automation.store import Automation


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


# --- the run lock ----------------------------------------------------------------

class TestRunLock:
    def test_the_first_claim_wins_and_the_second_is_refused(self, project):
        assert store.claim_run("report") is True
        assert store.claim_run("report") is False

    def test_releasing_lets_the_next_claim_through(self, project):
        store.claim_run("report")
        store.release_run("report")
        assert store.claim_run("report") is True

    def test_different_automations_do_not_block_each_other(self, project):
        assert store.claim_run("a") is True
        assert store.claim_run("b") is True

    def test_names_that_differ_only_in_odd_characters_get_separate_locks(self, project):
        assert store.claim_run("my report") is True
        assert store.claim_run("my/report") is True

    def test_a_lock_left_by_a_dead_process_is_reclaimed(self, project, monkeypatch):
        store.claim_run("report")
        monkeypatch.setattr(store, "_pid_alive", lambda pid: False)
        assert store.claim_run("report") is True

    def test_a_lock_held_by_a_live_process_is_respected(self, project, monkeypatch):
        store.claim_run("report")
        monkeypatch.setattr(store, "_pid_alive", lambda pid: True)
        assert store.claim_run("report") is False

    def test_a_lock_older_than_any_run_can_last_is_stale(self, project, monkeypatch):
        long_ago = datetime.now() - timedelta(seconds=store.RUN_LOCK_STALE_SECONDS + 60)
        store.claim_run("report", now=long_ago)
        monkeypatch.setattr(store, "_pid_alive", lambda pid: True)    # alive, but far too old
        assert store.claim_run("report") is True

    def test_a_torn_lock_file_does_not_wedge_the_job_forever(self, project):
        store.claim_run("report")
        lock = store._lock_path("report")
        lock.write_text("{not json", encoding="utf-8")
        old = time.time() - store.RUN_LOCK_STALE_SECONDS - 60
        os.utime(lock, (old, old))
        assert store.claim_run("report") is True

    def test_release_leaves_a_lock_that_someone_else_took_over(self, project):
        store.claim_run("report")
        lock = store._lock_path("report")
        lock.write_text(json.dumps({"pid": os.getpid() + 1, "since": datetime.now().isoformat()}),
                        encoding="utf-8")
        store.release_run("report")
        assert lock.exists()

    def test_releasing_what_was_never_claimed_is_harmless(self, project):
        store.release_run("never-claimed")


class TestTwoWindows:
    def test_only_one_of_two_schedulers_starts_a_due_job(self, project):
        """The terminal and the TUI each run a scheduler; both would otherwise
        fire the same automation."""
        store.upsert_automation(Automation(name="report", task="t", schedule="every 1m"))
        release = threading.Event()

        def slow(automation):
            release.wait(timeout=5)
            return {"status": "ok", "summary": ""}

        first = AutomationScheduler(runner=slow, tick_seconds=1)
        second = AutomationScheduler(runner=slow, tick_seconds=1)
        try:
            assert [a.name for a in first.tick()] == ["report"]
            assert second.tick() == []
        finally:
            release.set()

    def test_the_lock_is_released_when_the_job_ends(self, project):
        store.upsert_automation(Automation(name="report", task="t", schedule="every 1m"))
        done = threading.Event()
        sched = AutomationScheduler(
            runner=lambda a: {"status": "ok", "summary": ""}, tick_seconds=1,
            on_event=lambda ev: done.set() if ev.get("event") == "finished" else None)
        sched.tick()
        assert done.wait(timeout=5)
        deadline = time.time() + 2
        while store._lock_path("report").exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not store._lock_path("report").exists()


# --- jobs run in their own process ------------------------------------------------

def _child(script: str):
    """A stand-in for `python -m src.automation.runner`."""
    return lambda automation: [sys.executable, "-c", textwrap.dedent(script)]


def _result_line(**payload) -> str:
    return "print(" + repr(RESULT_MARKER + json.dumps(payload)) + ")"


class TestSubprocessRunner:
    def _automation(self, name="job"):
        return Automation(name=name, task="t", schedule="every 1h")

    def test_the_verdict_is_read_from_the_marker_not_from_the_noise(self, project):
        """The agent's tools print freely; only the marker line is the result."""
        script = f"""
            print("tool output that is not a result")
            print("{{not json")
            {_result_line(status="ok", summary="all good", denied_actions=[], new_items=2)}
            print("trailing noise")
        """
        result = SubprocessRunner(argv=_child(script))(self._automation())
        assert result["status"] == "ok"
        assert result["summary"] == "all good"
        assert result["new_items"] == 2

    def test_a_child_that_dies_silently_is_recorded_as_an_error(self, project):
        """A job that never reported must still show up in `/tasks runs`."""
        store.upsert_automation(self._automation())
        result = SubprocessRunner(argv=_child("import sys; sys.exit(3)"))(self._automation())
        assert result["status"] == "error"
        assert "exit code 3" in result["summary"]
        runs = store.load_runs(name="job")
        assert runs[-1]["status"] == "error"

    def test_a_crash_traceback_ends_up_in_the_summary(self, project):
        result = SubprocessRunner(argv=_child("raise RuntimeError('provider unreachable')"))(
            self._automation())
        assert result["status"] == "error"
        assert "provider unreachable" in result["summary"]

    def test_a_hung_job_is_killed_at_the_deadline(self, project):
        started = time.time()
        result = SubprocessRunner(timeout=1.5, argv=_child("import time; time.sleep(300)"))(
            self._automation())
        assert time.time() - started < 20
        assert result["status"] == "error"
        assert "timed out" in result["summary"]

    def test_the_child_works_in_the_project_directory(self, project):
        script = textwrap.dedent("""
            import json, os
            print(%r + json.dumps({"status": "ok", "summary": os.getcwd(),
                                   "denied_actions": [], "new_items": 0}))
        """) % RESULT_MARKER
        result = SubprocessRunner(argv=_child(script))(self._automation())
        assert os.path.normcase(result["summary"]) == os.path.normcase(str(project))

    def test_the_child_is_not_told_a_tui_is_attached(self, project, monkeypatch):
        """ARGENT_TUI makes prompts wait for a modal nobody can answer."""
        monkeypatch.setenv("ARGENT_TUI", "1")
        script = textwrap.dedent("""
            import json, os
            print(%r + json.dumps({"status": "ok", "summary": repr(os.environ.get("ARGENT_TUI")),
                                   "denied_actions": [], "new_items": 0}))
        """) % RESULT_MARKER
        result = SubprocessRunner(argv=_child(script))(self._automation())
        assert result["summary"] == "None"

    def test_the_child_can_import_argent_whatever_the_project_is(self, project):
        script = textwrap.dedent("""
            import json
            import src.automation.runner
            print(%r + json.dumps({"status": "ok", "summary": "imported",
                                   "denied_actions": [], "new_items": 0}))
        """) % RESULT_MARKER
        result = SubprocessRunner(argv=_child(script))(self._automation())
        assert result["summary"] == "imported"

    def test_terminate_all_stops_jobs_in_flight(self, project):
        runner = SubprocessRunner(argv=_child("import time; time.sleep(300)"), timeout=120)
        outcome = {}
        worker = threading.Thread(
            target=lambda: outcome.setdefault("r", runner(self._automation())), daemon=True)
        worker.start()
        deadline = time.time() + 10
        while not runner._procs and time.time() < deadline:
            time.sleep(0.05)
        assert runner._procs, "the job never started"

        runner.terminate_all()
        worker.join(15)

        assert not worker.is_alive(), "the host would hang waiting for a job it stopped"
        assert outcome["r"]["status"] == "error"

    def test_the_scheduler_stops_its_jobs_with_it(self, project):
        runner = SubprocessRunner(argv=_child("import time; time.sleep(300)"), timeout=120)
        sched = AutomationScheduler(runner=runner, tick_seconds=1)
        store.upsert_automation(self._automation("long"))
        sched.tick()
        deadline = time.time() + 10
        while not runner._procs and time.time() < deadline:
            time.sleep(0.05)
        assert runner._procs

        sched.stop()

        deadline = time.time() + 15
        while runner._procs and time.time() < deadline:
            time.sleep(0.1)
        assert not runner._procs


class TestChildEntryPoint:
    def test_it_reports_an_unknown_automation_without_a_traceback(self, project, capsys):
        assert runner_module.main(["nope"]) == 2
        assert "no automation named" in capsys.readouterr().err

    def test_it_needs_exactly_one_name(self, capsys):
        assert runner_module.main([]) == 2
        assert runner_module.main(["a", "b"]) == 2

    def test_it_prints_the_verdict_on_a_marker_line(self, project, monkeypatch, capsys):
        store.upsert_automation(self._a())
        monkeypatch.setattr(runner_module, "run_and_record", lambda a, **kw: {
            "status": "ok", "summary": "done", "denied_actions": [], "new_items": 1,
            "started": datetime.now(), "finished": datetime.now()})
        assert runner_module.main(["job"]) == 0
        line = [l for l in capsys.readouterr().out.splitlines() if l.startswith(RESULT_MARKER)][-1]
        assert json.loads(line[len(RESULT_MARKER):])["summary"] == "done"

    @staticmethod
    def _a():
        return Automation(name="job", task="t", schedule="every 1h")


# --- the host starts it ------------------------------------------------------------

class TestStartingTheScheduler:
    @pytest.fixture(autouse=True)
    def _fresh_singleton(self, monkeypatch):
        monkeypatch.setattr(scheduler_module, "_instance", None)

    def test_the_test_suite_never_runs_real_jobs(self):
        """conftest sets the switch: a bridge built by a test must not fire the
        developer's own schedule."""
        assert os.environ.get(scheduler_module.DISABLE_ENV) == "1"
        assert scheduler_module.start_background_scheduler() is None

    def test_it_starts_once_and_hands_back_the_same_scheduler(self, monkeypatch):
        monkeypatch.delenv(scheduler_module.DISABLE_ENV, raising=False)
        started = []
        monkeypatch.setattr(AutomationScheduler, "start", lambda self: started.append(self))
        monkeypatch.setattr(scheduler_module.atexit, "register", lambda fn: None)

        first = scheduler_module.start_background_scheduler(on_event=print)
        second = scheduler_module.start_background_scheduler()

        assert first is second
        assert started == [first]
        assert isinstance(first._runner, SubprocessRunner)
        assert first._on_event is print

    def test_it_is_stopped_at_exit(self, monkeypatch):
        monkeypatch.delenv(scheduler_module.DISABLE_ENV, raising=False)
        registered = []
        monkeypatch.setattr(AutomationScheduler, "start", lambda self: None)
        monkeypatch.setattr(scheduler_module.atexit, "register", registered.append)
        sched = scheduler_module.start_background_scheduler()
        assert registered == [sched.stop]

    def test_the_tui_bridge_starts_it(self, monkeypatch):
        """The caller whose absence made /tasks silently do nothing."""
        monkeypatch.delenv(scheduler_module.DISABLE_ENV, raising=False)
        called = threading.Event()
        seen = {}

        def spy(on_event=None):
            seen["on_event"] = on_event
            called.set()

        monkeypatch.setattr(scheduler_module, "start_background_scheduler", spy)
        from src.bridge.repl_bridge import ReplBridge
        bridge = ReplBridge()
        assert not called.is_set()               # nothing starts before a UI is attached
        bridge.set_event_sink(lambda ev: None)   # the TUI attaching runs the startup sequence
        assert called.wait(timeout=15), "ReplBridge never started the scheduler"
        assert callable(seen["on_event"])

    def test_the_tui_stays_quiet_about_an_uneventful_run(self, monkeypatch):
        """Only news, an error or a refused action is worth a line in the chat."""
        monkeypatch.delenv(scheduler_module.DISABLE_ENV, raising=False)
        called = threading.Event()
        seen = {}

        def spy(on_event=None):
            seen["on_event"] = on_event
            called.set()

        monkeypatch.setattr(scheduler_module, "start_background_scheduler", spy)
        from src.bridge.repl_bridge import ReplBridge
        bridge = ReplBridge()
        notices = []
        bridge.set_event_sink(notices.append)
        assert called.wait(timeout=15)

        on_event = seen["on_event"]
        on_event({"type": "automation", "event": "started", "name": "m"})
        on_event({"type": "automation", "event": "finished", "name": "m", "status": "ok",
                  "denied": 0, "new_items": 0})
        assert [n for n in notices if "«m»" in n.get("text", "")] == []

        on_event({"type": "automation", "event": "finished", "name": "m", "status": "ok",
                  "denied": 0, "new_items": 3})
        on_event({"type": "automation", "event": "finished", "name": "m", "status": "error",
                  "denied": 0, "new_items": 0})
        texts = [n["text"] for n in notices if "«m»" in n.get("text", "")]
        assert len(texts) == 2
        assert "нового: 3" in texts[0]
        assert "error" in texts[1]

    def test_the_tui_always_answers_a_run_you_asked_for(self, monkeypatch):
        monkeypatch.delenv(scheduler_module.DISABLE_ENV, raising=False)
        called = threading.Event()
        seen = {}

        def spy(on_event=None):
            seen["on_event"] = on_event
            called.set()

        monkeypatch.setattr(scheduler_module, "start_background_scheduler", spy)
        from src.bridge.repl_bridge import ReplBridge
        bridge = ReplBridge()
        notices = []
        bridge.set_event_sink(notices.append)
        assert called.wait(timeout=15)

        seen["on_event"]({"type": "automation", "event": "finished", "name": "m", "status": "ok",
                          "denied": 0, "new_items": 0, "alert": False, "trigger": "manual",
                          "summary": "all clear"})
        seen["on_event"]({"type": "automation", "event": "finished", "name": "tea", "status": "ok",
                          "alert": True, "mode": "remind", "trigger": "schedule", "summary": "Заварить чай"})
        texts = [n.get("text", "") for n in notices]
        assert any("«m»" in t and "all clear" in t for t in texts)
        assert any(t == "Напоминание «tea»: Заварить чай" for t in texts)
