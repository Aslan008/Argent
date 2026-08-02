"""Ambient automation: schedules, storage, and the unattended safety model."""

from datetime import datetime, timedelta

import pytest

import approval
from src.automation import store
from src.automation.runner import (
    UNATTENDED_TOOL_DENYLIST, UnattendedGate, resolve_tools, run_and_record, run_automation,
)
from src.automation.schedule import (
    ScheduleError, is_due, next_run_after, parse_schedule,
)
from src.automation.store import Automation


class TestParseSchedule:
    @pytest.mark.parametrize("text,seconds", [
        ("every 15m", 900), ("every 1 minute", 60), ("every 2h", 7200),
        ("EVERY 1d", 86400), ("every 30 mins", 1800),
    ])
    def test_intervals(self, text, seconds):
        assert parse_schedule(text) == {"kind": "interval", "seconds": seconds}

    def test_daily(self):
        assert parse_schedule("daily at 09:30") == {"kind": "daily", "hour": 9, "minute": 30}

    def test_sub_minute_is_refused(self):
        # An unattended loop at this rate is a token fire-hose nobody is watching.
        with pytest.raises(ScheduleError, match="too small"):
            parse_schedule("every 0m")

    @pytest.mark.parametrize("text", ["", "   ", "sometimes", "every banana", "daily at 25:00"])
    def test_garbage_is_rejected(self, text):
        with pytest.raises(ScheduleError):
            parse_schedule(text)


class TestDue:
    def test_interval_never_run_fires_immediately(self):
        now = datetime(2026, 7, 12, 10, 0)
        assert is_due("every 15m", None, now) is True

    def test_interval_waits_out_the_gap(self):
        now = datetime(2026, 7, 12, 10, 0)
        assert is_due("every 15m", now - timedelta(minutes=5), now) is False
        assert is_due("every 15m", now - timedelta(minutes=15), now) is True

    def test_daily_does_not_fire_at_a_surprising_hour(self):
        """A 09:00 report must not fire at 23:40 just because that is when the
        app happened to open — a surprise run is worse than a skipped one."""
        now = datetime(2026, 7, 12, 23, 40)
        assert is_due("daily at 09:00", None, now) is False
        assert next_run_after("daily at 09:00", None, now) == datetime(2026, 7, 13, 9, 0)

    def test_daily_catches_up_when_only_a_little_late(self):
        """Argent only runs while it is open, so 09:00 routinely finds nobody
        home; without a catch-up window a daily job would be skipped every day
        the app started late."""
        assert is_due("daily at 09:00", None, datetime(2026, 7, 12, 9, 5)) is True
        assert is_due("daily at 09:00", None, datetime(2026, 7, 12, 9, 59)) is True
        assert is_due("daily at 09:00", None, datetime(2026, 7, 12, 11, 0)) is False

    def test_daily_fires_once_per_day(self):
        ran = datetime(2026, 7, 12, 9, 0)
        assert is_due("daily at 09:00", ran, datetime(2026, 7, 12, 9, 5)) is False
        assert is_due("daily at 09:00", ran, datetime(2026, 7, 13, 9, 1)) is True

    def test_daily_still_ahead_today(self):
        assert is_due("daily at 09:00", None, datetime(2026, 7, 12, 8, 0)) is False

    def test_broken_schedule_never_fires(self):
        assert is_due("every banana", None, datetime.now()) is False


@pytest.fixture
def project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    root = home / "proj"
    (root / ".argent").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    monkeypatch.chdir(root)
    return root


class TestStore:
    def test_roundtrip(self, project):
        a = Automation(name="metrics", task="collect metrics", schedule="every 1h",
                       allowed_tools=["run_command", "write_file"])
        store.upsert_automation(a)
        loaded = store.get_automation("metrics")
        assert loaded.task == "collect metrics"
        assert loaded.allowed_tools == ["run_command", "write_file"]

    def test_anchored_to_project_root_not_cwd(self, project, monkeypatch):
        store.upsert_automation(Automation(name="x", task="t", schedule="every 1h"))
        nested = project / "src" / "deep"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)
        assert store.get_automation("x") is not None      # found from a subfolder

    def test_upsert_replaces_by_name(self, project):
        store.upsert_automation(Automation(name="a", task="v1", schedule="every 1h"))
        store.upsert_automation(Automation(name="a", task="v2", schedule="every 2h"))
        items = store.load_automations()
        assert len(items) == 1 and items[0].task == "v2"

    def test_remove(self, project):
        store.upsert_automation(Automation(name="a", task="t", schedule="every 1h"))
        assert store.remove_automation("a") is True
        assert store.remove_automation("a") is False

    def test_corrupt_file_is_not_fatal(self, project):
        store.defs_path().write_text("{not json", encoding="utf-8")
        assert store.load_automations() == []

    def test_unknown_fields_are_ignored(self, project):
        store.defs_path().write_text(
            '[{"name":"a","task":"t","schedule":"every 1h","from_the_future":1}]',
            encoding="utf-8")
        assert store.load_automations()[0].name == "a"

    def test_run_history_records_denials(self, project):
        store.upsert_automation(Automation(name="a", task="t", schedule="every 1h"))
        start = datetime(2026, 7, 12, 10, 0)
        store.record_run("a", "denied", "needed approval", start,
                         start + timedelta(seconds=3),
                         [{"action": "delete file x", "destructive": True}])
        runs = store.load_runs()
        assert runs[-1]["status"] == "denied"
        assert runs[-1]["denied_actions"][0]["action"] == "delete file x"
        assert store.get_automation("a").last_status == "denied"

    def test_torn_log_line_is_skipped(self, project):
        store.runs_path().parent.mkdir(parents=True, exist_ok=True)
        store.runs_path().write_text('{"name":"a","status":"ok"}\n{"name":"b"',
                                     encoding="utf-8")
        assert [r["name"] for r in store.load_runs()] == ["a"]


class TestUnattendedGate:
    def test_denies_everything_and_remembers(self):
        gate = UnattendedGate()
        assert gate("delete file x", True, None) == "deny"
        assert gate("run git push", False, "git") == "deny"
        assert [d["action"] for d in gate.denied] == ["delete file x", "run git push"]

    def test_gate_is_used_by_request_approval(self):
        """The real approval path must route through the gate — including the
        POLICY_AUTO case, which would otherwise silently approve."""
        approval.set_policy(approval.POLICY_AUTO)
        gate = UnattendedGate()
        approval.set_approval_backend(gate)
        try:
            assert approval.request_approval("rm -rf build", destructive=True) is False
        finally:
            approval.reset_approval_backend()
            approval.set_policy(approval.POLICY_ASK)
        assert gate.denied


class TestToolBounding:
    def test_declared_tools_are_filtered_by_the_denylist(self):
        tools = resolve_tools(["read_file", "run_command", "ask_user_questions"])
        assert "ask_user_questions" not in tools
        assert tools == ["read_file", "run_command"]

    def test_empty_means_default_set(self):
        assert resolve_tools([]) is None

    def test_interactive_and_escaping_tools_are_denied(self):
        for tool in ("ask_user_questions", "run_admin_command", "wait_heartbeat"):
            assert tool in UNATTENDED_TOOL_DENYLIST


class _FakeAgent:
    """Emits a scripted chunk stream, recording the tools it was given."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.allowed_tools = "unset"

    def process_user_input(self, text, allowed_tools=None):
        self.allowed_tools = allowed_tools
        for c in self._chunks:
            yield c


class TestRunAutomation:
    def test_collects_content_and_reports_ok(self):
        agent = _FakeAgent([{"type": "content_stream", "content": "CPU 42%"}])
        a = Automation(name="m", task="check cpu", schedule="every 1h")
        result = run_automation(a, agent=agent)
        assert result["status"] == "ok" and result["summary"] == "CPU 42%"

    def test_passes_the_bounded_toolset_to_the_agent(self):
        agent = _FakeAgent([{"type": "content_stream", "content": "x"}])
        a = Automation(name="m", task="t", schedule="every 1h",
                       allowed_tools=["read_file", "ask_user_questions"])
        run_automation(a, agent=agent)
        assert agent.allowed_tools == ["read_file"]

    def test_turn_budget_stops_the_run(self):
        chunks = [{"type": "tool_end", "name": "read_file", "result": "x"} for _ in range(50)]
        a = Automation(name="m", task="t", schedule="every 1h", max_turns=3)
        result = run_automation(a, agent=_FakeAgent(chunks))
        assert "3-turn budget" in result["summary"]

    def test_crash_is_contained(self):
        class Boom:
            def process_user_input(self, text, allowed_tools=None):
                raise RuntimeError("provider exploded")
                yield  # pragma: no cover

        result = run_automation(Automation(name="m", task="t", schedule="every 1h"),
                                agent=Boom())
        assert result["status"] == "error" and "provider exploded" in result["summary"]

    def test_silent_run_that_only_wanted_approval_is_not_a_success(self, monkeypatch):
        class NeedsApproval:
            def process_user_input(self, text, allowed_tools=None):
                approval.request_approval("delete everything", destructive=True)
                return
                yield  # pragma: no cover

        result = run_automation(Automation(name="m", task="t", schedule="every 1h"),
                                agent=NeedsApproval())
        assert result["status"] == "denied"
        assert result["denied_actions"][0]["action"] == "delete everything"

    def test_backend_is_reset_even_on_crash(self):
        class Boom:
            def process_user_input(self, text, allowed_tools=None):
                raise RuntimeError("x")
                yield  # pragma: no cover

        run_automation(Automation(name="m", task="t", schedule="every 1h"), agent=Boom())
        # A leaked deny-everything backend would break the user's next turn.
        assert approval._active_backend() is approval._terminal_decision

class TestScheduler:
    def _sched(self, runner=None, on_event=None):
        from src.automation.scheduler import AutomationScheduler
        return AutomationScheduler(runner=runner, on_event=on_event, tick_seconds=1)

    def test_runs_only_what_is_due_and_enabled(self, project):
        store.upsert_automation(Automation(name="due", task="t", schedule="every 1m"))
        store.upsert_automation(Automation(name="off", task="t", schedule="every 1m",
                                           enabled=False))
        store.upsert_automation(Automation(
            name="later", task="t", schedule="every 1h",
            last_run=datetime(2026, 7, 12, 10, 0).isoformat()))

        sched = self._sched(runner=lambda a: {"status": "ok", "summary": ""})
        started = [a.name for a in sched.tick(now=datetime(2026, 7, 12, 10, 30))]
        assert started == ["due"]

    def test_overlapping_run_is_skipped(self, project):
        """A job slower than its interval must not start a second copy of
        itself — concurrent writers to the same report would corrupt it."""
        store.upsert_automation(Automation(name="slow", task="t", schedule="every 1m"))
        release = __import__("threading").Event()

        def slow_runner(automation):
            release.wait(timeout=5)
            return {"status": "ok", "summary": ""}

        sched = self._sched(runner=slow_runner)
        assert [a.name for a in sched.tick()] == ["slow"]
        assert sched.tick() == []          # still running -> not started again
        release.set()

    def test_emits_started_and_finished_events(self, project):
        store.upsert_automation(Automation(name="a", task="t", schedule="every 1m"))
        events = []
        done = __import__("threading").Event()

        def listener(ev):
            events.append(ev)
            if ev.get("event") == "finished":
                done.set()

        sched = self._sched(runner=lambda a: {"status": "ok", "summary": "all good"},
                            on_event=listener)
        sched.tick()
        assert done.wait(timeout=5)
        assert [e["event"] for e in events] == ["started", "finished"]
        assert events[-1]["summary"] == "all good"

    def test_crashing_job_does_not_wedge_the_scheduler(self, project):
        store.upsert_automation(Automation(name="a", task="t", schedule="every 1m"))
        done = __import__("threading").Event()

        def boom(automation):
            raise RuntimeError("job exploded")

        sched = self._sched(runner=boom,
                            on_event=lambda ev: done.set() if ev.get("event") == "finished" else None)
        sched.tick()
        assert done.wait(timeout=5)
        # The name must be released, or the automation would never run again.
        assert sched.tick() != []

    def test_broken_listener_does_not_break_the_run(self, project):
        store.upsert_automation(Automation(name="a", task="t", schedule="every 1m"))
        ran = __import__("threading").Event()

        def runner(automation):
            ran.set()
            return {"status": "ok", "summary": ""}

        sched = self._sched(runner=runner,
                            on_event=lambda ev: (_ for _ in ()).throw(RuntimeError("ui gone")))
        sched.tick()
        assert ran.wait(timeout=5)


class TestRunAndRecord:
    def test_run_and_record_persists(self, project):
        a = Automation(name="m", task="t", schedule="every 1h")
        store.upsert_automation(a)
        run_and_record(a, agent=_FakeAgent([{"type": "content_stream", "content": "done"}]))
        runs = store.load_runs(name="m")
        assert runs[-1]["status"] == "ok" and runs[-1]["summary"] == "done"
