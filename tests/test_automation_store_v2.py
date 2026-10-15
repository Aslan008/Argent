"""The automation store beyond load/save: shared writes, self-protection, the inbox."""

import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.automation import report, store
from src.automation.store import Automation


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


def _a(name="job", **kw):
    kw.setdefault("task", "t")
    kw.setdefault("schedule", "every 1h")
    return Automation(name=name, **kw)


T0 = datetime(2026, 10, 8, 10, 0)


def _run(name, status="ok", trigger="schedule", at=T0, **kw):
    store.record_run(name, status, kw.pop("summary", "s"), at, at + timedelta(seconds=2),
                     trigger=trigger, **kw)


class TestDefinitions:
    def test_new_fields_round_trip(self, project):
        store.upsert_automation(_a(mode="command", notify="changes", channels=["telegram"],
                                   allowed_commands=["git log*"], timeout_minutes=5,
                                   report_file="r/{date}.md", retries=2, catchup_minutes=90))
        loaded = store.get_automation("job")
        assert (loaded.mode, loaded.notify, loaded.channels) == ("command", "changes", ["telegram"])
        assert loaded.allowed_commands == ["git log*"] and loaded.catchup_minutes == 90
        assert loaded.created                      # stamped on first save

    def test_a_definition_saved_by_an_older_version_still_loads(self, project):
        store.defs_path().parent.mkdir(parents=True, exist_ok=True)
        store.defs_path().write_text(
            '[{"name":"old","task":"t","schedule":"daily at 09:00","enabled":true,'
            '"allowed_tools":["read_file"],"max_turns":12,"last_run":null,"last_status":null}]',
            encoding="utf-8")
        a = store.get_automation("old")
        assert (a.mode, a.notify, a.retries, a.max_failures) == ("agent", "auto", 1, 5)

    def test_created_is_not_overwritten(self, project):
        store.upsert_automation(_a(created="2026-01-01T00:00:00"))
        assert store.get_automation("job").created == "2026-01-01T00:00:00"


class TestSharedWrites:
    def test_a_stale_copy_does_not_undo_a_later_edit(self, project):
        """The job finishing in another process must not put back the `enabled`
        flag you flipped while it ran."""
        store.upsert_automation(_a())
        stale = store.get_automation("job")            # what the running job holds
        store.update_automation("job", lambda a: setattr(a, "enabled", False))
        _run(stale.name)                               # job finishes, stamps its definition
        after = store.get_automation("job")
        assert after.enabled is False and after.last_status == "ok"

    def test_parallel_writers_lose_nothing(self, project):
        errors = []

        def writer(index):
            try:
                for round_ in range(6):
                    store.upsert_automation(_a(name=f"job{index}", task=f"v{round_}"))
            except Exception as e:                    # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert sorted(a.name for a in store.load_automations()) == [f"job{i}" for i in range(8)]

    def test_a_stuck_lock_is_broken_instead_of_wedging_everything(self, project):
        store.upsert_automation(_a())
        lock = store._argent_dir() / "automations.lock"
        lock.write_text("x", encoding="utf-8")
        import os
        old = datetime.now().timestamp() - 600
        os.utime(lock, (old, old))
        store.upsert_automation(_a(name="other"))
        assert {a.name for a in store.load_automations()} == {"job", "other"}

    def test_update_of_a_missing_automation_is_none(self, project):
        assert store.update_automation("ghost", lambda a: None) is None


class TestFindAutomation:
    def test_exact_prefix_and_case(self, project):
        store.upsert_automation(_a("Daily report"))
        store.upsert_automation(_a("Deploy check"))
        assert store.find_automation("Daily report")[0].name == "Daily report"
        assert store.find_automation("daily report")[0].name == "Daily report"
        assert store.find_automation("dep")[0].name == "Deploy check"

    def test_ambiguity_is_reported_not_guessed(self, project):
        store.upsert_automation(_a("report-a"))
        store.upsert_automation(_a("report-b"))
        found, candidates = store.find_automation("report")
        assert found is None and sorted(candidates) == ["report-a", "report-b"]

    def test_unknown(self, project):
        assert store.find_automation("zzz") == (None, [])


class TestValidation:
    @pytest.mark.parametrize("name", ["", "  ", " lead", "x" * 61, "a|b", 'a"b', "line\nbreak",
                                      "all", "ВСЕ"])
    def test_bad_names(self, name):
        assert store.validate_name(name)

    @pytest.mark.parametrize("name", ["report", "Ежедневный отчёт", "check-1.2", "a b"])
    def test_good_names(self, name):
        assert store.validate_name(name) is None

    def test_a_good_definition_has_no_problems(self, project):
        assert store.validate_automation(_a()) == []

    def test_every_problem_is_listed(self, project):
        bad = _a(name="", task="", schedule="sometimes", mode="x", notify="loud",
                 channels=["fax"], timeout_minutes=-1, max_turns=0, retries=9,
                 allowed_commands=["ok", " "], report_file="../outside.md")
        text = " | ".join(store.validate_automation(bad))
        for needle in ("имя", "режим", "задача", "sometimes", "notify", "каналы", "timeout",
                       "turns", "retries", "пустая строка", "внутри проекта"):
            assert needle in text, needle

    def test_task_wording_follows_the_mode(self, project):
        assert "команда" in " ".join(store.validate_automation(_a(task="", mode="command")))
        assert "напоминания" in " ".join(store.validate_automation(_a(task="", mode="remind")))


class TestOutcomeBookkeeping:
    def test_failures_pause_an_unattended_job_after_the_limit(self, project):
        store.upsert_automation(_a(max_failures=3))
        for _ in range(2):
            _run("job", "error")
        assert store.get_automation("job").enabled is True
        _run("job", "error")
        a = store.get_automation("job")
        assert a.enabled is False and "3 сбоев" in a.disabled_reason

    def test_a_success_resets_the_streak(self, project):
        store.upsert_automation(_a(max_failures=3))
        _run("job", "error"); _run("job", "error"); _run("job", "ok"); _run("job", "error")
        a = store.get_automation("job")
        assert a.enabled is True and a.fail_streak == 1

    def test_a_manual_failure_never_switches_the_job_off(self, project):
        store.upsert_automation(_a(max_failures=1))
        _run("job", "error", trigger="manual")
        assert store.get_automation("job").enabled is True

    def test_zero_means_never_pause(self, project):
        store.upsert_automation(_a(max_failures=0))
        for _ in range(10):
            _run("job", "error")
        assert store.get_automation("job").enabled is True

    def test_a_one_shot_is_done_after_its_scheduled_run(self, project):
        store.upsert_automation(_a(schedule="once at 2026-10-08 10:00"))
        _run("job")
        a = store.get_automation("job")
        assert a.enabled is False and "разовая" in a.disabled_reason

    def test_a_manual_test_run_does_not_use_up_a_one_shot(self, project):
        store.upsert_automation(_a(schedule="once at 2026-10-08 10:00"))
        _run("job", trigger="manual")
        assert store.get_automation("job").enabled is True

    def test_a_failed_one_shot_keeps_its_retries(self, project):
        store.upsert_automation(_a(schedule="once at 2026-10-08 10:00", retries=1))
        _run("job", "error")
        assert store.get_automation("job").enabled is True
        _run("job", "error")
        assert store.get_automation("job").enabled is False

    def test_run_count_and_digest(self, project):
        store.upsert_automation(_a())
        _run("job", digest="abc")
        _run("job")
        a = store.get_automation("job")
        assert a.run_count == 2 and a.last_digest == "abc"      # None keeps the old one


class TestHistory:
    def test_entries_carry_the_new_fields(self, project):
        store.upsert_automation(_a())
        _run("job", alert=True, silent=False, report="r.md", commands=["git log"], new_items=2)
        entry = store.load_runs()[-1]
        assert entry["trigger"] == "schedule" and entry["alert"] is True
        assert entry["report"] == "r.md" and entry["commands"] == ["git log"]

    def test_alert_defaults_to_anything_but_a_plain_ok(self, project):
        store.upsert_automation(_a())
        _run("job"); _run("job", "error"); _run("job", new_items=1)
        assert [e["alert"] for e in store.load_runs()] == [False, True, True]

    def test_the_log_is_actually_bounded(self, project, monkeypatch):
        """MAX_RUN_HISTORY used to limit only what was READ; the file grew forever."""
        monkeypatch.setattr(store, "_TRIM_AT_BYTES", 3000)
        monkeypatch.setattr(store, "MAX_RUN_HISTORY", 5)
        store.upsert_automation(_a())
        for i in range(120):
            _run("job", summary="x" * 100, at=T0 + timedelta(minutes=i))
        lines = store.runs_path().read_text(encoding="utf-8").splitlines()
        assert len(lines) < 15
        assert store.load_runs(limit=1)[-1]["started"].startswith("2026-10-08T11:59")

    def test_limit_zero_means_everything(self, project):
        store.upsert_automation(_a())
        for i in range(30):
            _run("job", at=T0 + timedelta(minutes=i))
        assert len(store.load_runs(limit=0)) == 30


class TestPauseAndInbox:
    def test_pause_until_resumed(self, project):
        assert store.pause_until() is None
        store.set_paused()
        assert store.pause_until() == datetime.max
        store.resume()
        assert store.pause_until() is None

    def test_a_timed_pause_expires_by_itself(self, project):
        store.set_paused(until=datetime.now() + timedelta(hours=1))
        assert store.pause_until() is not None
        store.set_paused(until=datetime.now() - timedelta(minutes=1))
        assert store.pause_until() is None

    def test_corrupt_state_reads_as_empty(self, project):
        store.state_path().parent.mkdir(parents=True, exist_ok=True)
        store.state_path().write_text("{nope", encoding="utf-8")
        assert store.load_state() == {} and store.pause_until() is None

    def test_unseen_runs_are_the_attention_worthy_ones_after_last_seen(self, project):
        store.upsert_automation(_a())
        _run("job", "error", at=T0 - timedelta(days=1))                 # before: already seen
        store.mark_seen(T0)
        _run("job", "ok", at=T0 + timedelta(hours=1))                   # plain ok: not news
        _run("job", "error", at=T0 + timedelta(hours=2))
        _run("job", "ok", at=T0 + timedelta(hours=3), new_items=3)
        _run("job", "error", trigger="manual", at=T0 + timedelta(hours=4))   # you were there
        got = store.unseen_runs()
        assert [e["status"] for e in got] == ["error", "ok"]

    def test_with_nothing_marked_seen_there_is_no_backlog(self, project):
        store.upsert_automation(_a())
        _run("job", "error")
        assert store.unseen_runs() == []

    def test_running_elsewhere_lists_live_locks_by_name(self, project, monkeypatch):
        monkeypatch.setattr(store, "_pid_alive", lambda pid: True)
        store.claim_run("my report")
        assert store.running_elsewhere() == ["my report"]
        store.release_run("my report")
        assert store.running_elsewhere() == []


class TestReportFiles:
    NOW = datetime(2026, 10, 8, 9, 30)

    def test_placeholders(self, project):
        path = report.resolve_report_path(".argent/reports/{name}-{date}.md", "Daily report",
                                          self.NOW, project)
        assert path == (project / ".argent" / "reports" / "Daily_report-2026-10-08.md").resolve()

    def test_unknown_placeholder_is_an_error(self, project):
        with pytest.raises(report.ReportPathError, match="подстановка"):
            report.resolve_report_path("r/{nope}.md", "j", self.NOW, project)

    @pytest.mark.parametrize("target", ["../escape.md", "/etc/passwd.md", "C:/Windows/x.md",
                                        "sub/../../escape.md"])
    def test_it_cannot_leave_the_project(self, project, target):
        with pytest.raises(report.ReportPathError):
            report.resolve_report_path(target, "j", self.NOW, project)

    @pytest.mark.parametrize("target", ["run.py", "run.bat", "notes", "x.json", "x.exe"])
    def test_only_plain_text_formats(self, project, target):
        with pytest.raises(report.ReportPathError, match="только"):
            report.resolve_report_path(target, "j", self.NOW, project)

    @pytest.mark.parametrize("target", [".git/hooks/x.md", ".argent/automation_runs.jsonl",
                                        ".git/info/exclude.log"])
    def test_argent_state_and_git_are_off_limits(self, project, target):
        with pytest.raises(report.ReportPathError):
            report.resolve_report_path(target, "j", self.NOW, project)

    def test_markdown_append(self, project):
        path = project / "out" / "r.md"
        report.append_report(path, "job", "ok", self.NOW, "all good")
        report.append_report(path, "job", "error", self.NOW, "broken")
        text = path.read_text(encoding="utf-8")
        assert "## 2026-10-08 09:30 — job [ok]" in text and text.count("##") == 2

    def test_jsonl_append_is_one_object_per_line(self, project):
        import json
        path = project / "r.jsonl"
        report.append_report(path, "job", "ok", self.NOW, "multi\nline")
        report.append_report(path, "job", "ok", self.NOW, "two")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["summary"] == "multi\nline" and len(rows) == 2

    def test_a_big_report_is_rotated_aside(self, project, monkeypatch):
        monkeypatch.setattr(report, "ROTATE_AT_BYTES", 50)
        path = project / "r.md"
        report.append_report(path, "job", "ok", self.NOW, "x" * 200)
        report.append_report(path, "job", "ok", self.NOW, "fresh")
        assert (project / "r.old.md").exists()
        assert "fresh" in path.read_text(encoding="utf-8")
        assert "x" * 200 not in path.read_text(encoding="utf-8")
