"""What a run IS: the agent's prompt and report, shell commands, reminders, the
one pre-approval a human can give, and who gets told.
"""

import sys
from datetime import datetime

import pytest

import approval
from src.automation import notify, runner, store
from src.automation.runner import (
    SILENT_MARKER, UnattendedGate, build_prompt, command_allowed, run_and_record,
    run_automation, split_silent,
)
from src.automation.store import Automation


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


def _a(name="job", **kw):
    kw.setdefault("task", "check things")
    kw.setdefault("schedule", "every 1h")
    return Automation(name=name, **kw)


class _Agent:
    """Emits a scripted chunk stream and remembers the prompt it was given."""

    def __init__(self, chunks):
        self.chunks, self.prompt, self.tools = chunks, None, "unset"

    def process_user_input(self, text, allowed_tools=None):
        self.prompt, self.tools = text, allowed_tools
        yield from self.chunks


def _say(text):
    return {"type": "content_stream", "content": text}


TOOL_END = {"type": "tool_end", "name": "read_file", "result": "x"}


# --- the one human pre-approval -----------------------------------------------------

class TestCommandPatterns:
    @pytest.mark.parametrize("command", [
        "git log", "git   log", "git log --oneline -5", "git log -n 3 --stat",
    ])
    def test_wildcards_match_arguments(self, command):
        assert command_allowed(command, ["git log*"])

    @pytest.mark.parametrize("command", [
        "git log; rm -rf build", "git log && echo hi", "git log | head", "git log > out.txt",
        "git log $(whoami)", "git log `whoami`", "git log\nrm x", "git status", "gitlog",
    ])
    def test_a_wildcard_never_swallows_a_shell_operator(self, command):
        """'git log*' must not be stretchable into 'git log; <anything>'."""
        assert not command_allowed(command, ["git log*"])

    def test_exact_commands(self):
        assert command_allowed("npm run stats", ["npm run stats"])
        assert not command_allowed("npm run stats --all", ["npm run stats"])

    def test_a_chained_command_must_be_listed_literally(self):
        assert command_allowed("npm run a && npm run b", ["npm run a && npm run b"])
        assert not command_allowed("npm run a && npm run c", ["npm run a && npm run b"])

    def test_nothing_is_allowed_by_default(self):
        assert not command_allowed("ls", [])
        assert not command_allowed("", ["*"])
        assert not command_allowed("ls", ["", "  "])

    def test_regex_characters_in_a_pattern_are_literal(self):
        assert command_allowed("a.b", ["a.b"]) and not command_allowed("axb", ["a.b"])
        assert command_allowed("echo (x)", ["echo (x)"])


class TestGate:
    def test_approves_a_listed_safe_command_and_remembers_it(self):
        gate = UnattendedGate(["git log*"])
        assert gate(approval.RUN_COMMAND_ACTION + "git log -3", False, "git") == "once"
        assert gate.ran == ["git log -3"] and gate.denied == []

    def test_a_background_start_of_a_listed_command_counts_too(self):
        gate = UnattendedGate(["npm run dev"])
        assert gate(approval.BACKGROUND_COMMAND_ACTION + "npm run dev", False, None) == "once"

    def test_never_approves_a_destructive_command_even_if_listed(self):
        gate = UnattendedGate(["rm -rf build"])
        assert gate(approval.RUN_COMMAND_ACTION + "rm -rf build", True, None) == "deny"
        assert gate.denied and not gate.ran

    def test_everything_else_is_still_refused(self):
        gate = UnattendedGate(["git log*"])
        for action in ("удалить файл 'a.txt'", approval.RUN_COMMAND_ACTION + "git push",
                       "выполнить команду с правами АДМИНИСТРАТОРА (UAC): git log",
                       "откатить ВСЕ изменения"):
            assert gate(action, False, None) == "deny"
        assert len(gate.denied) == 4

    def test_no_list_means_the_old_behaviour(self):
        assert UnattendedGate()(approval.RUN_COMMAND_ACTION + "ls", False, "ls") == "deny"

    def test_through_the_real_run_command_tool(self, tmp_path):
        """The coupling that matters: the text run_command asks approval with is
        the text the gate reads the command back out of."""
        from tools.command_ops import run_command
        gate = UnattendedGate(["echo argent_allowed*"])
        approval.set_approval_backend(gate)
        try:
            ran = run_command("echo argent_allowed_1", cwd=str(tmp_path))
            refused = run_command("echo something_else", cwd=str(tmp_path))
        finally:
            approval.reset_approval_backend()
        assert "argent_allowed_1" in ran and "aborted" not in ran.lower()
        assert "aborted" in refused.lower() and "something_else" not in refused.replace(
            "'echo something_else'", "")
        assert [d["action"] for d in gate.denied] == [approval.RUN_COMMAND_ACTION + "echo something_else"]
        assert gate.ran == ["echo argent_allowed_1"]

    def test_policy_auto_and_session_grants_are_still_ignored(self):
        approval.set_policy(approval.POLICY_AUTO)
        gate = UnattendedGate()
        approval.set_approval_backend(gate)
        try:
            assert approval.request_approval(approval.RUN_COMMAND_ACTION + "ls", grant_key="ls") is False
        finally:
            approval.reset_approval_backend()
            approval.set_policy(approval.POLICY_ASK)

    def test_unattended_runs_cannot_edit_automations(self):
        assert "manage_automations" in runner.UNATTENDED_TOOL_DENYLIST
        assert "manage_automations" not in runner.resolve_tools(["read_file", "manage_automations"])


# --- what the model is told ---------------------------------------------------------

class TestPrompt:
    NOW = datetime(2026, 10, 8, 9, 0)

    def test_it_carries_the_task_the_clock_and_the_rules(self):
        text = build_prompt(_a(name="digest", task="Summarise the news"), self.NOW)
        assert "Summarise the news" in text and "digest" in text
        assert "2026-10-08 09:00" in text and "Thursday" in text
        assert "Nobody is watching" in text and SILENT_MARKER in text

    def test_it_shows_the_previous_report(self):
        previous = {"finished": "2026-10-07T09:00:05", "status": "ok",
                    "summary": "3 open PRs\nand more"}
        text = build_prompt(_a(), self.NOW, previous)
        assert "Previous run (2026-10-07 09:00, ok): 3 open PRs and more" in text

    def test_no_previous_run_no_previous_line(self):
        assert "Previous run" not in build_prompt(_a(), self.NOW)

    def test_the_approved_commands_are_listed(self):
        text = build_prompt(_a(allowed_commands=["git log*", "npm test"]), self.NOW)
        assert "git log*" in text and "npm test" in text

    def test_without_commands_it_does_not_promise_any(self):
        assert "without approval" not in build_prompt(_a(), self.NOW)


class TestSilentMarker:
    @pytest.mark.parametrize("text,silent,rest", [
        ("NOTHING_NEW", True, ""), ("nothing_new", True, ""), ("  NOTHING_NEW.  ", True, ""),
        ("NOTHING_NEW: quiet night", True, "quiet night"), ("Found 3 things", False, "Found 3 things"),
        ("", False, ""), ("So NOTHING_NEW today", False, "So NOTHING_NEW today"),
    ])
    def test_split(self, text, silent, rest):
        assert split_silent(text) == (silent, rest)


# --- running the agent --------------------------------------------------------------

class TestAgentRun:
    def test_the_report_is_what_follows_the_last_tool_call(self, project):
        agent = _Agent([_say("Let me look at the page. "), TOOL_END,
                        _say("Now reading the second. "), TOOL_END, _say("3 new vacancies: A, B, C")])
        result = run_automation(_a(), agent=agent)
        assert result["summary"] == "3 new vacancies: A, B, C"

    def test_a_run_that_ends_on_a_tool_call_falls_back_to_the_last_thing_said(self, project):
        result = run_automation(_a(), agent=_Agent([_say("Looked, all fine."), TOOL_END]))
        assert result["summary"] == "Looked, all fine."

    def test_content_replace_replaces(self, project):
        agent = _Agent([_say('{"tool": "x"}'),
                        {"type": "content_replace", "content": "clean"}])
        assert run_automation(_a(), agent=agent)["summary"] == "clean"

    def test_the_marker_makes_a_quiet_run(self, project):
        result = run_automation(_a(), agent=_Agent([_say(SILENT_MARKER)]))
        assert result["silent"] is True and result["summary"] == "Ничего нового."

    def test_the_marker_after_an_error_note_is_still_quiet(self, project):
        agent = _Agent([_say(SILENT_MARKER), {"type": "error", "content": "slow provider"}])
        result = run_automation(_a(), agent=agent)
        assert result["silent"] is True and "slow provider" in result["summary"]

    def test_new_items_beat_the_marker(self, project):
        from tools.misc_tools import filter_new_items

        class Both:
            def process_user_input(self, text, allowed_tools=None):
                filter_new_items(["x"])
                yield _say(SILENT_MARKER)

        result = run_automation(_a(), agent=Both())
        assert result["new_items"] == 1 and result["silent"] is False

    def test_the_model_gets_the_wrapped_prompt(self, project):
        agent = _Agent([_say("ok")])
        run_automation(_a(task="the real task"), agent=agent)
        assert agent.prompt.rstrip().endswith("the real task") and "[Scheduled run]" in agent.prompt

    def test_the_previous_report_reaches_the_next_prompt(self, project):
        store.upsert_automation(_a())
        run_and_record(_a(), agent=_Agent([_say("first report")]), notify=False)
        agent = _Agent([_say("second")])
        run_automation(_a(), agent=agent)
        assert "first report" in agent.prompt

    def test_commands_run_through_the_gate_are_returned(self, project):
        class Runs:
            def process_user_input(self, text, allowed_tools=None):
                approval.request_approval(approval.RUN_COMMAND_ACTION + "git log -1")
                yield _say("done")

        result = run_automation(_a(allowed_commands=["git log*"]), agent=Runs())
        assert result["commands"] == ["git log -1"] and result["status"] == "ok"

    def test_budget_note_goes_after_the_report(self, project):
        chunks = [_say("partial")] + [TOOL_END] * 5
        result = run_automation(_a(max_turns=2), agent=_Agent(chunks))
        assert "2-turn budget" in result["summary"]


# --- the two modes that need no model -------------------------------------------------

def _script(project, body):
    path = project / "job.py"
    path.write_text(body, encoding="utf-8")
    return f'"{sys.executable}" "{path}"'


class TestCommandMode:
    def test_success_reports_the_output(self, project):
        a = _a(mode="command", task=_script(project, "print('backup ok: 42 files')"))
        result = run_automation(a)
        assert result["status"] == "ok" and "backup ok: 42 files" in result["summary"]

    def test_a_failing_exit_code_is_an_error_with_the_output(self, project):
        a = _a(mode="command", task=_script(project, "print('disk full'); raise SystemExit(3)"))
        result = run_automation(a)
        assert result["status"] == "error"
        assert "код 3" in result["summary"] and "disk full" in result["summary"]

    def test_a_silent_success_is_quiet(self, project):
        result = run_automation(_a(mode="command", task=_script(project, "pass")))
        assert result["status"] == "ok" and result["silent"] is True

    def test_it_runs_in_the_project_root(self, project):
        a = _a(mode="command", task=_script(project, "import os; print(os.getcwd())"))
        assert str(project.resolve()).lower() in run_automation(a)["summary"].lower()

    def test_a_hung_command_is_stopped(self, project, monkeypatch):
        monkeypatch.setattr(runner, "DEFAULT_COMMAND_TIMEOUT_MINUTES", 0.03)
        a = _a(mode="command", task=_script(project, "import time; time.sleep(30)"))
        started = datetime.now()
        result = run_automation(a)
        assert result["status"] == "error" and "stopped" in result["summary"]
        assert (datetime.now() - started).total_seconds() < 15

    def test_an_unstartable_command_is_an_error_not_a_crash(self, project):
        result = run_automation(_a(mode="command", task="definitely_not_a_program_xyz"))
        assert result["status"] == "error"

    def test_no_model_is_involved(self, project, monkeypatch):
        monkeypatch.setattr("agent.ArgentAgent", lambda *a, **k: pytest.fail("no agent for a command"))
        run_automation(_a(mode="command", task=_script(project, "print(1)")))

    def test_output_in_the_oem_code_page_is_still_readable(self):
        assert "привет" in runner._decode("привет".encode("cp866")) or sys.platform != "win32"


class TestRemindMode:
    def test_the_message_is_the_report(self, project, monkeypatch):
        monkeypatch.setattr("agent.ArgentAgent", lambda *a, **k: pytest.fail("no agent for a reminder"))
        result = run_automation(_a(mode="remind", task="Call Anna"))
        assert result["status"] == "ok" and result["summary"] == "Call Anna"

    def test_a_reminder_always_speaks(self, project, monkeypatch):
        sent = []
        monkeypatch.setattr(notify, "desktop_notify", lambda t, b: sent.append((t, b)) or True)
        result = run_and_record(_a(name="anna", mode="remind", task="Call Anna"))
        assert result["alert"] is True and sent == [("anna", "Call Anna")]

    def test_unless_you_silenced_it(self, project, monkeypatch):
        monkeypatch.setattr(notify, "desktop_notify", lambda t, b: pytest.fail("silenced"))
        result = run_and_record(_a(mode="remind", task="x", notify="never"))
        assert result["alert"] is False


# --- who gets told ------------------------------------------------------------------

class TestAlertPolicy:
    @pytest.mark.parametrize("policy,kw,expected", [
        ("auto", dict(status="ok"), False),
        ("auto", dict(status="ok", new_items=2), True),
        ("auto", dict(status="error"), True),
        ("auto", dict(status="denied"), True),
        ("auto", dict(status="ok", denied=1), True),
        ("auto", dict(status="ok", changed=True), False),
        ("failures", dict(status="ok", new_items=5), False),
        ("failures", dict(status="error"), True),
        ("failures", dict(status="denied"), True),
        ("changes", dict(status="ok", changed=True), True),
        ("changes", dict(status="ok", changed=True, silent=True), False),
        ("changes", dict(status="ok", changed=False), False),
        ("changes", dict(status="ok", new_items=1), True),
        ("always", dict(status="ok"), True),
        ("always", dict(status="ok", silent=True), True),
        ("never", dict(status="error"), False),
        ("never", dict(status="ok", new_items=9), False),
    ])
    def test_matrix(self, policy, kw, expected):
        assert notify.should_alert(policy, **kw) is expected


class TestRunAndRecordDelivery:
    def _capture(self, monkeypatch):
        sent = []
        monkeypatch.setattr(notify, "desktop_notify", lambda t, b: sent.append((t, b)) or True)
        return sent

    def test_auto_is_quiet_about_a_plain_report(self, project, monkeypatch):
        sent = self._capture(monkeypatch)
        result = run_and_record(_a(), agent=_Agent([_say("Everything normal.")]))
        assert sent == [] and result["alert"] is False

    def test_always_sends_a_digest(self, project, monkeypatch):
        sent = self._capture(monkeypatch)
        run_and_record(_a(name="digest", notify="always"), agent=_Agent([_say("3 PRs open")]))
        assert sent == [("digest", "3 PRs open")]

    def test_always_still_says_so_when_there_was_nothing(self, project, monkeypatch):
        sent = self._capture(monkeypatch)
        run_and_record(_a(name="digest", notify="always"), agent=_Agent([_say(SILENT_MARKER)]))
        assert sent == [("digest", "Ничего нового.")]

    def test_changes_speaks_once_per_distinct_report(self, project, monkeypatch):
        sent = self._capture(monkeypatch)
        a = _a(name="w", notify="changes")
        store.upsert_automation(a)
        run_and_record(store.get_automation("w"), agent=_Agent([_say("price: 100")]))
        run_and_record(store.get_automation("w"), agent=_Agent([_say("price: 100")]))
        run_and_record(store.get_automation("w"), agent=_Agent([_say("price: 90")]))
        assert [body for _, body in sent] == ["price: 100", "price: 90"]

    def test_changes_ignores_a_quiet_run(self, project, monkeypatch):
        sent = self._capture(monkeypatch)
        a = _a(name="w", notify="changes")
        store.upsert_automation(a)
        run_and_record(store.get_automation("w"), agent=_Agent([_say(SILENT_MARKER)]))
        assert sent == []

    def test_a_manual_run_never_notifies_but_is_recorded(self, project, monkeypatch):
        self._capture(monkeypatch)
        monkeypatch.setattr(notify, "desktop_notify", lambda *a: pytest.fail("no toast"))
        store.upsert_automation(_a())
        run_and_record(_a(), agent=_Agent([_say("x")]), notify=False)
        assert store.load_runs()[-1]["trigger"] == "manual"

    def test_the_run_log_carries_the_verdict(self, project, monkeypatch):
        self._capture(monkeypatch)
        store.upsert_automation(_a())
        run_and_record(_a(notify="always"), agent=_Agent([_say("hello")]))
        entry = store.load_runs()[-1]
        assert entry["alert"] is True and entry["silent"] is False and entry["trigger"] == "schedule"

    def test_the_digest_is_remembered_on_the_definition(self, project, monkeypatch):
        self._capture(monkeypatch)
        store.upsert_automation(_a())
        run_and_record(_a(), agent=_Agent([_say("hello")]))
        assert store.get_automation("job").last_digest

    def test_channels_reach_telegram(self, project, monkeypatch):
        self._capture(monkeypatch)
        sent = []
        monkeypatch.setattr(notify, "telegram_notify", lambda t, b: sent.append(t) or True)
        run_and_record(_a(name="t", notify="always", channels=["telegram"]),
                       agent=_Agent([_say("x")]))
        assert sent == ["t"]

    def test_no_channel_no_telegram(self, project, monkeypatch):
        self._capture(monkeypatch)
        monkeypatch.setattr(notify, "telegram_notify", lambda *a: pytest.fail("not asked for"))
        run_and_record(_a(notify="always"), agent=_Agent([_say("x")]))

    def test_a_broken_notifier_does_not_fail_the_run(self, project, monkeypatch):
        monkeypatch.setattr(notify, "desktop_notify",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("no display")))
        result = run_and_record(_a(notify="always"), agent=_Agent([_say("x")]))
        assert result["status"] == "ok"


class TestReportFile:
    def test_the_runner_writes_the_report_not_the_model(self, project):
        store.upsert_automation(_a(name="daily", report_file="reports/{name}-{date}.md"))
        result = run_and_record(store.get_automation("daily"), agent=_Agent([_say("all quiet")]),
                                notify=False)
        written = project / "reports" / f"daily-{datetime.now():%Y-%m-%d}.md"
        assert written.exists() and "all quiet" in written.read_text(encoding="utf-8")
        assert result["report"] == str(written.resolve())
        assert store.load_runs()[-1]["report"] == str(written.resolve())

    def test_the_agent_had_no_write_tool_for_it(self, project):
        store.upsert_automation(_a(report_file="r.md", allowed_tools=["read_file"]))
        agent = _Agent([_say("x")])
        run_and_record(store.get_automation("job"), agent=agent, notify=False)
        assert "write_file" not in agent.tools

    def test_a_bad_destination_is_logged_and_the_run_survives(self, project):
        a = _a(report_file="../elsewhere.md")
        result = run_and_record(a, agent=_Agent([_say("x")]), notify=False)
        assert result["status"] == "ok" and result["report"] == ""
        assert not (project.parent / "elsewhere.md").exists()


class TestTelegram:
    @pytest.fixture
    def configured(self, monkeypatch):
        monkeypatch.setattr("config.get_telegram_token", lambda: "123:SECRET")
        monkeypatch.setattr("config.get_telegram_chat_id", lambda: "42")

    def test_posts_to_the_bot_api(self, configured, monkeypatch):
        import json
        seen = {}

        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(request, timeout=None):
            seen["url"], seen["body"] = request.full_url, json.loads(request.data)
            return Response()

        monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
        assert notify.telegram_notify("Title", "Body") is True
        assert seen["url"] == "https://api.telegram.org/bot123:SECRET/sendMessage"
        assert seen["body"]["chat_id"] == "42" and "Title" in seen["body"]["text"]

    def test_unconfigured_is_a_quiet_false(self, monkeypatch):
        monkeypatch.setattr("config.get_telegram_token", lambda: None)
        monkeypatch.setattr("config.get_telegram_chat_id", lambda: None)
        monkeypatch.setattr(notify.urllib.request, "urlopen",
                            lambda *a, **k: pytest.fail("nothing to send to"))
        assert notify.telegram_notify("t", "b") is False

    def test_a_network_error_is_swallowed_and_the_token_never_logged(self, configured, monkeypatch, caplog):
        def boom(*a, **k):
            raise OSError("network down 123:SECRET")

        monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
        with caplog.at_level("DEBUG"):
            assert notify.telegram_notify("t", "b") is False
        assert "SECRET" not in caplog.text


class TestNotificationText:
    def test_titles_by_kind(self):
        assert notify.notification_for("j", {"status": "error", "summary": "boom", "alert": True}) \
            == ("j: сбой", "boom")
        assert notify.notification_for("j", {"status": "ok", "new_items": 2, "summary": "a, b",
                                             "alert": True})[0] == "j: 2 новых"
        assert notify.notification_for("j", {"status": "denied", "denied_actions": [{"action": "rm x"}],
                                             "alert": True})[0] == "j: нужен ты"

    def test_an_explicit_alert_flag_wins_over_the_old_derivation(self):
        assert notify.notification_for("j", {"status": "error", "alert": False}) is None

    def test_old_style_results_keep_working(self):
        assert notify.notification_for("j", {"status": "ok"}) is None
        assert notify.notification_for("j", {"status": "ok", "new_items": 1}) is not None
