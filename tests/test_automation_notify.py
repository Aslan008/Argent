"""Telling the user something happened — and, more importantly, staying quiet
when nothing did.

A toast after every tick trains you to dismiss them unread, and then the one
that mattered goes by unread too. So the interesting tests here are the ones
that assert silence.
"""

import pytest

from src.automation import notify


class TestWhenToSpeak:
    def test_a_quiet_run_says_nothing(self):
        assert notify.notification_for("jobs", {"status": "ok", "new_items": 0}) is None

    def test_new_items_are_worth_a_word(self):
        title, body = notify.notification_for(
            "jobs", {"status": "ok", "new_items": 3, "summary": "three postings"})
        assert "jobs" in title and "3" in title
        assert "three postings" in body

    def test_a_failure_outranks_everything(self):
        title, _ = notify.notification_for(
            "jobs", {"status": "error", "new_items": 5, "summary": "boom"})
        assert "сбой" in title

    def test_a_blocked_action_names_what_was_refused(self):
        """The whole point of refusing unattended is that you find out about it."""
        title, body = notify.notification_for("jobs", {
            "status": "ok", "new_items": 0,
            "denied_actions": [{"action": "delete_file report.md"}]})
        assert "нужен ты" in title and "delete_file" in body

    def test_singular_and_plural(self):
        one, _ = notify.notification_for("j", {"status": "ok", "new_items": 1})
        many, _ = notify.notification_for("j", {"status": "ok", "new_items": 4})
        assert "1 новое" in one and "4 новых" in many

    def test_missing_fields_do_not_crash(self):
        assert notify.notification_for("j", {}) is None
        assert notify.notification_for("j", None) is None


class TestDelivery:
    def test_disabled_in_config_means_no_subprocess(self, monkeypatch):
        monkeypatch.setattr("config.get_desktop_notifications", lambda: False)
        monkeypatch.setattr(notify.subprocess, "run",
                            lambda *a, **k: pytest.fail("must not shell out"))
        assert notify.desktop_notify("t", "b") is False

    def test_a_broken_notifier_is_not_a_failed_run(self, monkeypatch):
        """Delivery is best-effort; a missing notify-send must never turn a
        successful automation into a failed one."""
        monkeypatch.setattr("config.get_desktop_notifications", lambda: True)

        def boom(*a, **k):
            raise FileNotFoundError("notify-send")

        monkeypatch.setattr(notify.subprocess, "run", boom)
        assert notify.desktop_notify("t", "b") is False

    def test_long_text_is_trimmed(self):
        assert len(notify._clean("x" * 900, notify.MAX_BODY)) == notify.MAX_BODY
        assert notify._clean("a\n b   c", 50) == "a b c"

    def test_model_text_never_reaches_the_shell_body(self, monkeypatch):
        """A summary can contain quotes, $ and backticks. Interpolating it into
        a PowerShell command would be both broken and an injection."""
        monkeypatch.setattr("config.get_desktop_notifications", lambda: True)
        monkeypatch.setattr(notify.sys, "platform", "win32")
        seen = {}

        class _Ok:
            returncode = 0

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["env"] = kwargs.get("env") or {}
            return _Ok()

        monkeypatch.setattr(notify.subprocess, "run", fake_run)
        payload = '"; Remove-Item C:\\ -Recurse; $x = `whoami`'
        assert notify.desktop_notify("t", payload) is True
        assert payload not in " ".join(seen["cmd"])
        assert seen["env"]["ARGENT_TOAST_BODY"] == payload


class TestNotifyRun:
    def test_quiet_run_returns_false_without_touching_the_desktop(self, monkeypatch):
        monkeypatch.setattr(notify, "desktop_notify",
                            lambda *a: pytest.fail("nothing to say"))
        assert notify.notify_run("jobs", {"status": "ok", "new_items": 0}) is False

    def test_noisy_run_delivers(self, monkeypatch):
        sent = []
        monkeypatch.setattr(notify, "desktop_notify",
                            lambda t, b: sent.append((t, b)) or True)
        assert notify.notify_run("jobs", {"status": "ok", "new_items": 2}) is True
        assert sent
