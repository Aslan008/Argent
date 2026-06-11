import pytest

import approval


@pytest.fixture(autouse=True)
def reset_approval_state():
    approval.set_policy(approval.POLICY_ASK)
    approval.clear_session_grants()
    yield
    approval.set_policy(approval.POLICY_ASK)
    approval.clear_session_grants()


class TestDestructiveDetection:
    @pytest.mark.parametrize("command", [
        "rm -rf build",
        "del important.txt",
        "Remove-Item -Recurse -Force src",
        "ri -Recurse build",
        "rmdir /s /q dist",
        "erase data.db",
        "format D:",
        "Format-Volume -DriveLetter D",
        "git reset --hard HEAD~3",
        "git clean -fdx",
        "git push origin master --force",
        "reg delete HKCU\\Software\\Foo /f",
        "taskkill /IM python.exe /F",
        "Stop-Process -Name notepad",
        "shutdown /s /t 0",
        "Clear-Content log.txt",
    ])
    def test_destructive_commands_detected(self, command):
        assert approval.is_destructive_command(command)

    @pytest.mark.parametrize("command", [
        "python -m pytest -q",
        "git status --short",
        "git commit -m 'fix'",
        "npm install",
        "Get-ChildItem -Recurse",
        "pip list",
        "echo hello",
    ])
    def test_safe_commands_not_flagged(self, command):
        assert not approval.is_destructive_command(command)

    def test_empty_command_is_safe(self):
        assert not approval.is_destructive_command("")
        assert not approval.is_destructive_command(None)


class TestGrantKey:
    def test_first_token_lowercased(self):
        assert approval.command_grant_key("Git status --short") == "git"

    def test_empty_returns_none(self):
        assert approval.command_grant_key("") is None
        assert approval.command_grant_key("   ") is None


class TestPolicy:
    def test_default_policy_is_ask(self):
        assert approval.get_policy() == approval.POLICY_ASK

    def test_set_and_get_policy(self):
        approval.set_policy(approval.POLICY_AUTO)
        assert approval.get_policy() == approval.POLICY_AUTO

    def test_invalid_policy_rejected(self):
        with pytest.raises(ValueError):
            approval.set_policy("yolo")

    def test_auto_policy_approves_safe_action_without_prompt(self):
        approval.set_policy(approval.POLICY_AUTO)
        # No interactive prompt must happen: if it did, the test would hang.
        assert approval.request_approval("run command: git status", grant_key="git")

    def test_session_grant_approves_without_prompt(self):
        approval._session_grants.add("git")
        assert approval.request_approval("run command: git log", grant_key="git")

    def test_session_grant_does_not_cover_destructive(self, monkeypatch):
        approval._session_grants.add("git")
        approval.set_policy(approval.POLICY_AUTO)
        prompted = {}

        class FakePrompt:
            def ask(self):
                prompted["yes"] = True
                return False

        monkeypatch.setattr(approval.questionary, "confirm", lambda *a, **k: FakePrompt())
        result = approval.request_approval(
            "run command: git reset --hard", destructive=True, grant_key="git"
        )
        assert prompted.get("yes"), "destructive action must always prompt"
        assert result is False

    def test_interactive_denial(self, monkeypatch):
        class FakePrompt:
            def ask(self):
                return False

        monkeypatch.setattr(approval.questionary, "confirm", lambda *a, **k: FakePrompt())
        assert approval.request_approval("delete file 'x'", destructive=True) is False

    def test_interactive_approval(self, monkeypatch):
        class FakePrompt:
            def ask(self):
                return True

        monkeypatch.setattr(approval.questionary, "confirm", lambda *a, **k: FakePrompt())
        assert approval.request_approval("delete file 'x'", destructive=True) is True

    def test_select_always_adds_session_grant(self, monkeypatch):
        class FakeSelect:
            def __init__(self, choices):
                self.choices = choices

            def ask(self):
                return next(c for c in self.choices if c.startswith(approval._ALWAYS_PREFIX))

        monkeypatch.setattr(
            approval.questionary, "select",
            lambda message, choices: FakeSelect(choices),
        )
        assert approval.request_approval("run command: git status", grant_key="git")
        assert "git" in approval.get_session_grants()
