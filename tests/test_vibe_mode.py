"""/vibe: one switch that curates existing knobs — approval policy and
per-turn checkpoints. Nothing new, everything rewindable."""

import pytest

import argent.approval as approval
from pathlib import Path

from src.agent import checkpoints


@pytest.fixture
def restore_policy():
    yield
    approval.set_policy(approval.POLICY_ASK)


def test_toggle_on_enables_auto_policy_and_checkpoints(monkeypatch, restore_policy):
    import argent.main as main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)
    checkpoints.set_auto_checkpoint(False)

    assert main.toggle_vibe_mode(False) is True
    assert approval.get_policy() == approval.POLICY_AUTO
    assert checkpoints.auto_checkpoint_enabled() is True


def test_toggle_off_restores_ask_policy(monkeypatch, restore_policy):
    import argent.main as main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)

    assert main.toggle_vibe_mode(True) is False
    assert approval.get_policy() == approval.POLICY_ASK


def test_destructive_actions_still_prompt_in_vibe(monkeypatch, restore_policy):
    """POLICY_AUTO auto-denies destructive actions — the safety net
    /vibe relies on. Pin it here so a policy change can't silently break vibe."""
    import argent.main as main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)
    main.toggle_vibe_mode(False)                     # vibe ON

    asked = []
    approval.set_approval_backend(
        lambda action, destructive, grant_key: asked.append(action) or "deny")
    try:
        assert approval.request_approval("rm everything", destructive=True) is False
        assert asked == []                           # auto-denied, not prompted
        assert approval.request_approval("safe read", destructive=False) is True
        assert len(asked) == 0                       # safe one auto-approved
    finally:
        approval.reset_approval_backend()


def test_vibe_with_task_enables_mode_and_returns_task(monkeypatch, restore_policy):
    import argent.main as main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)
    monkeypatch.setattr(main, "offer_safety_checkpoint", lambda task: None)

    # Calling toggle_vibe_mode directly sets up state
    assert main.toggle_vibe_mode(False) is True
    assert approval.get_policy() == approval.POLICY_AUTO


class TestVibeAnswersByWhatACanBeDestroyed:
    """/vibe refuses deletion and the unreadable — not everything it cannot vet.

    The mode used to auto-deny any command graded "warn", and "warn" covered
    `python -c "print(1)"`, `taskkill` and `Remove-Item $env:TEMP\\probe.py`
    alike, which made it useless. What it refuses now is narrow: deleting the
    person's data, a command built at run time, and taking the machine down.
    """

    @pytest.fixture(autouse=True)
    def vibe(self, monkeypatch, restore_policy):
        import argent.main as main
        monkeypatch.setattr(main, "print_system", lambda *a, **k: None)
        main.set_vibe_mode(True)
        self.asked = []
        approval.set_approval_backend(self._record)
        yield
        approval.reset_approval_backend()

    def _record(self, action, destructive, grant_key):
        self.asked.append(action)
        return "deny"

    def _run(self, command):
        return approval.request_approval(
            f"{approval.RUN_COMMAND_ACTION}{command}",
            destructive=approval.is_destructive_command(command),
            grant_key=approval.command_grant_key(command))

    # -- what runs ----------------------------------------------------------

    def test_a_snippet_that_only_cannot_be_read_runs(self):
        assert self._run('python -c "import subprocess; subprocess.run([\'x\'])"') is True
        assert self._run('node -e "console.log(1)"') is True
        assert self._run('powershell -Command "Get-Date"') is True
        assert self._run("echo 'print(1)' | python") is True
        assert self.asked == []                    # never waits for a human

    def test_stopping_a_process_runs(self):
        # An autonomous run must be able to kill its own stuck interpreter.
        assert self._run("taskkill /F /PID 4321") is True
        assert self._run("Stop-Process -Name python") is True
        assert self.asked == []

    def test_housekeeping_in_temp_runs(self):
        import tempfile
        target = str(Path(tempfile.gettempdir()) / "argent-probe.log")
        assert self._run(f'Remove-Item "{target}"') is True
        assert self.asked == []

    def test_housekeeping_in_the_projects_own_scratch_runs(self):
        assert self._run("Remove-Item .argent/artifacts/probe.py") is True
        assert self.asked == []

    def test_the_sessions_and_checkpoints_are_not_scratch(self):
        # `.argent/` also holds sessions, checkpoints and permission rules:
        # deleting it would destroy what makes a mistake recoverable.
        assert self._run("Remove-Item -Recurse -Force .argent") is False
        assert self._run("Remove-Item .argent/checkpoints") is False

    # -- what is still refused ---------------------------------------------

    def test_deleting_the_persons_data_is_recoverable_or_refused(self):
        """In-project deletes run because the gate copies them first.

        `tests/test_vibe_autonomy.py` proves the copy and the restore. What is
        still refused is what cannot be put back: a git operation that throws
        work away, an overwrite, or a target outside the project.
        """
        assert self._run("Remove-Item .\\src\\main.rs") is True
        assert self._run("rm -rf build") is True
        assert self._run("git reset --hard HEAD~1") is False
        assert self._run("git clean -fd") is False
        assert self._run("git stash drop") is False
        assert self._run("truncate -s 0 notes.md") is False
        assert self.asked == []                    # decided, not left to a prompt

    def test_a_command_built_at_run_time_is_refused(self):
        # Nothing can be said about these, so they never run unattended.
        assert self._run("Invoke-Expression $payload") is False
        assert self._run("& $cmd") is False
        assert self._run("pwsh -EncodedCommand ZQBjAGgAbwA=") is False

    def test_a_deletion_hidden_in_a_snippet_is_refused(self):
        assert self._run('python -c "import shutil; shutil.rmtree(\'src\')"') is False
        assert self._run('bash -c "rm -f bar"') is False
        assert self._run("rm -rf /") is False

    # -- and the ordinary mode is untouched --------------------------------

    def test_the_ordinary_mode_still_asks(self):
        approval.set_policy(approval.POLICY_ASK)
        approval.request_approval(f"{approval.RUN_COMMAND_ACTION}python -c \"print(1)\"",
                                  destructive=False, grant_key="python")
        assert self.asked, "the ordinary mode must still ask a human"

    def test_the_transcript_says_why_it_was_allowed(self):
        # `console` keeps its own file, so ask it directly instead of relying on
        # whatever capture some other test in the session may have installed.
        import io
        from argent.ui import console

        buf = io.StringIO()
        original = console.file
        console.file = buf
        try:
            self._run('python -c "import subprocess; subprocess.run([\'x\'])"')
        finally:
            console.file = original
        assert "проверить нельзя" in buf.getvalue()

