"""A consent prompt for a destructive command must say how much it destroys.

`git checkout -- .` against a clean tree is harmless; against forty changed files it is
the afternoon's work. The prompt used to show only the command, so the count was the
person's job to find — and mine, once, mid-session, which is where this came from.

The number rides beside the decision, never inside `action`: the automation runner and
the subagent worker read the command back out of that string.
"""

import subprocess

import pytest

import argent.approval as approval


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.test")
    _git(tmp_path, "config", "user.name", "T")
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


class TestTheScopeNumber:
    def test_it_counts_the_work_that_would_be_lost(self, repo):
        (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
        (repo / "b.py").write_text("x = 3\n", encoding="utf-8")
        (repo / "new.txt").write_text("draft\n", encoding="utf-8")

        note = approval.destructive_scope("git checkout -- .")
        assert "2 изменённых" in note
        assert "1 неотслеживаемых" in note

    def test_a_clean_tree_says_there_is_nothing_to_lose(self, repo):
        assert "терять нечего" in approval.destructive_scope("git reset --hard HEAD")

    def test_an_ordinary_command_gets_no_note(self, repo):
        for command in ("pytest -q", "git status", "git commit -m x", "Get-ChildItem ."):
            assert approval.destructive_scope(command) == "", command

    def test_it_sees_through_a_chain(self, repo):
        (repo / "a.py").write_text("x = 9\n", encoding="utf-8")
        assert "1 изменённых" in approval.destructive_scope("cd repo && git checkout -- .")

    def test_outside_a_repository_it_stays_silent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert approval.destructive_scope("git checkout -- .") == ""


class TestItReachesThePrompt:
    def test_the_note_is_set_while_the_question_is_asked(self, repo, monkeypatch):
        (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
        seen = {}

        def backend(action, destructive, grant_key):
            seen["note"] = approval.current_scope_note()
            seen["action"] = action
            return "deny"

        monkeypatch.setattr(approval, "set_approval_backend", lambda fn: approval._local.__setattr__("backend", fn))
        approval.set_approval_backend(backend)
        try:
            action = f"{approval.RUN_COMMAND_ACTION}git checkout -- ."
            assert approval.request_approval(action) is False
        finally:
            approval.reset_approval_backend()

        assert "1 изменённых" in seen["note"]
        # The action string itself is untouched: the runner parses it back into a command.
        assert seen["action"] == action
        assert approval.command_from_action(seen["action"]) == "git checkout -- ."

    def test_no_note_for_a_harmless_command(self, repo):
        seen = {}

        def backend(action, destructive, grant_key):
            seen["note"] = approval.current_scope_note()
            return "deny"

        approval.set_approval_backend(backend)
        try:
            approval.request_approval(f"{approval.RUN_COMMAND_ACTION}pytest -q")
        finally:
            approval.reset_approval_backend()
        assert seen["note"] == ""
