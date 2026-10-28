"""The checkpoint tools' own surface: what a model sees, not what the module does.

`tests/test_checkpoints.py` covers the machinery (snapshots off the branch, rewind,
stash). These pin the answers the *tools* give, because those strings are what the
agent reads: an empty answer, a wrong "nothing to roll back to", or a rollback that
runs without asking are all failures a module-level test cannot see.
"""

import subprocess

import pytest

import argent.tools.misc_tools as mt


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.test")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


class TestCheckpointTool:
    def test_a_clean_tree_says_so_instead_of_claiming_a_checkpoint(self, repo):
        out = mt.git_checkpoint("before experiment")
        assert out == "No changes to checkpoint."

    def test_dirty_tree_creates_one_and_names_the_sha(self, repo):
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        out = mt.git_checkpoint("before experiment")
        assert out.startswith("Checkpoint created: 'before experiment' (")
        assert out.rstrip().endswith(")")

    def test_outside_a_repo_it_explains_instead_of_raising(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert mt.git_checkpoint("x").startswith("Error: Not a git repository")


class TestRollbackTool:
    def test_a_denied_approval_changes_nothing(self, repo, monkeypatch):
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        mt.git_checkpoint("cp1")
        (repo / "a.txt").write_text("three\n", encoding="utf-8")

        monkeypatch.setattr("argent.approval.request_approval", lambda *a, **k: False)

        out = mt.git_rollback()
        assert out == "Rollback aborted by user."
        assert (repo / "a.txt").read_text(encoding="utf-8") == "three\n"

    def test_an_approved_rollback_restores_the_files(self, repo, monkeypatch):
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        mt.git_checkpoint("cp1")
        (repo / "a.txt").write_text("three\n", encoding="utf-8")

        monkeypatch.setattr("argent.approval.request_approval", lambda *a, **k: True)

        out = mt.git_rollback()
        assert "Rewound to checkpoint" in out
        assert (repo / "a.txt").read_text(encoding="utf-8") == "two\n"

    def test_an_unknown_name_lists_what_exists(self, repo, monkeypatch):
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        mt.git_checkpoint("cp1")
        out = mt.git_rollback(to_checkpoint="nothing-like-this")
        assert out.startswith("Error: no checkpoint matches")
        assert "cp1" in out

    def test_without_checkpoints_it_says_there_is_nothing_to_do(self, repo):
        out = mt.git_rollback()
        assert out.startswith("Error:")
        assert "no Argent Checkpoints" in out
