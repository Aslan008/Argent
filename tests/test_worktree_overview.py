"""One line at startup: what is uncommitted in the tree the agent is about to touch.

I spent a round of a session finding this out by hand — the tree held hundreds of
changed files (a package move among them) and a 180 MB data folder that `.gitignore`
did not cover. The counts come from one `git status`, no disk walking: naming the
directory is enough for a person to look.
"""

import subprocess

import pytest

from argent.doctor import _check_git, worktree_summary


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.test")
    _git(tmp_path, "config", "user.name", "T")
    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


class TestWorktreeSummary:
    def test_a_clean_tree_says_clean(self, repo):
        assert worktree_summary() == {"inside": True, "tracked": 0, "untracked": 0, "dirs": []}
        assert _check_git() == ("ok", "repository detected, clean")

    def test_a_modified_file_is_counted_as_tracked(self, repo):
        (repo / "kept.py").write_text("x = 2\n", encoding="utf-8")
        summary = worktree_summary()
        assert summary["tracked"] == 1 and summary["untracked"] == 0
        assert "1 tracked change(s)" in _check_git()[1]

    def test_an_untracked_folder_is_named_not_sized(self, repo):
        (repo / "corpus").mkdir()
        (repo / "corpus" / "a.jsonl").write_text("{}\n", encoding="utf-8")
        summary = worktree_summary()
        assert summary["untracked"] == 1
        assert summary["dirs"] == ["corpus"]
        assert "corpus" in _check_git()[1]

    def test_outside_a_repository_it_says_so(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert worktree_summary()["inside"] is False
        level, text = _check_git()
        assert level == "warn" and "not a git repository" in text


class TestStatusShowsIt:
    def test_the_row_is_present_and_quiet_on_a_small_change(self, repo):
        from src.cli.status_overview import _worktree

        (repo / "kept.py").write_text("x = 2\n", encoding="utf-8")
        name, level, text = _worktree()
        assert name == "Рабочее дерево"
        assert level == "ok"                     # one file is not terrain
        assert "1 изменено" in text

    def test_hundreds_of_changes_are_flagged(self, repo, monkeypatch):
        import src.cli.status_overview as so

        monkeypatch.setattr("argent.doctor.worktree_summary", lambda: {
            "inside": True, "tracked": 738, "untracked": 135, "dirs": ["argent"],
        })
        name, level, text = so._worktree()
        assert level == "warn"
        assert "738" in text and "argent" in text
