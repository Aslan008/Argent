"""You must be told which file you are agreeing to destroy.

delete_file asked «удалить файл 'notes.md'» — the string the model passed, not
the path that would be unlinked. Those differed, because _resolve_path fell
back to a configured Obsidian vault when a relative path did not exist locally:
a name absent from the project but present in the vault resolved into an
unrelated directory tree, and the prompt named neither the directory nor the
fact that it had moved.

That fallback is gone with the rest of the Obsidian feature, so the two can no
longer disagree. The invariant outlives it: consent names an absolute path,
because "notes.md" alone does not say which notes.md, and the working
directory can change mid-session with /cd.
"""

from pathlib import Path

import pytest

import tools.file_ops as file_ops
import tools._helpers as helpers


class TestTheApprovalNamesTheResolvedPath:
    def test_a_delete_names_an_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "scratch.txt"
        target.write_text("x", encoding="utf-8")

        asked = []
        monkeypatch.setattr("approval.request_approval",
                            lambda action, **k: asked.append(action) or False)
        result = file_ops.delete_file("scratch.txt")

        assert asked, "удаление прошло без запроса подтверждения"
        named = Path(asked[0].split("'")[1])
        assert named.is_absolute() and named == target.resolve()
        assert "aborted" in result.lower()
        assert target.exists()

    def test_the_same_name_in_two_directories_reads_differently(self, tmp_path, monkeypatch):
        """The point of an absolute path: /cd changes what a relative one means."""
        asked = []
        monkeypatch.setattr("approval.request_approval",
                            lambda action, **k: asked.append(action) or False)
        for folder in ("a", "b"):
            here = tmp_path / folder
            here.mkdir()
            (here / "notes.md").write_text("x", encoding="utf-8")
            monkeypatch.chdir(here)
            file_ops.delete_file("notes.md")
        assert asked[0] != asked[1]


class TestThePathIsPlain:
    def test_resolution_stays_in_the_working_directory(self, tmp_path, monkeypatch):
        """No fallback anywhere else: a missing file resolves to where the
        caller said, so the caller finds out it is missing."""
        monkeypatch.chdir(tmp_path)
        assert helpers._resolve_path("nope.md") == (tmp_path / "nope.md").resolve()

    def test_a_missing_file_is_reported_not_hunted_for(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert "does not exist" in file_ops.read_file("nowhere/at/all.md")
