"""Deep mutation-killing tests for project_paths.py.

These tests are designed to kill mutants by verifying exact behaviour,
boundary conditions, and the internal logic of find_project_root and
project_root_or_cwd.  Every branch, every boundary, and every constant
is exercised.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from project_paths import find_project_root, project_root_or_cwd, PROJECT_MARKERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_marker(root: Path, marker: str) -> None:
    """Create a marker inside *root* — a directory for .argent/.git, a
    file for .argent_project.json."""
    target = root / marker
    if marker.endswith(".json"):
        target.write_text("{}", encoding="utf-8")
    else:
        target.mkdir()


def _can_symlink() -> bool:
    """True if the OS / user supports symlinks (Windows may require admin)."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            os.symlink(td, os.path.join(td, "link"), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def home(tmp_path, monkeypatch):
    """Create a fake home directory and patch Path.home() to return it."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: h))
    return h


# ---------------------------------------------------------------------------
# PROJECT_MARKERS constant
# ---------------------------------------------------------------------------

class TestProjectMarkers:
    """Verify the tuple's type, length, contents, and ordering."""

    def test_is_tuple(self):
        assert isinstance(PROJECT_MARKERS, tuple)

    def test_length_is_three(self):
        assert len(PROJECT_MARKERS) == 3

    def test_exact_contents_and_order(self):
        assert PROJECT_MARKERS == (".argent", ".argent_project.json", ".git")

    def test_argent_is_first(self):
        """.argent must be checked before .argent_project.json and .git."""
        assert PROJECT_MARKERS[0] == ".argent"

    def test_argent_project_json_is_second(self):
        assert PROJECT_MARKERS[1] == ".argent_project.json"

    def test_git_is_third(self):
        assert PROJECT_MARKERS[2] == ".git"

    def test_no_extra_markers(self):
        """Common non-Argent markers must NOT be present."""
        for foreign in (".hg", ".svn", "package.json", "pyproject.toml",
                        ".gitignore", "Makefile"):
            assert foreign not in PROJECT_MARKERS

    def test_all_markers_are_strings(self):
        assert all(isinstance(m, str) for m in PROJECT_MARKERS)

    def test_no_duplicate_markers(self):
        assert len(set(PROJECT_MARKERS)) == len(PROJECT_MARKERS)


# ---------------------------------------------------------------------------
# find_project_root — each marker individually
# ---------------------------------------------------------------------------

class TestFindProjectRootEachMarker:
    """Each marker in PROJECT_MARKERS must be detected at the start dir."""

    @pytest.mark.parametrize("marker", PROJECT_MARKERS)
    def test_marker_at_start_dir(self, home, monkeypatch, marker):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, marker)
        monkeypatch.chdir(root)
        assert find_project_root() == root

    @pytest.mark.parametrize("marker", PROJECT_MARKERS)
    def test_marker_found_from_deep_subdir(self, home, monkeypatch, marker):
        root = home / "proj"
        nested = root / "a" / "b" / "c" / "d"
        nested.mkdir(parents=True)
        _make_marker(root, marker)
        monkeypatch.chdir(nested)
        assert find_project_root() == root

    @pytest.mark.parametrize("marker", PROJECT_MARKERS)
    def test_marker_found_from_immediate_child(self, home, monkeypatch, marker):
        root = home / "proj"
        child = root / "child"
        child.mkdir(parents=True)
        _make_marker(root, marker)
        monkeypatch.chdir(child)
        assert find_project_root() == root


# ---------------------------------------------------------------------------
# find_project_root — home boundary
# ---------------------------------------------------------------------------

class TestFindProjectRootHomeBoundary:
    """The walk must stop at / above home — never treat home as a root."""

    def test_no_marker_returns_none(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert find_project_root() is None

    def test_home_with_argent_marker_returns_none(self, home, monkeypatch):
        """Home itself is never a root even with .argent (global state)."""
        _make_marker(home, ".argent")
        monkeypatch.chdir(home)
        assert find_project_root() is None

    def test_home_with_all_markers_returns_none(self, home, monkeypatch):
        """Multiple markers on home still must not make it a root."""
        for m in PROJECT_MARKERS:
            _make_marker(home, m)
        monkeypatch.chdir(home)
        assert find_project_root() is None

    def test_home_with_explicit_start_param(self, home):
        """Passing home as start should also return None."""
        _make_marker(home, ".git")
        assert find_project_root(home) is None

    def test_subdir_of_home_with_marker_on_home(self, home, monkeypatch):
        """Marker on home must not be detected from a child of home."""
        _make_marker(home, ".argent")
        work = home / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        assert find_project_root() is None

    def test_walk_does_not_escape_above_home(self, tmp_path, monkeypatch):
        """An .argent far above home must not capture the session."""
        _make_marker(tmp_path, ".argent")
        h = tmp_path / "home"
        work = h / "work"
        work.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: h))
        monkeypatch.chdir(work)
        assert find_project_root() is None

    def test_start_above_home_breaks_via_parents_check(self, tmp_path, monkeypatch):
        """Starting above home: d in stop.parents triggers the break.

        This specifically kills the mutation ``d in stop.parents`` →
        ``d not in stop.parents``.
        """
        h = tmp_path / "home"
        h.mkdir()
        _make_marker(tmp_path, ".argent")  # marker is above home
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: h))
        # tmp_path is in h.parents → break immediately
        assert find_project_root(tmp_path) is None

    def test_marker_between_start_and_home_is_found(self, home, monkeypatch):
        """A marker in a directory between start and home IS detected."""
        root = home / "proj"
        sub = root / "sub"
        sub.mkdir(parents=True)
        _make_marker(root, ".argent")
        monkeypatch.chdir(sub)
        assert find_project_root() == root


# ---------------------------------------------------------------------------
# find_project_root — start parameter handling
# ---------------------------------------------------------------------------

class TestFindProjectRootStartParam:
    """start=None uses cwd; start=path uses that path; types and resolution."""

    def test_start_none_uses_cwd(self, home, monkeypatch):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        monkeypatch.chdir(root)
        assert find_project_root(None) == root

    def test_start_none_no_marker_uses_cwd_walks_to_none(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert find_project_root(None) is None

    def test_start_with_path_object(self, home):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        assert find_project_root(root) == root

    def test_start_with_string_path(self, home):
        root = home / "proj"
        nested = root / "src"
        nested.mkdir(parents=True)
        _make_marker(root, ".argent")
        assert find_project_root(str(nested)) == root

    def test_start_with_relative_path_is_resolved(self, home, monkeypatch):
        """A relative start must be resolved to an absolute path."""
        root = home / "proj"
        sub = root / "sub"
        sub.mkdir(parents=True)
        _make_marker(root, ".argent")
        monkeypatch.chdir(root)
        result = find_project_root(Path("sub"))
        assert result is not None
        assert result == root
        assert result.is_absolute()

    def test_start_with_relative_string_is_resolved(self, home, monkeypatch):
        root = home / "proj"
        sub = root / "sub"
        sub.mkdir(parents=True)
        _make_marker(root, ".argent")
        monkeypatch.chdir(root)
        result = find_project_root("sub")
        assert result == root
        assert result.is_absolute()

    def test_returned_path_is_resolved_absolute(self, home):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        result = find_project_root(root)
        assert result == root.resolve()


# ---------------------------------------------------------------------------
# find_project_root — nearest root and marker priority
# ---------------------------------------------------------------------------

class TestFindProjectRootNearest:
    """The nearest ancestor with any marker wins."""

    def test_nearest_ancestor_wins_same_marker(self, home, monkeypatch):
        outer = home / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        _make_marker(outer, ".git")
        _make_marker(inner, ".git")
        monkeypatch.chdir(inner)
        assert find_project_root() == inner

    def test_inner_argent_beats_outer_git(self, home, monkeypatch):
        outer = home / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        _make_marker(outer, ".git")
        _make_marker(inner, ".argent")
        monkeypatch.chdir(inner)
        assert find_project_root() == inner

    def test_inner_git_beats_outer_argent(self, home, monkeypatch):
        """Nearest marker wins regardless of marker type."""
        outer = home / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        _make_marker(outer, ".argent")
        _make_marker(inner, ".git")
        monkeypatch.chdir(inner)
        assert find_project_root() == inner

    def test_three_levels_nearest_wins(self, home, monkeypatch):
        a = home / "a"
        b = a / "b"
        c = b / "c"
        c.mkdir(parents=True)
        _make_marker(a, ".git")
        _make_marker(b, ".argent")
        _make_marker(c, ".git")
        monkeypatch.chdir(c)
        assert find_project_root() == c

    def test_all_three_markers_same_dir_returns_that_dir(self, home, monkeypatch):
        """When all three markers coexist, the directory is returned."""
        root = home / "proj"
        root.mkdir()
        for m in PROJECT_MARKERS:
            _make_marker(root, m)
        monkeypatch.chdir(root)
        assert find_project_root() == root


# ---------------------------------------------------------------------------
# find_project_root — edge cases
# ---------------------------------------------------------------------------

class TestFindProjectRootEdgeCases:
    """Files, non-existent paths, symlinks, marker-as-file, empty markers."""

    def test_start_is_file_in_markered_dir(self, home):
        """Starting from a file path: walk checks the file (no marker), then
        its parent which has the marker."""
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        f = root / "file.txt"
        f.write_text("hello", encoding="utf-8")
        assert find_project_root(f) == root

    def test_start_is_file_with_git_marker(self, home):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".git")
        f = root / "notes.md"
        f.write_text("x", encoding="utf-8")
        assert find_project_root(f) == root

    def test_start_is_file_no_marker_returns_none(self, home):
        """File in a directory without markers → None."""
        d = home / "plain"
        d.mkdir()
        f = d / "file.txt"
        f.write_text("x", encoding="utf-8")
        assert find_project_root(f) is None

    def test_start_path_does_not_exist_with_marker_in_parent(self, home):
        """A non-existent path whose parent has a marker is still found."""
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        ghost = root / "nonexistent" / "deep"
        result = find_project_root(ghost)
        assert result == root

    def test_start_path_does_not_exist_no_marker(self, home):
        """A non-existent path with no markers → None (no crash)."""
        ghost = home / "nonexistent"
        assert find_project_root(ghost) is None

    def test_git_as_file_not_directory(self, home, monkeypatch):
        """.git can be a file (git worktree/submodule) — exists() still works."""
        root = home / "proj"
        root.mkdir()
        (root / ".git").write_text("gitdir: ../../.git/modules/sub",
                                   encoding="utf-8")
        monkeypatch.chdir(root)
        assert find_project_root() == root

    def test_argent_as_file_not_directory(self, home, monkeypatch):
        """.argent is normally a dir, but exists() checks files too."""
        root = home / "proj"
        root.mkdir()
        (root / ".argent").write_text("not a dir", encoding="utf-8")
        monkeypatch.chdir(root)
        assert find_project_root() == root

    def test_argent_project_json_as_directory(self, home, monkeypatch):
        """.argent_project.json is normally a file; if it's a dir, still
        detected."""
        root = home / "proj"
        root.mkdir()
        (root / ".argent_project.json").mkdir()
        monkeypatch.chdir(root)
        assert find_project_root() == root

    def test_empty_marker_file_still_detected(self, home, monkeypatch):
        """An empty .argent_project.json should still be detected."""
        root = home / "proj"
        root.mkdir()
        (root / ".argent_project.json").write_text("", encoding="utf-8")
        monkeypatch.chdir(root)
        assert find_project_root() == root

    def test_marker_with_content_still_detected(self, home, monkeypatch):
        """A .argent_project.json with real content is detected."""
        root = home / "proj"
        root.mkdir()
        (root / ".argent_project.json").write_text(
            '{"goal": "build a game"}', encoding="utf-8")
        monkeypatch.chdir(root)
        assert find_project_root() == root

    @pytest.mark.skipif(not _can_symlink(),
                        reason="symlinks not supported on this platform")
    def test_symlink_to_markered_dir(self, home, monkeypatch):
        """A symlink to a directory with markers resolves correctly."""
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        link = home / "link"
        link.symlink_to(root, target_is_directory=True)
        monkeypatch.chdir(link)
        result = find_project_root()
        assert result == root

    @pytest.mark.skipif(not _can_symlink(),
                        reason="symlinks not supported on this platform")
    def test_symlink_to_subdir_of_markered_root(self, home):
        """A symlink to a subdir of a markered root resolves to the root."""
        root = home / "proj"
        sub = root / "sub"
        sub.mkdir(parents=True)
        _make_marker(root, ".argent")
        link = home / "link_to_sub"
        link.symlink_to(sub, target_is_directory=True)
        result = find_project_root(link)
        assert result == root

    def test_deeply_nested_no_marker_returns_none(self, home, monkeypatch):
        """Deep nesting without any marker → None."""
        d = home / "a" / "b" / "c" / "d" / "e"
        d.mkdir(parents=True)
        monkeypatch.chdir(d)
        assert find_project_root() is None

    def test_marker_only_in_start_dir_not_parents(self, home, monkeypatch):
        """Marker exists only in the start directory, not in any parent."""
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".git")
        monkeypatch.chdir(root)
        assert find_project_root() == root

    def test_does_not_return_partial_match(self, home, monkeypatch):
        """A directory named '.arg' must NOT match '.argent'."""
        root = home / "proj"
        root.mkdir()
        (root / ".arg").mkdir()  # not .argent
        monkeypatch.chdir(root)
        assert find_project_root() is None

    def test_does_not_match_argent_project_txt(self, home, monkeypatch):
        """A file named .argent_project.txt must NOT match .argent_project.json."""
        root = home / "proj"
        root.mkdir()
        (root / ".argent_project.txt").write_text("{}", encoding="utf-8")
        monkeypatch.chdir(root)
        assert find_project_root() is None


# ---------------------------------------------------------------------------
# project_root_or_cwd
# ---------------------------------------------------------------------------

class TestProjectRootOrCwd:
    """project_root_or_cwd falls back to cwd / start when no marker is found."""

    def test_with_marker_returns_root(self, home, monkeypatch):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        monkeypatch.chdir(root)
        assert project_root_or_cwd() == root

    def test_without_marker_returns_cwd(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert project_root_or_cwd() == plain

    def test_without_marker_start_none_returns_cwd(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert project_root_or_cwd(None) == plain

    def test_with_start_and_marker_returns_root(self, home):
        root = home / "proj"
        nested = root / "src" / "deep"
        nested.mkdir(parents=True)
        _make_marker(root, ".argent")
        assert project_root_or_cwd(nested) == root

    def test_with_start_no_marker_returns_resolved_start(self, home):
        plain = home / "plain"
        plain.mkdir()
        result = project_root_or_cwd(plain)
        assert result == plain

    def test_with_string_start_no_marker(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(home)
        result = project_root_or_cwd(str(plain))
        assert result == plain

    def test_with_string_start_and_marker(self, home):
        root = home / "proj"
        nested = root / "src"
        nested.mkdir(parents=True)
        _make_marker(root, ".git")
        result = project_root_or_cwd(str(nested))
        assert result == root

    def test_returns_absolute_path_with_marker(self, home, monkeypatch):
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".argent")
        monkeypatch.chdir(root)
        result = project_root_or_cwd()
        assert result.is_absolute()

    def test_returns_absolute_path_without_marker(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        result = project_root_or_cwd()
        assert result.is_absolute()

    def test_start_with_marker_returns_root_not_start(self, home):
        """When a marker is found, the root is returned, not the start dir."""
        root = home / "proj"
        nested = root / "src" / "deep"
        nested.mkdir(parents=True)
        _make_marker(root, ".argent")
        result = project_root_or_cwd(nested)
        assert result == root
        assert result != nested

    def test_no_marker_start_none_falls_back_to_cwd(self, home, monkeypatch):
        """No marker, no start → cwd is returned."""
        d = home / "work"
        d.mkdir()
        monkeypatch.chdir(d)
        result = project_root_or_cwd()
        assert result == d

    def test_no_marker_start_given_falls_back_to_start(self, home):
        """No marker, start given → resolved start is returned."""
        d = home / "lonely"
        d.mkdir()
        result = project_root_or_cwd(d)
        assert result == d.resolve()

    def test_marker_at_cwd_with_start_none(self, home, monkeypatch):
        """Marker at cwd, start=None → root returned (not cwd fallback)."""
        root = home / "proj"
        root.mkdir()
        _make_marker(root, ".git")
        monkeypatch.chdir(root)
        result = project_root_or_cwd()
        assert result == root

    def test_different_markers_all_trigger_root_return(self, home, monkeypatch):
        """All three markers cause project_root_or_cwd to return the root."""
        for marker in PROJECT_MARKERS:
            root = home / f"proj_{marker.replace('.', '')}"
            root.mkdir()
            _make_marker(root, marker)
            monkeypatch.chdir(root)
            assert project_root_or_cwd() == root
            # clean up for next iteration
            monkeypatch.chdir(home)

    def test_fallback_uses_resolved_path(self, home, monkeypatch):
        """The fallback path should be resolved (absolute, no symlinks)."""
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(home)
        result = project_root_or_cwd(str(plain))
        assert result == plain.resolve()
        assert result.is_absolute()
# ---------------------------------------------------------------------------
# find_project_root — OSError handling (return None, not 0)
# ---------------------------------------------------------------------------

class TestOSErrorReturn:
    """When Path.home() or Path.cwd() raises OSError, find_project_root must
    return ``None`` — never ``0`` (which is falsy but not ``None``).

    These tests kill the ``return None`` → ``return 0`` mutation on line 27
    by using ``is None`` identity checks instead of truthiness.
    """

    def test_find_project_root_oserror_returns_none(self, monkeypatch):
        """Path.home() raising OSError → find_project_root returns None."""
        def _raise_oserror(cls):
            raise OSError("home unavailable")

        monkeypatch.setattr("pathlib.Path.home", classmethod(_raise_oserror))
        result = find_project_root()
        assert result is None

    def test_find_project_root_oserror_is_not_falsy_zero(self, monkeypatch):
        """The return value must be None, not 0 (both are falsy)."""
        def _raise_oserror(cls):
            raise OSError("home unavailable")

        monkeypatch.setattr("pathlib.Path.home", classmethod(_raise_oserror))
        result = find_project_root()
        assert result is None
        assert result != 0
        assert not (result is None and result == 0)

    def test_project_root_or_cwd_oserror_falls_back(self, monkeypatch, tmp_path):
        """project_root_or_cwd must fall back to cwd when home() raises.

        find_project_root returns None on OSError, so project_root_or_cwd
        uses its ``or`` fallback to Path.cwd().
        """
        def _raise_oserror(cls):
            raise OSError("home unavailable")

        monkeypatch.setattr("pathlib.Path.home", classmethod(_raise_oserror))
        monkeypatch.chdir(tmp_path)
        result = project_root_or_cwd()
        assert result is not None
        assert result == tmp_path.resolve()