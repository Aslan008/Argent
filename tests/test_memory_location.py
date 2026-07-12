"""Project memory is found by walking up to the nearest .argent (not CWD-only),
stopping at the home directory so ~/.argent is never mistaken for it."""

import pytest

import memory_manager
from memory_manager import resolve_memory_file


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A monkeypatched home directory that IS an ancestor of the work dirs —
    the realistic layout (projects live under the user profile)."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: h))
    return h


def test_finds_argent_from_project_root(home, monkeypatch):
    root = home / "proj"
    (root / ".argent").mkdir(parents=True)
    (root / ".argent" / "memory.json").write_text('{"objective": "x"}', encoding="utf-8")
    monkeypatch.chdir(root)
    assert resolve_memory_file() == root / ".argent" / "memory.json"


def test_finds_argent_from_nested_subfolder(home, monkeypatch):
    # The bug: cd src used to create a fresh empty .argent and forget context.
    root = home / "proj"
    (root / ".argent").mkdir(parents=True)
    nested = root / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert resolve_memory_file() == root / ".argent" / "memory.json"


def test_new_project_falls_back_to_cwd(home, monkeypatch):
    fresh = home / "fresh"
    fresh.mkdir()
    monkeypatch.chdir(fresh)
    assert resolve_memory_file() == fresh / ".argent" / "memory.json"


def test_home_argent_is_not_adopted_as_project_memory(home, monkeypatch):
    # ~/.argent exists (Argent's global state) but must not become project memory.
    (home / ".argent").mkdir()
    work = home / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert resolve_memory_file() == work / ".argent" / "memory.json"


def test_manager_resolves_once_at_construction(home, monkeypatch):
    root = home / "proj"
    (root / ".argent").mkdir(parents=True)
    (root / ".argent" / "memory.json").write_text('{"objective": "build a game"}', encoding="utf-8")
    nested = root / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    mgr = memory_manager.MemoryManager()
    assert mgr.data.get("objective") == "build a game"    # loaded the real memory
