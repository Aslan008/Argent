"""Project state is anchored to the project root, not the current directory."""

import json

import pytest

from project_paths import find_project_root, project_root_or_cwd


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: h))
    return h


class TestFindProjectRoot:
    @pytest.mark.parametrize("marker", [".argent", ".argent_project.json", ".git"])
    def test_each_marker_identifies_the_root(self, home, monkeypatch, marker):
        root = home / "proj"
        nested = root / "src" / "deep"
        nested.mkdir(parents=True)
        target = root / marker
        if marker.endswith(".json"):
            target.write_text("{}", encoding="utf-8")
        else:
            target.mkdir()
        monkeypatch.chdir(nested)
        assert find_project_root() == root

    def test_none_without_markers(self, home, monkeypatch):
        plain = home / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert find_project_root() is None
        assert project_root_or_cwd() == plain

    def test_home_itself_is_never_a_root(self, home, monkeypatch):
        (home / ".argent").mkdir()          # Argent's own global state
        work = home / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        assert find_project_root() is None

    def test_walk_does_not_escape_above_home(self, tmp_path, monkeypatch):
        # An .argent far above home must not capture the session.
        (tmp_path / ".argent").mkdir()
        h = tmp_path / "home"
        work = h / "work"
        work.mkdir(parents=True)
        monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: h))
        monkeypatch.chdir(work)
        assert find_project_root() is None

    def test_nearest_root_wins(self, home, monkeypatch):
        outer = home / "outer"
        inner = outer / "inner"
        (outer / ".git").mkdir(parents=True)
        (inner / ".argent").mkdir(parents=True)
        monkeypatch.chdir(inner / ".argent")
        assert find_project_root() == inner


class TestProjectManagerAnchoring:
    def test_project_file_found_from_subfolder(self, home, monkeypatch):
        import project_manager
        root = home / "proj"
        (root / ".argent").mkdir(parents=True)
        (root / ".argent_project.json").write_text(
            json.dumps({"goal": "build a game"}), encoding="utf-8")
        nested = root / "src"
        nested.mkdir()

        monkeypatch.chdir(nested)
        pm = project_manager.ProjectManager()
        assert pm.data is not None and pm.data.get("goal") == "build a game"

    def test_save_writes_to_the_root_not_the_cwd(self, home, monkeypatch):
        import project_manager
        root = home / "proj"
        (root / ".argent").mkdir(parents=True)
        nested = root / "src"
        nested.mkdir()

        monkeypatch.chdir(nested)
        pm = project_manager.ProjectManager()
        pm.data = {"goal": "x"}
        pm._save()

        assert (root / ".argent_project.json").exists()
        assert not (nested / ".argent_project.json").exists()


class TestAgentsMemoryAnchoring:
    def test_project_agents_md_found_from_subfolder(self, home, monkeypatch):
        from agent import build_agents_memory
        root = home / "proj"
        (root / ".argent").mkdir(parents=True)
        (root / ".argent" / "AGENTS.md").write_text("PROJECT RULES", encoding="utf-8")
        nested = root / "src" / "deep"
        nested.mkdir(parents=True)

        monkeypatch.chdir(nested)
        out = build_agents_memory(12000)
        assert any("PROJECT RULES" in section for section in out)
