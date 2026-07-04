"""Global + project AGENTS.md loading with a shared, tier-scaled budget."""

import pathlib
from pathlib import Path

import pytest

from agent import build_agents_memory


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.chdir(proj)
    return home, proj


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestBuildAgentsMemory:
    def test_none_when_no_files(self, env):
        assert build_agents_memory(12000) == []

    def test_project_only(self, env):
        _, proj = env
        _write(proj / ".argent" / "AGENTS.md", "project rules here")
        out = build_agents_memory(12000)
        assert len(out) == 1
        assert "PROJECT INSTRUCTIONS" in out[0] and "project rules here" in out[0]

    def test_global_only(self, env):
        home, _ = env
        _write(home / ".argent" / "AGENTS.md", "always be concise")
        out = build_agents_memory(12000)
        assert len(out) == 1
        assert "GLOBAL INSTRUCTIONS" in out[0] and "always be concise" in out[0]

    def test_global_then_project_order(self, env):
        home, proj = env
        _write(home / ".argent" / "AGENTS.md", "GLOBALTEXT")
        _write(proj / ".argent" / "AGENTS.md", "PROJECTTEXT")
        out = build_agents_memory(12000)
        assert len(out) == 2
        assert "GLOBAL INSTRUCTIONS" in out[0] and "GLOBALTEXT" in out[0]
        assert "PROJECT INSTRUCTIONS" in out[1] and "PROJECTTEXT" in out[1]

    def test_bare_agents_md_in_root(self, env):
        _, proj = env
        _write(proj / "AGENTS.md", "root level rules")
        out = build_agents_memory(12000)
        assert len(out) == 1 and "root level rules" in out[0]

    def test_dot_argent_preferred_over_root(self, env):
        _, proj = env
        _write(proj / ".argent" / "AGENTS.md", "DOTARGENT")
        _write(proj / "AGENTS.md", "ROOT")
        out = build_agents_memory(12000)
        assert len(out) == 1 and "DOTARGENT" in out[0] and "ROOT" not in out[0]

    def test_shared_budget_truncates_global_and_starves_project(self, env):
        home, proj = env
        _write(home / ".argent" / "AGENTS.md", "G" * 5000)
        _write(proj / ".argent" / "AGENTS.md", "P" * 5000)
        out = build_agents_memory(2000)  # tiny tier
        # Global consumes the whole budget (truncated); the project file is
        # starved entirely (remaining == 0), so only one section comes back.
        assert len(out) == 1
        assert "GLOBAL INSTRUCTIONS" in out[0] and "truncated" in out[0]
        # Truncated to the 2000-char budget, nowhere near the full 5000.
        assert len(out[0]) < 3000

    def test_project_gets_remaining_budget(self, env):
        home, proj = env
        _write(home / ".argent" / "AGENTS.md", "G" * 1500)
        _write(proj / ".argent" / "AGENTS.md", "P" * 5000)
        out = build_agents_memory(2000)
        assert len(out) == 2
        assert "GLOBAL INSTRUCTIONS" in out[0]
        # Project got only the leftover ~500 chars and was truncated.
        assert "truncated" in out[1]
        assert len(out[1]) < 1500  # ~500 content + header/note, nowhere near 5000
