"""`create_artifact` used to write wherever the name pointed.

`Path(".argent") / "artifacts" / filename` looks contained and is not: a name of
`../../AGENTS.md` leaves the folder, and an absolute path replaces it. The tool is
reachable by a model (and by a sub-agent) that was given a name in prose, so the
containment has to be enforced here, not assumed.
"""

from pathlib import Path

import pytest

import argent.tools.misc_tools as mt


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestContainment:
    def test_a_plain_name_lands_in_the_artifacts_folder(self, project):
        out = mt.create_artifact("plan.md", "# Plan")
        assert out.startswith("Artifact created")
        assert (project / ".argent" / "artifacts" / "plan.md").read_text(encoding="utf-8") == "# Plan"

    @pytest.mark.parametrize("name", [
        "../escape.md",
        "../../AGENTS.md",
        "sub/../../escape.md",
    ])
    def test_dot_dot_cannot_walk_out(self, project, name):
        out = mt.create_artifact(name, "x")
        assert out.startswith("Error:")
        assert "'..'" in out or ".." in out
        assert not (project / "escape.md").exists()
        assert not (project / "AGENTS.md").exists()

    @pytest.mark.parametrize("name", ["C:/Windows/Temp/evil.md", "/etc/passwd", "C:evil.md"])
    def test_an_absolute_path_is_refused(self, project, name):
        out = mt.create_artifact(name, "x")
        assert out.startswith("Error:")
        assert "relative" in out or "absolute" in out

    def test_an_empty_name_says_what_it_wants(self, project):
        assert mt.create_artifact("", "x").startswith("Error:")
        assert "filename is required" in mt.create_artifact("", "x")


class TestWriting:
    def test_a_subfolder_is_created(self, project):
        out = mt.create_artifact("notes/2026/summary.md", "body")
        assert out.startswith("Artifact created")
        assert (project / ".argent" / "artifacts" / "notes" / "2026" / "summary.md").exists()

    def test_overwriting_is_allowed_and_reported(self, project):
        """Rewriting your own plan is normal — but the result must say so, or the
        previous content is lost silently."""
        mt.create_artifact("plan.md", "v1")
        out = mt.create_artifact("plan.md", "v2")
        assert out.startswith("Artifact overwrote")
        assert (project / ".argent" / "artifacts" / "plan.md").read_text(encoding="utf-8") == "v2"

    def test_the_result_points_at_a_readable_path(self, project):
        out = mt.create_artifact("p.md", "x")
        rel = out.split("read_file('", 1)[1].split("'", 1)[0]
        assert Path(rel).exists()
        assert rel == ".argent/artifacts/p.md"
