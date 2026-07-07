"""Refuse to create a Unity script outside Assets/, where Unity never sees it."""

from tools._helpers import _unity_project_root, _unity_script_placement_error
from tools.file_ops import write_file


def _make_unity_project(tmp_path):
    root = tmp_path / "Proj"
    (root / "Assets" / "Scripts").mkdir(parents=True)
    (root / "ProjectSettings").mkdir()
    (root / "Packages").mkdir()
    return root


def test_detects_project_root(tmp_path):
    root = _make_unity_project(tmp_path)
    assert _unity_project_root(root / "Scripts" / "Enemy.cs") == root.resolve()


def test_plain_dir_has_no_root(tmp_path):
    (tmp_path / "src").mkdir()
    assert _unity_project_root(tmp_path / "src" / "a.cs") is None


def test_script_outside_assets_is_refused(tmp_path):
    root = _make_unity_project(tmp_path)
    bad = root / "Scripts" / "Enemy.cs"          # sibling of Assets, NOT inside it
    out = write_file(str(bad), "public class Enemy {}")
    assert "OUTSIDE the Assets/ folder" in out
    assert "Reimport All" in out
    assert not bad.exists()                       # nothing written


def test_script_inside_assets_is_allowed(tmp_path):
    root = _make_unity_project(tmp_path)
    good = root / "Assets" / "Scripts" / "Enemy.cs"
    out = write_file(str(good), "public class Enemy {}")
    assert "Successfully wrote" in out and good.exists()


def test_script_inside_packages_is_allowed(tmp_path):
    root = _make_unity_project(tmp_path)
    good = root / "Packages" / "my.pkg" / "Runtime" / "Thing.cs"
    out = write_file(str(good), "public class Thing {}")
    assert "Successfully wrote" in out and good.exists()


def test_cs_outside_any_unity_project_is_allowed(tmp_path):
    f = tmp_path / "plain" / "Program.cs"
    out = write_file(str(f), "class Program {}")
    assert "Successfully wrote" in out and f.exists()


def test_non_script_outside_assets_is_allowed(tmp_path):
    root = _make_unity_project(tmp_path)
    # A README next to Assets/ is fine — the guard is script-only.
    f = root / "README.md"
    out = write_file(str(f), "# notes")
    assert "Successfully wrote" in out and f.exists()


def test_overwriting_existing_misplaced_script_is_not_blocked(tmp_path):
    root = _make_unity_project(tmp_path)
    bad = root / "Scripts" / "Old.cs"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("public class Old {}", encoding="utf-8")   # already exists
    out = write_file(str(bad), "public class Old { int x; }", overwrite=True)
    assert "Successfully wrote" in out                        # guard is creation-only
