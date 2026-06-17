import shutil
import types
from pathlib import Path

import pytest

import skill_manager as sm_module


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(sm_module, "get_skills_dir", lambda: str(tmp_path))
    return sm_module.SkillManager()


class TestCreateAndRead:
    def test_create_and_read_body(self, manager):
        manager.create_skill("greet", "Say hello, then ask a question.", "Greeting flow")
        assert manager.read_skill("greet") == "Say hello, then ask a question."

    def test_description_round_trips(self, manager):
        manager.create_skill("s", "Body.", "Simple description")
        meta = next(s for s in manager.list_skills() if s["name"] == "s")
        assert meta["description"] == "Simple description"

    def test_description_with_quotes_and_colon_preserved(self, manager):
        # Regression: a hand-built YAML f-string corrupted these, losing the
        # description on read ("No description provided.").
        tricky = 'Skill for: parsing "tricky" input'
        manager.create_skill("t", "Body.", tricky)
        meta = next(s for s in manager.list_skills() if s["name"] == "t")
        assert meta["description"] == tricky

    def test_unicode_description(self, manager):
        manager.create_skill("u", "Тело.", "Навык: работа с проектом")
        meta = next(s for s in manager.list_skills() if s["name"] == "u")
        assert meta["description"] == "Навык: работа с проектом"

    def test_read_missing_returns_none(self, manager):
        assert manager.read_skill("nope") is None


class TestListAndDelete:
    def test_list_multiple(self, manager):
        manager.create_skill("a", "x", "first")
        manager.create_skill("b", "y", "second")
        names = {s["name"] for s in manager.list_skills()}
        assert {"a", "b"} <= names

    def test_delete(self, manager):
        manager.create_skill("temp", "x", "d")
        assert "deleted" in manager.delete_skill("temp")
        assert manager.read_skill("temp") is None

    def test_delete_missing(self, manager):
        assert "not found" in manager.delete_skill("ghost")


def _make_skill_folder(root, name, frontmatter, body, resources=None):
    folder = root / name
    folder.mkdir(parents=True)
    fm = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    (folder / "SKILL.md").write_text(f"---\n{fm}\n---\n\n{body}", encoding="utf-8")
    for sub, files in (resources or {}).items():
        d = folder / sub
        d.mkdir()
        for fn, content in files.items():
            (d / fn).write_text(content, encoding="utf-8")
    return folder


class TestSkillMdFormat:
    def test_folder_skill_listed_with_name_and_kind(self, manager, tmp_path):
        _make_skill_folder(tmp_path, "pdf-tools",
                           {"name": "PDF Tools", "description": "Work with PDFs"},
                           "Do PDF things.")
        entry = next(s for s in manager.list_skills() if s["name"] == "pdf-tools")
        assert entry["kind"] == "folder"
        assert entry["display_name"] == "PDF Tools"
        assert entry["description"] == "Work with PDFs"

    def test_read_folder_skill_returns_body(self, manager, tmp_path):
        _make_skill_folder(tmp_path, "greet-folder",
                           {"name": "greet", "description": "d"}, "Say hello nicely.")
        assert "Say hello nicely." in manager.read_skill("greet-folder")

    def test_read_surfaces_bundled_resources_and_allowed_tools(self, manager, tmp_path):
        _make_skill_folder(
            tmp_path, "builder",
            {"name": "builder", "description": "d", "allowed-tools": "run_command, read_file"},
            "Run the build script.",
            resources={"scripts": {"build.sh": "echo build"}},
        )
        out = manager.read_skill("builder")
        assert "BUNDLED RESOURCES" in out
        assert "build.sh" in out
        assert "ALLOWED TOOLS" in out and "run_command" in out

    def test_delete_folder_skill_removes_directory(self, manager, tmp_path):
        _make_skill_folder(tmp_path, "temp-folder", {"description": "d"}, "body")
        assert "deleted" in manager.delete_skill("temp-folder")
        assert not (tmp_path / "temp-folder").exists()

    def test_flat_and_folder_coexist(self, manager, tmp_path):
        manager.create_skill("flatone", "x", "flat skill")
        _make_skill_folder(tmp_path, "folderone", {"description": "folder skill"}, "body")
        names = {s["name"] for s in manager.list_skills()}
        assert {"flatone", "folderone"} <= names


class TestImport:
    def test_import_skill_folder(self, manager, tmp_path):
        src = tmp_path / "_external"
        _make_skill_folder(src, "imported",
                           {"name": "imported", "description": "ext"}, "body",
                           resources={"references": {"doc.md": "ref"}})
        result = manager.import_skill(str(src / "imported"))
        assert "Imported skill folder 'imported'" in result
        assert manager.read_skill("imported") is not None
        # Bundled resource came along.
        assert (manager.skills_dir / "imported" / "references" / "doc.md").exists()

    def test_import_duplicate_rejected(self, manager, tmp_path):
        src = tmp_path / "_external2"
        _make_skill_folder(src, "dup", {"description": "d"}, "body")
        manager.import_skill(str(src / "dup"))
        assert "already exists" in manager.import_skill(str(src / "dup"))

    def test_import_non_skill_rejected(self, manager, tmp_path):
        d = tmp_path / "_notaskill"
        d.mkdir()
        (d / "readme.txt").write_text("nope", encoding="utf-8")
        assert "no SKILL.md" in manager.import_skill(str(d)) or "not a skill" in manager.import_skill(str(d))


class TestRepoReferenceParsing:
    @pytest.mark.parametrize("ref,expected", [
        ("owner/repo", ("https://github.com/owner/repo.git", None, None)),
        ("https://github.com/owner/repo", ("https://github.com/owner/repo.git", None, None)),
        ("https://github.com/owner/repo/", ("https://github.com/owner/repo.git", None, None)),
        ("https://github.com/owner/repo.git", ("https://github.com/owner/repo.git", None, None)),
        ("https://github.com/owner/repo/tree/main",
         ("https://github.com/owner/repo.git", "main", None)),
        ("https://github.com/AyanbekDos/unfairgaps-os/tree/main/skills/unfairgaps",
         ("https://github.com/AyanbekDos/unfairgaps-os.git", "main", "skills/unfairgaps")),
        ("git@github.com:owner/repo.git", ("git@github.com:owner/repo.git", None, None)),
    ])
    def test_parse_repo_ref(self, ref, expected):
        assert sm_module.SkillManager._parse_repo_ref(ref) == expected

    def test_parse_repo_ref_garbage(self):
        assert sm_module.SkillManager._parse_repo_ref("just some text") is None

    def test_looks_like_repo(self):
        f = sm_module.SkillManager._looks_like_repo
        assert f("owner/repo")
        assert f("https://github.com/owner/repo")
        assert f("git@github.com:owner/repo.git")
        assert f("https://gitlab.com/a/b.git")
        # local paths must NOT be mistaken for repos
        assert not f("C:/Users/x/skills/mine")
        assert not f(r"C:\Users\x\skills")

    def test_sanitize_name(self):
        s = sm_module.SkillManager._sanitize_name
        assert s("My Skill") == "My-Skill"
        assert s("a/b c") == "a-b-c"
        assert s("  ..weird.. ") == "weird"
        assert s("") == "imported_skill"


def _fake_clone_from(remote: Path):
    """Build a fake subprocess.run that 'clones' `remote` into the target dir."""
    def fake_run(cmd, *a, **k):
        target = Path(cmd[-1])
        target.mkdir(parents=True, exist_ok=True)
        for item in remote.iterdir():
            dst = target / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy(item, dst)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return fake_run


class TestInstallFromTree:
    def test_subfolder_skill_uses_its_own_name(self, manager, tmp_path):
        root = tmp_path / "_tree"
        _make_skill_folder(root / "skills", "alpha",
                           {"name": "alpha", "description": "d"}, "body",
                           resources={"references": {"r.md": "ref"}})
        msgs = manager._install_skills_from_tree(root, "fallback")
        assert any("installed 'alpha'" in m for m in msgs)
        assert manager.read_skill("alpha") is not None
        assert (manager.skills_dir / "alpha" / "references" / "r.md").exists()

    def test_root_level_skill_uses_fallback_and_skips_junk(self, manager, tmp_path):
        root = tmp_path / "_root"
        root.mkdir()
        (root / "SKILL.md").write_text("---\ndescription: d\n---\nbody", encoding="utf-8")
        (root / "references").mkdir()
        (root / "references" / "x.md").write_text("ref", encoding="utf-8")
        # a .git checkout and stray sources must NOT be copied into the skill
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("junk", encoding="utf-8")
        (root / "unrelated.py").write_text("print()", encoding="utf-8")
        msgs = manager._install_skills_from_tree(root, "myrepo")
        assert any("installed 'myrepo'" in m for m in msgs)
        assert (manager.skills_dir / "myrepo" / "references" / "x.md").exists()
        assert not (manager.skills_dir / "myrepo" / ".git").exists()
        assert not (manager.skills_dir / "myrepo" / "unrelated.py").exists()


class TestRepoImport:
    def test_import_clones_and_installs(self, manager, tmp_path, monkeypatch):
        remote = tmp_path / "_remote"
        _make_skill_folder(remote / "skills", "unfairgaps",
                           {"name": "unfairgaps", "description": "find pains"}, "body",
                           resources={"references": {"r.md": "ref"}})
        monkeypatch.setattr(sm_module.subprocess, "run", _fake_clone_from(remote))
        result = manager.import_skill("AyanbekDos/unfairgaps-os")
        assert "Installed 1 skill" in result
        assert manager.read_skill("unfairgaps") is not None
        assert (manager.skills_dir / "unfairgaps" / "references" / "r.md").exists()

    def test_import_subpath_url(self, manager, tmp_path, monkeypatch):
        # a /tree/<branch>/<subpath> URL clones, then narrows to that subfolder
        remote = tmp_path / "_remote2"
        _make_skill_folder(remote / "skills", "wanted",
                           {"name": "wanted", "description": "d"}, "body")
        _make_skill_folder(remote / "other", "ignored",
                           {"name": "ignored", "description": "d"}, "body")
        monkeypatch.setattr(sm_module.subprocess, "run", _fake_clone_from(remote))
        result = manager.import_skill(
            "https://github.com/owner/repo/tree/main/skills/wanted")
        assert "Installed 1 skill" in result
        assert manager.read_skill("wanted") is not None
        assert manager.read_skill("ignored") is None

    def test_import_git_missing(self, manager, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError()
        monkeypatch.setattr(sm_module.subprocess, "run", boom)
        assert "git is not installed" in manager.import_skill("owner/repo")

    def test_import_clone_failure(self, manager, monkeypatch):
        monkeypatch.setattr(
            sm_module.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(
                returncode=128, stdout="", stderr="fatal: repository not found"),
        )
        out = manager.import_skill("owner/missing")
        assert "git clone failed" in out and "repository not found" in out

    def test_import_repo_without_skill(self, manager, monkeypatch):
        def fake_run(cmd, *a, **k):
            target = Path(cmd[-1])
            target.mkdir(parents=True, exist_ok=True)
            (target / "README.md").write_text("nothing", encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        monkeypatch.setattr(sm_module.subprocess, "run", fake_run)
        assert "No SKILL.md" in manager.import_skill("owner/empty")

    def test_import_repo_duplicate_skipped(self, manager, tmp_path, monkeypatch):
        remote = tmp_path / "_remote3"
        _make_skill_folder(remote / "skills", "dup",
                           {"name": "dup", "description": "d"}, "body")
        monkeypatch.setattr(sm_module.subprocess, "run", _fake_clone_from(remote))
        manager.import_skill("owner/repo")
        again = manager.import_skill("owner/repo")
        assert "skipped 'dup'" in again


class TestForeignToolNameTranslation:
    def test_distinctive_names_translated(self, manager, tmp_path):
        _make_skill_folder(
            tmp_path, "research",
            {"name": "research", "description": "d"},
            "Use WebSearch to find sources, then WebFetch each page.")
        out = manager.read_skill("research")
        assert "[TOOL NAMES]" in out
        assert "WebSearch -> search_web" in out
        assert "WebFetch -> read_webpage" in out

    def test_ambiguous_name_needs_a_cue(self, manager, tmp_path):
        # "Read"/"write" as plain prose must NOT trigger a translation
        _make_skill_folder(
            tmp_path, "prose",
            {"name": "prose", "description": "d"},
            "Read through the ledger and write a short summary of it.")
        out = manager.read_skill("prose")
        assert "[TOOL NAMES]" not in out

    def test_ambiguous_name_with_tool_cue_translated(self, manager, tmp_path):
        _make_skill_folder(
            tmp_path, "pdf",
            {"name": "pdf", "description": "d"},
            "If WebFetch returns binary, use the Read tool on the cache path.")
        out = manager.read_skill("pdf")
        assert "Read -> read_file" in out
        assert "WebFetch -> read_webpage" in out

    def test_allowed_tools_line_triggers_translation(self, manager, tmp_path):
        _make_skill_folder(
            tmp_path, "builder",
            {"name": "builder", "description": "d", "allowed-tools": "Bash, Read"},
            "Run the build.")
        out = manager.read_skill("builder")
        assert "Bash -> run_command" in out
        assert "Read -> read_file" in out

    def test_no_foreign_names_no_note(self, manager, tmp_path):
        _make_skill_folder(
            tmp_path, "clean",
            {"name": "clean", "description": "d"},
            "Call search_web and read_webpage as usual.")
        assert "[TOOL NAMES]" not in manager.read_skill("clean")

    def test_flat_skill_also_translated(self, manager):
        manager.create_skill("flatcc", "First WebSearch, then read with Read tool.", "d")
        out = manager.read_skill("flatcc")
        assert "WebSearch -> search_web" in out
        assert "Read -> read_file" in out


class TestShippedSourcedResearcher:
    """The sourced-researcher skill ships with Argent — guard its shape."""

    @pytest.fixture
    def repo_manager(self, monkeypatch):
        repo_skills = Path(sm_module.__file__).resolve().parent / "skills"
        monkeypatch.setattr(sm_module, "get_skills_dir", lambda: str(repo_skills))
        return sm_module.SkillManager()

    def test_listed_as_folder_skill(self, repo_manager):
        entry = next((s for s in repo_manager.list_skills()
                      if s["name"] == "sourced-researcher"), None)
        assert entry is not None
        assert entry["kind"] == "folder"
        assert "source" in entry["description"].lower()

    def test_read_has_phases_and_rules(self, repo_manager):
        body = repo_manager.read_skill("sourced-researcher")
        assert body is not None
        for marker in ("Phase 1", "Phase 5", "Hard rules", "fetch budget"):
            assert marker in body

    def test_surfaces_bundled_reference(self, repo_manager):
        body = repo_manager.read_skill("sourced-researcher")
        assert "BUNDLED RESOURCES" in body and "example-report.md" in body

    def test_uses_native_tools_no_translation_note(self, repo_manager):
        body = repo_manager.read_skill("sourced-researcher")
        assert "search_web" in body and "read_webpage" in body
        assert "[TOOL NAMES]" not in body
