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
