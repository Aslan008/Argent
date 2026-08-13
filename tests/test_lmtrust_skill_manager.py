"""Blind-spot tests for skill_manager.py.

Covers frontmatter parsing, name sanitization, repo detection/parsing,
cross-agent tool-name translation, CRUD operations against a tmp_path
skills directory, and tree-based skill installation.
"""

from pathlib import Path

import pytest

import skill_manager
from skill_manager import SkillManager, SKILL_FILE, _TOOL_NAME_MAP


# ── _parse_frontmatter ──────────────────────────────────────────────────────


class TestParseFrontmatter:
    def test_valid_yaml(self):
        content = "---\nname: test\n---\nbody"
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {"name": "test"}
        assert body == "body"

    def test_no_frontmatter(self):
        content = "Just some markdown.\nNo frontmatter here."
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {}
        assert body == content.strip()

    def test_invalid_yaml(self):
        content = "---\n: : :\n  bad: yaml: here\n---\nbody text"
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {}
        assert body == "body text"

    def test_frontmatter_no_body(self):
        content = "---\nname: test\n---\n"
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {"name": "test"}
        assert body == ""

    def test_frontmatter_multiple_fields(self):
        content = (
            "---\nname: my-skill\ndescription: A test skill\n"
            "allowed-tools: Read, Write\n---\nInstructions here."
        )
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta["name"] == "my-skill"
        assert meta["description"] == "A test skill"
        assert meta["allowed-tools"] == "Read, Write"
        assert body == "Instructions here."

    def test_frontmatter_not_dict(self):
        """YAML that parses to a list/scalar should produce empty meta."""
        content = "---\n- item1\n- item2\n---\nbody"
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta == {}
        assert body == "body"

    def test_frontmatter_only_delimiter(self):
        """A single '---' with nothing after is not valid frontmatter."""
        content = "---\njust text"
        meta, body = SkillManager._parse_frontmatter(content)
        # Only one '---' → split gives 2 parts, not >= 3, so no frontmatter
        assert meta == {}
        assert body == content.strip()


# ── _sanitize_name ──────────────────────────────────────────────────────────


class TestSanitizeName:
    def test_spaces_to_hyphens(self):
        assert SkillManager._sanitize_name("My Skill Name") == "My-Skill-Name"

    def test_slash_to_hyphen(self):
        assert SkillManager._sanitize_name("skill/name") == "skill-name"

    def test_empty_after_strip(self):
        assert SkillManager._sanitize_name("!!!") == "imported_skill"

    def test_leading_trailing_spaces(self):
        assert SkillManager._sanitize_name("  leading-trailing  ") == "leading-trailing"

    def test_strips_leading_trailing_dots_and_hyphens(self):
        assert SkillManager._sanitize_name("---name---") == "name"

    def test_preserves_alphanumeric_and_hyphens(self):
        assert SkillManager._sanitize_name("my-skill-123") == "my-skill-123"

    def test_empty_string(self):
        assert SkillManager._sanitize_name("") == "imported_skill"

    def test_only_special_chars(self):
        assert SkillManager._sanitize_name("@#$%") == "imported_skill"

    def test_preserves_dots(self):
        assert SkillManager._sanitize_name("my.skill.name") == "my.skill.name"

    def test_mixed_special_and_alnum(self):
        assert SkillManager._sanitize_name("my skill (v2)") == "my-skill-v2"


# ── _looks_like_repo ────────────────────────────────────────────────────────


class TestLooksLikeRepo:
    def test_https_github_url(self):
        assert SkillManager._looks_like_repo("https://github.com/x/y") is True

    def test_ssh_git_url(self):
        assert SkillManager._looks_like_repo("git@github.com:x/y.git") is True

    def test_owner_repo_shorthand(self):
        assert SkillManager._looks_like_repo("owner/repo") is True

    def test_owner_repo_when_path_exists(self, monkeypatch):
        original_exists = Path.exists

        def fake_exists(self):
            if str(self) == "owner" + "\\" + "repo":
                return True
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        assert SkillManager._looks_like_repo("owner/repo") is False

    def test_local_path(self, monkeypatch):
        original_exists = Path.exists

        def fake_exists(self):
            if str(self) == "local" + "\\" + "path":
                return True
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        assert SkillManager._looks_like_repo("local/path") is False

    def test_https_example_dot_git(self):
        assert SkillManager._looks_like_repo("https://example.com/repo.git") is True

    def test_ssh_protocol(self):
        assert SkillManager._looks_like_repo("ssh://git@github.com/x/y") is True

    def test_plain_text_not_repo(self):
        assert SkillManager._looks_like_repo("just some text") is False

    def test_http_url(self):
        assert SkillManager._looks_like_repo("http://example.com/repo") is True

    def test_empty_string(self):
        assert SkillManager._looks_like_repo("") is False

    def test_single_word(self):
        assert SkillManager._looks_like_repo("justaword") is False

    def test_three_part_path(self):
        assert SkillManager._looks_like_repo("a/b/c") is False


# ── _parse_repo_ref ─────────────────────────────────────────────────────────


class TestParseRepoRef:
    def test_github_url(self):
        result = SkillManager._parse_repo_ref("https://github.com/owner/repo")
        assert result == ("https://github.com/owner/repo.git", None, None)

    def test_github_url_with_branch_and_subpath(self):
        result = SkillManager._parse_repo_ref(
            "https://github.com/owner/repo/tree/main/skills/foo"
        )
        assert result == ("https://github.com/owner/repo.git", "main", "skills/foo")

    def test_ssh_git_url(self):
        result = SkillManager._parse_repo_ref("git@github.com:x/y.git")
        assert result == ("git@github.com:x/y.git", None, None)

    def test_owner_repo_shorthand(self):
        result = SkillManager._parse_repo_ref("owner/repo")
        assert result == ("https://github.com/owner/repo.git", None, None)

    def test_non_repo_string(self):
        result = SkillManager._parse_repo_ref("just some text")
        assert result is None

    def test_ssh_protocol_url(self):
        result = SkillManager._parse_repo_ref("ssh://git@github.com/x/y")
        assert result == ("ssh://git@github.com/x/y", None, None)

    def test_https_with_dot_git(self):
        result = SkillManager._parse_repo_ref("https://example.com/repo.git")
        assert result == ("https://example.com/repo.git", None, None)

    def test_github_url_with_trailing_slash(self):
        result = SkillManager._parse_repo_ref("https://github.com/owner/repo/")
        assert result == ("https://github.com/owner/repo.git", None, None)

    def test_github_url_branch_no_subpath(self):
        result = SkillManager._parse_repo_ref(
            "https://github.com/owner/repo/tree/main"
        )
        assert result == ("https://github.com/owner/repo.git", "main", None)

    def test_github_url_with_dot_git_suffix(self):
        result = SkillManager._parse_repo_ref("https://github.com/owner/repo.git")
        assert result == ("https://github.com/owner/repo.git", None, None)

    def test_http_url_non_github(self):
        result = SkillManager._parse_repo_ref("http://gitlab.com/owner/repo")
        assert result == ("http://gitlab.com/owner/repo", None, None)

    def test_deep_subpath(self):
        result = SkillManager._parse_repo_ref(
            "https://github.com/owner/repo/tree/dev/skills/my-skill"
        )
        assert result == (
            "https://github.com/owner/repo.git",
            "dev",
            "skills/my-skill",
        )

    def test_empty_string(self):
        assert SkillManager._parse_repo_ref("") is None


# ── _mentions_tool ──────────────────────────────────────────────────────────


class TestMentionsTool:
    def test_backticked_read(self):
        assert SkillManager._mentions_tool("Use `Read` to get the file.", "Read") is True

    def test_read_tool_phrase(self):
        assert SkillManager._mentions_tool("Use the Read tool to get the file.", "Read") is True

    def test_allowed_tools_line(self):
        assert SkillManager._mentions_tool("allowed-tools: Read, Write", "Read") is True

    def test_ambiguous_no_cue(self):
        assert SkillManager._mentions_tool("read this file", "Read") is False

    def test_non_ambiguous_anywhere(self):
        assert SkillManager._mentions_tool("Use WebSearch to find info.", "WebSearch") is True

    def test_no_tool_names(self):
        assert SkillManager._mentions_tool("Just some regular text.", "Read") is False

    def test_non_ambiguous_not_present(self):
        assert SkillManager._mentions_tool("No tools here.", "WebSearch") is False

    def test_backticked_write(self):
        assert SkillManager._mentions_tool("Use `Write` to save.", "Write") is True

    def test_write_no_cue(self):
        assert SkillManager._mentions_tool("write this file", "Write") is False

    def test_allowed_tools_case_insensitive_label(self):
        assert SkillManager._mentions_tool("Allowed Tools: Read, Write", "Read") is True

    def test_non_ambiguous_word_boundary(self):
        assert SkillManager._mentions_tool("Use the WebSearch tool.", "WebSearch") is True

    def test_grep_non_ambiguous(self):
        assert SkillManager._mentions_tool("Run Grep to search.", "Grep") is True

    def test_edit_ambiguous_no_cue(self):
        assert SkillManager._mentions_tool("edit the file", "Edit") is False

    def test_edit_with_cue(self):
        assert SkillManager._mentions_tool("Use `Edit` to change it.", "Edit") is True

    def test_bash_ambiguous_no_cue(self):
        assert SkillManager._mentions_tool("run bash commands", "Bash") is False

    def test_bash_with_tool_phrase(self):
        assert SkillManager._mentions_tool("Use the Bash tool.", "Bash") is True

    def test_allowed_tools_with_write(self):
        assert SkillManager._mentions_tool("allowed-tools: Read, Write", "Write") is True

    def test_task_ambiguous_no_cue(self):
        assert SkillManager._mentions_tool("complete the task", "Task") is False

    def test_ls_ambiguous_no_cue(self):
        assert SkillManager._mentions_tool("ls the directory", "LS") is False


# ── _tool_name_hints ────────────────────────────────────────────────────────


class TestToolNameHints:
    def test_multiple_tools(self):
        text = "Use `Read` and WebSearch to find information."
        hints = SkillManager._tool_name_hints(text)
        assert "WebSearch -> search_web" in hints
        assert "Read -> read_file" in hints

    def test_no_tool_names(self):
        hints = SkillManager._tool_name_hints("Just some regular text with no tools.")
        assert hints == ""

    def test_empty_text(self):
        assert SkillManager._tool_name_hints("") == ""

    def test_only_ambiguous_without_cue(self):
        hints = SkillManager._tool_name_hints("read this and write that")
        assert hints == ""

    def test_all_non_ambiguous_tools(self):
        text = "Use WebSearch and WebFetch and Grep."
        hints = SkillManager._tool_name_hints(text)
        assert "WebSearch -> search_web" in hints
        assert "WebFetch -> read_webpage" in hints
        assert "Grep -> grep_search" in hints

    def test_hint_header_present(self):
        text = "Use `Read` to get files."
        hints = SkillManager._tool_name_hints(text)
        assert "[TOOL NAMES]" in hints
        assert "Argent" in hints

    def test_ambiguous_with_allowed_tools_line(self):
        text = "allowed-tools: Read, Write, Bash"
        hints = SkillManager._tool_name_hints(text)
        assert "Read -> read_file" in hints
        assert "Write -> write_file" in hints
        assert "Bash -> run_command" in hints

    def test_tool_name_map_completeness(self):
        """Every entry in _TOOL_NAME_MAP should be translatable."""
        text = ""
        for name in _TOOL_NAME_MAP:
            if name in ("Read", "Write", "Edit", "Task", "LS", "Bash"):
                text += f"`{name}` "
            else:
                text += f"{name} "
        hints = SkillManager._tool_name_hints(text)
        for cc_name, ar_name in _TOOL_NAME_MAP.items():
            assert f"{cc_name} -> {ar_name}" in hints


# ── SkillManager CRUD with tmp_path ─────────────────────────────────────────


@pytest.fixture
def mgr(monkeypatch, tmp_path):
    """A SkillManager whose skills_dir points at a clean tmp_path."""
    monkeypatch.setattr(skill_manager, "get_skills_dir", lambda: str(tmp_path))
    return SkillManager()


class TestListSkills:
    def test_empty_dir(self, mgr):
        assert mgr.list_skills() == []

    def test_flat_skill_listed(self, mgr):
        (mgr.skills_dir / "my-skill.md").write_text(
            "---\nname: my-skill\ndescription: A test.\n---\nBody.",
            encoding="utf-8",
        )
        skills = mgr.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "my-skill"
        assert skills[0]["kind"] == "flat"
        assert skills[0]["description"] == "A test."

    def test_folder_skill_listed(self, mgr):
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\ndescription: Folder test.\n---\nBody.",
            encoding="utf-8",
        )
        skills = mgr.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "folder-skill"
        assert skills[0]["kind"] == "folder"

    def test_mixed_flat_and_folder(self, mgr):
        (mgr.skills_dir / "flat.md").write_text(
            "---\nname: flat\ndescription: Flat.\n---\nBody.", encoding="utf-8"
        )
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\ndescription: Folder.\n---\nBody.",
            encoding="utf-8",
        )
        skills = mgr.list_skills()
        assert len(skills) == 2
        names = {s["name"] for s in skills}
        assert names == {"flat", "folder-skill"}

    def test_non_md_files_ignored(self, mgr):
        (mgr.skills_dir / "readme.txt").write_text("not a skill", encoding="utf-8")
        (mgr.skills_dir / "data.json").write_text("{}", encoding="utf-8")
        assert mgr.list_skills() == []

    def test_dir_without_skill_file_ignored(self, mgr):
        (mgr.skills_dir / "empty-folder").mkdir()
        assert mgr.list_skills() == []

    def test_display_name_from_frontmatter(self, mgr):
        (mgr.skills_dir / "my-skill.md").write_text(
            "---\nname: My Display Name\ndescription: Test.\n---\nBody.",
            encoding="utf-8",
        )
        skills = mgr.list_skills()
        assert skills[0]["display_name"] == "My Display Name"

    def test_display_name_fallback_to_filename(self, mgr):
        (mgr.skills_dir / "my-skill.md").write_text(
            "---\ndescription: Test.\n---\nBody.", encoding="utf-8"
        )
        skills = mgr.list_skills()
        assert skills[0]["display_name"] == "my-skill"

    def test_skills_sorted_alphabetically(self, mgr):
        for name in ("zeta", "alpha", "mid"):
            (mgr.skills_dir / f"{name}.md").write_text(
                f"---\nname: {name}\ndescription: x.\n---\nBody.", encoding="utf-8"
            )
        skills = mgr.list_skills()
        assert [s["name"] for s in skills] == ["alpha", "mid", "zeta"]


class TestReadSkill:
    def test_flat_skill_returns_body(self, mgr):
        (mgr.skills_dir / "my-skill.md").write_text(
            "---\nname: my-skill\ndescription: Test.\n---\nDo the thing.",
            encoding="utf-8",
        )
        result = mgr.read_skill("my-skill")
        assert result is not None
        assert "Do the thing." in result

    def test_folder_skill_returns_body(self, mgr):
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\ndescription: Test.\n---\nDo the thing.",
            encoding="utf-8",
        )
        result = mgr.read_skill("folder-skill")
        assert result is not None
        assert "Do the thing." in result

    def test_folder_skill_with_allowed_tools(self, mgr):
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\nallowed-tools: Read, Write\n---\nBody text.",
            encoding="utf-8",
        )
        result = mgr.read_skill("folder-skill")
        assert result is not None
        assert "[ALLOWED TOOLS]: Read, Write" in result

    def test_folder_skill_with_allowed_tools_underscore(self, mgr):
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\nallowed_tools: Read, Write\n---\nBody text.",
            encoding="utf-8",
        )
        result = mgr.read_skill("folder-skill")
        assert result is not None
        assert "[ALLOWED TOOLS]: Read, Write" in result

    def test_folder_skill_with_when_to_use(self, mgr):
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\nwhen_to_use: When you need it.\n---\nBody.",
            encoding="utf-8",
        )
        result = mgr.read_skill("folder-skill")
        assert result is not None
        assert "[WHEN TO USE]: When you need it." in result

    def test_nonexistent_skill_returns_none(self, mgr):
        assert mgr.read_skill("does-not-exist") is None

    def test_flat_skill_with_md_suffix(self, mgr):
        (mgr.skills_dir / "my-skill.md").write_text(
            "---\nname: my-skill\ndescription: Test.\n---\nBody.", encoding="utf-8"
        )
        result = mgr.read_skill("my-skill.md")
        assert result is not None
        assert "Body." in result

    def test_folder_skill_with_bundled_resources(self, mgr):
        folder = mgr.skills_dir / "folder-skill"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: folder-skill\ndescription: Test.\n---\nBody.", encoding="utf-8"
        )
        scripts_dir = folder / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text("print('hi')", encoding="utf-8")
        result = mgr.read_skill("folder-skill")
        assert result is not None
        assert "[BUNDLED RESOURCES]" in result
        assert "helper.py" in result

    def test_flat_skill_with_tool_name_hint(self, mgr):
        (mgr.skills_dir / "my-skill.md").write_text(
            "---\nname: my-skill\ndescription: Test.\n---\nUse `Read` to get files.",
            encoding="utf-8",
        )
        result = mgr.read_skill("my-skill")
        assert result is not None
        assert "Read -> read_file" in result


class TestCreateSkill:
    def test_creates_file_with_frontmatter(self, mgr):
        msg = mgr.create_skill("new-skill", "Do the thing.", "A description")
        assert "created successfully" in msg
        fp = mgr.skills_dir / "new-skill.md"
        assert fp.exists()
        content = fp.read_text(encoding="utf-8")
        assert content.startswith("---")
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta["name"] == "new-skill"
        assert meta["description"] == "A description"
        assert body == "Do the thing."

    def test_creates_file_without_description(self, mgr):
        mgr.create_skill("no-desc", "Just instructions.")
        fp = mgr.skills_dir / "no-desc.md"
        assert fp.exists()
        content = fp.read_text(encoding="utf-8")
        meta, body = SkillManager._parse_frontmatter(content)
        assert meta["name"] == "no-desc"
        assert meta["description"] == ""
        assert body == "Just instructions."

    def test_create_overwrites_existing(self, mgr):
        mgr.create_skill("overwrite-me", "Original.", "First")
        mgr.create_skill("overwrite-me", "Updated.", "Second")
        fp = mgr.skills_dir / "overwrite-me.md"
        content = fp.read_text(encoding="utf-8")
        meta, body = SkillManager._parse_frontmatter(content)
        assert body == "Updated."
        assert meta["description"] == "Second"

    def test_create_with_md_suffix_name(self, mgr):
        mgr.create_skill("with-suffix.md", "Body.", "Desc.")
        fp = mgr.skills_dir / "with-suffix.md"
        assert fp.exists()
        meta, _ = SkillManager._parse_frontmatter(fp.read_text(encoding="utf-8"))
        assert meta["name"] == "with-suffix"


class TestDeleteSkill:
    def test_delete_flat_skill(self, mgr):
        (mgr.skills_dir / "doomed.md").write_text(
            "---\nname: doomed\ndescription: x.\n---\nBody.", encoding="utf-8"
        )
        msg = mgr.delete_skill("doomed")
        assert "deleted" in msg
        assert not (mgr.skills_dir / "doomed.md").exists()

    def test_delete_folder_skill(self, mgr):
        folder = mgr.skills_dir / "doomed-folder"
        folder.mkdir()
        (folder / SKILL_FILE).write_text(
            "---\nname: doomed-folder\ndescription: x.\n---\nBody.", encoding="utf-8"
        )
        msg = mgr.delete_skill("doomed-folder")
        assert "deleted" in msg
        assert not folder.exists()

    def test_delete_nonexistent(self, mgr):
        msg = mgr.delete_skill("no-such-skill")
        assert "not found" in msg

    def test_delete_flat_with_md_suffix(self, mgr):
        (mgr.skills_dir / "doomed.md").write_text(
            "---\nname: doomed\ndescription: x.\n---\nBody.", encoding="utf-8"
        )
        msg = mgr.delete_skill("doomed.md")
        assert "deleted" in msg
        assert not (mgr.skills_dir / "doomed.md").exists()


# ── _install_skills_from_tree ───────────────────────────────────────────────


class TestInstallSkillsFromTree:
    @pytest.fixture
    def skill_tree(self, tmp_path):
        """Create a tmp tree with SKILL.md at root and in a subfolder."""
        root = tmp_path / "source-tree"
        root.mkdir()

        # Root SKILL.md
        (root / SKILL_FILE).write_text(
            "---\nname: root-skill\ndescription: Root skill.\n---\nRoot body.",
            encoding="utf-8",
        )

        # Subfolder SKILL.md
        sub = root / "sub-skill"
        sub.mkdir()
        (sub / SKILL_FILE).write_text(
            "---\nname: sub-skill\ndescription: Sub skill.\n---\nSub body.",
            encoding="utf-8",
        )
        return root

    def test_both_root_and_sub_installed(self, mgr, skill_tree):
        messages = mgr._install_skills_from_tree(skill_tree, "fallback-name")
        assert any("installed 'root-skill'" in m for m in messages)
        assert any("installed 'sub-skill'" in m for m in messages)
        # Verify files exist in skills_dir
        assert (mgr.skills_dir / "root-skill" / SKILL_FILE).exists()
        assert (mgr.skills_dir / "sub-skill" / SKILL_FILE).exists()

    def test_existing_skill_skipped(self, mgr, skill_tree):
        # Pre-create root-skill in skills_dir
        existing = mgr.skills_dir / "root-skill"
        existing.mkdir()
        (existing / SKILL_FILE).write_text(
            "---\nname: root-skill\ndescription: Pre-existing.\n---\nOld body.",
            encoding="utf-8",
        )
        messages = mgr._install_skills_from_tree(skill_tree, "fallback-name")
        assert any("skipped 'root-skill'" in m for m in messages)
        assert any("installed 'sub-skill'" in m for m in messages)
        # The pre-existing content should be untouched
        content = (existing / SKILL_FILE).read_text(encoding="utf-8")
        assert "Old body." in content

    def test_root_skill_uses_fallback_when_no_name(self, mgr, tmp_path):
        root = tmp_path / "no-name-tree"
        root.mkdir()
        (root / SKILL_FILE).write_text(
            "---\ndescription: No name field.\n---\nBody.", encoding="utf-8"
        )
        messages = mgr._install_skills_from_tree(root, "fallback-name")
        assert any("installed 'fallback-name'" in m for m in messages)
        assert (mgr.skills_dir / "fallback-name" / SKILL_FILE).exists()

    def test_subfolder_uses_folder_name_when_no_name(self, mgr, tmp_path):
        root = tmp_path / "no-name-sub-tree"
        root.mkdir()
        sub = root / "my-subfolder"
        sub.mkdir()
        (sub / SKILL_FILE).write_text(
            "---\ndescription: No name field.\n---\nBody.", encoding="utf-8"
        )
        messages = mgr._install_skills_from_tree(root, "fallback-name")
        assert any("installed 'my-subfolder'" in m for m in messages)
        assert (mgr.skills_dir / "my-subfolder" / SKILL_FILE).exists()

    def test_root_skill_copies_resource_dirs(self, mgr, tmp_path):
        root = tmp_path / "resource-tree"
        root.mkdir()
        (root / SKILL_FILE).write_text(
            "---\nname: res-skill\ndescription: Test.\n---\nBody.", encoding="utf-8"
        )
        scripts = root / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_text("print('hi')", encoding="utf-8")
        messages = mgr._install_skills_from_tree(root, "fallback")
        assert any("installed 'res-skill'" in m for m in messages)
        assert (mgr.skills_dir / "res-skill" / "scripts" / "run.py").exists()

    def test_subfolder_skill_copies_entire_folder(self, mgr, tmp_path):
        root = tmp_path / "sub-resource-tree"
        root.mkdir()
        sub = root / "sub-with-resources"
        sub.mkdir()
        (sub / SKILL_FILE).write_text(
            "---\nname: sub-res\ndescription: Test.\n---\nBody.", encoding="utf-8"
        )
        refs = sub / "references"
        refs.mkdir()
        (refs / "doc.md").write_text("reference doc", encoding="utf-8")
        messages = mgr._install_skills_from_tree(root, "fallback")
        assert any("installed 'sub-res'" in m for m in messages)
        assert (mgr.skills_dir / "sub-res" / "references" / "doc.md").exists()

    def test_no_skill_files(self, mgr, tmp_path):
        root = tmp_path / "empty-tree"
        root.mkdir()
        (root / "readme.txt").write_text("no skills here", encoding="utf-8")
        messages = mgr._install_skills_from_tree(root, "fallback")
        assert messages == []

    def test_all_existing_skipped(self, mgr, skill_tree):
        for name in ("root-skill", "sub-skill"):
            d = mgr.skills_dir / name
            d.mkdir()
            (d / SKILL_FILE).write_text(
                f"---\nname: {name}\ndescription: Pre.\n---\nOld.", encoding="utf-8"
            )
        messages = mgr._install_skills_from_tree(skill_tree, "fallback")
        assert all("skipped" in m for m in messages)
        assert len(messages) == 2