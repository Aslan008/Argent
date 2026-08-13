"""Blind-spot tests for tools/_helpers.py — pure functions only.

Every test is self-contained: monkeypatches are applied per-test, file-system
fixtures use pytest's ``tmp_path``, and the Rich console is replaced with a
MagicMock so nothing is printed to the terminal.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools._helpers as helpers
from tools._helpers import (
    _resolve_path,
    _is_plugin_path_restricted,
    _is_unity_meta_restricted,
    _is_unity_project_file,
    _unity_script_placement_error,
    _maybe_unescape_content,
    _strip_read_line_numbers,
    _changed_region_preview,
    _shift_indent,
    _build_match_hint,
    drain_diff_events,
    _print_diff,
)


@pytest.fixture(autouse=True)
def _silence_console(monkeypatch):
    """Replace the module-level Rich console with a MagicMock for every test."""
    monkeypatch.setattr(helpers, "console", MagicMock())


# ---------------------------------------------------------------------------
# 1. _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_forward_slashes_resolve(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_path("sub/dir/file.txt")
        assert result == (tmp_path / "sub" / "dir" / "file.txt").resolve()

    def test_relative_path_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_path("hello.py")
        assert result == (tmp_path / "hello.py").resolve()

    def test_tilde_expands(self, monkeypatch):
        monkeypatch.setenv("USERPROFILE", str(Path("C:/Users/testuser")))
        result = _resolve_path("~/somefile.txt")
        # expanduser on Windows honours USERPROFILE
        assert str(result).replace("\\", "/").endswith("somefile.txt")
        assert "~" not in str(result)

    def test_nonexistent_path_still_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _resolve_path("does/not/exist/at/all.py")
        assert result == (tmp_path / "does" / "not" / "exist" / "at" / "all.py").resolve()
        assert not result.exists()


# ---------------------------------------------------------------------------
# 2. _is_plugin_path_restricted
# ---------------------------------------------------------------------------

class TestIsPluginPathRestricted:
    @pytest.fixture
    def plugins_dir(self, tmp_path, monkeypatch):
        d = tmp_path / "plugins"
        d.mkdir()
        monkeypatch.setattr(helpers, "get_hooks_dir", lambda: str(d))
        return d

    def test_py_inside_plugins_returns_restriction_error(self, plugins_dir):
        f = plugins_dir / "evil.py"
        result = _is_plugin_path_restricted(str(f))
        assert result is not None
        assert "restricted" in result.lower()
        assert "create_plugin" in result

    def test_md_inside_plugins_returns_python_only_error(self, plugins_dir):
        f = plugins_dir / "notes.md"
        result = _is_plugin_path_restricted(str(f))
        assert result is not None
        assert "Python plugins only" in result

    def test_file_outside_plugins_returns_none(self, plugins_dir, tmp_path):
        f = tmp_path / "regular.py"
        result = _is_plugin_path_restricted(str(f))
        assert result is None

    def test_non_py_inside_plugins_returns_python_only_error(self, plugins_dir):
        f = plugins_dir / "data.txt"
        result = _is_plugin_path_restricted(str(f))
        assert result is not None
        assert "Python plugins only" in result


# ---------------------------------------------------------------------------
# 3. _is_unity_meta_restricted
# ---------------------------------------------------------------------------

class TestIsUnityMetaRestricted:
    def test_meta_suffix_returns_error(self):
        result = _is_unity_meta_restricted("Player.cs.meta")
        assert result is not None
        assert ".meta" in result

    def test_cs_file_returns_none(self):
        assert _is_unity_meta_restricted("Player.cs") is None

    def test_uppercase_meta_suffix_returns_error(self):
        result = _is_unity_meta_restricted("Player.cs.META")
        assert result is not None
        assert ".meta" in result

    def test_meta_in_middle_returns_none(self):
        assert _is_unity_meta_restricted("some.meta.folder/file.cs") is None


# ---------------------------------------------------------------------------
# 4. _is_unity_project_file
# ---------------------------------------------------------------------------

class TestIsUnityProjectFile:
    def test_sibling_meta_returns_true(self, tmp_path):
        f = tmp_path / "Player.cs"
        f.write_text("x")
        (tmp_path / "Player.cs.meta").write_text("guid")
        assert _is_unity_project_file(str(f)) is True

    def test_under_assets_returns_true(self, tmp_path):
        assets = tmp_path / "MyProj" / "Assets" / "Scripts"
        assets.mkdir(parents=True)
        f = assets / "Player.cs"
        f.write_text("x")
        assert _is_unity_project_file(str(f)) is True

    def test_not_under_assets_and_no_meta_returns_false(self, tmp_path):
        f = tmp_path / "random.py"
        f.write_text("x")
        assert _is_unity_project_file(str(f)) is False


# ---------------------------------------------------------------------------
# 5. _unity_script_placement_error
# ---------------------------------------------------------------------------

class TestUnityScriptPlacementError:
    @pytest.fixture
    def unity_project(self, tmp_path):
        root = tmp_path / "UnityProj"
        (root / "Assets").mkdir(parents=True)
        (root / "ProjectSettings").mkdir(parents=True)
        return root

    def test_cs_inside_assets_returns_none(self, unity_project):
        f = unity_project / "Assets" / "Player.cs"
        f.write_text("x")
        assert _unity_script_placement_error(str(f)) is None

    def test_cs_outside_assets_but_inside_project_returns_error(self, unity_project):
        f = unity_project / "Player.cs"
        f.write_text("x")
        result = _unity_script_placement_error(str(f))
        assert result is not None
        assert "OUTSIDE the Assets" in result

    def test_cs_outside_any_unity_project_returns_none(self, tmp_path):
        f = tmp_path / "lonely.cs"
        f.write_text("x")
        assert _unity_script_placement_error(str(f)) is None

    def test_txt_file_returns_none(self, unity_project):
        f = unity_project / "notes.txt"
        f.write_text("x")
        assert _unity_script_placement_error(str(f)) is None


# ---------------------------------------------------------------------------
# 6. _maybe_unescape_content
# ---------------------------------------------------------------------------

class TestMaybeUnescapeContent:
    def test_real_newlines_unchanged(self):
        text = "line1\nline2"
        assert _maybe_unescape_content(text) == text

    def test_literal_backslash_n_unescaped(self):
        text = "line1\\nline2"
        assert _maybe_unescape_content(text) == "line1\nline2"

    def test_literal_n_inside_string_literal_stays(self):
        # Odd number of quotes on a resulting line → treated as string literal
        text = 'print("a\\nb")'
        assert _maybe_unescape_content(text) == text

    def test_literal_backslash_t_unescaped(self):
        text = "col1\\tcol2"
        assert _maybe_unescape_content(text) == "col1\tcol2"

    def test_empty_string_returns_empty(self):
        assert _maybe_unescape_content("") == ""

    def test_double_backslash_unescaped_to_single(self):
        # \\\\ is only unescaped when \n or \t is also present (guard condition)
        text = "line1\\npath\\\\to\\\\file"
        result = _maybe_unescape_content(text)
        assert "\\" in result and "\\\\" not in result


# ---------------------------------------------------------------------------
# 7. _strip_read_line_numbers
# ---------------------------------------------------------------------------

class TestStripReadLineNumbers:
    def test_proper_gutter_stripped(self):
        text = "  1\talpha\n  2\tbeta\n  3\tgamma"
        result = _strip_read_line_numbers(text)
        assert result == "alpha\nbeta\ngamma"

    def test_non_consecutive_numbers_stays(self):
        text = "  1\talpha\n  3\tbeta\n  5\tgamma"
        assert _strip_read_line_numbers(text) == text

    def test_no_padding_stays(self):
        # No right-alignment (no leading spaces before digits)
        text = "1\talpha\n2\tbeta\n3\tgamma"
        assert _strip_read_line_numbers(text) == text

    def test_fewer_than_two_matched_lines_stays(self):
        text = "  1\talpha"
        assert _strip_read_line_numbers(text) == text

    def test_tab_separated_table_non_consecutive_stays(self):
        text = "10\tAlice\n20\tBob\n30\tCarol"
        assert _strip_read_line_numbers(text) == text

    def test_empty_string_returns_empty(self):
        assert _strip_read_line_numbers("") == ""


# ---------------------------------------------------------------------------
# 8. _changed_region_preview
# ---------------------------------------------------------------------------

class TestChangedRegionPreview:
    def test_basic_preview_with_arrow_markers(self):
        content = "a\nb\nc\nd\ne"
        result = _changed_region_preview(content, start_line=1, new_line_count=1)
        lines = result.split("\n")
        # The changed line (line 2, 0-indexed 1) should have →
        assert any(ln.startswith("→") for ln in lines)
        assert any(ln.startswith(" ") for ln in lines)

    def test_context_lines_around_change(self):
        content = "l0\nl1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9"
        result = _changed_region_preview(content, start_line=4, new_line_count=1, context=2)
        lines = result.split("\n")
        # Should include lines before and after the change
        assert "l2" in result  # context before
        assert "l6" in result  # context after
        # Arrow only on the changed line
        arrow_lines = [ln for ln in lines if ln.startswith("→")]
        assert len(arrow_lines) == 1

    def test_truncation_when_region_too_large(self):
        content = "\n".join(f"line{i}" for i in range(100))
        result = _changed_region_preview(content, start_line=10, new_line_count=50, context=3, max_lines=10)
        assert "truncated" in result.lower()

    def test_empty_content_returns_empty_string(self):
        assert _changed_region_preview("", start_line=0, new_line_count=1) == ""


# ---------------------------------------------------------------------------
# 9. _shift_indent
# ---------------------------------------------------------------------------

class TestShiftIndent:
    def test_swap_4space_to_2space(self):
        text = "    def foo():\n        return 1"
        result = _shift_indent(text, "    ", "  ")
        # Only the first old_indent prefix is swapped per line:
        # 8 spaces -> 2 + 4 = 6 spaces
        assert result == "  def foo():\n      return 1"

    def test_same_indent_returns_unchanged(self):
        text = "    def foo():\n        return 1"
        assert _shift_indent(text, "    ", "    ") == text

    def test_blank_lines_preserved(self):
        text = "    line1\n\n    line2"
        result = _shift_indent(text, "    ", "  ")
        assert result == "  line1\n\n  line2"

    def test_lines_without_old_indent_preserved(self):
        text = "    indented\nnot_indented"
        result = _shift_indent(text, "    ", "  ")
        assert result == "  indented\nnot_indented"


# ---------------------------------------------------------------------------
# 10. _build_match_hint
# ---------------------------------------------------------------------------

class TestBuildMatchHint:
    def test_exact_first_line_found_returns_hint_with_snippet(self):
        content = "import os\nimport sys\n\ndef main():\n    pass\n"
        target = "def main():\n    pass"
        result = _build_match_hint(target, content)
        assert result != ""
        assert "def main()" in result

    def test_no_match_returns_empty(self):
        content = "import os\nimport sys\n"
        target = "class NonExistent:\n    pass"
        assert _build_match_hint(target, content) == ""

    def test_very_short_first_line_returns_empty(self):
        content = "x = 1\n"
        target = "x\nmore"
        assert _build_match_hint(target, content) == ""


# ---------------------------------------------------------------------------
# 11. drain_diff_events
# ---------------------------------------------------------------------------

class TestDrainDiffEvents:
    def test_empty_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(helpers, "_PENDING_DIFFS", [])
        assert drain_diff_events() == []

    def test_after_print_diff_returns_event(self, monkeypatch):
        monkeypatch.setattr(helpers, "_PENDING_DIFFS", [])
        old = "line1\nline2\n"
        new = "line1\nlineX\n"
        _print_diff(old, new, "test.py")
        events = drain_diff_events()
        assert len(events) == 1
        assert events[0]["file"] == "test.py"
        assert "diff" in events[0]

    def test_second_drain_returns_empty(self, monkeypatch):
        monkeypatch.setattr(helpers, "_PENDING_DIFFS", [])
        _print_diff("a\n", "b\n", "f.py")
        drain_diff_events()
        assert drain_diff_events() == []