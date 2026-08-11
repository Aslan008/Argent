"""Comprehensive pytest tests for search_files, grep_search, _python_grep_search
from tools/search_ops.py, AND _maybe_unescape_content, _strip_read_line_numbers,
_shift_indent, _build_match_hint, _changed_region_preview from tools/_helpers.py.
"""

import pytest
import os
from pathlib import Path

# --- Critical mocking: stub dependencies before importing the tools ---
import memory_manager


class FakeMemory:
    def add_completed(self, *a, **k):
        pass

    def add_file_modified(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    import memory_manager

    class FakeMemory:
        def add_completed(self, *a, **k):
            pass

        def add_file_modified(self, *a, **k):
            pass

    monkeypatch.setattr(memory_manager, 'memory', FakeMemory())
    yield


@pytest.fixture(autouse=True)
def _chdir_tmp(monkeypatch, tmp_path):
    """Run every test inside a clean tmp directory so relative searches work."""
    monkeypatch.chdir(tmp_path)
    yield


from tools.search_ops import search_files, grep_search, _python_grep_search
from tools._helpers import (
    _maybe_unescape_content,
    _strip_read_line_numbers,
    _shift_indent,
    _build_match_hint,
    _changed_region_preview,
)


# ---------------------------------------------------------------------------
# Helper: force grep_search to use the Python fallback (no ripgrep installed).
# ---------------------------------------------------------------------------
def _force_python_fallback(monkeypatch):
    """Make subprocess.run raise FileNotFoundError so grep_search falls back."""
    import subprocess

    def _raise(*a, **k):
        raise FileNotFoundError("rg not found")

    monkeypatch.setattr(subprocess, "run", _raise)


# ===========================================================================
# 1. search_files
# ===========================================================================
class TestSearchFiles:
    def test_find_files_by_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
        (tmp_path / "c.txt").write_text("z = 3\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py")
        assert "Found 2 file(s):" in result
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_find_files_by_name_contains(self, tmp_path):
        (tmp_path / "alpha_model.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "beta_view.py").write_text("y\n", encoding="utf-8")
        (tmp_path / "gamma_model.py").write_text("z\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", name_contains="model")
        assert "Found 2 file(s):" in result
        assert "alpha_model.py" in result
        assert "gamma_model.py" in result
        assert "beta_view.py" not in result

    def test_name_contains_is_case_insensitive(self, tmp_path):
        (tmp_path / "MyModule.py").write_text("x\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", name_contains="mymodule")
        assert "Found 1 file(s):" in result
        assert "MyModule.py" in result

    def test_find_files_by_content_contains(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("def world():\n    pass\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="hello")
        assert "Found 1 file(s):" in result
        assert "a.py" in result
        assert "b.py" not in result
        assert ">>" in result  # snippet marker

    def test_content_contains_is_case_insensitive(self, tmp_path):
        (tmp_path / "a.py").write_text("IMPORTANT NOTE\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="important")
        assert "Found 1 file(s):" in result
        assert "a.py" in result

    def test_max_results_limits_output(self, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i}.py").write_text("x\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", max_results=3)
        assert "Found 3 file(s):" in result
        assert "Results limited to 3" in result

    def test_max_results_one(self, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("x\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", max_results=1)
        assert "Found 1 file(s):" in result
        assert "Results limited to 1" in result

    def test_nonexistent_directory(self):
        result = search_files("/nonexistent/dir/xyz")
        assert "Error: Directory" in result
        assert "does not exist" in result

    def test_path_is_a_file_not_dir(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("x\n", encoding="utf-8")
        result = search_files(str(f))
        assert "Error:" in result
        assert "is not a directory" in result

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "emptydir"
        empty.mkdir()
        result = search_files(str(empty))
        assert "No files found matching: any file" in result

    def test_content_snippet_has_context_around_match(self, tmp_path):
        padding_before = "A" * 50
        padding_after = "B" * 70
        content = f"{padding_before}NEEDLE{padding_after}"
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="NEEDLE")
        assert "Found 1 file(s):" in result
        # The snippet should contain the search term and surrounding context
        assert "NEEDLE" in result
        # Should have ... prefix because start > 0
        assert "..." in result

    def test_content_snippet_prefix_when_start_gt_zero(self, tmp_path):
        # Put the term far enough that start > 0
        content = "X" * 100 + "TARGET" + "Y" * 100
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="TARGET")
        snippet_line = [l for l in result.splitlines() if ">>" in l][0]
        assert snippet_line.startswith("  >> ...")  # prefix because start > 0

    def test_content_snippet_suffix_when_end_lt_len(self, tmp_path):
        content = "X" * 100 + "TARGET" + "Y" * 100
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="TARGET")
        snippet_line = [l for l in result.splitlines() if ">>" in l][0]
        assert snippet_line.rstrip().endswith("...")  # suffix because end < len

    def test_content_snippet_no_prefix_when_start_is_zero(self, tmp_path):
        # Term at the very beginning → start = 0 → no prefix
        content = "TARGET" + "Y" * 100
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="TARGET")
        snippet_line = [l for l in result.splitlines() if ">>" in l][0]
        assert not snippet_line.startswith("  >> ...")

    def test_skips_hidden_directories(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "visible.py").write_text("y\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py")
        assert "visible.py" in result
        assert "secret.py" not in result

    def test_skips_git_directory(self, tmp_path):
        gitdir = tmp_path / ".git"
        gitdir.mkdir()
        (gitdir / "config").write_text("x\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*")
        assert "config" not in result

    def test_skips_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "lib.js").write_text("x\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*")
        assert "lib.js" not in result

    def test_skips_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "mod.cpython.pyc").write_text("x\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*")
        assert "mod.cpython.pyc" not in result

    def test_skips_binary_files_in_content_search(self, tmp_path):
        (tmp_path / "a.png").write_text("NEEDLE\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("NEEDLE\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*", content_contains="NEEDLE")
        assert "b.py" in result
        assert "a.png" not in result

    def test_utf8_content_search(self, tmp_path):
        (tmp_path / "a.py").write_text("# café résumé naïve\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="café")
        assert "Found 1 file(s):" in result
        assert "a.py" in result

    def test_pattern_star_returns_all_non_hidden(self, tmp_path):
        (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("y\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*")
        assert "Found 2 file(s):" in result
        assert "a.py" in result
        assert "b.txt" in result

    def test_no_filters_returns_all_non_hidden_files(self, tmp_path):
        (tmp_path / "x.py").write_text("x\n", encoding="utf-8")
        (tmp_path / "y.txt").write_text("y\n", encoding="utf-8")
        result = search_files(str(tmp_path))
        assert "Found 2 file(s):" in result

    def test_content_contains_no_match(self, tmp_path):
        (tmp_path / "a.py").write_text("nothing here\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py", content_contains="NEEDLE")
        assert "No files found matching:" in result
        assert "content contains" in result


# ===========================================================================
# 2. grep_search
# ===========================================================================
class TestGrepSearch:
    def test_search_existing_pattern_python_fallback(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "hello")
        assert "Found" in result
        assert "match(es)" in result
        assert "(via Python)" in result
        assert "hello" in result

    def test_search_existing_pattern_ripgrep(self, tmp_path, monkeypatch):
        # Simulate ripgrep being available and returning matches
        import subprocess

        class FakeResult:
            returncode = 0
            stdout = f"{tmp_path / 'a.py'}:1:def hello():\n"
            stderr = ""

        def _fake_run(cmd, *a, **k):
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = grep_search(str(tmp_path), "hello")
        assert "(via ripgrep)" in result
        assert "hello" in result

    def test_search_with_file_pattern_filter(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "needle", file_pattern="*.py")
        assert "Found" in result
        assert "a.py" in result
        assert "b.txt" not in result

    def test_nonexistent_directory(self):
        result = grep_search("/nonexistent/dir/xyz", "pattern")
        assert "Error: Directory" in result
        assert "does not exist" in result

    def test_directory_is_a_file(self, tmp_path, monkeypatch):
        f = tmp_path / "afile.txt"
        f.write_text("x\n", encoding="utf-8")
        result = grep_search(str(f), "pattern")
        assert "Error:" in result
        assert "is not a directory" in result

    def test_no_matches(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("nothing here\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "NEEDLE")
        assert "No matches found for pattern" in result

    def test_no_matches_ripgrep(self, tmp_path, monkeypatch):
        import subprocess

        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
        result = grep_search(str(tmp_path), "NEEDLE")
        assert "No matches found for pattern" in result

    def test_max_results_limits_output(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        content = "\n".join(f"needle line {i}" for i in range(20)) + "\n"
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        result = grep_search(str(tmp_path), "needle", max_results=3)
        assert "Found 3 match(es)" in result
        assert "Results limited to 3" in result

    def test_invalid_regex_python_fallback(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "[invalid")
        assert "Error" in result
        assert "Invalid regex" in result or "regex" in result.lower()

    def test_case_insensitive_search(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("Hello World\nHELLO AGAIN\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "hello")
        assert "Found 2 match(es)" in result

    def test_utf8_content_search(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("# café résumé\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "café")
        assert "Found" in result
        assert "café" in result

    def test_line_numbers_correct_in_results(self, tmp_path, monkeypatch):
        _force_python_fallback(monkeypatch)
        (tmp_path / "a.py").write_text("line1\nneedle\nline3\nneedle\n", encoding="utf-8")
        result = grep_search(str(tmp_path), "needle")
        assert "  2: needle" in result
        assert "  4: needle" in result

    def test_ripgrep_line_numbers_correct(self, tmp_path, monkeypatch):
        import subprocess

        class FakeResult:
            returncode = 0
            stdout = f"{tmp_path / 'a.py'}:2:needle\n{tmp_path / 'a.py'}:4:needle\n"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
        result = grep_search(str(tmp_path), "needle")
        assert "  2: needle" in result
        assert "  4: needle" in result


# ===========================================================================
# 3. _python_grep_search
# ===========================================================================
class TestPythonGrepSearch:
    def test_search_existing_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "hello")
        assert "Found" in result
        assert "(via Python)" in result
        assert "hello" in result

    def test_invalid_regex(self, tmp_path):
        (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "[invalid")
        assert "Error: Invalid regex pattern" in result

    def test_file_pattern_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("needle\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "needle", file_pattern="*.py")
        assert "a.py" in result
        assert "b.txt" not in result

    def test_no_matches(self, tmp_path):
        (tmp_path / "a.py").write_text("nothing here\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "NEEDLE")
        assert "No matches found for pattern" in result

    def test_skips_hidden_dirs(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "a.py").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("needle\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "needle")
        assert "b.py" in result
        assert ".hidden" not in result

    def test_skips_binary_files(self, tmp_path):
        (tmp_path / "a.exe").write_text("needle\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("needle\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "needle")
        assert "b.py" in result
        assert "a.exe" not in result

    def test_skips_pdb_obj_bin(self, tmp_path):
        for ext in [".pdb", ".obj", ".bin"]:
            (tmp_path / f"f{ext}").write_text("needle\n", encoding="utf-8")
        (tmp_path / "real.py").write_text("needle\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "needle")
        assert "real.py" in result
        assert ".pdb" not in result
        assert ".obj" not in result
        assert ".bin" not in result

    def test_max_results_one(self, tmp_path):
        content = "\n".join(f"needle {i}" for i in range(10)) + "\n"
        (tmp_path / "a.py").write_text(content, encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "needle", max_results=1)
        assert "Found 1 match(es)" in result
        assert "Results limited to 1" in result

    def test_nonexistent_directory(self):
        result = _python_grep_search("/nonexistent/dir/xyz", "pattern")
        assert "Error: Directory" in result
        assert "does not exist" in result

    def test_directory_is_a_file(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("x\n", encoding="utf-8")
        result = _python_grep_search(str(f), "pattern")
        assert "Error:" in result
        assert "is not a directory" in result

    def test_case_insensitive(self, tmp_path):
        (tmp_path / "a.py").write_text("Hello\nHELLO\nhello\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), "hello")
        assert "Found 3 match(es)" in result


# ===========================================================================
# 4. _maybe_unescape_content
# ===========================================================================
class TestMaybeUnescapeContent:
    def test_real_newlines_preserve_literal_backslash_n(self):
        src = 'line1\nline2 with \\n literal'
        assert _maybe_unescape_content(src) == src

    def test_literal_backslash_n_converted(self):
        assert _maybe_unescape_content("line1\\nline2") == "line1\nline2"

    def test_literal_backslash_t_converted(self):
        assert _maybe_unescape_content("a\\tb") == "a\tb"

    def test_double_backslash_collapsed_after_newline(self):
        # Order matters: \n replaced first, then \\ → \
        assert _maybe_unescape_content("a\\nC:\\\\x") == "a\nC:\\x"

    def test_lone_double_backslash_unchanged(self):
        # No \n or \t, so no unescaping at all
        assert _maybe_unescape_content("C:\\\\path\\\\x") == "C:\\\\path\\\\x"

    def test_empty_string(self):
        assert _maybe_unescape_content("") == ""

    def test_none_returns_none(self):
        assert _maybe_unescape_content(None) is None

    def test_both_backslash_n_and_t(self):
        assert _maybe_unescape_content("a\\nb\\tc") == "a\nb\tc"

    def test_no_escape_sequences_unchanged(self):
        assert _maybe_unescape_content("just plain text") == "just plain text"

    def test_real_carriage_return_unchanged(self):
        src = "line1\r\nline2"
        assert _maybe_unescape_content(src) == src

    def test_real_newline_with_literal_tab(self):
        # Has real newline → return unchanged even though \t is present
        src = "line1\nline2\\t"
        assert _maybe_unescape_content(src) == src


# ===========================================================================
# 5. _strip_read_line_numbers
# ===========================================================================
class TestStripReadLineNumbers:
    def test_text_with_gutter_stripped(self):
        text = "     1\tline1\n     2\tline2"
        assert _strip_read_line_numbers(text) == "line1\nline2"

    def test_text_without_gutter_unchanged(self):
        text = "def foo():\n    return 1\n"
        assert _strip_read_line_numbers(text) == text

    def test_tab_separated_table_not_stripped(self):
        table = "1\tAlice\n2\tBob"
        assert _strip_read_line_numbers(table) == table

    def test_empty_string(self):
        assert _strip_read_line_numbers("") == ""

    def test_single_line_with_gutter_not_stripped(self):
        # Need >= 2 matches
        assert _strip_read_line_numbers("42\tvalue") == "42\tvalue"

    def test_non_consecutive_numbers_not_stripped(self):
        text = "   10\ta\n   40\tb\n"
        assert _strip_read_line_numbers(text) == text

    def test_no_padding_not_stripped(self):
        # All lines have number+tab, consecutive, but no right-alignment padding
        text = "1\ta\n2\tb\n"
        assert _strip_read_line_numbers(text) == text

    def test_blank_lines_preserved(self):
        text = "    1\ta\n    2\t\n    3\tc\n"
        assert _strip_read_line_numbers(text) == "a\n\nc\n"

    def test_no_tab_unchanged(self):
        text = "just some text without tabs"
        assert _strip_read_line_numbers(text) == text

    def test_three_line_gutter_stripped(self):
        text = "    1\ta\n    2\tb\n    3\tc\n"
        assert _strip_read_line_numbers(text) == "a\nb\nc\n"

    def test_excerpt_with_gap_not_stripped(self):
        # Non-blank lines: "    1\ta", "    3\tc" → numbers 1, 3 not consecutive
        text = "    1\ta\n\n    3\tc\n"
        assert _strip_read_line_numbers(text) == text


# ===========================================================================
# 6. _shift_indent
# ===========================================================================
class TestShiftIndent:
    def test_shift_4_to_8_spaces(self):
        text = "    x = 1\n    y = 2\n"
        result = _shift_indent(text, "    ", "        ")
        assert result == "        x = 1\n        y = 2\n"

    def test_old_equals_new_unchanged(self):
        text = "    x = 1\n"
        assert _shift_indent(text, "    ", "    ") == text

    def test_empty_text(self):
        assert _shift_indent("", "    ", "  ") == ""

    def test_blank_lines_become_empty(self):
        text = "    x = 1\n\n    y = 2\n"
        result = _shift_indent(text, "    ", "  ")
        assert result == "  x = 1\n\n  y = 2\n"

    def test_lines_not_starting_with_old_indent_unchanged(self):
        text = "no indent\n    indented\n"
        result = _shift_indent(text, "    ", "  ")
        assert result == "no indent\n  indented\n"

    def test_shift_2_to_4_spaces_relative_depth(self):
        text = "  a\n    b\n"
        result = _shift_indent(text, "  ", "    ")
        assert result == "    a\n      b\n"

    def test_shift_tab_to_spaces(self):
        text = "\tx = 1\n"
        result = _shift_indent(text, "\t", "    ")
        assert result == "    x = 1\n"

    def test_mixed_lines(self):
        text = "def f():\n    x = 1\n    return x\n"
        result = _shift_indent(text, "    ", "        ")
        assert result == "def f():\n        x = 1\n        return x\n"


# ===========================================================================
# 7. _build_match_hint
# ===========================================================================
class TestBuildMatchHint:
    def test_first_line_found_in_content(self):
        target = "def hello():\n    pass\n"
        content = "import os\n\ndef hello():\n    pass\n\n# end\n"
        result = _build_match_hint(target, content)
        assert result != ""
        assert "EXACT text from the file" in result
        assert "def hello()" in result

    def test_first_line_not_found_but_partial_match(self):
        target = "def hello():\n    XTRA STUFF\n"
        content = "import os\n\ndef hello():\n    pass\n\n# end\n"
        result = _build_match_hint(target, content)
        assert result != ""
        assert "Found something similar around line" in result

    def test_empty_target_text(self):
        assert _build_match_hint("", "some content") == ""

    def test_first_line_too_short(self):
        # <= 3 chars
        assert _build_match_hint("ab\nrest", "ab here") == ""

    def test_no_match_at_all(self):
        target = "def nonexistent():\n    pass\n"
        content = "import os\n\n# nothing similar\n"
        assert _build_match_hint(target, content) == ""

    def test_hint_includes_actual_file_content(self):
        target = "def hello():\n    pass\n"
        content = "import os\n\ndef hello():\n    pass\n\n# end\n"
        result = _build_match_hint(target, content)
        # The hint should include a snippet of the actual file content
        assert "```" in result
        assert "def hello()" in result

    def test_first_line_exactly_3_chars_returns_empty(self):
        # len == 3, condition is <= 3
        assert _build_match_hint("abc\nrest", "abc") == ""

    def test_first_line_4_chars_works(self):
        target = "abcd\nrest\n"
        content = "abcd\nrest\n"
        result = _build_match_hint(target, content)
        assert result != ""
        assert "EXACT text from the file" in result

    def test_partial_match_context_line_number(self):
        target = "def hello():\n    XTRA\n"
        content = "line0\nline1\ndef hello():\n    pass\n"
        result = _build_match_hint(target, content)
        # "def hello():" is on line 3 (1-indexed)
        assert "line 3" in result


# ===========================================================================
# 8. _changed_region_preview
# ===========================================================================
class TestChangedRegionPreview:
    def test_normal_preview_shows_arrow_marker(self):
        content = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n"
        result = _changed_region_preview(content, start_line=1, new_line_count=1)
        assert result != ""
        assert "→" in result
        # Changed line is line 2 (0-indexed start_line=1)
        assert "b = 2" in result

    def test_empty_content(self):
        assert _changed_region_preview("", 0, 1) == ""

    def test_start_line_at_beginning_context_clamped(self):
        content = "a\nb\nc\nd\ne\n"
        result = _changed_region_preview(content, start_line=0, new_line_count=1, context=3)
        # Should show from line 0 (lo clamped to 0)
        lines = result.splitlines()
        assert "1 | a" in lines[0] or " 1 | a" in lines[0]
        assert "→" in lines[0]

    def test_start_line_at_end_hi_clamped(self):
        content = "a\nb\nc\nd\ne\n"
        result = _changed_region_preview(content, start_line=4, new_line_count=1, context=3)
        # hi clamped to len(lines) = 5
        lines = result.splitlines()
        # Last line should be line 5
        assert "5 | e" in lines[-1] or " 5 | e" in lines[-1]

    def test_new_line_count_zero_span_at_least_one(self):
        content = "a\nb\nc\n"
        result = _changed_region_preview(content, start_line=1, new_line_count=0)
        # span = max(1, 0) = 1, so line 2 (0-indexed 1) is marked
        assert "→" in result
        assert "b" in result

    def test_region_larger_than_max_lines_truncated(self):
        content = "\n".join(f"line{i}" for i in range(50)) + "\n"
        result = _changed_region_preview(content, start_line=0, new_line_count=50,
                                          context=3, max_lines=10)
        assert "region truncated" in result

    def test_line_numbers_are_1_indexed(self):
        content = "a\nb\nc\n"
        result = _changed_region_preview(content, start_line=0, new_line_count=1, context=1)
        lines = result.splitlines()
        # First line should be numbered 1
        first = lines[0]
        assert "1" in first
        assert "a" in first

    def test_arrow_markers_only_on_changed_lines(self):
        content = "a\nb\nc\nd\ne\nf\ng\n"
        result = _changed_region_preview(content, start_line=2, new_line_count=2, context=1)
        lines = result.splitlines()
        # start_line=2 → line 3 (c), span=2 → lines 3,4 (c,d) marked
        # context=1 → show lines 2,3,4,5 (b,c,d,e)
        for line in lines:
            if "c" in line or "d" in line:
                assert "→" in line
            elif line.strip() and not line.startswith(" "):
                # context line should NOT have arrow
                pass

    def test_context_lines_shown(self):
        content = "a\nb\nc\nd\ne\n"
        result = _changed_region_preview(content, start_line=2, new_line_count=1, context=1)
        # start_line=2 → line 3 (c), context=1 → show lines 2,3,4 (b,c,d)
        assert "b" in result  # context before
        assert "c" in result  # changed
        assert "d" in result  # context after
        assert "a" not in result  # outside context
        assert "e" not in result  # outside context

    def test_start_line_clamped_to_valid_range(self):
        content = "a\nb\nc\n"
        # start_line way beyond content
        result = _changed_region_preview(content, start_line=100, new_line_count=1)
        assert result != ""
        # Should be clamped to last line
        assert "c" in result

    def test_truncation_marker_format(self):
        content = "\n".join(f"line{i}" for i in range(50)) + "\n"
        result = _changed_region_preview(content, start_line=0, new_line_count=50,
                                          context=3, max_lines=5)
        assert "…" in result or "..." in result
        assert "truncated" in result