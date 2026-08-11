"""Comprehensive pytest tests for read_file, list_directory, and create_directory.

These cover line-number formatting, range reads, char/line budgets, error
paths, UTF-8 fidelity, directory listing with size formatting, and directory
creation including nested parents and already-existing cases.
"""

import pytest
import os
import sys
from pathlib import Path

# --- Critical mocking: stub dependencies before importing the tools ---
import memory_manager


class FakeMemory:
    def add_completed(self, *a, **k):
        pass

    def add_file_modified(self, *a, **k):
        pass


import file_tracker

import config
from tools import file_ops


@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    """Stub external dependencies so the tools run in isolation."""
    monkeypatch.setattr(memory_manager, 'memory', FakeMemory())
    monkeypatch.setattr(file_tracker, 'snapshot', lambda *a, **k: None)
    monkeypatch.setattr(config, 'get_current_model', lambda: 'test-model')
    monkeypatch.setattr(config, 'get_model_size_category', lambda m: 'default')


@pytest.fixture(autouse=True)
def _in_tmp_dir(monkeypatch, tmp_path):
    """Run every test inside a throwaway directory so relative paths resolve there."""
    monkeypatch.chdir(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# read_file
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFileLineNumbers:
    def test_small_file_line_number_format(self, tmp_path):
        """Line numbers are prefixed as '{number:>5}\\t{line}'."""
        (tmp_path / "a.py").write_text("import os\nx = 1\n", encoding="utf-8")
        out = file_ops.read_file("a.py")
        assert out == "    1\timport os\n    2\tx = 1\n", (
            f"Expected right-aligned 5-width line numbers; got: {out!r}")

    def test_single_line_file(self, tmp_path):
        (tmp_path / "one.txt").write_text("hello\n", encoding="utf-8")
        out = file_ops.read_file("one.txt")
        assert out == "    1\thello\n", f"Single line should be numbered 1; got: {out!r}"

    def test_content_matches_exactly_no_corruption(self, tmp_path):
        """L4xD10: Read back content exactly as written."""
        content = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
        (tmp_path / "exact.txt").write_text(content, encoding="utf-8")
        out = file_ops.read_file("exact.txt")
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert plain == content, f"Content corrupted: expected {content!r}, got {plain!r}"


class TestReadFileRanges:
    def test_range_with_start_and_end(self, tmp_path):
        """start_line=2, end_line=5 → '[Lines 2-5 of N]' header."""
        (tmp_path / "r.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)),
                                         encoding="utf-8")
        out = file_ops.read_file("r.txt", start_line=2, end_line=5)
        assert out.startswith("[Lines 2-5 of 10]"), f"Expected range header; got: {out[:40]!r}"
        assert "    2\tline2\n" in out
        assert "    5\tline5\n" in out
        assert "line1\n" not in out.replace("    2\tline2\n", "")
        assert "line6" not in out

    def test_only_start_line_reads_to_end(self, tmp_path):
        """start_line=3 with no end_line reads from line 3 to the end."""
        (tmp_path / "r.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)),
                                         encoding="utf-8")
        out = file_ops.read_file("r.txt", start_line=3)
        assert out.startswith("[Lines 3-10 of 10]"), f"Expected [Lines 3-10 of 10]; got: {out[:40]!r}"
        assert "    3\tline3\n" in out
        assert "   10\tline10\n" in out
        # Use tab-prefixed form so "line1" is not matched inside "line10"
        assert "\tline1\n" not in out and "\tline2\n" not in out

    def test_only_end_line_reads_from_start(self, tmp_path):
        """end_line=5 with no start_line reads from line 1 to 5."""
        (tmp_path / "r.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)),
                                         encoding="utf-8")
        out = file_ops.read_file("r.txt", end_line=5)
        assert out.startswith("[Lines 1-5 of 10]"), f"Expected [Lines 1-5 of 10]; got: {out[:40]!r}"
        assert "    1\tline1\n" in out
        assert "    5\tline5\n" in out
        assert "line6" not in out


class TestReadFileErrors:
    def test_nonexistent_file(self, tmp_path):
        out = file_ops.read_file("does_not_exist.py")
        assert out == "Error: File 'does_not_exist.py' does not exist.", (
            f"Expected 'does not exist' error; got: {out!r}")

    def test_path_is_a_directory(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        out = file_ops.read_file("subdir")
        assert out == "Error: 'subdir' is not a file.", (
            f"Expected 'is not a file' error; got: {out!r}")


class TestReadFileEdgeCases:
    def test_empty_file(self, tmp_path):
        """L3: An empty (0-byte) file returns an empty body."""
        (tmp_path / "empty.txt").write_bytes(b"")
        out = file_ops.read_file("empty.txt")
        assert isinstance(out, str)
        assert "Error" not in out
        # No lines were read, so body is empty; no header (total 0 <= max_lines)
        assert out == ""

    def test_start_line_beyond_file_length(self, tmp_path):
        """L3xD1: start_line beyond file length → '[Lines X-0 of N]'.

        shown_to = first + len(numbered) - 1 = first - 1 when no lines match.
        For start_line=100 on a 5-line file: first=100, shown_to=99.
        """
        (tmp_path / "short.txt").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        out = file_ops.read_file("short.txt", start_line=100)
        assert out.startswith("[Lines 100-99 of 5]"), (
            f"Expected '[Lines 100-99 of 5]'; got: {out[:40]!r}")

    def test_start_line_zero_treated_as_one(self, tmp_path):
        """L3xD1: start_line=0 → max(1, 0) = 1."""
        (tmp_path / "z.txt").write_text("a\nb\nc\n", encoding="utf-8")
        out = file_ops.read_file("z.txt", start_line=0)
        assert out.startswith("[Lines 1-3 of 3]"), (
            f"start_line=0 should be clamped to 1; got: {out[:40]!r}")
        assert "    1\ta\n" in out

    def test_negative_start_line_treated_as_one(self, tmp_path):
        """L3xD1: negative start_line → max(1, -5) = 1."""
        (tmp_path / "n.txt").write_text("a\nb\nc\n", encoding="utf-8")
        out = file_ops.read_file("n.txt", start_line=-5)
        assert out.startswith("[Lines 1-3 of 3]"), (
            f"Negative start_line should be clamped to 1; got: {out[:40]!r}")
        assert "    1\ta\n" in out

    def test_no_trailing_newline_gets_one_appended(self, tmp_path):
        """L3: A file without a trailing newline gets one appended to the body."""
        (tmp_path / "nontl.txt").write_text("no newline here", encoding="utf-8")
        out = file_ops.read_file("nontl.txt")
        assert out == "    1\tno newline here\n", (
            f"Body should end with newline; got: {out!r}")


class TestReadFileUTF8:
    def test_utf8_content_preserved(self, tmp_path):
        """D4: Cyrillic, emoji, and CJK characters are preserved."""
        content = "Привет мир\n你好世界\n😀 emoji here\n"
        (tmp_path / "utf8.txt").write_text(content, encoding="utf-8")
        out = file_ops.read_file("utf8.txt")
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert plain == content, f"UTF-8 content corrupted; got: {plain!r}"
        assert "Привет мир" in out
        assert "你好世界" in out
        assert "😀 emoji here" in out


class TestReadFileBudgets:
    def test_large_file_shows_exceeds_header(self, tmp_path):
        """L3: A file with >1500 lines (default budget) shows 'File exceeds' header."""
        (tmp_path / "big.py").write_text(
            "".join(f"# line {i}\n" for i in range(1, 1601)), encoding="utf-8")
        out = file_ops.read_file("big.py")
        assert "File exceeds 1500 lines (1600 total)" in out, (
            f"Expected 'exceeds' header for 1600-line file; got: {out[:80]!r}")
        # The first 1500 lines should be present (500 is width-5: "  500")
        assert "  500\t# line 500\n" in out
        assert " 1500\t# line 1500\n" in out
        # Line 1501 should NOT be in the body
        assert " 1501\t# line 1501\n" not in out

    def test_exactly_max_lines_no_exceeds_header(self, tmp_path):
        """L3: A file with exactly 1500 lines does NOT show 'exceeds' (total == max_lines, not >)."""
        (tmp_path / "exact_max.py").write_text(
            "".join(f"# {i}\n" for i in range(1, 1501)), encoding="utf-8")
        out = file_ops.read_file("exact_max.py")
        assert "exceeds" not in out, (
            f"File with exactly max_lines should not show 'exceeds'; got: {out[:80]!r}")
        assert " 1500\t# 1500\n" in out

    def test_max_lines_plus_one_shows_exceeds_header(self, tmp_path):
        """L3: A file with max_lines+1 lines shows 'exceeds' header."""
        (tmp_path / "over.py").write_text(
            "".join(f"# {i}\n" for i in range(1, 1502)), encoding="utf-8")
        out = file_ops.read_file("over.py")
        assert "File exceeds 1500 lines (1501 total)" in out, (
            f"Expected 'exceeds' header for 1501-line file; got: {out[:80]!r}")

    def test_very_long_lines_hit_char_budget(self, tmp_path):
        """L3: A file with very long lines hits the char budget → 'Stopped at' message."""
        # One line that far exceeds MAX_READ_CHARS (40000)
        long_line = "A" * 50_000
        (tmp_path / "longline.txt").write_text(long_line, encoding="utf-8")
        out = file_ops.read_file("longline.txt")
        assert "Stopped at" in out, (
            f"Expected 'Stopped at' for very long line; got: {out[:80]!r}")
        # The output should be bounded
        assert len(out) < file_ops.MAX_READ_CHARS + 500


# ══════════════════════════════════════════════════════════════════════════════
# list_directory
# ══════════════════════════════════════════════════════════════════════════════

class TestListDirectory:
    def test_dirs_come_before_files(self, tmp_path):
        """L2: Subdirectories are listed before files."""
        (tmp_path / "zfile.txt").write_text("data", encoding="utf-8")
        (tmp_path / "adir").mkdir()
        (tmp_path / "mfile.txt").write_text("more", encoding="utf-8")
        (tmp_path / "bdir").mkdir()
        out = file_ops.list_directory(".")
        lines = out.splitlines()
        # First line is the header
        assert lines[0] == "Contents of .:"
        # Find positions of dirs and files
        dir_lines = [l for l in lines[1:] if l.startswith("[DIR]")]
        file_lines = [l for l in lines[1:] if l.startswith("[FILE]")]
        # All dir lines should appear before all file lines
        first_file_idx = lines.index(file_lines[0]) if file_lines else len(lines)
        last_dir_idx = lines.index(dir_lines[-1]) if dir_lines else 0
        assert last_dir_idx < first_file_idx, (
            f"Dirs should come before files; got: {lines!r}")
        # Dirs should be sorted
        assert dir_lines == sorted(dir_lines, key=lambda l: l.split()[1]), (
            f"Dirs should be sorted; got: {dir_lines!r}")

    def test_file_size_formatting_bytes_and_kb(self, tmp_path):
        """L2: Small file shows bytes, larger file shows KB."""
        (tmp_path / "small.txt").write_text("x" * 100, encoding="utf-8")  # 100 bytes
        (tmp_path / "large.txt").write_text("y" * 2048, encoding="utf-8")  # 2048 bytes = 2.0 KB
        out = file_ops.list_directory(".")
        assert "small.txt  (100 B)" in out, f"Expected 100 B; got: {out!r}"
        assert "large.txt  (2.0 KB)" in out, f"Expected 2.0 KB; got: {out!r}"

    def test_nonexistent_directory(self, tmp_path):
        out = file_ops.list_directory("no_such_dir")
        assert out == "Error: Directory 'no_such_dir' does not exist.", (
            f"Expected 'does not exist' error; got: {out!r}")

    def test_path_is_a_file_not_dir(self, tmp_path):
        (tmp_path / "afile.txt").write_text("hi", encoding="utf-8")
        out = file_ops.list_directory("afile.txt")
        assert out == "Error: 'afile.txt' is not a directory.", (
            f"Expected 'is not a directory' error; got: {out!r}")

    def test_empty_directory(self, tmp_path):
        (tmp_path / "emptydir").mkdir()
        out = file_ops.list_directory("emptydir")
        assert out == "Directory 'emptydir' is empty.", (
            f"Expected empty message; got: {out!r}")

    def test_all_items_present(self, tmp_path):
        """L4xD10: Verify all items are listed, none missing."""
        names = ["alpha.txt", "beta.py", "gamma.md", "sub1", "sub2", "zeta.json"]
        for n in names:
            p = tmp_path / n
            if "." in n:
                p.write_text("c", encoding="utf-8")
            else:
                p.mkdir()
        out = file_ops.list_directory(".")
        for n in names:
            assert n in out, f"Item '{n}' missing from listing: {out!r}"

    def test_file_exactly_1024_bytes_shows_kb(self, tmp_path):
        """D1: 1024 bytes → 1024/1024 = 1.0 KB (boundary: size < 1024 is False, so KB branch)."""
        (tmp_path / "boundary.txt").write_text("x" * 1024, encoding="utf-8")
        out = file_ops.list_directory(".")
        assert "boundary.txt  (1.0 KB)" in out, (
            f"1024 bytes should show as 1.0 KB; got: {out!r}")
        assert "1024 B" not in out

    def test_file_exactly_1023_bytes_shows_b(self, tmp_path):
        """D1: 1023 bytes → 1023 B (size < 1024 is True)."""
        (tmp_path / "just_under.txt").write_text("x" * 1023, encoding="utf-8")
        out = file_ops.list_directory(".")
        assert "just_under.txt  (1023 B)" in out, (
            f"1023 bytes should show as 1023 B; got: {out!r}")


# ══════════════════════════════════════════════════════════════════════════════
# create_directory
# ══════════════════════════════════════════════════════════════════════════════

class TestCreateDirectory:
    def test_create_new_directory(self, tmp_path):
        out = file_ops.create_directory("newdir")
        assert "Successfully created directory" in out, (
            f"Expected success message; got: {out!r}")
        assert (tmp_path / "newdir").is_dir(), "Directory was not actually created"

    def test_create_nested_directory(self, tmp_path):
        """L2: Creating a/b/c creates all parents."""
        out = file_ops.create_directory("a/b/c")
        assert "Successfully created directory" in out, (
            f"Expected success for nested dir; got: {out!r}")
        assert (tmp_path / "a").is_dir(), "Parent 'a' not created"
        assert (tmp_path / "a" / "b").is_dir(), "Parent 'a/b' not created"
        assert (tmp_path / "a" / "b" / "c").is_dir(), "Leaf 'a/b/c' not created"

    def test_directory_already_exists(self, tmp_path):
        (tmp_path / "existing").mkdir()
        out = file_ops.create_directory("existing")
        assert out == "Directory 'existing' already exists.", (
            f"Expected 'already exists' message; got: {out!r}")

    def test_invalid_path_returns_error(self, tmp_path):
        """L1: A path with a null byte triggers an error (ValueError from Path)."""
        out = file_ops.create_directory("bad\x00dir")
        assert "Error" in out, f"Expected an error for invalid path; got: {out!r}"