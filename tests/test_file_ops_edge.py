"""File operations edge-case tests.

These tests cover boundary conditions that could silently corrupt data or
return wrong results. The existing tests (test_read_file_bounds.py,
test_edit_preview.py) cover truncation and basic edits, but miss:

- Empty files (0 bytes)
- BOM handling
- CRLF / mixed line endings
- multi_replace_in_file_chunk boundary conditions (first line, last line,
  end_line beyond file, overlapping ranges)
- write → read roundtrip fidelity
"""

import json

import pytest

from tools import file_ops


@pytest.fixture(autouse=True)
def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# read_file edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestReadFileEdgeCases:
    def test_empty_file_returns_empty(self, tmp_path):
        """A 0-byte file should not crash."""
        (tmp_path / "empty.py").write_bytes(b"")
        out = file_ops.read_file("empty.py")
        # Should return empty string or a header with 0 lines, not crash
        assert isinstance(out, str)
        assert "Error" not in out

    def test_file_with_only_newlines(self, tmp_path):
        """A file with only newlines should return each as a numbered line."""
        (tmp_path / "newlines.py").write_text("\n\n\n", encoding="utf-8")
        out = file_ops.read_file("newlines.py")
        # Should have 3 lines (all empty), not crash
        assert "\t\n" in out or out.strip() == ""

    def test_start_line_beyond_end(self, tmp_path):
        """start_line=1000 on a 5-line file should not crash."""
        (tmp_path / "short.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        out = file_ops.read_file("short.py", start_line=1000)
        assert isinstance(out, str)
        assert "Error" not in out
        # Should report the range and total
        assert "1000" in out

    def test_start_greater_than_end(self, tmp_path):
        """start_line=5, end_line=2 should not crash (no lines match)."""
        (tmp_path / "rev.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        out = file_ops.read_file("rev.py", start_line=5, end_line=2)
        assert isinstance(out, str)
        assert "Error" not in out

    def test_utf8_bom_not_garbled(self, tmp_path):
        """A file with UTF-8 BOM should be readable without garbled output."""
        (tmp_path / "bom.py").write_bytes(b"\xef\xbb\xbf# coding: utf-8\nx = 1\n")
        out = file_ops.read_file("bom.py")
        assert isinstance(out, str)
        # The content should be readable (BOM may appear as \ufeff but
        # the actual code should be visible)
        assert "coding" in out or "x = 1" in out

    def test_crlf_line_endings(self, tmp_path):
        """Windows CRLF should be split into correct lines."""
        (tmp_path / "crlf.py").write_bytes(b"line1\r\nline2\r\nline3\r\n")
        out = file_ops.read_file("crlf.py")
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        # Universal newlines should convert \r\n to \n
        assert "line1" in plain
        assert "line2" in plain
        assert "line3" in plain
        assert "\r" not in plain, "CRLF should be normalized to LF"

    def test_mixed_line_endings(self, tmp_path):
        """Mixed \n and \r\n should be handled consistently."""
        (tmp_path / "mixed.py").write_bytes(b"line1\nline2\r\nline3\n")
        out = file_ops.read_file("mixed.py")
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "line1" in plain
        assert "line2" in plain
        assert "line3" in plain
        assert "\r" not in plain


# ══════════════════════════════════════════════════════════════════════════════
# multi_replace_in_file_chunk edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiReplaceEdgeCases:
    def _make_file(self, tmp_path, name="m.txt", content=None):
        if content is None:
            content = "line1\nline2\nline3\nline4\nline5\n"
        (tmp_path / name).write_text(content, encoding="utf-8")
        return str(tmp_path / name)

    def test_replace_last_line(self, tmp_path):
        """end_line = last line should replace to the end."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 5, "end_line": 5,
            "target_content": "line5",
            "replacement_content": "REPLACED",
        }])
        out = file_ops.multi_replace_in_file_chunk(fp, changes)
        assert "Successfully" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert "REPLACED" in result
        assert "line4" in result
        assert "line5" not in result.replace("REPLACED", "")

    def test_end_line_beyond_file_clamps(self, tmp_path):
        """end_line=999 on a 5-line file should clamp, not crash."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 4, "end_line": 999,
            "target_content": "line4\nline5",
            "replacement_content": "REPLACED",
        }])
        out = file_ops.multi_replace_in_file_chunk(fp, changes)
        assert "Successfully" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert "REPLACED" in result
        assert "line3" in result
        # Lines 4 and 5 should be replaced
        assert "line4" not in result.replace("REPLACED", "")
        assert "line5" not in result.replace("REPLACED", "")

    def test_overlapping_ranges_produce_error(self, tmp_path):
        """Two changes with overlapping line ranges should produce a
        target_content mismatch error (since the first change modifies
        lines the second change expects to match)."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([
            {
                "start_line": 2, "end_line": 3,
                "target_content": "line2\nline3",
                "replacement_content": "REPLACED_A",
            },
            {
                "start_line": 3, "end_line": 4,
                "target_content": "line3\nline4",
                "replacement_content": "REPLACED_B",
            },
        ])
        out = file_ops.multi_replace_in_file_chunk(fp, changes)
        # The second change should fail because line3 was already replaced
        # by the first change. The result should contain an error message.
        assert "Error" in out or "does not match" in out, (
            f"Expected error for overlapping ranges, got: {out}"
        )

    def test_multiple_non_overlapping_changes(self, tmp_path):
        """Two non-overlapping changes should both apply successfully."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([
            {
                "start_line": 1, "end_line": 1,
                "target_content": "line1",
                "replacement_content": "AAA",
            },
            {
                "start_line": 5, "end_line": 5,
                "target_content": "line5",
                "replacement_content": "BBB",
            },
        ])
        out = file_ops.multi_replace_in_file_chunk(fp, changes)
        assert "Successfully" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        lines = result.strip().split("\n")
        assert lines[0] == "AAA"
        assert lines[4] == "BBB"
        assert lines[1] == "line2"
        assert lines[2] == "line3"
        assert lines[3] == "line4"


# ══════════════════════════════════════════════════════════════════════════════
# write_file + read_file roundtrip
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteReadRoundtrip:
    def test_write_then_read_preserves_content(self, tmp_path):
        """Content written by write_file should be readable by read_file
        without corruption."""
        content = "def hello():\n    return 'world'\n"
        file_ops.write_file("roundtrip.py", content)
        out = file_ops.read_file("roundtrip.py")
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert plain == content, (
            f"Roundtrip corrupted content.\nExpected: {content!r}\nGot: {plain!r}"
        )

    def test_write_unicode_content(self, tmp_path):
        """Unicode content (Cyrillic, emoji) should survive write→read."""
        content = "# Комментарий\nx = 'привет 🌍'\n"
        file_ops.write_file("unicode.py", content)
        out = file_ops.read_file("unicode.py")
        from tools._helpers import _strip_read_line_numbers
        plain = _strip_read_line_numbers(out)
        assert "привет" in plain
        assert "🌍" in plain

    def test_write_nested_path_creates_dirs(self, tmp_path):
        """write_file to a deeply nested path should create parent dirs."""
        file_ops.write_file("a/b/c/deep.py", "x = 1\n")
        assert (tmp_path / "a" / "b" / "c" / "deep.py").exists()
        assert (tmp_path / "a" / "b" / "c" / "deep.py").read_text(encoding="utf-8") == "x = 1\n"


# ══════════════════════════════════════════════════════════════════════════════
# replace_in_file edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestReplaceInFileEdgeCases:
    def test_partial_match_does_not_corrupt(self, tmp_path):
        """A target that partially matches (substring of a line) should
        fail cleanly, not corrupt the file."""
        (tmp_path / "f.py").write_text("abcdef\nghijkl\n", encoding="utf-8")
        out = file_ops.replace_in_file(str(tmp_path / "f.py"), "abc", "REPLACED")
        # Should either succeed (exact match) or fail cleanly
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        # File should not be corrupted regardless of outcome
        assert "ghijkl" in result, "Unrelated content should be preserved"

    def test_empty_replacement_deletes_target(self, tmp_path):
        """Replacing with empty string should remove the target text."""
        (tmp_path / "f.txt").write_text("keep\ndelete\nkeep2\n", encoding="utf-8")
        out = file_ops.replace_in_file(str(tmp_path / "f.txt"), "delete\n", "")
        result = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert "delete" not in result
        assert "keep" in result
        assert "keep2" in result