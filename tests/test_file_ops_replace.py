"""Comprehensive pytest tests for replace_in_file, multi_replace_in_file,
and multi_replace_in_file_chunk from tools/file_ops.py.

Covers:
  * replace_in_file: exact match, multi-line target/replacement, not-found,
    multiple-match, empty/whitespace target, fuzzy fallback (wrong indentation,
    extra blank lines, ambiguous), line-number stripping, UTF-8, whole-file
    replace, non-existent file, directory, plugin/.meta restriction, syntax
    validation revert, no-op replace, CRLF.
  * multi_replace_in_file: single, multiple files, multiple changes to same
    file, invalid JSON, non-array JSON, missing file_path, partial failure,
    empty array.
  * multi_replace_in_file_chunk: single chunk, multiple chunks, invalid JSON,
    non-array JSON, missing start/end_line, invalid range, target mismatch,
    empty replacement, first/last line, multiple non-overlapping, non-existent
    file, syntax validation revert, empty target_content.
"""

import pytest
import json
import os
from pathlib import Path

from tools.file_ops import (
    replace_in_file,
    multi_replace_in_file,
    multi_replace_in_file_chunk,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures / dependency stubs
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    import memory_manager

    class FakeMemory:
        def __init__(self):
            self.completed = []
            self.modified = []

        def add_completed(self, s):
            self.completed.append(s)

        def add_file_modified(self, s):
            self.modified.append(s)

    monkeypatch.setattr(memory_manager, "memory", FakeMemory())

    import file_tracker
    monkeypatch.setattr(file_tracker, "snapshot", lambda *a, **k: None)

    import tools._helpers as helpers
    monkeypatch.setattr(helpers, "_validate_code_syntax", lambda fp: None)

    # replace_in_file uses the top-level import, not a local re-import, so
    # patch the name in file_ops' own namespace too.
    import tools.file_ops as file_ops_mod
    monkeypatch.setattr(file_ops_mod, "_validate_code_syntax", lambda fp: None)

    yield


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# replace_in_file
# ══════════════════════════════════════════════════════════════════════════════

class TestReplaceInFileExact:
    def test_simple_exact_replacement(self, tmp_path):
        """L2: Simple exact replacement returns success and changes content."""
        (tmp_path / "f.txt").write_text("hello world\nfoo bar\n", encoding="utf-8")
        out = replace_in_file("f.txt", "hello world", "goodbye world")
        assert out.startswith("Successfully replaced text in")
        result = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert "goodbye world" in result
        assert "hello world" not in result
        assert "foo bar" in result

    def test_exact_content_after_replace(self, tmp_path):
        """L4×D10: After replace, file content is exactly expected."""
        (tmp_path / "f.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        replace_in_file("f.txt", "beta", "BETA")
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"

    def test_replace_multiline_target(self, tmp_path):
        """L2: Replace a multi-line target."""
        original = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        (tmp_path / "f.py").write_text(original, encoding="utf-8")
        out = replace_in_file("f.py", "def foo():\n    return 1", "def foo():\n    return 10")
        assert out.startswith("Successfully replaced text in")
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        assert "return 10" in result
        assert "return 1\n" not in result
        assert "def bar" in result

    def test_replace_with_multiline_replacement(self, tmp_path):
        """L2: Replace with a multi-line replacement."""
        (tmp_path / "f.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        out = replace_in_file("f.py", "x = 1", "x = 1\nz = 3")
        assert out.startswith("Successfully replaced text in")
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        assert "x = 1\nz = 3\ny = 2\n" == result

    def test_replace_entire_file_content(self, tmp_path):
        """L3: Replace entire file content."""
        (tmp_path / "f.txt").write_text("old content line 1\nold content line 2\n", encoding="utf-8")
        out = replace_in_file("f.txt", "old content line 1\nold content line 2", "brand new")
        assert out.startswith("Successfully replaced text in")
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "brand new\n"

    def test_target_equals_replacement(self, tmp_path):
        """L3: target_text == replacement_text → file unchanged but success."""
        (tmp_path / "f.txt").write_text("same text\nother\n", encoding="utf-8")
        original = (tmp_path / "f.txt").read_text(encoding="utf-8")
        out = replace_in_file("f.txt", "same text", "same text")
        assert out.startswith("Successfully replaced text in")
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == original


class TestReplaceInFileErrors:
    def test_target_not_found(self, tmp_path):
        """L1: Target not found → error mentioning not found."""
        (tmp_path / "f.txt").write_text("hello\nworld\n", encoding="utf-8")
        out = replace_in_file("f.txt", "nonexistent text", "whatever")
        assert "Error" in out
        assert "not found" in out

    def test_target_appears_multiple_times(self, tmp_path):
        """L1: Target appears 2+ times → error with count and line locations."""
        (tmp_path / "f.txt").write_text("dup\nother\ndup\n", encoding="utf-8")
        out = replace_in_file("f.txt", "dup", "unique")
        assert "Error" in out
        assert "2 times" in out
        assert "line 1" in out
        assert "line 3" in out

    def test_empty_target_text(self, tmp_path):
        """L1×D2: Empty target_text → error about empty target."""
        (tmp_path / "f.txt").write_text("content\n", encoding="utf-8")
        out = replace_in_file("f.txt", "", "replacement")
        assert "Error" in out
        assert "empty" in out.lower()

    def test_whitespace_only_target_text(self, tmp_path):
        """L1×D2: Whitespace-only target_text → error about empty target."""
        (tmp_path / "f.txt").write_text("content\n", encoding="utf-8")
        out = replace_in_file("f.txt", "   \n\t  ", "replacement")
        assert "Error" in out
        assert "empty" in out.lower()

    def test_nonexistent_file(self, tmp_path):
        """L1: Non-existent file → error."""
        out = replace_in_file("does_not_exist.txt", "target", "replacement")
        assert "Error" in out
        assert "does not exist" in out

    def test_path_is_directory(self, tmp_path):
        """L1: Path is directory → error."""
        (tmp_path / "subdir").mkdir()
        out = replace_in_file("subdir", "target", "replacement")
        assert "Error" in out
        assert "not a file" in out


class TestReplaceInFileRestrictions:
    def test_plugin_restricted_path(self, tmp_path, monkeypatch):
        """L1×D7: Plugin-restricted path → restriction error."""
        import config
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(config, "get_hooks_dir", lambda: str(plugins_dir))
        monkeypatch.setattr(config, "get_current_model", lambda: "qwen2.5:3b")
        monkeypatch.setattr(config, "get_model_category_override", lambda: "small")
        target = plugins_dir / "restricted.py"
        target.write_text("x = 1\n", encoding="utf-8")
        out = replace_in_file(str(target), "x = 1", "x = 2")
        assert "Error" in out
        assert "restricted" in out.lower() or "plugins" in out.lower()
        # File should be unchanged
        assert target.read_text(encoding="utf-8") == "x = 1\n"

    def test_meta_file_restricted(self, tmp_path):
        """L1×D7: .meta file → restriction error."""
        target = tmp_path / "asset.cs.meta"
        target.write_text("guid: 123\n", encoding="utf-8")
        out = replace_in_file(str(target), "guid: 123", "guid: 456")
        assert "Error" in out
        assert ".meta" in out
        assert target.read_text(encoding="utf-8") == "guid: 123\n"


class TestReplaceInFileFuzzy:
    def test_fuzzy_wrong_indentation_corrected(self, tmp_path):
        """L2: Fuzzy match — target with different indentation → matches, re-indents."""
        content = "def greet(name):\n    if name:\n        print('hi')\n    return name\n"
        (tmp_path / "f.py").write_text(content, encoding="utf-8")
        # Target has NO indentation but file has 4-space indent. The
        # replacement carries relative depth (4 spaces on the nested line)
        # so _shift_indent re-bases it to the file's 4-space block indent.
        out = replace_in_file(
            "f.py",
            "if name:\nprint('hi')",
            "if name:\n    print('hello')",
        )
        assert out.startswith("Successfully replaced text in")
        assert "fuzzy" in out
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        assert "    if name:" in result
        assert "        print('hello')" in result
        assert "print('hi')" not in result

    def test_fuzzy_extra_blank_lines_in_file(self, tmp_path):
        """L2: Fuzzy match — target with extra blank lines → matches (blank lines skipped)."""
        content = "line_a\n\n\nline_b\nline_c\n"
        (tmp_path / "f.txt").write_text(content, encoding="utf-8")
        # Target without the blank lines should still fuzzy-match
        out = replace_in_file("f.txt", "line_a\nline_b", "LINE_A\nLINE_B")
        assert out.startswith("Successfully replaced text in")
        assert "fuzzy" in out
        result = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert "LINE_A" in result
        assert "LINE_B" in result
        assert "line_c" in result

    def test_fuzzy_multiple_matches_error(self, tmp_path):
        """L1: Fuzzy match with multiple matches → error about multiple places."""
        content = "    x = 1\n\n    x = 1\n"
        (tmp_path / "f.txt").write_text(content, encoding="utf-8")
        out = replace_in_file("f.txt", "x = 1", "x = 2")
        assert "Error" in out
        # The error mentions multiple matches (exact or fuzzy)
        assert "2 times" in out or "2 places" in out or "matches" in out


class TestReplaceInFileLineNumbers:
    def test_line_numbers_in_target_stripped(self, tmp_path):
        """L2: Line numbers in target_text (from read_file output) → stripped, match succeeds."""
        content = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        (tmp_path / "f.py").write_text(content, encoding="utf-8")
        # Simulate what the model pastes back from read_file: lines 4-5 with
        # the gutter (right-aligned number + tab). _strip_read_line_numbers
        # requires ≥2 numbered lines, consecutive, with padding to trigger.
        target_with_line_numbers = "    4\tdef bar():\n    5\t    return 2"
        out = replace_in_file("f.py", target_with_line_numbers, "def bar():\n    return 20")
        assert out.startswith("Successfully replaced text in")
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        assert "return 20" in result
        assert "return 2\n" not in result
        assert "def foo" in result


class TestReplaceInFileUTF8:
    def test_utf8_replacement_preserved(self, tmp_path):
        """L4×D10: Replacement with UTF-8 content → preserved correctly."""
        (tmp_path / "f.txt").write_text("placeholder\nother\n", encoding="utf-8")
        replacement = "# Комментарий\nx = 'привет 🌍'\n"
        out = replace_in_file("f.txt", "placeholder", replacement)
        assert out.startswith("Successfully replaced text in")
        result = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert "привет" in result
        assert "🌍" in result
        assert "Комментарий" in result
        assert "other" in result


class TestReplaceInFileSyntaxValidation:
    def test_syntax_validation_fails_reverts(self, tmp_path, monkeypatch):
        """L1: Syntax validation fails after replace → 'Modification aborted', file reverted."""
        import tools.file_ops as file_ops_mod
        original = "def foo():\n    return 1\n"
        (tmp_path / "f.py").write_text(original, encoding="utf-8")
        monkeypatch.setattr(
            file_ops_mod, "_validate_code_syntax", lambda fp: "SyntaxError: invalid syntax"
        )
        out = replace_in_file("f.py", "return 1", "return )(")
        assert "Modification aborted" in out or "aborted" in out.lower()
        # File should be reverted to original
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == original


class TestReplaceInFileCRLF:
    def test_replace_in_crlf_file(self, tmp_path):
        """D4: Replace in file with Windows line endings (\\r\\n) → handle correctly."""
        (tmp_path / "f.txt").write_bytes(b"line1\r\nline2\r\nline3\r\n")
        out = replace_in_file("f.txt", "line2", "REPLACED")
        assert out.startswith("Successfully replaced text in")
        result = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert "REPLACED" in result
        assert "line1" in result
        assert "line3" in result


# ══════════════════════════════════════════════════════════════════════════════
# multi_replace_in_file
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiReplaceInFile:
    def test_single_file_replacement(self, tmp_path):
        """L2: Single file replacement → success."""
        (tmp_path / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
        changes = json.dumps([
            {"file_path": "a.txt", "target_text": "hello", "replacement_text": "HI"}
        ])
        out = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in out
        assert "Successfully replaced" in out
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "HI\nworld\n"

    def test_multiple_files_in_one_call(self, tmp_path):
        """L2: Multiple files in one call → all succeed."""
        (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
        changes = json.dumps([
            {"file_path": "a.txt", "target_text": "alpha", "replacement_text": "ALPHA"},
            {"file_path": "b.txt", "target_text": "beta", "replacement_text": "BETA"},
        ])
        out = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in out
        assert "Successfully replaced" in out
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "ALPHA\n"
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "BETA\n"

    def test_multiple_changes_same_file_in_order(self, tmp_path):
        """L2: Multiple changes to same file → all applied (in order)."""
        (tmp_path / "f.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        changes = json.dumps([
            {"file_path": "f.txt", "target_text": "one", "replacement_text": "ONE"},
            {"file_path": "f.txt", "target_text": "two", "replacement_text": "TWO"},
        ])
        out = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in out
        result = (tmp_path / "f.txt").read_text(encoding="utf-8")
        assert result == "ONE\nTWO\nthree\n"

    def test_invalid_json(self, tmp_path):
        """L1: Invalid JSON string → error."""
        out = multi_replace_in_file("not valid json {{{")
        assert "Error" in out

    def test_json_object_not_array(self, tmp_path):
        """L1: JSON is object not array → error."""
        out = multi_replace_in_file(json.dumps({"file_path": "a.txt"}))
        assert "Error" in out
        assert "array" in out.lower()

    def test_missing_file_path(self, tmp_path):
        """L1: Missing file_path in entry → 'Skipping change: file_path is missing.'"""
        (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
        changes = json.dumps([
            {"target_text": "hello", "replacement_text": "HI"}
        ])
        out = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in out
        assert "file_path" in out and "missing" in out.lower()

    def test_one_fails_one_succeeds(self, tmp_path):
        """L1: One file fails, one succeeds → report shows both results."""
        (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
        changes = json.dumps([
            {"file_path": "a.txt", "target_text": "hello", "replacement_text": "HI"},
            {"file_path": "nonexistent.txt", "target_text": "x", "replacement_text": "y"},
        ])
        out = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in out
        assert "Successfully replaced" in out
        assert "Error" in out
        # The successful one should have changed the file
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "HI\n"

    def test_empty_array(self, tmp_path):
        """L3: Empty array → 'Multi-replace execution finished:' with no entries."""
        out = multi_replace_in_file(json.dumps([]))
        assert "Multi-replace execution finished" in out


# ══════════════════════════════════════════════════════════════════════════════
# multi_replace_in_file_chunk
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiReplaceInFileChunk:
    def _make_file(self, tmp_path, name="m.txt", content=None):
        if content is None:
            content = "line1\nline2\nline3\nline4\nline5\n"
        (tmp_path / name).write_text(content, encoding="utf-8")
        return str(tmp_path / name)

    def test_single_chunk_replacement(self, tmp_path):
        """L2: Single chunk replacement → success."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 2, "end_line": 2,
            "target_content": "line2",
            "replacement_content": "REPLACED",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert "REPLACED" in result
        assert "line1" in result
        assert "line3" in result

    def test_exact_content_after_chunk_replace(self, tmp_path):
        """L4×D10: After replace, file content matches expected exactly."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 2, "end_line": 3,
            "target_content": "line2\nline3",
            "replacement_content": "REPLACED",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        assert (tmp_path / "m.txt").read_text(encoding="utf-8") == "line1\nREPLACED\nline4\nline5\n"

    def test_multiple_chunks_reverse_order(self, tmp_path):
        """L2: Multiple chunks in one file → all applied (reverse order preserves line numbers)."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([
            {
                "start_line": 1, "end_line": 1,
                "target_content": "line1",
                "replacement_content": "AAA",
            },
            {
                "start_line": 4, "end_line": 5,
                "target_content": "line4\nline5",
                "replacement_content": "BBB",
            },
        ])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert result == "AAA\nline2\nline3\nBBB\n"

    def test_invalid_json(self, tmp_path):
        """L1: Invalid JSON → error."""
        fp = self._make_file(tmp_path)
        out = multi_replace_in_file_chunk(fp, "not valid json {{{")
        assert "Error" in out
        assert "not valid JSON" in out

    def test_non_array_json(self, tmp_path):
        """L1: Non-array JSON → error."""
        fp = self._make_file(tmp_path)
        out = multi_replace_in_file_chunk(fp, json.dumps({"start_line": 1}))
        assert "Error" in out
        assert "array" in out.lower()

    def test_missing_start_line(self, tmp_path):
        """L1: Missing start_line → error."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "end_line": 2,
            "target_content": "line2",
            "replacement_content": "X",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Error" in out
        assert "start_line" in out and "end_line" in out

    def test_missing_end_line(self, tmp_path):
        """L1: Missing end_line → error."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 2,
            "target_content": "line2",
            "replacement_content": "X",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Error" in out
        assert "start_line" in out and "end_line" in out

    def test_start_greater_than_end(self, tmp_path):
        """L1: start_line > end_line → error."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 5, "end_line": 2,
            "target_content": "",
            "replacement_content": "X",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Error" in out
        assert "Invalid line range" in out

    def test_target_content_mismatch(self, tmp_path):
        """L1: target_content doesn't match actual content → error."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 2, "end_line": 2,
            "target_content": "WRONG CONTENT",
            "replacement_content": "X",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Error" in out
        assert "does not match" in out

    def test_empty_replacement_removes_lines(self, tmp_path):
        """L3: Empty replacement_content → lines removed (empty list)."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 2, "end_line": 3,
            "target_content": "line2\nline3",
            "replacement_content": "",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert result == "line1\nline4\nline5\n"

    def test_start_line_1(self, tmp_path):
        """D1: start_line=1 (first line) → works correctly."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 1, "end_line": 1,
            "target_content": "line1",
            "replacement_content": "FIRST",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert result.startswith("FIRST\n")
        assert "line2" in result

    def test_end_line_last(self, tmp_path):
        """D1: end_line = last line → works correctly."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 5, "end_line": 5,
            "target_content": "line5",
            "replacement_content": "LAST",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert result.endswith("LAST\n")
        assert "line4" in result

    def test_multiple_non_overlapping_chunks_exact(self, tmp_path):
        """L4×D10: Multiple non-overlapping chunks → verify all applied correctly."""
        fp = self._make_file(tmp_path, content="a\nb\nc\nd\ne\nf\ng\nh\n")
        changes = json.dumps([
            {
                "start_line": 2, "end_line": 2,
                "target_content": "b",
                "replacement_content": "B",
            },
            {
                "start_line": 5, "end_line": 6,
                "target_content": "e\nf",
                "replacement_content": "EF",
            },
            {
                "start_line": 8, "end_line": 8,
                "target_content": "h",
                "replacement_content": "H",
            },
        ])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        assert (tmp_path / "m.txt").read_text(encoding="utf-8") == "a\nB\nc\nd\nEF\ng\nH\n"

    def test_nonexistent_file(self, tmp_path):
        """L1: Non-existent file → error."""
        changes = json.dumps([{
            "start_line": 1, "end_line": 1,
            "target_content": "x",
            "replacement_content": "y",
        }])
        out = multi_replace_in_file_chunk("does_not_exist.txt", changes)
        assert "Error" in out
        assert "does not exist" in out

    def test_syntax_validation_fails_reverts(self, tmp_path, monkeypatch):
        """L1: Syntax validation fails → 'Modification aborted', file reverted."""
        # multi_replace_in_file_chunk does a LOCAL import of
        # _validate_code_syntax from tools._helpers, so patch it there.
        import tools._helpers as helpers
        original = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        (tmp_path / "f.py").write_text(original, encoding="utf-8")
        monkeypatch.setattr(
            helpers, "_validate_code_syntax", lambda fp: "SyntaxError: invalid syntax"
        )
        changes = json.dumps([{
            "start_line": 2, "end_line": 2,
            "target_content": "    return 1",
            "replacement_content": "    return )(",
        }])
        out = multi_replace_in_file_chunk("f.py", changes)
        assert "Modification aborted" in out or "aborted" in out.lower()
        # File should be reverted
        assert (tmp_path / "f.py").read_text(encoding="utf-8") == original

    def test_empty_target_content_skips_verification(self, tmp_path):
        """L3: Changes with empty target_content (norm_target empty) → skips target verification, applies replacement."""
        fp = self._make_file(tmp_path)
        changes = json.dumps([{
            "start_line": 3, "end_line": 3,
            "target_content": "",
            "replacement_content": "INJECTED",
        }])
        out = multi_replace_in_file_chunk(fp, changes)
        assert "Successfully applied chunk replacements" in out
        result = (tmp_path / "m.txt").read_text(encoding="utf-8")
        assert "INJECTED" in result
        assert "line2" in result
        assert "line4" in result


# ══════════════════════════════════════════════════════════════════════════════
# Bounded line-range replace_in_file (Antigravity-style start_line/end_line)
# ══════════════════════════════════════════════════════════════════════════════

class TestReplaceInFileBounded:
    def test_bounded_replace_resolves_duplicate_lines(self, tmp_path):
        """Bounded replace isolates the edit to the window, resolving duplicates outside."""
        content = (
            "def first():\n"
            "    return True\n"   # line 2
            "\n"
            "def second():\n"
            "    return True\n"   # line 5
            "\n"
            "def third():\n"
            "    return True\n"   # line 8
        )
        file_p = tmp_path / "app.py"
        file_p.write_text(content, encoding="utf-8")

        # Global replace fails due to 3 duplicate occurrences
        res_global = replace_in_file(str(file_p), "    return True", "    return False")
        assert "Error: The target text appears 3 times" in res_global

        # Bounded replace targeting only line 5 succeeds
        res_bounded = replace_in_file(
            str(file_p), "    return True", "    return False",
            start_line=4, end_line=6
        )
        assert "Successfully replaced text" in res_bounded

        updated = file_p.read_text(encoding="utf-8").splitlines()
        assert updated[1] == "    return True"   # line 2 unchanged
        assert updated[4] == "    return False"  # line 5 modified
        assert updated[7] == "    return True"   # line 8 unchanged

    def test_bounded_replace_fuzzy_within_range(self, tmp_path):
        """Fuzzy fallback works cleanly within the bounded range."""
        content = (
            "class Service:\n"
            "    def run(self):\n"
            "        x = 10\n"
            "        y = 20\n"
        )
        file_p = tmp_path / "srv.py"
        file_p.write_text(content, encoding="utf-8")

        # Model sends unindented multi-line block with broken whitespace within lines 2-4
        target = "def run(self):\nx = 10\ny = 20"
        replacement = "def run(self):\nx = 99\ny = 20"
        res = replace_in_file(
            str(file_p), target, replacement,
            start_line=2, end_line=4
        )
        assert "Successfully replaced text" in res
        assert "fuzzy match within lines 2-4" in res
        assert "x = 99" in file_p.read_text(encoding="utf-8")

    def test_bounded_replace_not_found_in_range(self, tmp_path):
        """Returns helpful error when target exists elsewhere in file but not in range."""
        content = "line1\nline2\nline3\ntarget_here\nline5\n"
        file_p = tmp_path / "doc.txt"
        file_p.write_text(content, encoding="utf-8")

        res = replace_in_file(str(file_p), "target_here", "new_val", start_line=1, end_line=3)
        assert "Error: The target text was not found within lines 1-3" in res
        assert "omit start_line/end_line to search the whole file" in res
        assert file_p.read_text(encoding="utf-8") == content

    def test_bounded_replace_invalid_ranges(self, tmp_path):
        """Rejects inverted or completely out-of-bound ranges cleanly."""
        file_p = tmp_path / "demo.txt"
        file_p.write_text("a\nb\nc\n", encoding="utf-8")

        res_inv = replace_in_file(str(file_p), "b", "B", start_line=3, end_line=1)
        assert "Error: 'end_line' (1) cannot be less than 'start_line' (3)" in res_inv

        res_oob = replace_in_file(str(file_p), "b", "B", start_line=50, end_line=60)
        assert "Error: 'start_line' (50) exceeds total lines in file (3)" in res_oob

    def test_bounded_replace_duplicate_within_range(self, tmp_path):
        """Reports ambiguity when target repeats multiple times even inside the window."""
        content = "x = 1\nx = 1\nx = 1\n"
        file_p = tmp_path / "dup.py"
        file_p.write_text(content, encoding="utf-8")

        res = replace_in_file(str(file_p), "x = 1", "x = 2", start_line=1, end_line=3)
        assert "appears 3 times within lines 1-3" in res
        assert "Narrow down start_line/end_line" in res

    def test_bounded_replace_arg_coercion_and_aliases(self):
        """Verifies aliases (from_line, to_line, start, end) and int coercion for replace_in_file."""
        from src.agent.arg_coercion import coerce_and_log

        raw = {
            "file_path": "test.py",
            "target": "old_code",
            "replacement": "new_code",
            "from_line": "12",
            "to_line": "18"
        }
        coerced = coerce_and_log("replace_in_file", raw)
        assert coerced["target_text"] == "old_code"
        assert coerced["replacement_text"] == "new_code"
        assert coerced["start_line"] == 12
        assert coerced["end_line"] == 18
        assert isinstance(coerced["start_line"], int)
        assert isinstance(coerced["end_line"], int)

    @pytest.mark.lmtrust(layer="L0", direction="D7")
    def test_l0_smoke_bounded_replace(self, tmp_path):
        """L0 Smoke: Basic single-line bounded replacement succeeds and preserves file."""
        file_p = tmp_path / "smoke.txt"
        file_p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        res = replace_in_file(str(file_p), "beta", "BETA", start_line=2, end_line=3)
        assert "Successfully replaced text" in res
        assert file_p.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_l1_contract_defaults_and_coordinates(self, tmp_path):
        """L1 Contract: start_line defaults to 1, end_line defaults to EOF, coords respected."""
        content = "first\nsecond\nthird\nfourth\n"
        file_p = tmp_path / "contract.txt"
        file_p.write_text(content, encoding="utf-8")

        # start_line omitted -> defaults to line 1
        res1 = replace_in_file(str(file_p), "first", "1st", end_line=2)
        assert "Successfully replaced text" in res1
        assert file_p.read_text(encoding="utf-8") == "1st\nsecond\nthird\nfourth\n"

        # end_line omitted -> defaults to total_lines (4)
        res2 = replace_in_file(str(file_p), "fourth", "4th", start_line=3)
        assert "Successfully replaced text" in res2
        assert file_p.read_text(encoding="utf-8") == "1st\nsecond\nthird\n4th\n"

    @pytest.mark.lmtrust(layer="L1", direction="D3")
    def test_l1_contract_unicode_and_blank_lines(self, tmp_path):
        """L1 Coordinates: Multi-byte UTF-8 (Cyrillic, Emoji) and blank lines slice accurately."""
        content = "Привет мир 🌟\n\nСтрока для замены\nОкончание файла\n"
        file_p = tmp_path / "unicode.txt"
        file_p.write_text(content, encoding="utf-8")

        res = replace_in_file(
            str(file_p),
            "Строка для замены",
            "Успешно заменено",
            start_line=2,
            end_line=4,
        )
        assert "Successfully replaced text" in res
        expected = "Привет мир 🌟\n\nУспешно заменено\nОкончание файла\n"
        assert file_p.read_text(encoding="utf-8") == expected

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_l2_boundary_single_line_window(self, tmp_path):
        """L2 Boundary: Single-line range (start_line == end_line)."""
        content = "line1\ntarget\nline3\n"
        file_p = tmp_path / "single_line.txt"
        file_p.write_text(content, encoding="utf-8")

        res = replace_in_file(str(file_p), "target", "REPLACED", start_line=2, end_line=2)
        assert "Successfully replaced text" in res
        assert file_p.read_text(encoding="utf-8") == "line1\nREPLACED\nline3\n"

    @pytest.mark.lmtrust(layer="L2", direction="D10")
    def test_l2_boundary_target_spans_past_end_line(self, tmp_path):
        """L2 Boundary: Target starting in window but spanning past end_line must fail."""
        content = "line1\nline2\nline3\nline4\n"
        file_p = tmp_path / "cross_end.txt"
        file_p.write_text(content, encoding="utf-8")

        # target spans lines 2 and 3, but end_line is 2
        res = replace_in_file(str(file_p), "line2\nline3", "NEW", start_line=1, end_line=2)
        assert "Error: The target text was not found within lines 1-2" in res
        assert file_p.read_text(encoding="utf-8") == content

    @pytest.mark.lmtrust(layer="L2", direction="D10")
    def test_l2_boundary_target_spans_before_start_line(self, tmp_path):
        """L2 Boundary: Target starting before start_line and ending inside window must fail."""
        content = "line1\nline2\nline3\nline4\n"
        file_p = tmp_path / "cross_start.txt"
        file_p.write_text(content, encoding="utf-8")

        # target spans lines 1 and 2, but start_line is 2
        res = replace_in_file(str(file_p), "line1\nline2", "NEW", start_line=2, end_line=4)
        assert "Error: The target text was not found within lines 2-4" in res
        assert file_p.read_text(encoding="utf-8") == content

    @pytest.mark.lmtrust(layer="L2", direction="D3")
    def test_l2_boundary_crlf_preservation(self, tmp_path):
        """L2 Boundary: CRLF newlines are preserved after bounded replacement."""
        content_crlf = "header\r\ntarget_val\r\nfooter\r\n"
        file_p = tmp_path / "crlf_bounded.txt"
        file_p.write_bytes(content_crlf.encode("utf-8"))

        res = replace_in_file(str(file_p), "target_val", "new_val", start_line=1, end_line=3)
        assert "Successfully replaced text" in res
        raw_bytes = file_p.read_bytes()
        assert b"\r\n" in raw_bytes
        assert b"target_val" not in raw_bytes
        assert b"new_val" in raw_bytes
        assert raw_bytes == b"header\r\nnew_val\r\nfooter\r\n"

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_l4_adversarial_invalid_types_and_defensive_coercion(self, tmp_path):
        """L4 Adversarial: Non-integer line bounds rejected; string integers gracefully handled."""
        content = "a\nb\nc\n"
        file_p = tmp_path / "adversarial.txt"
        file_p.write_text(content, encoding="utf-8")

        # Invalid string non-number
        res_err = replace_in_file(str(file_p), "b", "B", start_line="not_an_int")
        assert "Error: 'start_line' and 'end_line' must be valid integers." in res_err

        # String integer representation should be coerced gracefully
        res_ok = replace_in_file(str(file_p), "b", "B", start_line="2", end_line="3")
        assert "Successfully replaced text" in res_ok
        assert file_p.read_text(encoding="utf-8") == "a\nB\nc\n"

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_l4b_fallback_fuzzy_blocked_at_boundary(self, tmp_path):
        """L4b Fallback: Fuzzy matcher must not leak across end_line boundary."""
        content = (
            "def outer():\n"
            "    step1()\n"
            "    step2()\n"
            "    step3()\n"
        )
        file_p = tmp_path / "fuzzy_bound.py"
        file_p.write_text(content, encoding="utf-8")

        # Target spans step1 and step2, but user specified end_line=2 (which only contains step1)
        # Even with mismatched indentation, fuzzy match must not match into step2 (line 3)
        target = "step1()\nstep2()"
        replacement = "step1_new()\nstep2_new()"
        res = replace_in_file(str(file_p), target, replacement, start_line=1, end_line=2)
        assert "Error: The target text was not found within lines 1-2" in res
        assert file_p.read_text(encoding="utf-8") == content

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_l9_negative_space_strictly_contained(self, tmp_path):
        """L9 Negative Space: Lines strictly outside [start_line, end_line] remain untouched."""
        lines = [f"line_{i:02d} = {i}" for i in range(1, 11)]
        content = "\n".join(lines) + "\n"
        file_p = tmp_path / "neg_space.py"
        file_p.write_text(content, encoding="utf-8")

        res = replace_in_file(
            str(file_p),
            "line_05 = 5",
            "line_05 = 999",
            start_line=4,
            end_line=6,
        )
        assert "Successfully replaced text" in res
        new_lines = file_p.read_text(encoding="utf-8").splitlines()

        # Lines 1-3 (indices 0-2) are 100% identical
        assert new_lines[:3] == lines[:3]
        # Line 5 changed
        assert new_lines[4] == "line_05 = 999"
        # Lines 7-10 (indices 6-9) are 100% identical
        assert new_lines[6:] == lines[6:]