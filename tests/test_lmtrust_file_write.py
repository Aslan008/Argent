"""LMTrust blind-spot tests for write_file, append_to_file, and delete_file.

Covers edge cases that are easy to overlook:
  * write_file: overwrite flag on large files, empty content, directory paths,
    parent directory creation, None content
  * append_to_file: new file creation, newline handling (with/without trailing
    newline), empty content, directory paths
  * delete_file: non-existent file, directory, approval gate
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

from tools.file_ops import write_file, append_to_file, delete_file


@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    """Neutralise external dependencies so tests run in isolation."""
    import memory_manager

    class FakeMemory:
        def __init__(self):
            self.completed = []
            self.modified = []
            self.errors = []

        def add_completed(self, s):
            self.completed.append(s)

        def add_file_modified(self, s):
            self.modified.append(s)

        def add_error(self, s):
            self.errors.append(s)

    fake_mem = FakeMemory()
    monkeypatch.setattr(memory_manager, "memory", fake_mem)
    monkeypatch.setattr("tools.file_ops.memory", fake_mem)

    # No-op snapshot (file_tracker)
    monkeypatch.setattr("tools.file_ops.snapshot", lambda fp: None)

    # No syntax validation errors
    monkeypatch.setattr("tools.file_ops._validate_code_syntax", lambda fp: None)

    # No path restrictions
    monkeypatch.setattr("tools.file_ops._is_plugin_path_restricted", lambda fp: None)
    monkeypatch.setattr("tools.file_ops._is_unity_meta_restricted", lambda fp: None)
    monkeypatch.setattr("tools.file_ops._unity_script_placement_error", lambda fp: None)

    # Mock rag_engine to avoid indexing side-effects
    mock_rag = MagicMock()
    mock_rag.update_file_index = lambda fp: None
    mock_rag.remove_file_index = lambda fp: None
    monkeypatch.setitem(sys.modules, "rag_engine", mock_rag)

    yield


# ══════════════════════════════════════════════════════════════════════════════
# write_file — LMTrust blind spots
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteFileBlindSpots:
    """LMTrust Layer: File Creation & Overwrite Safety"""

    def test_overwrite_true_on_large_file_succeeds(self, tmp_path):
        """write_file with overwrite=True should succeed even on a large existing file."""
        large_file = tmp_path / "big.py"
        large_content = "\n".join(f"line {i}" for i in range(200))
        large_file.write_text(large_content, encoding="utf-8")

        result = write_file(str(large_file), "# overwritten\n", overwrite=True)
        assert "Successfully wrote" in result
        assert large_file.read_text(encoding="utf-8") == "# overwritten\n"

    def test_no_overwrite_on_large_file_refuses(self, tmp_path):
        """write_file without overwrite on a large file (>150 lines) should refuse."""
        large_file = tmp_path / "big.py"
        large_content = "\n".join(f"line {i}" for i in range(200))
        large_file.write_text(large_content, encoding="utf-8")

        result = write_file(str(large_file), "# should not overwrite\n", overwrite=False)
        assert "Error" in result
        assert "overwrite" in result.lower() or "replace_in_file" in result.lower()
        # File should be unchanged
        assert large_file.read_text(encoding="utf-8") == large_content

    def test_no_overwrite_on_large_file_by_bytes_refuses(self, tmp_path):
        """A file with >10000 bytes but <=150 lines should also be refused without overwrite."""
        large_file = tmp_path / "big.txt"
        # One line but >10000 bytes
        large_content = "x" * 11000
        large_file.write_text(large_content, encoding="utf-8")

        result = write_file(str(large_file), "short", overwrite=False)
        assert "Error" in result
        assert large_file.read_text(encoding="utf-8") == large_content

    def test_empty_content_creates_empty_file(self, tmp_path):
        """write_file with empty content should create an empty file and succeed."""
        new_file = tmp_path / "empty.txt"
        result = write_file(str(new_file), "")
        assert "Successfully wrote" in result
        assert new_file.exists()
        assert new_file.read_text(encoding="utf-8") == ""

    def test_write_to_directory_path_errors(self, tmp_path):
        """write_file to a path that is a directory should return an error, not crash."""
        dir_path = tmp_path / "somedir"
        dir_path.mkdir()
        result = write_file(str(dir_path), "content")
        assert "Error" in result

    def test_creates_parent_directories(self, tmp_path):
        """write_file should create parent directories that don't exist yet."""
        nested_file = tmp_path / "a" / "b" / "c" / "deep.txt"
        result = write_file(str(nested_file), "deep content")
        assert "Successfully wrote" in result
        assert nested_file.exists()
        assert nested_file.read_text(encoding="utf-8") == "deep content"

    def test_content_none_errors_gracefully(self, tmp_path):
        """write_file with content=None should return an error string, not crash."""
        new_file = tmp_path / "none_test.txt"
        result = write_file(str(new_file), None)
        assert "Error" in result
        # If the file was created (truncated before the write failed), it should
        # not contain the literal string "None".
        if new_file.exists():
            assert new_file.read_text(encoding="utf-8") != "None"

    def test_small_file_no_overwrite_succeeds(self, tmp_path):
        """A small existing file (<150 lines, <10000 bytes) should be writable without overwrite."""
        small_file = tmp_path / "small.txt"
        small_file.write_text("hello\n", encoding="utf-8")
        result = write_file(str(small_file), "world\n")
        assert "Successfully wrote" in result
        assert small_file.read_text(encoding="utf-8") == "world\n"


# ══════════════════════════════════════════════════════════════════════════════
# append_to_file — LMTrust blind spots
# ══════════════════════════════════════════════════════════════════════════════

class TestAppendToFileBlindSpots:
    """LMTrust Layer: Append Semantics & Newline Integrity"""

    def test_append_to_new_file_creates_it(self, tmp_path):
        """append_to_file to a path that doesn't exist should create the file."""
        new_file = tmp_path / "new_append.txt"
        result = append_to_file(str(new_file), "first line\n")
        assert "Successfully appended" in result
        assert new_file.exists()
        assert new_file.read_text(encoding="utf-8") == "first line\n"

    def test_append_without_trailing_newline_adds_newline(self, tmp_path):
        """Appending to a file without a trailing newline should insert a newline separator."""
        existing = tmp_path / "no_newline.txt"
        existing.write_text("hello", encoding="utf-8")  # no trailing \n

        result = append_to_file(str(existing), "world")
        assert "Successfully appended" in result
        content = existing.read_text(encoding="utf-8")
        assert content == "hello\nworld"

    def test_append_with_trailing_newline_no_extra_blank_line(self, tmp_path):
        """Appending to a file that already ends with a newline should NOT add an extra blank line."""
        existing = tmp_path / "with_newline.txt"
        existing.write_text("hello\n", encoding="utf-8")  # trailing \n

        result = append_to_file(str(existing), "world")
        assert "Successfully appended" in result
        content = existing.read_text(encoding="utf-8")
        # Should be "hello\nworld", NOT "hello\n\nworld"
        assert content == "hello\nworld"
        assert "\n\n" not in content

    def test_append_empty_content_succeeds(self, tmp_path):
        """Appending empty content should succeed without error."""
        existing = tmp_path / "empty_append.txt"
        existing.write_text("base\n", encoding="utf-8")
        result = append_to_file(str(existing), "")
        assert "Successfully appended" in result

    def test_append_to_directory_path_errors(self, tmp_path):
        """append_to_file to a directory path should return an error, not crash."""
        dir_path = tmp_path / "adir"
        dir_path.mkdir()
        result = append_to_file(str(dir_path), "content")
        assert "Error" in result

    def test_append_creates_parent_directories(self, tmp_path):
        """append_to_file should create parent directories if needed."""
        nested = tmp_path / "x" / "y" / "nested.txt"
        result = append_to_file(str(nested), "content\n")
        assert "Successfully appended" in result
        assert nested.exists()
        assert nested.read_text(encoding="utf-8") == "content\n"

    def test_append_multiple_times_accumulates(self, tmp_path):
        """Multiple appends should accumulate content correctly without extra blank lines."""
        f = tmp_path / "multi.txt"
        append_to_file(str(f), "line1\n")
        append_to_file(str(f), "line2\n")
        append_to_file(str(f), "line3\n")
        content = f.read_text(encoding="utf-8")
        assert content == "line1\nline2\nline3\n"


# ══════════════════════════════════════════════════════════════════════════════
# delete_file — LMTrust blind spots
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteFileBlindSpots:
    """LMTrust Layer: Deletion Safety & Approval Gate"""

    def test_delete_nonexistent_file_errors(self, tmp_path):
        """delete_file on a file that doesn't exist should return an error."""
        result = delete_file(str(tmp_path / "ghost.txt"))
        assert "Error" in result
        assert "does not exist" in result.lower() or "not exist" in result.lower()

    def test_delete_directory_errors(self, tmp_path):
        """delete_file on a directory should return an error, not delete the directory."""
        dir_path = tmp_path / "mydir"
        dir_path.mkdir()
        result = delete_file(str(dir_path))
        assert "Error" in result
        assert "not a file" in result.lower()
        assert dir_path.exists()  # directory should still be there

    def test_delete_with_approval_succeeds(self, tmp_path, monkeypatch):
        """delete_file should succeed when approval is granted."""
        monkeypatch.setattr("approval.request_approval", lambda *a, **kw: True)

        target = tmp_path / "deleteme.txt"
        target.write_text("bye", encoding="utf-8")
        result = delete_file(str(target))
        assert "Successfully deleted" in result
        assert not target.exists()

    def test_delete_without_approval_aborts(self, tmp_path, monkeypatch):
        """delete_file should abort when approval is denied, leaving the file intact."""
        monkeypatch.setattr("approval.request_approval", lambda *a, **kw: False)

        target = tmp_path / "keepme.txt"
        target.write_text("stay", encoding="utf-8")
        result = delete_file(str(target))
        assert "aborted" in result.lower()
        assert target.exists()  # file should still be there

    def test_delete_calls_memory_on_success(self, tmp_path, monkeypatch):
        """On successful deletion, memory.add_completed should be called."""
        monkeypatch.setattr("approval.request_approval", lambda *a, **kw: True)

        calls = []

        class TrackMemory:
            def add_completed(self, s):
                calls.append(s)

            def add_file_modified(self, s):
                pass

            def add_error(self, s):
                pass

        monkeypatch.setattr("tools.file_ops.memory", TrackMemory())

        target = tmp_path / "track.txt"
        target.write_text("x", encoding="utf-8")
        result = delete_file(str(target))
        assert "Successfully deleted" in result
        assert any("Deleted" in c or "track" in c for c in calls)