"""Comprehensive pytest tests for write_file, append_to_file, and delete_file
from tools/file_ops.py.

Covers:
  * write_file: new files, nested dirs, overwrite guard (>150 lines / >10000 bytes),
    boundary conditions (exactly 150 lines / 10000 bytes), UTF-8 fidelity, empty
    content, no trailing newline, plugin restriction, .meta restriction, and the
    syntax-validation safety net (revert / no-create).
  * append_to_file: existing file, non-existent file, nested path, newline
    insertion logic (file ending with newline, not ending with newline, content
    starting with \n, empty file, multiple appends), UTF-8.
  * delete_file: existing file, non-existent file, directory, approval denied /
    granted, plugin restriction, filesystem verification.
"""

import os
import sys
import pytest
from pathlib import Path

from tools.file_ops import write_file, append_to_file, delete_file


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

    fake_mem = FakeMemory()
    monkeypatch.setattr(memory_manager, "memory", fake_mem)

    import file_tracker
    monkeypatch.setattr(file_tracker, "snapshot", lambda *a, **k: None)

    # Stub approval to auto-approve
    import approval
    monkeypatch.setattr(approval, "request_approval", lambda *a, **k: True)

    # Stub _validate_code_syntax to always pass (no syntax errors)
    import tools._helpers as helpers
    monkeypatch.setattr(helpers, "_validate_code_syntax", lambda fp: None)

    # Stub _unity_script_placement_error to always pass
    monkeypatch.setattr(helpers, "_unity_script_placement_error", lambda p: None)

    yield fake_mem


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# write_file
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteFileNew:
    def test_write_new_file_success(self, tmp_path):
        out = write_file("new.txt", "hello world\n")
        assert "Successfully wrote to" in out
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello world\n"

    def test_write_nested_path_creates_dirs(self, tmp_path):
        out = write_file("a/b/c/deep.txt", "deep content\n")
        assert "Successfully wrote to" in out
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").read_text(encoding="utf-8") == "deep content\n"

    def test_write_empty_content(self, tmp_path):
        out = write_file("empty.txt", "")
        assert "Successfully wrote to" in out
        assert (tmp_path / "empty.txt").exists()
        assert (tmp_path / "empty.txt").read_text(encoding="utf-8") == ""
        assert (tmp_path / "empty.txt").stat().st_size == 0

    def test_write_no_trailing_newline(self, tmp_path):
        out = write_file("nonewline.txt", "no newline here")
        assert "Successfully wrote to" in out
        content = (tmp_path / "nonewline.txt").read_text(encoding="utf-8")
        assert content == "no newline here"
        assert not content.endswith("\n")

    def test_write_utf8_cyrillic_emoji(self, tmp_path):
        content = "# Комментарий\nx = 'привет 🌍'\n"
        out = write_file("utf8.txt", content)
        assert "Successfully wrote to" in out
        on_disk = (tmp_path / "utf8.txt").read_text(encoding="utf-8")
        assert on_disk == content
        assert "привет" in on_disk
        assert "🌍" in on_disk

    def test_write_content_matches_exactly(self, tmp_path):
        content = "line1\nline2\nline3\n"
        out = write_file("exact.txt", content)
        assert "Successfully wrote to" in out
        assert (tmp_path / "exact.txt").read_text(encoding="utf-8") == content


class TestWriteFileOverwrite:
    def test_overwrite_small_file_no_overwrite_flag(self, tmp_path):
        """A small existing file (<=150 lines, <=10000 bytes) can be overwritten
        without overwrite=True."""
        (tmp_path / "small.txt").write_text("original\n", encoding="utf-8")
        out = write_file("small.txt", "replaced\n")
        assert "Successfully wrote to" in out
        assert (tmp_path / "small.txt").read_text(encoding="utf-8") == "replaced\n"

    def test_overwrite_large_file_with_overwrite_flag(self, tmp_path):
        """A large file can be overwritten with overwrite=True."""
        big = "line\n" * 200  # 200 lines > 150
        (tmp_path / "big.txt").write_text(big, encoding="utf-8")
        out = write_file("big.txt", "new content\n", overwrite=True)
        assert "Successfully wrote to" in out
        assert (tmp_path / "big.txt").read_text(encoding="utf-8") == "new content\n"

    def test_overwrite_large_file_by_lines_refused(self, tmp_path):
        """Overwriting a file with >150 lines without overwrite=True is refused,
        and the file is NOT modified."""
        big = "line\n" * 200
        (tmp_path / "big_lines.txt").write_text(big, encoding="utf-8")
        out = write_file("big_lines.txt", "should not write\n")
        assert "already exists" in out
        assert "overwrite" in out.lower()
        # File unchanged
        assert (tmp_path / "big_lines.txt").read_text(encoding="utf-8") == big

    def test_overwrite_large_file_by_bytes_refused(self, tmp_path):
        """Overwriting a file >10000 bytes (but few lines) without overwrite=True
        is refused, and the file is NOT modified."""
        big = "x" * 10001 + "\n"  # 10002 bytes, 1 line
        (tmp_path / "big_bytes.txt").write_text(big, encoding="utf-8")
        out = write_file("big_bytes.txt", "should not write\n")
        assert "already exists" in out
        # File unchanged
        assert (tmp_path / "big_bytes.txt").read_text(encoding="utf-8") == big

    def test_exactly_150_lines_writable_without_overwrite(self, tmp_path):
        """150 lines is NOT > 150, so writable without overwrite=True."""
        content = "line\n" * 150
        (tmp_path / "exactly150.txt").write_text(content, encoding="utf-8")
        out = write_file("exactly150.txt", "replaced\n")
        assert "Successfully wrote to" in out
        assert (tmp_path / "exactly150.txt").read_text(encoding="utf-8") == "replaced\n"

    def test_151_lines_requires_overwrite(self, tmp_path):
        """151 lines IS > 150, so requires overwrite=True."""
        content = "line\n" * 151
        (tmp_path / "exactly151.txt").write_text(content, encoding="utf-8")
        out = write_file("exactly151.txt", "replaced\n")
        assert "already exists" in out
        # With overwrite it succeeds
        out2 = write_file("exactly151.txt", "replaced\n", overwrite=True)
        assert "Successfully wrote to" in out2

    def test_exactly_10000_bytes_writable_without_overwrite(self, tmp_path):
        """10000 bytes is NOT > 10000, so writable without overwrite=True.

        Uses write_bytes to control the exact on-disk size, since write_text
        on Windows translates \\n to \\r\\n (changing the byte count).
        """
        (tmp_path / "exactly10000.txt").write_bytes(b"x" * 10000)
        assert (tmp_path / "exactly10000.txt").stat().st_size == 10000
        out = write_file("exactly10000.txt", "replaced\n")
        assert "Successfully wrote to" in out
        assert (tmp_path / "exactly10000.txt").read_text(encoding="utf-8") == "replaced\n"

    def test_10001_bytes_requires_overwrite(self, tmp_path):
        """10001 bytes IS > 10000, so requires overwrite=True.

        Uses write_bytes to control the exact on-disk size.
        """
        (tmp_path / "exactly10001.txt").write_bytes(b"x" * 10001)
        assert (tmp_path / "exactly10001.txt").stat().st_size == 10001
        out = write_file("exactly10001.txt", "replaced\n")
        assert "already exists" in out
        # With overwrite it succeeds
        out2 = write_file("exactly10001.txt", "replaced\n", overwrite=True)
        assert "Successfully wrote to" in out2


class TestWriteFileRestrictions:
    def test_plugin_restricted_py_path(self, tmp_path, monkeypatch):
        """A .py file inside the plugins/ directory is restricted."""
        from config import get_hooks_dir
        # get_hooks_dir defaults to cwd/plugins; with chdir(tmp_path) it is
        # tmp_path/plugins. Ensure the plugins dir exists so the path resolves.
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        target = plugins_dir / "something.py"
        out = write_file(str(target), "x = 1\n")
        assert "restricted" in out.lower() or "plugins" in out.lower()
        assert "Error" in out
        assert not target.exists()

    def test_meta_file_restricted(self, tmp_path):
        """A .meta file is refused (Unity restriction)."""
        target = tmp_path / "asset.cs.meta"
        out = write_file(str(target), "guid: 123\n")
        assert "Refusing to modify a Unity .meta" in out
        assert not target.exists()


class TestWriteFileSyntaxValidation:
    def test_new_file_syntax_error_not_created(self, tmp_path, monkeypatch):
        """When syntax validation fails on a NEW file, the file is NOT created
        and the message says 'The new file was not created.'"""
        import tools._helpers as helpers
        monkeypatch.setattr(
            helpers, "_validate_code_syntax", lambda fp: "SyntaxError: invalid syntax"
        )
        out = write_file("broken.py", "def (\n")
        assert "Edit REJECTED" in out
        assert "not created" in out.lower()
        assert not (tmp_path / "broken.py").exists()

    def test_existing_file_syntax_error_restored(self, tmp_path, monkeypatch):
        """When syntax validation fails on an EXISTING file, the file is
        restored to its previous version."""
        import tools._helpers as helpers
        original = "x = 1\n"
        (tmp_path / "existing.py").write_text(original, encoding="utf-8")
        monkeypatch.setattr(
            helpers, "_validate_code_syntax", lambda fp: "SyntaxError: invalid syntax"
        )
        out = write_file("existing.py", "def (\n", overwrite=True)
        assert "Edit REJECTED" in out
        assert "restored" in out.lower()
        # File restored to original
        assert (tmp_path / "existing.py").read_text(encoding="utf-8") == original


# ══════════════════════════════════════════════════════════════════════════════
# append_to_file
# ══════════════════════════════════════════════════════════════════════════════

class TestAppendToFile:
    def test_append_to_existing_file(self, tmp_path):
        (tmp_path / "app.txt").write_text("first\n", encoding="utf-8")
        out = append_to_file("app.txt", "second\n")
        assert "Successfully appended" in out
        assert (tmp_path / "app.txt").read_text(encoding="utf-8") == "first\nsecond\n"

    def test_append_to_nonexistent_file(self, tmp_path):
        out = append_to_file("new_append.txt", "created\n")
        assert "Successfully appended" in out
        assert (tmp_path / "new_append.txt").exists()
        assert (tmp_path / "new_append.txt").read_text(encoding="utf-8") == "created\n"

    def test_append_nested_path_creates_dirs(self, tmp_path):
        out = append_to_file("x/y/z/nested.txt", "nested\n")
        assert "Successfully appended" in out
        assert (tmp_path / "x" / "y" / "z" / "nested.txt").exists()
        assert (tmp_path / "x" / "y" / "z" / "nested.txt").read_text(encoding="utf-8") == "nested\n"

    def test_append_to_file_ending_with_newline_no_extra_blank(self, tmp_path):
        """Appending to a file that ends with a newline should NOT add an extra
        blank line."""
        (tmp_path / "nl.txt").write_text("line1\n", encoding="utf-8")
        append_to_file("nl.txt", "line2\n")
        content = (tmp_path / "nl.txt").read_text(encoding="utf-8")
        assert content == "line1\nline2\n"
        # No double newline
        assert "\n\n" not in content

    def test_append_to_file_not_ending_with_newline(self, tmp_path):
        """Appending to a file that does NOT end with a newline should add a
        newline before the content."""
        (tmp_path / "nonl.txt").write_text("line1", encoding="utf-8")
        append_to_file("nonl.txt", "line2\n")
        content = (tmp_path / "nonl.txt").read_text(encoding="utf-8")
        assert content == "line1\nline2\n"

    def test_append_content_starting_with_newline_no_extra(self, tmp_path):
        """If the content itself starts with \\n, no extra newline is added."""
        (tmp_path / "startnl.txt").write_text("line1", encoding="utf-8")
        append_to_file("startnl.txt", "\nline2\n")
        content = (tmp_path / "startnl.txt").read_text(encoding="utf-8")
        assert content == "line1\nline2\n"
        # No double newline from the auto-insertion
        assert "\n\n" not in content

    def test_append_to_empty_file_no_newline_prefix(self, tmp_path):
        """Appending to a 0-byte file should NOT prefix a newline."""
        (tmp_path / "empty.txt").write_bytes(b"")
        append_to_file("empty.txt", "content\n")
        content = (tmp_path / "empty.txt").read_text(encoding="utf-8")
        assert content == "content\n"
        assert not content.startswith("\n")

    def test_multiple_appends_no_double_newlines(self, tmp_path):
        """Repeated appends should not accumulate blank lines at the seams."""
        (tmp_path / "multi.txt").write_text("a\n", encoding="utf-8")
        append_to_file("multi.txt", "b\n")
        append_to_file("multi.txt", "c\n")
        append_to_file("multi.txt", "d\n")
        content = (tmp_path / "multi.txt").read_text(encoding="utf-8")
        assert content == "a\nb\nc\nd\n"
        assert "\n\n" not in content

    def test_append_utf8_content_preserved(self, tmp_path):
        (tmp_path / "utf8.txt").write_text("привет\n", encoding="utf-8")
        append_to_file("utf8.txt", "мир 🌍\n")
        content = (tmp_path / "utf8.txt").read_text(encoding="utf-8")
        assert "привет" in content
        assert "мир" in content
        assert "🌍" in content

    def test_append_empty_string(self, tmp_path):
        """Appending an empty string to a file ending with newline should leave
        the file unchanged (no extra newline added)."""
        (tmp_path / "emp.txt").write_text("line1\n", encoding="utf-8")
        append_to_file("emp.txt", "")
        content = (tmp_path / "emp.txt").read_text(encoding="utf-8")
        assert content == "line1\n"


# ══════════════════════════════════════════════════════════════════════════════
# delete_file
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteFile:
    def test_delete_existing_file(self, tmp_path):
        (tmp_path / "del.txt").write_text("bye\n", encoding="utf-8")
        out = delete_file("del.txt")
        assert "Successfully deleted" in out
        assert not (tmp_path / "del.txt").exists()

    def test_delete_nonexistent_file(self, tmp_path):
        out = delete_file("nope.txt")
        assert "does not exist" in out
        assert "Error" in out

    def test_delete_directory_error(self, tmp_path):
        (tmp_path / "adir").mkdir()
        out = delete_file("adir")
        assert "not a file" in out
        assert "Error" in out
        assert (tmp_path / "adir").exists()

    def test_delete_approval_denied(self, tmp_path, monkeypatch):
        """When approval is denied, the file is NOT deleted."""
        import approval
        monkeypatch.setattr(approval, "request_approval", lambda *a, **k: False)
        (tmp_path / "denied.txt").write_text("keep me\n", encoding="utf-8")
        out = delete_file("denied.txt")
        assert "aborted" in out.lower()
        assert (tmp_path / "denied.txt").exists()
        assert (tmp_path / "denied.txt").read_text(encoding="utf-8") == "keep me\n"

    def test_delete_approval_granted(self, tmp_path, monkeypatch):
        """When approval is granted, the file is deleted."""
        import approval
        monkeypatch.setattr(approval, "request_approval", lambda *a, **k: True)
        (tmp_path / "granted.txt").write_text("delete me\n", encoding="utf-8")
        out = delete_file("granted.txt")
        assert "Successfully deleted" in out
        assert not (tmp_path / "granted.txt").exists()

    def test_delete_plugin_restricted_before_approval(self, tmp_path, monkeypatch):
        """A plugin-restricted path returns the restriction error BEFORE the
        approval check is even reached."""
        import approval
        called = {"approval": False}
        monkeypatch.setattr(
            approval, "request_approval", lambda *a, **k: called.__setitem__("approval", True) or False
        )
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir(exist_ok=True)
        target = plugins_dir / "restricted.py"
        target.write_text("x = 1\n", encoding="utf-8")
        out = delete_file(str(target))
        assert "Error" in out
        assert "restricted" in out.lower() or "plugins" in out.lower()
        # approval was never called
        assert called["approval"] is False
        # file still exists
        assert target.exists()

    def test_delete_file_gone_from_filesystem(self, tmp_path):
        """After a successful delete, the file is truly gone from the filesystem."""
        (tmp_path / "gone.txt").write_text("gone\n", encoding="utf-8")
        delete_file("gone.txt")
        assert not (tmp_path / "gone.txt").exists()
        assert not (tmp_path / "gone.txt").is_file()