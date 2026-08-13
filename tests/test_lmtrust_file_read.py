"""Blind-spot LMTrust tests for read_file, list_directory, get_file_outline,
and create_directory.

These exercise line-range clamping, empty/edge-range reads, directory-vs-file
confusion, outline depth limits, and directory-creation edge cases that a
happy-path smoke test would miss.

Pattern: monkeypatch tools.file_ops.snapshot to a no-op and replace
memory_manager.memory (and the already-imported tools.file_ops.memory) with a
FakeMemory so no real I/O side-effects leak into the test session.
"""

import pytest

import tools.file_ops
import memory_manager
from tools.file_ops import (
    read_file,
    list_directory,
    get_file_outline,
    create_directory,
)


# --------------------------------------------------------------------------- #
# Fakes & isolation fixture
# --------------------------------------------------------------------------- #
class FakeMemory:
    """Minimal stand-in for _MemoryProxy — swallows all state-mutating calls."""

    def __init__(self):
        self.completed = []
        self.files_modified = []
        self.facts = []
        self.errors = []

    def add_completed(self, s):
        self.completed.append(s)

    def add_file_modified(self, s):
        self.files_modified.append(s)

    def add_fact(self, s):
        self.facts.append(s)

    def add_error(self, s):
        self.errors.append(s)

    def set_objective(self, s):
        pass

    def set_current_task(self, s):
        pass

    def build_context_note(self):
        return ""

    def clear(self):
        pass


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Prevent snapshot I/O and memory persistence during tests."""
    monkeypatch.setattr(tools.file_ops, "snapshot", lambda *a, **kw: None)
    fake = FakeMemory()
    monkeypatch.setattr(memory_manager, "memory", fake)
    monkeypatch.setattr(tools.file_ops, "memory", fake)
    yield


# --------------------------------------------------------------------------- #
# L0 — Smoke: every tool works on a happy-path input
# --------------------------------------------------------------------------- #
class TestSmoke:
    """Minimal sanity checks: each tool succeeds on a valid input."""

    def test_read_file_basic(self, tmp_path):
        f = tmp_path / "smoke.txt"
        f.write_text("hello\n", encoding="utf-8")
        result = read_file(str(f))
        assert "hello" in result

    def test_list_directory_basic(self, tmp_path):
        (tmp_path / "child.txt").write_text("x", encoding="utf-8")
        result = list_directory(str(tmp_path))
        assert "child.txt" in result

    def test_get_file_outline_basic(self, tmp_path):
        f = tmp_path / "smoke.py"
        f.write_text("def foo():\n    pass\n", encoding="utf-8")
        result = get_file_outline(str(f))
        assert "foo" in result

    def test_create_directory_basic(self, tmp_path):
        target = tmp_path / "newdir"
        result = create_directory(str(target))
        assert "Successfully created" in result
        assert target.is_dir()


# --------------------------------------------------------------------------- #
# L1 — read_file line-range edge cases
# --------------------------------------------------------------------------- #
class TestReadFileEdgeCases:
    """Boundary conditions for start_line / end_line parameters."""

    def test_start_line_zero_clamps_to_one(self, tmp_path):
        """start_line=0 is falsy, so `0 or 1` → 1; line 1 must still appear."""
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = read_file(str(f), start_line=0)
        assert "line1" in result, "start_line=0 should clamp to 1, not skip line 1"

    def test_end_line_zero_returns_empty_body(self, tmp_path):
        """end_line=0 is not None, so last=0; every line number > 0 is skipped."""
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = read_file(str(f), end_line=0)
        assert "line1" not in result
        assert "line2" not in result
        assert "line3" not in result

    def test_end_line_below_start_line_returns_empty_body(self, tmp_path):
        """When end_line < start_line, the range is empty — no content shown."""
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = read_file(str(f), start_line=3, end_line=1)
        assert "line1" not in result
        assert "line2" not in result
        assert "line3" not in result

    def test_negative_start_line_clamps_to_one(self, tmp_path):
        """start_line=-5 → max(1, -5) = 1; line 1 must appear."""
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\n", encoding="utf-8")
        result = read_file(str(f), start_line=-5)
        assert "line1" in result, "negative start_line should clamp to 1"

    def test_empty_file_returns_empty_string(self, tmp_path):
        """A zero-byte file yields no lines; body is '' and no header is added."""
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = read_file(str(f))
        assert result == ""

    def test_single_line_file(self, tmp_path):
        """A file with exactly one line returns that line with a line number."""
        f = tmp_path / "one.txt"
        f.write_text("solo\n", encoding="utf-8")
        result = read_file(str(f))
        assert "solo" in result
        # The line number prefix should be present.
        assert "1" in result

    def test_crlf_line_endings_normalized(self, tmp_path):
        """Universal-newline mode strips \\r; output must not contain \\r."""
        f = tmp_path / "crlf.txt"
        f.write_bytes(b"line1\r\nline2\r\n")
        result = read_file(str(f))
        assert "\r" not in result
        assert "line1" in result
        assert "line2" in result

    def test_start_line_beyond_file_length_returns_empty_body(self, tmp_path):
        """start_line=100 on a 3-line file: all lines skipped, no content."""
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = read_file(str(f), start_line=100)
        assert "line1" not in result
        assert "line2" not in result
        assert "line3" not in result

    def test_end_line_beyond_file_length_returns_all_lines(self, tmp_path):
        """end_line=100 on a 3-line file: all 3 lines shown, no truncation."""
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = read_file(str(f), start_line=1, end_line=100)
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result


# --------------------------------------------------------------------------- #
# L2 — read_file adversarial / error paths
# --------------------------------------------------------------------------- #
class TestReadFileAdversarial:
    """Invalid or hostile inputs that should produce an error, not a crash."""

    def test_reading_a_directory_path_errors(self, tmp_path):
        """A directory is not a file — must return an error string."""
        result = read_file(str(tmp_path))
        assert "Error" in result
        assert "not a file" in result

    def test_reading_nonexistent_file_errors(self, tmp_path):
        """A path that doesn't exist must report an error."""
        result = read_file(str(tmp_path / "ghost.txt"))
        assert "Error" in result
        assert "does not exist" in result


# --------------------------------------------------------------------------- #
# L3 — list_directory edge cases
# --------------------------------------------------------------------------- #
class TestListDirectoryEdgeCases:
    """Boundary conditions for directory listing."""

    def test_empty_directory(self, tmp_path):
        """An empty directory reports that it is empty."""
        result = list_directory(str(tmp_path))
        assert "empty" in result.lower()

    def test_mixed_files_and_subdirs(self, tmp_path):
        """Both files and subdirectories are listed with type tags."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
        (tmp_path / "beta.txt").write_text("b", encoding="utf-8")
        result = list_directory(str(tmp_path))
        assert "subdir" in result
        assert "alpha.txt" in result
        assert "beta.txt" in result
        assert "[DIR]" in result
        assert "[FILE]" in result


# --------------------------------------------------------------------------- #
# L4 — list_directory adversarial
# --------------------------------------------------------------------------- #
class TestListDirectoryAdversarial:
    """Invalid inputs that should produce an error, not a crash."""

    def test_list_directory_on_file_path_errors(self, tmp_path):
        """Listing a file path must return an error, not crash."""
        f = tmp_path / "not_a_dir.txt"
        f.write_text("content", encoding="utf-8")
        result = list_directory(str(f))
        assert "Error" in result
        assert "not a directory" in result

    def test_list_directory_nonexistent_errors(self, tmp_path):
        """A nonexistent directory must report an error."""
        result = list_directory(str(tmp_path / "ghost_dir"))
        assert "Error" in result
        assert "does not exist" in result


# --------------------------------------------------------------------------- #
# L5 — get_file_outline edge cases
# --------------------------------------------------------------------------- #
class TestGetFileOutlineEdgeCases:
    """Boundary conditions for the Python outline parser."""

    def test_empty_python_file(self, tmp_path):
        """An empty .py file has no structures — reports 'No significant'."""
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        result = get_file_outline(str(f))
        assert "No significant structures" in result

    def test_nested_classes_not_shown(self, tmp_path):
        """The outline only descends one level into a class; a nested class
        and its methods must NOT appear (this is the blind spot)."""
        f = tmp_path / "nested.py"
        f.write_text(
            "class Outer:\n"
            "    def method(self):\n"
            "        pass\n"
            "    class Inner:\n"
            "        def inner_method(self):\n"
            "            pass\n",
            encoding="utf-8",
        )
        result = get_file_outline(str(f))
        # Outer class and its direct method ARE shown.
        assert "Outer" in result
        assert "method" in result
        # The nested class and its method are the blind spot — NOT shown.
        assert "Inner" not in result, "nested class should not appear in outline"
        assert "inner_method" not in result, (
            "nested class method should not appear in outline"
        )


# --------------------------------------------------------------------------- #
# L6 — get_file_outline adversarial
# --------------------------------------------------------------------------- #
class TestGetFileOutlineAdversarial:
    """Invalid inputs that should produce an error, not a crash."""

    def test_js_file_errors(self, tmp_path):
        """A .js file is neither .py nor .cs — must return an error."""
        f = tmp_path / "script.js"
        f.write_text("function foo() { return 1; }\n", encoding="utf-8")
        result = get_file_outline(str(f))
        assert "Error" in result
        assert ".js" in result

    def test_syntax_error_file_errors(self, tmp_path):
        """A .py file with a SyntaxError must report it, not crash."""
        f = tmp_path / "broken.py"
        f.write_text("def broken(:\n    pass\n", encoding="utf-8")
        result = get_file_outline(str(f))
        assert "Error" in result
        assert "SyntaxError" in result

    def test_outline_on_directory_errors(self, tmp_path):
        """A directory path is not a file — must return an error."""
        result = get_file_outline(str(tmp_path))
        assert "Error" in result
        assert "not a file" in result


# --------------------------------------------------------------------------- #
# L7 — create_directory edge cases
# --------------------------------------------------------------------------- #
class TestCreateDirectoryEdgeCases:
    """Boundary conditions for directory creation."""

    def test_path_already_exists(self, tmp_path):
        """Creating a directory that already exists reports 'already exists'."""
        d = tmp_path / "existing"
        d.mkdir()
        result = create_directory(str(d))
        assert "already exists" in result
        # Still a directory, unchanged.
        assert d.is_dir()

    def test_nested_path_creation(self, tmp_path):
        """A deeply nested path is created in one call via parents=True."""
        target = tmp_path / "a" / "b" / "c" / "d"
        result = create_directory(str(target))
        assert "Successfully created" in result
        assert target.is_dir()


# --------------------------------------------------------------------------- #
# L8 — create_directory adversarial
# --------------------------------------------------------------------------- #
class TestCreateDirectoryAdversarial:
    """Invalid inputs that should produce an error or safe message, not crash."""

    def test_path_is_a_file(self, tmp_path):
        """When the path is an existing file, create_directory must NOT
        overwrite it and must report that the path already exists."""
        f = tmp_path / "iam_a_file.txt"
        f.write_text("content", encoding="utf-8")
        result = create_directory(str(f))
        # It detects the path exists and bails — it must not convert the file.
        assert "already exists" in result or "Error" in result
        # The path must still be a file, not a directory.
        assert f.is_file()
        assert f.read_text(encoding="utf-8") == "content"