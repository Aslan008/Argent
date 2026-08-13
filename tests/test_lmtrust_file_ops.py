"""LMTrust blind-spot tests for move_file, copy_file, and replace_python_function.

Covers edge cases that are easy to overlook:
  * move_file: non-existent source, existing destination, directory source,
    content preservation, parent-directory creation
  * copy_file: non-existent source, existing destination, directory source,
    content preservation, parent-directory creation
  * replace_python_function: non-existent file, function not found, SyntaxError
    file, decorator preservation, async functions, empty new_code, dotted
    Class.method path, nested Outer.Inner.method path, indentation adjustment

Pattern: monkeypatch tools.file_ops.snapshot to a no-op and replace
memory_manager.memory (and the already-imported tools.file_ops.memory) with a
FakeMemory so no real I/O side-effects leak into the test session.
"""

import sys
import textwrap
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import tools.file_ops
import memory_manager
from tools.file_ops import move_file, copy_file, replace_python_function


# --------------------------------------------------------------------------- #
# Fakes & isolation fixture
# --------------------------------------------------------------------------- #
class FakeMemory:
    """Minimal stand-in for the real MemoryManager — swallows all calls."""

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
    """Prevent snapshot I/O, memory persistence, path restrictions, and
    rag-engine indexing during tests."""
    monkeypatch.setattr(tools.file_ops, "snapshot", lambda *a, **kw: None)

    fake = FakeMemory()
    monkeypatch.setattr(memory_manager, "memory", fake)
    monkeypatch.setattr(tools.file_ops, "memory", fake)

    # No syntax validation errors (replace_python_function validates after write)
    monkeypatch.setattr(tools.file_ops, "_validate_code_syntax", lambda fp: None)

    # No path restrictions
    monkeypatch.setattr(tools.file_ops, "_is_plugin_path_restricted", lambda fp: None)
    monkeypatch.setattr(tools.file_ops, "_is_unity_meta_restricted", lambda fp: None)

    # Mock rag_engine to avoid indexing side-effects
    mock_rag = MagicMock()
    mock_rag.update_file_index = lambda fp: None
    mock_rag.remove_file_index = lambda fp: None
    monkeypatch.setitem(sys.modules, "rag_engine", mock_rag)

    yield


# ══════════════════════════════════════════════════════════════════════════════
# move_file — LMTrust blind spots
# ══════════════════════════════════════════════════════════════════════════════
class TestMoveFileBlindSpots:
    """LMTrust Layer: Move/Rename Safety & Content Integrity"""

    def test_move_nonexistent_source_errors(self, tmp_path):
        """move_file on a source that does not exist should return an error."""
        src = tmp_path / "ghost.txt"
        dst = tmp_path / "dest.txt"
        result = move_file(str(src), str(dst))
        assert "Error" in result
        assert "does not exist" in result.lower() or "not exist" in result.lower()
        assert not dst.exists()

    def test_move_destination_already_exists_errors(self, tmp_path):
        """move_file should refuse to overwrite an existing destination."""
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("source content", encoding="utf-8")
        dst.write_text("dest content", encoding="utf-8")

        result = move_file(str(src), str(dst))
        assert "Error" in result
        assert "already exists" in result.lower()
        # Both files should be intact
        assert src.exists()
        assert dst.read_text(encoding="utf-8") == "dest content"

    def test_move_source_is_directory_errors(self, tmp_path):
        """move_file on a directory source should error (is_file check)."""
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        dst = tmp_path / "dest.txt"

        result = move_file(str(src_dir), str(dst))
        assert "Error" in result
        assert "not a file" in result.lower()
        assert src_dir.exists()  # directory should still be there
        assert not dst.exists()

    def test_move_preserves_content_exactly(self, tmp_path):
        """move_file should move the file with byte-exact content."""
        src = tmp_path / "original.txt"
        dst = tmp_path / "moved.txt"
        content = "line1\nline2\nline3\nspecial: ünïcödé\n"
        src.write_text(content, encoding="utf-8")

        result = move_file(str(src), str(dst))
        assert "Successfully moved" in result
        assert not src.exists()
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == content

    def test_move_creates_parent_dirs_of_destination(self, tmp_path):
        """move_file should create parent directories of the destination if missing."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "a" / "b" / "c" / "destination.txt"
        src.write_text("content to move", encoding="utf-8")

        result = move_file(str(src), str(dst))
        assert "Successfully moved" in result
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "content to move"
        assert not src.exists()


# ══════════════════════════════════════════════════════════════════════════════
# copy_file — LMTrust blind spots
# ══════════════════════════════════════════════════════════════════════════════
class TestCopyFileBlindSpots:
    """LMTrust Layer: Copy Safety & Content Integrity"""

    def test_copy_nonexistent_source_errors(self, tmp_path):
        """copy_file on a source that does not exist should return an error."""
        src = tmp_path / "ghost.txt"
        dst = tmp_path / "dest.txt"
        result = copy_file(str(src), str(dst))
        assert "Error" in result
        assert "does not exist" in result.lower() or "not exist" in result.lower()
        assert not dst.exists()

    def test_copy_destination_already_exists_errors(self, tmp_path):
        """copy_file should refuse to overwrite an existing destination."""
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("source content", encoding="utf-8")
        dst.write_text("dest content", encoding="utf-8")

        result = copy_file(str(src), str(dst))
        assert "Error" in result
        assert "already exists" in result.lower()
        # Source should be untouched, destination unchanged
        assert src.read_text(encoding="utf-8") == "source content"
        assert dst.read_text(encoding="utf-8") == "dest content"

    def test_copy_source_is_directory_errors(self, tmp_path):
        """copy_file on a directory source should error (is_file check)."""
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        dst = tmp_path / "dest.txt"

        result = copy_file(str(src_dir), str(dst))
        assert "Error" in result
        assert "not a file" in result.lower()
        assert src_dir.exists()
        assert not dst.exists()

    def test_copy_preserves_content_exactly(self, tmp_path):
        """copy_file should copy the file with byte-exact content."""
        src = tmp_path / "original.txt"
        dst = tmp_path / "copied.txt"
        content = "line1\nline2\nline3\nspecial: ünïcödé\n"
        src.write_text(content, encoding="utf-8")

        result = copy_file(str(src), str(dst))
        assert "Successfully copied" in result
        # Source should still exist (it's a copy, not a move)
        assert src.exists()
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == content
        assert src.read_text(encoding="utf-8") == content

    def test_copy_creates_parent_dirs_of_destination(self, tmp_path):
        """copy_file should create parent directories of the destination if missing."""
        src = tmp_path / "source.txt"
        dst = tmp_path / "x" / "y" / "z" / "destination.txt"
        src.write_text("content to copy", encoding="utf-8")

        result = copy_file(str(src), str(dst))
        assert "Successfully copied" in result
        assert dst.exists()
        assert dst.read_text(encoding="utf-8") == "content to copy"
        assert src.exists()  # source still present


# ══════════════════════════════════════════════════════════════════════════════
# replace_python_function — LMTrust blind spots
# ══════════════════════════════════════════════════════════════════════════════
class TestReplacePythonFunctionBlindSpots:
    """LMTrust Layer: Surgical Function Replacement Safety"""

    def test_replace_in_nonexistent_file_errors(self, tmp_path):
        """replace_python_function on a file that doesn't exist should error."""
        target = tmp_path / "missing.py"
        result = replace_python_function(str(target), "my_func", "def my_func():\n    pass\n")
        assert "Error" in result
        assert "does not exist" in result.lower() or "not exist" in result.lower()

    def test_replace_function_not_found_errors(self, tmp_path):
        """replace_python_function with a function name not in the file should error."""
        target = tmp_path / "mod.py"
        target.write_text("def existing_func():\n    return 1\n", encoding="utf-8")

        result = replace_python_function(str(target), "nonexistent_func",
                                         "def nonexistent_func():\n    return 2\n")
        assert "Error" in result
        assert "not found" in result.lower()
        # File should be unchanged
        assert target.read_text(encoding="utf-8") == "def existing_func():\n    return 1\n"

    def test_replace_in_file_with_syntax_error_errors(self, tmp_path):
        """replace_python_function on a file with a SyntaxError should error."""
        target = tmp_path / "broken.py"
        # Deliberately invalid Python (unmatched paren)
        target.write_text("def good_func():\n    return (1\n", encoding="utf-8")

        result = replace_python_function(str(target), "good_func",
                                         "def good_func():\n    return 1\n")
        assert "Error" in result
        assert "syntax" in result.lower()

    def test_replace_preserves_decorators(self, tmp_path):
        """replace_python_function should include decorators in the replaced region."""
        target = tmp_path / "decorated.py"
        target.write_text(textwrap.dedent("""\
            def my_decorator(func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper

            @my_decorator
            def decorated_func():
                return 42

            def other_func():
                return 0
            """), encoding="utf-8")

        new_body = textwrap.dedent("""\
            @my_decorator
            def decorated_func():
                return 99
            """)

        result = replace_python_function(str(target), "decorated_func", new_body)
        assert "Successfully replaced" in result
        content = target.read_text(encoding="utf-8")
        # The old decorator should NOT appear twice (old function fully removed)
        assert content.count("@my_decorator") == 1
        # The new body should be present
        assert "return 99" in content
        # other_func should be untouched
        assert "def other_func" in content

    def test_replace_async_function(self, tmp_path):
        """replace_python_function should handle async functions."""
        target = tmp_path / "async_mod.py"
        target.write_text(textwrap.dedent("""\
            async def fetch_data():
                return "old"

            def sync_func():
                return 1
            """), encoding="utf-8")

        new_body = textwrap.dedent("""\
            async def fetch_data():
                return "new"
            """)

        result = replace_python_function(str(target), "fetch_data", new_body)
        assert "Successfully replaced" in result
        content = target.read_text(encoding="utf-8")
        assert 'return "new"' in content
        assert 'return "old"' not in content
        assert "def sync_func" in content

    def test_replace_with_empty_new_code(self, tmp_path):
        """replace_python_function with empty new_code should remove the function."""
        target = tmp_path / "removable.py"
        target.write_text(textwrap.dedent("""\
            def to_remove():
                return 1

            def keep():
                return 2
            """), encoding="utf-8")

        result = replace_python_function(str(target), "to_remove", "")
        assert "Successfully replaced" in result
        content = target.read_text(encoding="utf-8")
        assert "def to_remove" not in content
        assert "def keep" in content

    def test_replace_dotted_class_method(self, tmp_path):
        """replace_python_function with Class.method dotted path should work."""
        target = tmp_path / "cls_mod.py"
        target.write_text(textwrap.dedent("""\
            class MyClass:
                def my_method(self):
                    return "old"

            def standalone():
                return 0
            """), encoding="utf-8")

        new_body = textwrap.dedent("""\
            def my_method(self):
                return "new"
            """)

        result = replace_python_function(str(target), "MyClass.my_method", new_body)
        assert "Successfully replaced" in result
        content = target.read_text(encoding="utf-8")
        assert 'return "new"' in content
        assert 'return "old"' not in content
        assert "class MyClass" in content
        assert "def standalone" in content

    def test_replace_nested_class_method(self, tmp_path):
        """replace_python_function with Outer.Inner.method nested path should work."""
        target = tmp_path / "nested.py"
        target.write_text(textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        return "old"

            def top_level():
                return 0
            """), encoding="utf-8")

        new_body = textwrap.dedent("""\
            def method(self):
                return "new"
            """)

        result = replace_python_function(str(target), "Outer.Inner.method", new_body)
        assert "Successfully replaced" in result
        content = target.read_text(encoding="utf-8")
        assert 'return "new"' in content
        assert 'return "old"' not in content
        assert "class Outer" in content
        assert "class Inner" in content
        assert "def top_level" in content

    def test_replace_adjusts_indentation_correctly(self, tmp_path):
        """replace_python_function should re-indent the new code to match the
        original function's indentation level."""
        target = tmp_path / "indent_mod.py"
        target.write_text(textwrap.dedent("""\
            class Container:
                def indented_method(self):
                    return "old"

            def top_func():
                return 0
            """), encoding="utf-8")

        # New code provided at column 0 (no leading indentation) — the tool
        # should shift it to 8 spaces (two levels) to sit inside the class.
        new_body = "def indented_method(self):\n    return 'new'\n"

        result = replace_python_function(str(target), "Container.indented_method", new_body)
        assert "Successfully replaced" in result
        content = target.read_text(encoding="utf-8")

        # The new method should be indented at 4 spaces (one level inside the
        # class).  The new_code was given at column 0, so the tool must have
        # shifted it to match the original method's base indentation.
        lines = content.splitlines()
        method_line_idx = None
        for i, line in enumerate(lines):
            if "def indented_method" in line:
                method_line_idx = i
                break
        assert method_line_idx is not None
        # Method def at 4 spaces (inside class)
        assert lines[method_line_idx].startswith("    def indented_method")
        # Return statement at 8 spaces (inside method, inside class)
        assert lines[method_line_idx + 1].startswith("        return")
        assert "return 'new'" in content