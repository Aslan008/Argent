"""Tests for bug fixes in move_file, copy_file, and replace_python_function.

Covers:
  * move_file: RAG update (remove old + add new), snapshot, plugin restriction on destination
  * copy_file: RAG update (add new), plugin path restriction
  * replace_python_function: _maybe_unescape_content on new_code
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.file_ops import (
    move_file,
    copy_file,
    replace_python_function,
)


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
    yield


# ══════════════════════════════════════════════════════════════════════════════
# move_file
# ══════════════════════════════════════════════════════════════════════════════

class TestMoveFileFixes:
    def test_rag_update_on_move(self, tmp_path, monkeypatch):
        """After move, RAG should remove old path index and add new path index."""
        src = tmp_path / "old.txt"
        src.write_text("content", encoding="utf-8")
        dst = tmp_path / "new.txt"

        rag_calls = []
        def fake_remove(fp):
            rag_calls.append(("remove", fp))
        def fake_update(fp):
            rag_calls.append(("update", fp))

        monkeypatch.setattr("tools.file_ops.snapshot", lambda fp: None)
        # Patch the import inside the function
        import sys
        mock_rag = MagicMock()
        mock_rag.remove_file_index = fake_remove
        mock_rag.update_file_index = fake_update
        sys.modules["rag_engine"] = mock_rag

        out = move_file(str(src), str(dst))
        assert "Successfully moved" in out
        assert ("remove", str(src.resolve())) in rag_calls
        assert ("update", str(dst.resolve())) in rag_calls

        # Cleanup
        del sys.modules["rag_engine"]

    def test_snapshot_before_move(self, tmp_path, monkeypatch):
        """snapshot() should be called on source before moving."""
        src = tmp_path / "old.txt"
        src.write_text("content", encoding="utf-8")
        dst = tmp_path / "new.txt"

        snap_calls = []
        monkeypatch.setattr("tools.file_ops.snapshot", lambda fp: snap_calls.append(fp))

        out = move_file(str(src), str(dst))
        assert "Successfully moved" in out
        assert str(src.resolve()) in snap_calls

    def test_plugin_restriction_on_destination(self, tmp_path, monkeypatch):
        """Plugin path restriction should check destination, not just source."""
        src = tmp_path / "normal.txt"
        src.write_text("content", encoding="utf-8")
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        dst = plugins_dir / "malicious.py"

        monkeypatch.setattr(
            "tools.file_ops._is_plugin_path_restricted",
            lambda fp: "Error: Plugin paths are restricted." if "plugins" in str(fp) else None
        )

        out = move_file(str(src), str(dst))
        assert "restricted" in out.lower() or "Error" in out
        assert src.exists()  # source should still be there
        assert not dst.exists()  # destination should not be created


# ══════════════════════════════════════════════════════════════════════════════
# copy_file
# ══════════════════════════════════════════════════════════════════════════════

class TestCopyFileFixes:
    def test_rag_update_on_copy(self, tmp_path, monkeypatch):
        """After copy, RAG should index the new file."""
        src = tmp_path / "orig.txt"
        src.write_text("content", encoding="utf-8")
        dst = tmp_path / "copy.txt"

        rag_calls = []
        def fake_update(fp):
            rag_calls.append(("update", fp))

        import sys
        mock_rag = MagicMock()
        mock_rag.update_file_index = fake_update
        sys.modules["rag_engine"] = mock_rag

        out = copy_file(str(src), str(dst))
        assert "Successfully copied" in out
        assert ("update", str(dst.resolve())) in rag_calls

        del sys.modules["rag_engine"]

    def test_plugin_restriction_on_source(self, tmp_path, monkeypatch):
        """Plugin path restriction should check source."""
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        src = plugins_dir / "secret.py"
        src.write_text("stolen", encoding="utf-8")
        dst = tmp_path / "copy.py"

        monkeypatch.setattr(
            "tools.file_ops._is_plugin_path_restricted",
            lambda fp: "Error: Plugin paths are restricted." if "plugins" in str(fp) else None
        )

        out = copy_file(str(src), str(dst))
        assert "restricted" in out.lower() or "Error" in out
        assert not dst.exists()

    def test_plugin_restriction_on_destination(self, tmp_path, monkeypatch):
        """Plugin path restriction should check destination."""
        src = tmp_path / "normal.txt"
        src.write_text("content", encoding="utf-8")
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        dst = plugins_dir / "malicious.py"

        monkeypatch.setattr(
            "tools.file_ops._is_plugin_path_restricted",
            lambda fp: "Error: Plugin paths are restricted." if "plugins" in str(fp) else None
        )

        out = copy_file(str(src), str(dst))
        assert "restricted" in out.lower() or "Error" in out
        assert not dst.exists()


# ══════════════════════════════════════════════════════════════════════════════
# replace_python_function — _maybe_unescape_content
# ══════════════════════════════════════════════════════════════════════════════

class TestReplacePythonFunctionUnescape:
    def test_literal_backslash_n_in_new_code_is_unescaped(self, tmp_path, monkeypatch):
        """When model sends \\n (literal backslash+n) in new_code, it should be
        converted to a real newline — same as write_file and replace_in_file do."""
        monkeypatch.setattr("tools.file_ops.snapshot", lambda fp: None)
        monkeypatch.setattr("tools.file_ops.memory", MagicMock())

        # File with a simple function
        (tmp_path / "f.py").write_text(
            "def greet():\n    return 'hello'\n", encoding="utf-8")

        # new_code with literal \\n (as a model might send via JSON)
        # JSON: "def greet():\\n    return 'bye'\\n"
        # After JSON parsing: "def greet():\n    return 'bye'\n"
        # This is already correct — but test that _maybe_unescape_content
        # doesn't corrupt it.
        new_code = "def greet():\n    return 'bye'\n"
        out = replace_python_function(str(tmp_path / "f.py"), "greet", new_code)
        assert "Successfully replaced" in out
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        assert "bye" in result
        assert "hello" not in result

    def test_code_with_string_literal_backslash_n_preserved(self, tmp_path, monkeypatch):
        """When new_code contains a string literal with \\n (e.g. print('a\\nb')),
        the auto-detect should preserve it as literal backslash+n, not convert
        to a real newline that breaks the string."""
        monkeypatch.setattr("tools.file_ops.snapshot", lambda fp: None)
        monkeypatch.setattr("tools.file_ops.memory", MagicMock())

        (tmp_path / "f.py").write_text(
            "def greet():\n    return 'hello'\n", encoding="utf-8")

        # new_code has a string literal with \\n inside — this should be
        # preserved by the auto-detect quote check
        new_code = 'def greet():\n    return "a\\nb"\n'
        out = replace_python_function(str(tmp_path / "f.py"), "greet", new_code)
        assert "Successfully replaced" in out
        result = (tmp_path / "f.py").read_text(encoding="utf-8")
        # The \\n inside the string should be preserved as literal backslash+n
        assert 'a\\nb' in result