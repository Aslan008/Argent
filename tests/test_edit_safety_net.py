"""Safety net: a broken edit must never be left on disk.

write_file and replace_python_function now auto-revert when the result would
not compile — matching the behaviour replace_in_file / multi_replace_in_file_chunk
already had.
"""

import pytest

from tools.file_ops import write_file, replace_python_function

GOOD = "def foo():\n    return 1\n"
BROKEN = "def foo(:\n    return 1\n"  # SyntaxError py_compile reliably catches


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # Resolve relative paths and write memory/snapshots under tmp, not the repo/home.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("file_tracker.HISTORY_DIR", tmp_path / "_hist", raising=False)


class TestWriteFileSafetyNet:
    def test_new_broken_file_is_not_created(self, tmp_path):
        target = tmp_path / "broken_new.py"
        result = write_file(str(target), BROKEN)
        assert "REJECTED" in result
        assert not target.exists()

    def test_overwrite_broken_restores_previous(self, tmp_path):
        target = tmp_path / "keep.py"
        assert "Successfully" in write_file(str(target), GOOD)
        result = write_file(str(target), BROKEN, overwrite=True)
        assert "REJECTED" in result
        assert target.read_text(encoding="utf-8") == GOOD

    def test_valid_write_succeeds(self, tmp_path):
        target = tmp_path / "ok.py"
        assert "Successfully" in write_file(str(target), GOOD)
        assert target.read_text(encoding="utf-8") == GOOD

    def test_non_python_is_not_gated(self, tmp_path):
        # The validator only checks .py / .cs — odd text in a .txt must still write.
        target = tmp_path / "notes.txt"
        assert "Successfully" in write_file(str(target), "def foo(:")
        assert target.exists()


class TestReplacePythonFunctionSafetyNet:
    def test_broken_replacement_reverts(self, tmp_path):
        target = tmp_path / "mod.py"
        target.write_text(GOOD, encoding="utf-8")
        result = replace_python_function(str(target), "foo", "def foo(:\n    pass")
        assert "REJECTED" in result
        assert target.read_text(encoding="utf-8") == GOOD

    def test_valid_replacement_applies(self, tmp_path):
        target = tmp_path / "mod2.py"
        target.write_text(GOOD, encoding="utf-8")
        result = replace_python_function(str(target), "foo", "def foo():\n    return 2")
        assert "Successfully" in result
        assert "return 2" in target.read_text(encoding="utf-8")
