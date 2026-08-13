"""Blind-spot tests for tools/search_ops.py.

Covers search_files, _python_grep_search, and grep_search with realistic
temporary file structures, edge cases, and monkeypatched subprocess for
the ripgrep fallback path.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.search_ops import search_files, grep_search, _python_grep_search


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------
class TestSearchFiles:
    """Tests for the search_files function."""

    def _make_standard_tree(self, tmp_path: Path) -> Path:
        """Create a standard set of files under tmp_path and return it."""
        (tmp_path / "alpha.py").write_text("import os\nprint('hello')\n", encoding="utf-8")
        (tmp_path / "beta.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "test_helper.py").write_text("import sys\n", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# Readme\nimport nothing\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("some notes here\n", encoding="utf-8")
        (tmp_path / "test_file.txt").write_text("import json\n", encoding="utf-8")
        return tmp_path

    # -- pattern filtering ------------------------------------------------
    def test_pattern_py_returns_only_py(self, tmp_path):
        self._make_standard_tree(tmp_path)
        result = search_files(str(tmp_path), pattern="*.py")
        assert "Found" in result
        assert "alpha.py" in result
        assert "beta.py" in result
        assert "test_helper.py" in result
        assert "readme.md" not in result
        assert "notes.txt" not in result
        assert "test_file.txt" not in result

    # -- name_contains filtering -----------------------------------------
    def test_name_contains_test(self, tmp_path):
        self._make_standard_tree(tmp_path)
        result = search_files(str(tmp_path), name_contains="test")
        assert "test_helper.py" in result
        assert "test_file.txt" in result
        assert "alpha.py" not in result
        assert "beta.py" not in result

    def test_name_contains_is_case_insensitive(self, tmp_path):
        (tmp_path / "MyFile.py").write_text("x=1\n", encoding="utf-8")
        result = search_files(str(tmp_path), name_contains="myfile")
        assert "MyFile.py" in result

    # -- content_contains filtering --------------------------------------
    def test_content_contains_import_returns_snippet(self, tmp_path):
        self._make_standard_tree(tmp_path)
        result = search_files(str(tmp_path), content_contains="import")
        assert "alpha.py" in result
        assert "test_helper.py" in result
        assert "readme.md" in result
        assert "test_file.txt" in result
        # snippet should be present (>> prefix)
        assert ">>" in result
        # beta.py and notes.txt do not contain 'import'
        assert "beta.py" not in result
        assert "notes.txt" not in result

    def test_content_contains_no_match(self, tmp_path):
        self._make_standard_tree(tmp_path)
        result = search_files(str(tmp_path), content_contains="zzznotfound")
        assert "No files found" in result

    # -- max_results ------------------------------------------------------
    def test_max_results_limit(self, tmp_path):
        self._make_standard_tree(tmp_path)
        result = search_files(str(tmp_path), max_results=2)
        # Should mention the limit message
        assert "Results limited to 2" in result
        # Should only have found 2 files
        assert "Found 2 file(s)" in result

    # -- non-existent directory ------------------------------------------
    def test_nonexistent_dir_returns_error(self, tmp_path):
        bogus = tmp_path / "does_not_exist"
        result = search_files(str(bogus))
        assert "Error" in result
        assert "does not exist" in result

    # -- file instead of directory ---------------------------------------
    def test_file_not_dir_returns_error(self, tmp_path):
        filepath = tmp_path / "afile.txt"
        filepath.write_text("hi", encoding="utf-8")
        result = search_files(str(filepath))
        assert "Error" in result
        assert "not a directory" in result

    # -- hidden dirs skipped ---------------------------------------------
    def test_hidden_dirs_skipped(self, tmp_path):
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "visible.py").write_text("import os\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py")
        assert "visible.py" in result
        assert "secret.py" not in result

    # -- node_modules skipped --------------------------------------------
    def test_node_modules_skipped(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "package.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("import os\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py")
        assert "main.py" in result
        assert "package.py" not in result

    # -- __pycache__ skipped ---------------------------------------------
    def test_pycache_skipped(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "cached.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("import os\n", encoding="utf-8")
        result = search_files(str(tmp_path), pattern="*.py")
        assert "app.py" in result
        assert "cached.py" not in result

    # -- binary extensions skipped for content search --------------------
    def test_binary_ext_skipped_for_content(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n import")
        (tmp_path / "app.exe").write_bytes(b"\x90\x00import\x00")
        (tmp_path / "real.py").write_text("import os\n", encoding="utf-8")
        result = search_files(str(tmp_path), content_contains="import")
        assert "real.py" in result
        assert "image.png" not in result
        assert "app.exe" not in result

    def test_binary_ext_not_skipped_for_name_search(self, tmp_path):
        """Binary files should still be found when only searching by name."""
        (tmp_path / "photo.png").write_bytes(b"\x89PNG")
        result = search_files(str(tmp_path), name_contains="photo")
        assert "photo.png" in result

    # -- empty directory --------------------------------------------------
    def test_empty_dir_returns_no_files(self, tmp_path):
        result = search_files(str(tmp_path))
        assert "No files found" in result


# ---------------------------------------------------------------------------
# _python_grep_search
# ---------------------------------------------------------------------------
class TestPythonGrepSearch:
    """Tests for the internal _python_grep_search fallback function."""

    def _make_grep_tree(self, tmp_path: Path) -> Path:
        (tmp_path / "app.py").write_text(
            "import os\n"
            "def hello():\n"
            "    print('Hello World')\n"
            "    return None\n",
            encoding="utf-8",
        )
        (tmp_path / "utils.py").write_text(
            "import sys\n"
            "VALUE = 42\n",
            encoding="utf-8",
        )
        (tmp_path / "notes.md").write_text(
            "# Notes\n"
            "Hello from markdown\n",
            encoding="utf-8",
        )
        return tmp_path

    # -- basic regex match ------------------------------------------------
    def test_basic_regex_match(self, tmp_path):
        self._make_grep_tree(tmp_path)
        result = _python_grep_search(str(tmp_path), r"import")
        assert "Found" in result
        assert "import os" in result
        assert "import sys" in result

    def test_regex_pattern_with_groups(self, tmp_path):
        self._make_grep_tree(tmp_path)
        result = _python_grep_search(str(tmp_path), r"def (\w+)")
        assert "Found" in result
        assert "def hello" in result

    # -- file_pattern filter ---------------------------------------------
    def test_file_pattern_filter_py(self, tmp_path):
        self._make_grep_tree(tmp_path)
        result = _python_grep_search(str(tmp_path), r"Hello", file_pattern="*.py")
        assert "Found" in result
        assert "Hello World" in result
        # markdown file should be excluded
        assert "markdown" not in result

    def test_file_pattern_filter_md(self, tmp_path):
        self._make_grep_tree(tmp_path)
        result = _python_grep_search(str(tmp_path), r"Hello", file_pattern="*.md")
        assert "Found" in result
        assert "markdown" in result
        assert "Hello World" not in result

    # -- invalid regex ----------------------------------------------------
    def test_invalid_regex_returns_error(self, tmp_path):
        self._make_grep_tree(tmp_path)
        result = _python_grep_search(str(tmp_path), r"[invalid(")
        assert "Error" in result
        assert "Invalid regex" in result

    # -- max_results ------------------------------------------------------
    def test_max_results_limit(self, tmp_path):
        # Create many files with matches
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text("import os\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), r"import", max_results=3)
        assert "Found 3 match(es)" in result
        assert "Results limited to 3" in result

    # -- skip_dirs excluded ----------------------------------------------
    def test_skip_dirs_excluded(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("import os\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), r"import")
        assert "main.py" in result
        assert "dep.py" not in result

    def test_pycache_excluded(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "cached.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("import os\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), r"import")
        assert "app.py" in result
        assert "cached.py" not in result

    def test_hidden_dir_excluded(self, tmp_path):
        hd = tmp_path / ".hidden"
        hd.mkdir()
        (hd / "secret.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "visible.py").write_text("import os\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), r"import")
        assert "visible.py" in result
        assert "secret.py" not in result

    # -- skip_ext excluded ------------------------------------------------
    def test_skip_ext_excluded(self, tmp_path):
        (tmp_path / "data.png").write_text("import os\n", encoding="utf-8")
        (tmp_path / "code.py").write_text("import os\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), r"import")
        assert "code.py" in result
        assert "data.png" not in result

    def test_skip_ext_exe_excluded(self, tmp_path):
        (tmp_path / "app.exe").write_text("import os\n", encoding="utf-8")
        (tmp_path / "real.py").write_text("import os\n", encoding="utf-8")
        result = _python_grep_search(str(tmp_path), r"import")
        assert "real.py" in result
        assert "app.exe" not in result

    # -- no matches -------------------------------------------------------
    def test_no_matches_returns_message(self, tmp_path):
        self._make_grep_tree(tmp_path)
        result = _python_grep_search(str(tmp_path), r"zzznomatch")
        assert "No matches found" in result

    # -- case-insensitive -------------------------------------------------
    def test_case_insensitive_search(self, tmp_path):
        (tmp_path / "mixed.py").write_text(
            "Hello World\n"
            "HELLO AGAIN\n"
            "hello there\n",
            encoding="utf-8",
        )
        result = _python_grep_search(str(tmp_path), r"hello")
        assert "Found" in result
        # All three lines should match due to re.IGNORECASE
        assert "Hello World" in result
        assert "HELLO AGAIN" in result
        assert "hello there" in result

    # -- non-existent dir -------------------------------------------------
    def test_nonexistent_dir_returns_error(self, tmp_path):
        bogus = tmp_path / "nope"
        result = _python_grep_search(str(bogus), r"test")
        assert "Error" in result
        assert "does not exist" in result

    def test_file_not_dir_returns_error(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("hi", encoding="utf-8")
        result = _python_grep_search(str(f), r"hi")
        assert "Error" in result
        assert "not a directory" in result


# ---------------------------------------------------------------------------
# grep_search
# ---------------------------------------------------------------------------
class TestGrepSearch:
    """Tests for grep_search with monkeypatched subprocess.run."""

    def _make_tree(self, tmp_path: Path) -> Path:
        (tmp_path / "app.py").write_text("import os\nprint('hello')\n", encoding="utf-8")
        (tmp_path / "utils.py").write_text("import sys\n", encoding="utf-8")
        return tmp_path

    # -- ripgrep not found -> fallback to python --------------------------
    def test_ripgrep_not_found_falls_back(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("rg not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = grep_search(str(tmp_path), r"import")
        # Should fall back to python implementation
        assert "Found" in result
        assert "via Python" in result
        assert "import os" in result

    # -- ripgrep rc=1 (no matches) ---------------------------------------
    def test_ripgrep_rc1_no_matches(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"import")
        assert "No matches found" in result

    # -- ripgrep rc=0 with output -> parsed correctly --------------------
    def test_ripgrep_rc0_parses_output(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        class FakeResult:
            returncode = 0
            stdout = f"{tmp_path / 'app.py'}:1:import os\n{tmp_path / 'utils.py'}:1:import sys\n"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"import")
        assert "Found" in result
        assert "via ripgrep" in result
        assert "import os" in result
        assert "import sys" in result

    # -- Windows drive letter parsing ------------------------------------
    def test_windows_drive_letter_parsing(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        # Simulate Windows-style ripgrep output with drive letter C:\...
        # The regex must handle the colon in the drive letter.
        win_path = f"C:\\fake\\path\\app.py"
        fake_stdout = f"{win_path}:1:import os\n"

        class FakeResult:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"import")
        assert "Found" in result
        assert "via ripgrep" in result
        assert "import os" in result
        # The file path should be parsed correctly (not split at drive colon)
        assert "C:\\fake\\path\\app.py" in result

    def test_windows_drive_letter_line_number_correct(self, tmp_path, monkeypatch):
        """Ensure the line number is parsed correctly, not the drive letter portion."""
        self._make_tree(tmp_path)

        win_path = "C:\\Users\\test\\file.py"
        fake_stdout = f"{win_path}:42:some code here\n"

        class FakeResult:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"code")
        assert "42: some code here" in result
        assert win_path in result

    # -- ripgrep rc=0 but empty stdout -----------------------------------
    def test_ripgrep_rc0_empty_stdout(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        class FakeResult:
            returncode = 0
            stdout = "   \n  \n"  # whitespace only
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"import")
        assert "No matches found" in result

    # -- ripgrep other error code -> fallback ----------------------------
    def test_ripgrep_error_code_falls_back(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        class FakeResult:
            returncode = 2
            stdout = ""
            stderr = "some rg error"

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"import")
        # Should fall back to python
        assert "via Python" in result
        assert "import os" in result

    # -- non-existent dir -------------------------------------------------
    def test_nonexistent_dir_returns_error(self, tmp_path, monkeypatch):
        # This should return error before even calling subprocess
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            raise FileNotFoundError("should not reach here")

        monkeypatch.setattr(subprocess, "run", fake_run)
        bogus = tmp_path / "nonexistent"
        result = grep_search(str(bogus), r"test")
        assert "Error" in result
        assert "does not exist" in result
        # subprocess should not have been called
        assert call_count[0] == 0

    def test_file_not_dir_returns_error(self, tmp_path, monkeypatch):
        f = tmp_path / "afile.txt"
        f.write_text("hi", encoding="utf-8")

        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            raise FileNotFoundError("should not reach here")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = grep_search(str(f), r"hi")
        assert "Error" in result
        assert "not a directory" in result
        assert call_count[0] == 0

    # -- file_pattern passed to ripgrep ----------------------------------
    def test_file_pattern_in_command(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)
        captured_cmd = []

        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = ""

        def fake_run(cmd, *args, **kwargs):
            captured_cmd.extend(cmd)
            return FakeResult()

        monkeypatch.setattr(subprocess, "run", fake_run)
        grep_search(str(tmp_path), r"import", file_pattern="*.py")
        # The -g flag with the file pattern should be in the command
        assert "-g" in captured_cmd
        assert "*.py" in captured_cmd

    # -- max_results respected in ripgrep parsing ------------------------
    def test_max_results_in_ripgrep_output(self, tmp_path, monkeypatch):
        self._make_tree(tmp_path)

        # Generate more lines than max_results
        lines = []
        for i in range(10):
            lines.append(f"{tmp_path / 'app.py'}:{i+1}:import os")
        fake_stdout = "\n".join(lines) + "\n"

        class FakeResult:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())
        result = grep_search(str(tmp_path), r"import", max_results=3)
        assert "Found 3 match(es)" in result