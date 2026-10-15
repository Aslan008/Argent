"""
LMTrust deep verification suite for file operations audit.
Verifies all 14 defect fixes, universal handling, and regressions prevention.
"""

import json
import os
import stat
from pathlib import Path
import pytest

from tools.file_ops import (
    read_file,
    write_file,
    replace_in_file,
    multi_replace_in_file,
    move_file,
    copy_file,
    delete_file,
    list_directory,
)
from tools._helpers import (
    _validate_code_syntax,
    _maybe_unescape_content,
    _strip_read_line_numbers,
)
import file_tracker
from tools.schemas import AVAILABLE_TOOLS


class TestFuzzyFallbackNoDuplicates:
    def test_fuzzy_fallback_with_leading_blank_lines(self, tmp_path, monkeypatch):
        """DEFECT-FILE-01: Fuzzy fallback must not false-alarm duplicate matches
        when target text starts with empty lines matching empty lines in content."""
        monkeypatch.chdir(tmp_path)
        content = (
            "import os\n\n\n"
            "def calculate(a, b):\n"
            "    return a + b\n\n\n"
            "def main():\n"
            "    pass\n"
        )
        p = tmp_path / "calc.py"
        p.write_bytes(content.encode("utf-8"))

        # Target with leading blank lines and slightly adjusted indentation in body
        target = "\n\ndef calculate(a, b):\n   return a + b"
        replacement = "def calculate(a, b):\n    return a * b"

        result = replace_in_file("calc.py", target, replacement)
        assert "Successfully replaced" in result
        assert "Error: Found" not in result
        updated = p.read_text(encoding="utf-8")
        assert "return a * b" in updated


class TestEolPreservationWindows:
    def test_write_file_preserves_pure_lf_on_windows(self, tmp_path, monkeypatch):
        """DEFECT-FILE-02: write_file with newline='' must preserve pure LF without CRLF injection."""
        monkeypatch.chdir(tmp_path)
        content = "line 1\nline 2\nline 3\n"
        res = write_file("pure_lf.txt", content)
        assert "Successfully wrote" in res
        raw = (tmp_path / "pure_lf.txt").read_bytes()
        assert b"\r\n" not in raw
        assert raw == content.encode("utf-8")

    def test_replace_in_file_preserves_pure_lf(self, tmp_path, monkeypatch):
        """DEFECT-FILE-02: replace_in_file on pure LF file must not introduce CRLF."""
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "lf_code.py"
        p.write_bytes(b"x = 1\ny = 2\nz = 3\n")

        res = replace_in_file("lf_code.py", "y = 2", "y = 20")
        assert "Successfully replaced" in res
        raw = p.read_bytes()
        assert b"\r\n" not in raw
        assert raw == b"x = 1\ny = 20\nz = 3\n"

    def test_replace_in_file_preserves_crlf(self, tmp_path, monkeypatch):
        """DEFECT-FILE-02: replace_in_file on CRLF file keeps CRLF line endings."""
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "crlf_code.py"
        p.write_bytes(b"x = 1\r\ny = 2\r\nz = 3\r\n")

        # LLM usually provides target and replacement with \n
        res = replace_in_file("crlf_code.py", "y = 2", "y = 20")
        assert "Successfully replaced" in res
        raw = p.read_bytes()
        assert b"y = 20\r\n" in raw
        assert b"\r\n" in raw


class TestMoveAndCopyDirectoryTarget:
    def test_move_file_into_existing_directory(self, tmp_path, monkeypatch):
        """DEFECT-FILE-03: move_file into directory path retains original file name."""
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "hello.txt"
        src.write_text("content", encoding="utf-8")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        res = move_file("hello.txt", "dest")
        assert "Successfully moved" in res
        assert not src.exists()
        assert (dest_dir / "hello.txt").exists()
        assert (dest_dir / "hello.txt").read_text(encoding="utf-8") == "content"

    def test_copy_file_into_existing_directory(self, tmp_path, monkeypatch):
        """DEFECT-FILE-03: copy_file into directory path retains original file name."""
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "hello.txt"
        src.write_text("content", encoding="utf-8")
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        res = copy_file("hello.txt", "dest")
        assert "Successfully copied" in res
        assert src.exists()
        assert (dest_dir / "hello.txt").exists()
        assert (dest_dir / "hello.txt").read_text(encoding="utf-8") == "content"


class TestBinaryFileDetection:
    def test_read_file_binary_detection(self, tmp_path, monkeypatch):
        """DEFECT-FILE-04: read_file detects null bytes and returns clean error instead of garbage."""
        monkeypatch.chdir(tmp_path)
        bin_file = tmp_path / "binary.dat"
        bin_file.write_bytes(b"GIF89a\x00\x01\x02\x03fake-image-bytes")

        res = read_file("binary.dat")
        assert "file appears to be binary" in res
        assert "binary.dat" in res


class TestJsonWithBomValidation:
    def test_validate_code_syntax_json_with_utf8_bom(self, tmp_path):
        """DEFECT-FILE-05: _validate_code_syntax parses JSON files with UTF-8 BOM."""
        p = tmp_path / "config.json"
        p.write_bytes(b"\xef\xbb\xbf{\"status\": \"ok\", \"count\": 42}")

        err = _validate_code_syntax(str(p))
        assert err is None


class TestUnescapeContentApostrophes:
    def test_maybe_unescape_content_preserves_apostrophes(self):
        """DEFECT-FILE-06: Apostrophes in comments must not break unescaping of newlines."""
        raw = "def foo():\\n    # don't remove this comment\\n    return 42"
        unescaped = _maybe_unescape_content(raw)
        assert unescaped == "def foo():\n    # don't remove this comment\n    return 42"


class TestStripReadLineNumbers:
    def test_strip_read_line_numbers_consecutive(self):
        """DEFECT-FILE-07: Consecutive lines with gutter numbers and tabs are stripped cleanly."""
        text = "    1\tprint('hello')\n    2\tprint('world')"
        stripped = _strip_read_line_numbers(text)
        assert stripped == "print('hello')\nprint('world')"


class TestFileTrackerCaseInsensitiveKey:
    def test_file_tracker_normcase_key(self, tmp_path):
        """DEFECT-FILE-08: Path with varying drive letter cases produces identical keys on Windows."""
        p = tmp_path / "track_test.txt"
        p.write_text("v1", encoding="utf-8")

        key1 = file_tracker._key(p)
        p_str = str(p)
        if len(p_str) > 1 and p_str[1] == ":":
            flipped_drive = p_str[0].lower() if p_str[0].isupper() else p_str[0].upper()
            p_flipped = Path(flipped_drive + p_str[1:])
            key2 = file_tracker._key(p_flipped)
            assert key1 == key2


class TestMultiReplacePolymorphism:
    def test_multi_replace_accepts_list_directly(self, tmp_path, monkeypatch):
        """DEFECT-FILE-09: multi_replace_in_file accepts Python list directly."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.txt").write_text("aaa\n", encoding="utf-8")
        changes = [
            {"file_path": "a.txt", "target_text": "aaa", "replacement_text": "AAA"}
        ]
        out = multi_replace_in_file(changes)
        assert out.startswith("Successfully")
        assert "Multi-replace execution finished" in out
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "AAA\n"

    def test_multi_replace_accepts_dict_with_changes(self, tmp_path, monkeypatch):
        """DEFECT-FILE-09: multi_replace_in_file accepts dict with 'changes' key."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "b.txt").write_text("bbb\n", encoding="utf-8")
        payload = {
            "changes": [
                {"file_path": "b.txt", "target_text": "bbb", "replacement_text": "BBB"}
            ]
        }
        out = multi_replace_in_file(payload)
        assert out.startswith("Successfully")
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "BBB\n"


class TestUnityMetaGuards:
    def test_delete_file_blocks_unity_meta(self, tmp_path, monkeypatch):
        """DEFECT-FILE-10: delete_file blocks deleting Unity .meta files."""
        monkeypatch.chdir(tmp_path)
        res = delete_file("Assets/Scripts/Player.cs.meta")
        assert "Error" in res
        assert ".meta" in res

    def test_move_file_blocks_unity_meta(self, tmp_path, monkeypatch):
        """DEFECT-FILE-10: move_file blocks moving Unity .meta files."""
        monkeypatch.chdir(tmp_path)
        res = move_file("Assets/Scripts/Player.cs.meta", "dest.meta")
        assert "Error" in res
        assert ".meta" in res

    def test_copy_file_blocks_unity_meta(self, tmp_path, monkeypatch):
        """DEFECT-FILE-10: copy_file blocks copying Unity .meta files."""
        monkeypatch.chdir(tmp_path)
        res = copy_file("Assets/Scripts/Player.cs.meta", "dest.meta")
        assert "Error" in res
        assert ".meta" in res


class TestDeleteReadOnlyFile:
    def test_delete_readonly_file_windows(self, tmp_path, monkeypatch):
        """DEFECT-FILE-11: delete_file clears Read-Only attribute on Windows and deletes successfully."""
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "readonly.txt"
        p.write_text("readonly content", encoding="utf-8")
        os.chmod(p, stat.S_IREAD)  # Make read-only

        monkeypatch.setattr("approval.request_approval", lambda *args, **kwargs: True)
        res = delete_file("readonly.txt")
        assert "Successfully deleted" in res
        assert not p.exists()


class TestListDirectoryBounded:
    def test_list_directory_caps_at_200_entries(self, tmp_path, monkeypatch):
        """DEFECT-FILE-12: list_directory caps at 200 entries and warns user without crashing."""
        monkeypatch.chdir(tmp_path)
        big_dir = tmp_path / "huge"
        big_dir.mkdir()
        for i in range(250):
            (big_dir / f"file_{i:03d}.txt").write_text("x", encoding="utf-8")

        res = list_directory("huge")
        assert "Showing first 200" in res
        assert "and 50 more items" in res


class TestAgentDropReadCacheForMultiReplace:
    def test_agent_drops_cache_for_multi_replace(self):
        """DEFECT-FILE-13: agent._drop_read_cache handles polymorphic changes_json."""
        from agent import ArgentAgent
        from tools._helpers import _resolve_path

        p1 = str(_resolve_path("some/path/file1.py"))
        p2 = str(_resolve_path("some/path/file2.py"))

        class DummyAgent:
            _read_cache = {p1: "sig1", p2: "sig2"}
            _drop_read_cache = ArgentAgent._drop_read_cache

        dummy = DummyAgent()
        changes = [
            {"file_path": "some/path/file1.py", "target_text": "a", "replacement_text": "b"}
        ]
        dummy._drop_read_cache({"changes_json": changes})
        assert p1 not in dummy._read_cache
        assert p2 in dummy._read_cache


class TestToolRegistryCanonicalCleanliness:
    def test_deprecated_tools_not_in_available_tools(self):
        """DEFECT-FILE-14: append_to_file, multi_replace_in_file_chunk, and multi_replace_in_file must be absent from AVAILABLE_TOOLS."""
        assert "append_to_file" not in AVAILABLE_TOOLS
        assert "multi_replace_in_file_chunk" not in AVAILABLE_TOOLS
        assert "multi_replace_in_file" not in AVAILABLE_TOOLS
        assert "write_file" in AVAILABLE_TOOLS
        assert "replace_in_file" in AVAILABLE_TOOLS

