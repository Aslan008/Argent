"""Blind-spot tests for atomic_io.py.

Covers crash-safety, encoding, unicode paths, and the binary writer callback —
paths that a happy-path smoke test would miss.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from atomic_io import atomic_write_bytes, atomic_write_text


# ---------------------------------------------------------------------------
# L0 — Smoke
# ---------------------------------------------------------------------------
class TestSmoke:
    """Minimal sanity checks: the import works and a basic call succeeds."""

    def test_atomic_write_bytes_import_and_basic_call(self, tmp_path):
        """atomic_write_bytes is importable and writes bytes via a callback."""
        target = tmp_path / "smoke.bin"
        atomic_write_bytes(target, lambda f: f.write(b"hello"))
        assert target.read_bytes() == b"hello"


# ---------------------------------------------------------------------------
# L1 — atomic_write_bytes behaviour
# ---------------------------------------------------------------------------
class TestAtomicWriteBytes:
    """Verify the binary writer produces correct files and side-effects."""

    def test_writes_correct_binary_content(self, tmp_path):
        target = tmp_path / "data.bin"
        payload = bytes(range(256))
        atomic_write_bytes(target, lambda f: f.write(payload))
        assert target.read_bytes() == payload

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "dir" / "out.bin"
        atomic_write_bytes(target, lambda f: f.write(b"\x00\x01\x02"))
        assert target.exists()
        assert target.read_bytes() == b"\x00\x01\x02"

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "replace.bin"
        target.write_bytes(b"old-content")
        atomic_write_bytes(target, lambda f: f.write(b"new-content"))
        assert target.read_bytes() == b"new-content"


# ---------------------------------------------------------------------------
# L4×D4 — Writer callback crashes mid-write
# ---------------------------------------------------------------------------
class TestWriterCrash:
    """If the writer raises, the temp file must be cleaned up and the
    original left untouched."""

    def test_temp_cleaned_and_original_intact_on_writer_crash(self, tmp_path):
        target = tmp_path / "crash.bin"
        target.write_text("original")

        def bad_writer(f):
            f.write(b"partial")
            raise ValueError("simulated writer failure")

        with pytest.raises(ValueError, match="simulated writer failure"):
            atomic_write_bytes(target, bad_writer)

        # No leftover .tmp files in the directory
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"

        # Original content untouched
        assert target.read_text() == "original"


# ---------------------------------------------------------------------------
# L4×D5 — fsync fails (disk error / broken handle)
# ---------------------------------------------------------------------------
class TestFsyncFailure:
    """If os.fsync raises, cleanup must still happen and the original survive."""

    def test_temp_cleaned_and_original_intact_on_fsync_failure(self, tmp_path):
        target = tmp_path / "fsync.bin"
        target.write_text("original")

        with patch("os.fsync", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                atomic_write_bytes(target, lambda f: f.write(b"new"))

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"

        assert target.read_text() == "original"


# ---------------------------------------------------------------------------
# L2×D1 — Empty content edge case
# ---------------------------------------------------------------------------
class TestEmptyContent:
    """atomic_write_text with an empty string must still create the file."""

    def test_empty_string_creates_empty_file(self, tmp_path):
        target = tmp_path / "empty.txt"
        atomic_write_text(target, "")
        assert target.exists()
        assert target.read_text() == ""
        assert target.stat().st_size == 0


# ---------------------------------------------------------------------------
# L2×D3 — Unicode path
# ---------------------------------------------------------------------------
class TestUnicodePath:
    """Paths with non-ASCII characters must be handled correctly."""

    def test_unicode_path_roundtrip(self, tmp_path):
        target = tmp_path / "файл_данные.json"
        payload = '{"ключ": "значение"}'
        atomic_write_text(target, payload)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == payload


# ---------------------------------------------------------------------------
# L1×D7 — Encoding parameter
# ---------------------------------------------------------------------------
class TestEncodingParameter:
    """atomic_write_text must honour the encoding argument."""

    def test_cp1251_encoding_roundtrip(self, tmp_path):
        target = tmp_path / "cyrillic.txt"
        text = "Привет, мир!"
        atomic_write_text(target, text, encoding="cp1251")
        # Read back with the same encoding to verify
        assert target.read_text(encoding="cp1251") == text
        # And confirm the raw bytes are cp1251-encoded, not utf-8
        assert target.read_bytes() == text.encode("cp1251")