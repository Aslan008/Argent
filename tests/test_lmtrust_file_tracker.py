"""Blind-spot tests for file_tracker — edge cases not covered by test_file_tracker."""

import pytest
from pathlib import Path

import file_tracker
from file_tracker import (
    get_diff, get_pending_changes, snapshot, undo, undo_all,
)


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tracker, "HISTORY_DIR", tmp_path / "hist")
    yield


# --------------------------------------------------------------------------- #
# 1. get_diff on a file that does not exist at all
# --------------------------------------------------------------------------- #
class TestGetDiffFileNotExists:
    def test_get_diff_missing_file(self, tmp_path):
        result = get_diff(str(tmp_path / "ghost.py"))
        assert "does not exist" in result


# --------------------------------------------------------------------------- #
# 2. get_diff on a real file that was never snapshotted
# --------------------------------------------------------------------------- #
class TestGetDiffNoSnapshots:
    def test_get_diff_no_snapshots(self, tmp_path):
        f = tmp_path / "fresh.py"
        f.write_text("hello\n", encoding="utf-8")
        result = get_diff(str(f))
        assert "No previous snapshots" in result


# --------------------------------------------------------------------------- #
# 3. undo on a file that has no snapshots
# --------------------------------------------------------------------------- #
class TestUndoNoSnapshots:
    def test_undo_no_snapshots(self, tmp_path):
        f = tmp_path / "novel.py"
        f.write_text("content\n", encoding="utf-8")
        result = undo(str(f))
        assert "No snapshots found" in result


# --------------------------------------------------------------------------- #
# 4. get_diff when the file hasn't changed since the snapshot
# --------------------------------------------------------------------------- #
class TestGetDiffNoChanges:
    def test_get_diff_no_changes(self, tmp_path):
        f = tmp_path / "stable.py"
        f.write_text("same\n", encoding="utf-8")
        snapshot(str(f))
        # don't modify the file
        result = get_diff(str(f))
        assert "No changes detected" in result


# --------------------------------------------------------------------------- #
# 5. undo deletes the snapshot it restored from; second undo finds none
# --------------------------------------------------------------------------- #
class TestUndoRemovesSnapshot:
    def test_undo_then_no_snapshots(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("v1\n", encoding="utf-8")
        snapshot(str(f))
        f.write_text("v2\n", encoding="utf-8")

        first = undo(str(f))
        assert "Restored" in first
        assert f.read_text(encoding="utf-8") == "v1\n"

        # The snapshot should be gone now.
        second = undo(str(f))
        assert "No snapshots found" in second


# --------------------------------------------------------------------------- #
# 6. With multiple snapshots, undo restores the *latest* one (v2, not v1)
# --------------------------------------------------------------------------- #
class TestMultipleSnapshotsUndoLatest:
    def test_undo_restores_latest_snapshot(self, tmp_path):
        f = tmp_path / "multi.py"
        f.write_text("v1\n", encoding="utf-8")
        snapshot(str(f))

        f.write_text("v2\n", encoding="utf-8")
        snapshot(str(f))

        f.write_text("v3\n", encoding="utf-8")

        result = undo(str(f))
        assert "Restored" in result
        assert f.read_text(encoding="utf-8") == "v2\n"


# --------------------------------------------------------------------------- #
# 7. get_diff on a binary file that can't be decoded as UTF-8
# --------------------------------------------------------------------------- #
class TestGetDiffBinaryFile:
    def test_get_diff_binary_raises_handled(self, tmp_path):
        f = tmp_path / "blob.bin"
        f.write_bytes(b"\x80\xff\x00")
        snapshot(str(f))
        f.write_bytes(b"\x80\xff\x01\xfe")

        result = get_diff(str(f))
        assert "Error" in result


# --------------------------------------------------------------------------- #
# 8. undo_all where one target file was deleted — partial failure
# --------------------------------------------------------------------------- #
class TestUndoAllPartialFailure:
    def test_partial_failure_reports_both(self, tmp_path):
        import shutil as _shutil
        f1 = tmp_path / "alive.py"
        f2 = tmp_path / "sub" / "doomed.py"
        f2.parent.mkdir()
        for f in (f1, f2):
            f.write_text("good\n", encoding="utf-8")
            snapshot(str(f))
            f.write_text("bad\n", encoding="utf-8")

        # Remove the target's parent dir so copy2 cannot write the file.
        _shutil.rmtree(f2.parent)

        result = undo_all()
        assert "Restored" in result
        assert "Failed" in result
        # The surviving file should have been restored.
        assert f1.read_text(encoding="utf-8") == "good\n"


# --------------------------------------------------------------------------- #
# 9. _key returns a 16-char lowercase hex string
# --------------------------------------------------------------------------- #
class TestKeyIsHex:
    def test_key_is_hex(self):
        k = file_tracker._key(Path("/some/path"))
        assert len(k) == 16
        assert all(c in "0123456789abcdef" for c in k)


# --------------------------------------------------------------------------- #
# 10. _key is deterministic — same path → same key
# --------------------------------------------------------------------------- #
class TestKeyDeterministic:
    def test_key_deterministic(self):
        k1 = file_tracker._key(Path("/a/b.py"))
        k2 = file_tracker._key(Path("/a/b.py"))
        assert k1 == k2


# --------------------------------------------------------------------------- #
# 11. _key differs for different paths
# --------------------------------------------------------------------------- #
class TestKeyDifferent:
    def test_key_different(self):
        k1 = file_tracker._key(Path("/a/b.py"))
        k2 = file_tracker._key(Path("/a/c.py"))
        assert k1 != k2


# --------------------------------------------------------------------------- #
# 12. snapshot writes a <key>.path sidecar containing the resolved path
# --------------------------------------------------------------------------- #
class TestSnapshotCreatesPathSidecar:
    def test_path_sidecar_exists(self, tmp_path):
        f = tmp_path / "side.py"
        f.write_text("data\n", encoding="utf-8")
        snapshot(str(f))

        session = file_tracker._session_dir()
        key = file_tracker._key(f.resolve())
        sidecar = session / f"{key}.path"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").strip() == str(f.resolve())


# --------------------------------------------------------------------------- #
# 13. get_pending_changes with no snapshots returns an empty list
# --------------------------------------------------------------------------- #
class TestGetPendingChangesNoSnapshots:
    def test_empty_pending(self):
        assert get_pending_changes() == []


# --------------------------------------------------------------------------- #
# 14. get_pending_changes skips entries whose .path sidecar has no snapshots
# --------------------------------------------------------------------------- #
class TestGetPendingChangesCorruptedPath:
    def test_corrupted_path_skipped(self, tmp_path):
        f = tmp_path / "corrupt.py"
        f.write_text("x\n", encoding="utf-8")
        snapshot(str(f))

        # Delete the snapshot file(s) but keep the .path sidecar.
        session = file_tracker._session_dir()
        key = file_tracker._key(f.resolve())
        for snap in session.glob(f"{key}__*"):
            snap.unlink()

        assert get_pending_changes() == []