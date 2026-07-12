"""File snapshot tracker: path-length-safe keys, rotation, undo, undo_all."""

import pytest

import file_tracker
from file_tracker import (
    get_diff, get_pending_changes, snapshot, undo, undo_all,
)


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tracker, "HISTORY_DIR", tmp_path / "hist")
    yield


def test_snapshot_and_diff(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert snapshot(str(f)) is True
    f.write_text("x = 2\n", encoding="utf-8")

    diff = get_diff(str(f))
    assert "-x = 1" in diff and "+x = 2" in diff


def test_undo_restores_latest(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("original\n", encoding="utf-8")
    snapshot(str(f))
    f.write_text("broken\n", encoding="utf-8")

    assert "Restored" in undo(str(f))
    assert f.read_text(encoding="utf-8") == "original\n"


def test_missing_file_snapshot_is_false(tmp_path):
    assert snapshot(str(tmp_path / "nope.py")) is False


class TestPathLengthSafety:
    def test_deeply_nested_path_snapshots_ok(self, tmp_path):
        # A path long enough to blow the old whole-path-as-filename scheme.
        deep = tmp_path
        for i in range(15):
            deep = deep / f"very_long_directory_segment_number_{i}_padding"
        deep.mkdir(parents=True)
        f = deep / "DeeplyNestedUnityScript.cs"
        f.write_text("// unity\n", encoding="utf-8")

        assert snapshot(str(f)) is True
        f.write_text("// edited\n", encoding="utf-8")
        assert "Restored" in undo(str(f))
        assert f.read_text(encoding="utf-8") == "// unity\n"

    def test_snapshot_filenames_are_bounded(self, tmp_path):
        deep = tmp_path / ("d" * 200) / ("e" * 200)
        deep.mkdir(parents=True)
        f = deep / "file.cs"
        f.write_text("x\n", encoding="utf-8")
        snapshot(str(f))

        session = file_tracker._session_dir()
        names = [p.name for p in session.iterdir()]
        assert names and all(len(n) < 100 for n in names)   # no giant filenames


class TestRotation:
    def test_keeps_only_max_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_tracker, "MAX_SNAPSHOTS_PER_FILE", 3)
        f = tmp_path / "a.py"
        for i in range(8):
            f.write_text(f"v{i}\n", encoding="utf-8")
            snapshot(str(f))

        session = file_tracker._session_dir()
        key = file_tracker._key(f.resolve())
        snaps = list(session.glob(f"{key}__*"))
        assert len(snaps) == 3     # older ones rotated out


class TestListingAndUndoAll:
    def test_pending_changes_shows_real_path(self, tmp_path):
        f = tmp_path / "sub" / "game.py"
        f.parent.mkdir()
        f.write_text("a\n", encoding="utf-8")
        snapshot(str(f))

        pending = get_pending_changes()
        assert len(pending) == 1
        assert pending[0]["path"] == str(f.resolve())
        assert pending[0]["snapshot_count"] == 1

    def test_undo_all_restores_to_original_location(self, tmp_path):
        f1 = tmp_path / "one.py"
        f2 = tmp_path / "deep" / "two.py"
        f2.parent.mkdir()
        for f in (f1, f2):
            f.write_text("good\n", encoding="utf-8")
            snapshot(str(f))
            f.write_text("bad\n", encoding="utf-8")

        out = undo_all()
        assert "Restored" in out
        assert f1.read_text(encoding="utf-8") == "good\n"
        assert f2.read_text(encoding="utf-8") == "good\n"

    def test_undo_all_empty_is_explicit(self):
        assert undo_all() == "No pending changes to undo."
