"""Atomic writes: config/project state survives a crash or a racing writer."""

import json
import os
from unittest.mock import patch

import pytest

from atomic_io import atomic_write_text


def test_writes_content(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_text(target, '{"a": 1}')
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


def test_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "state.json"
    atomic_write_text(target, "hi")
    assert target.read_text(encoding="utf-8") == "hi"


def test_overwrites_existing(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_crash_mid_write_leaves_original_intact(tmp_path):
    """If the swap fails, the original file must be untouched (not truncated)
    and no temp file is left behind."""
    target = tmp_path / "state.json"
    target.write_text("original", encoding="utf-8")

    with patch("atomic_io.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            atomic_write_text(target, "new data that never lands")

    assert target.read_text(encoding="utf-8") == "original"   # not truncated
    assert list(tmp_path.glob(".*tmp")) == []                 # temp cleaned up


def test_no_temp_files_left_after_success(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_text(target, "x")
    leftovers = [p for p in tmp_path.iterdir() if p.name != "state.json"]
    assert leftovers == []


class TestConfigUsesAtomicWrite:
    def test_save_config_is_atomic(self, tmp_path, monkeypatch):
        import config
        cfg_file = tmp_path / ".argent_coder_config.json"
        monkeypatch.setattr(config, "CONFIG_FILE", cfg_file)
        monkeypatch.setattr(config, "_CONFIG_CACHE", None)

        config.save_config({"model": "test", "provider": "ollama"})
        assert json.loads(cfg_file.read_text(encoding="utf-8"))["model"] == "test"
        # No temp residue in the home dir.
        assert [p.name for p in tmp_path.iterdir()] == [".argent_coder_config.json"]
