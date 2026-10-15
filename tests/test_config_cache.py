"""The config cache must be coherent when there is no config file.

`load_config` trusted its cache only when it could compare a real modification
time, so with no file on disk (a first run, a read-only home, a test) it built a
brand-new default dict on every call. A setting changed in memory was gone on the
next read. Tests that assumed otherwise only passed on a machine where the
developer's own config file happened to exist.
"""

import json

import pytest

import config


@pytest.fixture
def no_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config, "_CONFIG_MTIME", None)
    return tmp_path / "config.json"


class TestNoFileOnDisk:
    def test_the_same_dict_comes_back(self, no_config):
        assert config.load_config() is config.load_config()

    def test_a_change_that_could_not_be_saved_survives_the_next_read(self, no_config, monkeypatch):
        monkeypatch.setattr(config, "save_config", lambda c: None)    # a read-only home
        config.set_sound_notification(False)
        assert config.get_sound_notification() is False

    def test_defaults_apply_until_something_is_set(self, no_config):
        assert config.load_config() == {"model": config.DEFAULT_MODEL}


class TestFileAppearingAndDisappearing:
    def test_a_file_written_by_another_process_wins_over_the_empty_cache(self, no_config):
        assert config.get_sound_notification() is True            # default, cache primed
        no_config.write_text(json.dumps({"sound_notification": False}), encoding="utf-8")
        assert config.get_sound_notification() is False

    def test_a_deleted_file_falls_back_to_defaults(self, no_config):
        no_config.write_text(json.dumps({"sound_notification": False}), encoding="utf-8")
        assert config.get_sound_notification() is False
        no_config.unlink()
        assert config.get_sound_notification() is True

    def test_a_quarantined_file_leaves_a_stable_default_state(self, no_config):
        no_config.write_text("{broken", encoding="utf-8")
        first = config.load_config()
        assert first == {"model": config.DEFAULT_MODEL}
        assert not no_config.exists()
        assert config.load_config() is first      # not re-created on every call
        assert len(list(no_config.parent.glob("*.broken-*"))) == 1

    def test_saving_then_reading_goes_through_the_file(self, no_config):
        config.set_sound_notification(False)
        assert json.loads(no_config.read_text(encoding="utf-8"))["sound_notification"] is False
        assert config.get_sound_notification() is False
