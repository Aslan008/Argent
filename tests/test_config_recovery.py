"""A config Argent cannot parse must not be a config Argent deletes.

load_config() swallowed json.JSONDecodeError and returned {"model": ...}. The
session then ran with no provider, no API keys, no MCP servers and no knowledge
bases — and the first setting the user touched called save_config(), which
writes the whole dict, overwriting the damaged-but-readable file with those
defaults. A truncated write during a crash therefore cost every stored setting,
silently, with the recoverable text destroyed a minute later.
"""

import json

import pytest

import config


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / ".argent_coder_config.json")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    yield
    config._CONFIG_CACHE = None


def _broken(text: str):
    config.CONFIG_FILE.write_text(text, encoding="utf-8")
    return config.load_config()


class TestDamagedConfig:
    def test_settings_are_kept_on_disk(self):
        """The exact loss: a half-written file, then defaults written over it."""
        original = '{"provider": "zai", "zai_api_key": "sk-live", "mcp_servers": [{'
        loaded = _broken(original)
        assert loaded == {"model": config.DEFAULT_MODEL}

        backups = list(config.CONFIG_FILE.parent.glob("*.broken-*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == original

    def test_a_later_save_cannot_overwrite_the_backup(self):
        _broken('{"zai_api_key": "sk-live",')
        config.set_current_model("llama3.1")
        backup = next(config.CONFIG_FILE.parent.glob("*.broken-*"))
        assert "sk-live" in backup.read_text(encoding="utf-8")

    def test_the_user_is_told(self, capsys):
        _broken("{oops")
        err = capsys.readouterr().err
        assert "Конфиг нечитаем" in err
        assert "broken-" in err

    @pytest.mark.parametrize("payload", ["[1, 2, 3]", "null", '"a string"', "42"])
    def test_valid_json_of_the_wrong_shape_is_also_quarantined(self, payload):
        """No exception is raised here — it fails later, at the first _get(),
        from wherever that happens to be."""
        assert _broken(payload) == {"model": config.DEFAULT_MODEL}
        assert list(config.CONFIG_FILE.parent.glob("*.broken-*"))
        assert config.get_provider()          # must not raise AttributeError


class TestTheOrdinaryPaths:
    def test_a_good_config_is_loaded_untouched(self):
        config.CONFIG_FILE.write_text(json.dumps({"provider": "ollama", "temperature": 0.3}),
                                      encoding="utf-8")
        assert config.load_config()["provider"] == "ollama"
        assert not list(config.CONFIG_FILE.parent.glob("*.broken-*"))

    def test_no_config_at_all_is_not_an_error(self, capsys):
        assert config.load_config() == {"model": config.DEFAULT_MODEL}
        assert "нечитаем" not in capsys.readouterr().err

    def test_an_empty_file_is_treated_as_damage(self):
        """Zero bytes is what a killed process leaves behind."""
        assert _broken("") == {"model": config.DEFAULT_MODEL}
        assert list(config.CONFIG_FILE.parent.glob("*.broken-*"))
