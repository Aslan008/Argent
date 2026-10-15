"""A provider's own documented environment variable must work as its API key.

Only the Claude key fell back to the environment. For every other provider the
standard `export OPENROUTER_API_KEY=...` was ignored, and the only way to supply a
key was to write it into a plain-text config file.
"""

import pytest

import config

# (getter, saved-key name in the config file, environment variables it honours)
KEYS = [
    ("get_zai_api_key", "zai_api_key", ["ZAI_API_KEY"]),
    ("get_openrouter_api_key", "openrouter_api_key", ["OPENROUTER_API_KEY"]),
    ("get_deepseek_api_key", "deepseek_api_key", ["DEEPSEEK_API_KEY"]),
    ("get_claude_api_key", "claude_api_key", ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]),
    ("get_brave_api_key", "brave_api_key", ["BRAVE_API_KEY"]),
    ("get_ollama_api_key", "ollama_api_key", ["OLLAMA_API_KEY"]),
]


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config, "_CONFIG_MTIME", None)
    for _, _, names in KEYS:
        for name in names:
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("getter, stored, envs", KEYS)
class TestEnvironmentFallback:
    def test_the_environment_supplies_a_key_nothing_else_does(self, monkeypatch, getter, stored, envs):
        monkeypatch.setenv(envs[0], "from-environment")
        assert getattr(config, getter)() == "from-environment"

    def test_a_saved_key_wins(self, monkeypatch, getter, stored, envs):
        config._set(stored, "saved-with-key-command")
        monkeypatch.setenv(envs[0], "from-environment")
        assert getattr(config, getter)() == "saved-with-key-command"

    def test_an_empty_saved_key_does_not_shadow_the_environment(self, monkeypatch, getter, stored, envs):
        config._set(stored, "")
        monkeypatch.setenv(envs[0], "from-environment")
        assert getattr(config, getter)() == "from-environment"

    def test_a_blank_variable_is_not_a_key(self, monkeypatch, getter, stored, envs):
        monkeypatch.setenv(envs[0], "   ")
        assert not getattr(config, getter)()

    def test_without_either_there_is_no_key(self, getter, stored, envs):
        assert not getattr(config, getter)()


def test_whitespace_around_an_exported_key_is_trimmed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-123\n")
    assert config.get_openrouter_api_key() == "sk-or-123"


def test_both_claude_spellings_work(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-claude")
    assert config.get_claude_api_key() == "sk-claude"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    assert config.get_claude_api_key() == "sk-anthropic"      # the first one listed wins
