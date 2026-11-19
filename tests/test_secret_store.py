"""API keys: the OS credential store first, the config file only as a fallback.

The real vault is never touched here: `_backend` is replaced with an in-memory
double, and the root conftest sets ARGENT_DISABLE_KEYRING=1 for every other test.
"""
import pytest

import argent.config as config
from src.agent import secret_store


class FakeKeyring:
    """The two methods `secret_store` uses, plus a working `get_keyring`."""

    def __init__(self):
        self.values = {}

    def get_password(self, service, name):
        return self.values.get((service, name))

    def set_password(self, service, name, value):
        self.values[(service, name)] = value

    def delete_password(self, service, name):
        self.values.pop((service, name), None)

    def get_keyring(self):
        return self


@pytest.fixture
def vault(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(secret_store, "_backend", lambda: fake)
    return fake


def test_a_saved_key_lives_in_the_vault_and_not_in_the_config_file(vault):
    config.set_zai_api_key("sk-vault-123")

    assert vault.values[("argent", "zai_api_key")] == "sk-vault-123"
    assert "zai_api_key" not in config.load_config()
    assert config.get_zai_api_key() == "sk-vault-123"


def test_a_legacy_plaintext_key_is_migrated_on_first_read(vault):
    config._set("openrouter_api_key", "sk-legacy-xyz")

    assert config.get_openrouter_api_key() == "sk-legacy-xyz"
    assert vault.values[("argent", "openrouter_api_key")] == "sk-legacy-xyz"
    assert "openrouter_api_key" not in config.load_config()


def test_without_a_vault_the_key_still_works_from_the_file(monkeypatch):
    monkeypatch.setattr(secret_store, "_backend", lambda: None)

    config.set_zai_api_key("sk-file-456")

    assert config.load_config()["zai_api_key"] == "sk-file-456"
    assert config.get_zai_api_key() == "sk-file-456"


def test_clearing_a_key_removes_both_copies(vault, monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    config.set_zai_api_key("sk-to-delete")
    config.set_zai_api_key("")

    assert vault.values == {}
    assert "zai_api_key" not in config.load_config()
    assert config.get_zai_api_key() in (None, "")


def test_the_environment_is_used_when_nothing_was_saved(vault, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-from-env")

    assert config.get_zai_api_key() == "sk-from-env"


class FailKeyring:
    """Named like keyring's own `fail.Keyring`: installed, but stores nothing."""


def test_a_fail_backend_is_not_treated_as_a_vault():
    class Shim:
        def get_keyring(self):
            return FailKeyring()

    assert secret_store._usable(Shim()) is False


def test_a_working_backend_is_usable():
    assert secret_store._usable(FakeKeyring()) is True
