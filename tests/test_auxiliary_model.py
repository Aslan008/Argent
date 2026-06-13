import pytest

import config
import providers
from providers import OllamaProvider, OpenRouterProvider


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {"provider": "openrouter", "model": "anthropic/claude-3.7-sonnet",
             "openrouter_api_key": "sk-or-x"}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestConfig:
    def test_unset_by_default(self):
        assert config.get_auxiliary_model() is None
        assert config.get_auxiliary_provider() is None

    def test_set_and_clear(self):
        config.set_auxiliary_provider("ollama")
        config.set_auxiliary_model("qwen2.5:1.5b")
        assert config.get_auxiliary_provider() == "ollama"
        assert config.get_auxiliary_model() == "qwen2.5:1.5b"
        config.set_auxiliary_model(None)
        config.set_auxiliary_provider(None)
        assert config.get_auxiliary_model() is None
        assert config.get_auxiliary_provider() is None


class TestCreateServiceProvider:
    def test_falls_back_to_main_when_unset(self):
        provider, model = providers.create_service_provider()
        assert isinstance(provider, OpenRouterProvider)
        assert model == "anthropic/claude-3.7-sonnet"

    def test_uses_auxiliary_when_set(self):
        # Main provider is paid OpenRouter; service tasks go to local Ollama.
        config.set_auxiliary_provider("ollama")
        config.set_auxiliary_model("qwen2.5:1.5b")
        provider, model = providers.create_service_provider()
        assert isinstance(provider, OllamaProvider)
        assert model == "qwen2.5:1.5b"

    def test_auxiliary_model_without_explicit_provider_reuses_main(self):
        # Aux model set but no aux provider -> same provider, different model.
        config.set_auxiliary_model("anthropic/claude-3.5-haiku")
        provider, model = providers.create_service_provider()
        assert isinstance(provider, OpenRouterProvider)
        assert model == "anthropic/claude-3.5-haiku"


class TestCreateProviderOverride:
    def test_named_override_builds_specific_provider(self):
        # Explicit name ignores the configured main provider.
        assert isinstance(providers.create_provider("ollama"), OllamaProvider)
        assert isinstance(providers.create_provider(), OpenRouterProvider)  # default
