import pytest

import config
import providers
from providers import OpenRouterProvider, OllamaProvider, ZAIProvider
from src.agent.strategy import get_model_strategy, CloudStrategy


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestConfig:
    def test_defaults(self):
        assert config.get_openrouter_url() == "https://openrouter.ai/api/v1"
        assert config.get_openrouter_api_key() is None

    def test_set_and_get_key(self):
        config.set_openrouter_api_key("sk-or-abc")
        assert config.get_openrouter_api_key() == "sk-or-abc"

    def test_custom_url(self):
        config.set_openrouter_url("https://proxy.local/v1")
        assert config.get_openrouter_url() == "https://proxy.local/v1"


class TestFactory:
    def test_create_openrouter_provider(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "openrouter")
        config.set_openrouter_api_key("sk-or-xyz")
        provider = providers.create_provider()
        assert isinstance(provider, OpenRouterProvider)
        assert provider.name == "openrouter"
        assert provider._api_key == "sk-or-xyz"
        assert provider._base_url == "https://openrouter.ai/api/v1"

    def test_other_providers_unaffected(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "ollama")
        assert isinstance(providers.create_provider(), OllamaProvider)


class TestValidation:
    def test_missing_key_reports_actionable_error(self):
        p = OpenRouterProvider(api_key=None, base_url="https://openrouter.ai/api/v1")
        msg = p.validate_config()
        assert msg and "API key" in msg and "openrouter.ai" in msg

    def test_present_key_validates(self):
        p = OpenRouterProvider(api_key="sk-or", base_url="https://openrouter.ai/api/v1")
        assert p.validate_config() is None


class TestAttributionHeaders:
    def test_default_headers_are_attached(self):
        p = OpenRouterProvider(api_key="sk-or", base_url="https://openrouter.ai/api/v1")
        assert p._default_headers["X-Title"] == "Argent"
        assert "HTTP-Referer" in p._default_headers

    def test_other_openai_providers_have_no_default_headers(self):
        z = ZAIProvider(api_key="k", base_url="https://api.z.ai/api/paas/v4/")
        assert z._default_headers is None


class TestListModels:
    def test_falls_back_to_curated_list_without_network(self):
        p = OpenRouterProvider(api_key=None, base_url="https://openrouter.ai/api/v1")
        models = p.list_models()
        assert "anthropic/claude-3.5-sonnet" in models
        assert all("/" in m for m in models)


class TestStrategy:
    def test_openrouter_always_uses_cloud_strategy(self):
        # Even a model whose name parses as a small local size must run the
        # cloud profile when served through OpenRouter.
        for model in ["anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct",
                      "qwen/qwen-2.5-7b-instruct"]:
            assert isinstance(get_model_strategy(model, "openrouter"), CloudStrategy)

    def test_cloud_strategy_is_unburdened_by_small_model_accommodations(self):
        s = get_model_strategy("deepseek/deepseek-chat", "openrouter")
        assert s.supports_native_tools() is True
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is False
