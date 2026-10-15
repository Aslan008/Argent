import pytest
from unittest.mock import patch, MagicMock

import config
import providers
from providers import WormSoftProvider, OllamaProvider
from src.agent.strategy import get_model_strategy, CloudStrategy
from src.agent.critic import parse_critic_model


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestWormSoftConfig:
    def test_defaults(self):
        assert config.get_wormsoft_url() == "https://ai.wormsoft.ru/api/gpt"
        assert config.get_wormsoft_api_key() is None

    def test_set_and_get_key(self):
        config.set_wormsoft_api_key("sk-wormsoft-test-key")
        assert config.get_wormsoft_api_key() == "sk-wormsoft-test-key"

    def test_custom_url(self):
        config.set_wormsoft_url("https://custom.wormsoft.proxy/api/gpt")
        assert config.get_wormsoft_url() == "https://custom.wormsoft.proxy/api/gpt"

    def test_auto_migrates_legacy_urls(self):
        config.set_wormsoft_url("https://ai.wormsoft.ru/v1")
        assert config.get_wormsoft_url() == "https://ai.wormsoft.ru/api/gpt"
        config.set_wormsoft_url("https://ai.wormsoft.ru/api/v1")
        assert config.get_wormsoft_url() == "https://ai.wormsoft.ru/api/gpt"


class TestWormSoftFactory:
    def test_create_wormsoft_provider_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "wormsoft")
        config.set_wormsoft_api_key("sk-ws-factory")
        provider = providers.create_provider()
        assert isinstance(provider, WormSoftProvider)
        assert provider.name == "wormsoft"
        assert provider._api_key == "sk-ws-factory"
        assert provider._base_url == "https://ai.wormsoft.ru/api/gpt"

    def test_create_wormsoft_provider_explicit_name(self):
        config.set_wormsoft_api_key("sk-ws-explicit")
        provider = providers.create_provider("wormsoft")
        assert isinstance(provider, WormSoftProvider)
        assert provider.name == "wormsoft"
        assert provider._api_key == "sk-ws-explicit"


class TestWormSoftValidation:
    def test_missing_key_reports_actionable_error(self):
        p = WormSoftProvider(api_key=None, base_url="https://ai.wormsoft.ru/api/gpt")
        msg = p.validate_config()
        assert msg and "API key" in msg and "ai.wormsoft.ru" in msg

    def test_present_key_validates(self):
        p = WormSoftProvider(api_key="sk-ws-valid", base_url="https://ai.wormsoft.ru/api/gpt")
        assert p.validate_config() is None


class TestWormSoftListModels:
    def test_falls_back_to_curated_list_without_network(self):
        p = WormSoftProvider(api_key=None, base_url="https://ai.wormsoft.ru/api/gpt")
        models = p.list_models()
        assert "wormsoft/code/high" in models
        assert "anthropic/claude-sonnet-5" in models
        assert "deepseek-ai/deepseek-v4-pro" in models

    def test_fetches_models_from_api_endpoint(self):
        p = WormSoftProvider(api_key="sk-ws-test", base_url="https://ai.wormsoft.ru/api/gpt")
        mock_client = MagicMock()
        mock_model_1 = MagicMock()
        mock_model_1.id = "anthropic/claude-sonnet-5"
        mock_model_2 = MagicMock()
        mock_model_2.id = "deepseek-ai/deepseek-v4-pro"
        mock_client.models.list.return_value = MagicMock(data=[mock_model_1, mock_model_2])

        with patch.object(p, "_get_client", return_value=mock_client):
            models = p.list_models()
            assert models == [
                "anthropic/claude-sonnet-5",
                "deepseek-ai/deepseek-v4-pro",
            ]


class TestWormSoftErrorHandling:
    def test_handles_401_invalid_key(self):
        p = WormSoftProvider(api_key="sk-ws-bad", base_url="https://ai.wormsoft.ru/api/gpt")
        err = MagicMock()
        err.status_code = 401
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Invalid API key" in str(excinfo.value)
        assert "WormSoft" in str(excinfo.value)
        assert "ai.wormsoft.ru" in str(excinfo.value)

    def test_handles_402_insufficient_credits(self):
        p = WormSoftProvider(api_key="sk-ws-empty", base_url="https://ai.wormsoft.ru/api/gpt")
        err = MagicMock()
        err.status_code = 402
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Insufficient credits" in str(excinfo.value)
        assert "ai.wormsoft.ru" in str(excinfo.value)

    def test_handles_429_rate_limit(self):
        p = WormSoftProvider(api_key="sk-ws-valid", base_url="https://ai.wormsoft.ru/api/gpt")
        err = MagicMock()
        err.status_code = 429
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Rate limited" in str(excinfo.value)
        assert excinfo.value.retryable is True


class TestWormSoftStrategy:
    def test_wormsoft_always_uses_cloud_strategy(self):
        for model in ["wormsoft/code/high", "openai/gpt-6-astra", "deepseek-ai/deepseek-v4-pro"]:
            assert isinstance(get_model_strategy(model, "wormsoft"), CloudStrategy)

    def test_cloud_strategy_properties(self):
        s = get_model_strategy("wormsoft/code/high", "wormsoft")
        assert s.supports_native_tools() is True
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is False


class TestWormSoftCriticParser:
    def test_critic_wormsoft_prefix(self):
        provider, model = parse_critic_model("wormsoft:openai/gpt-6-astra")
        assert provider == "wormsoft"
        assert model == "openai/gpt-6-astra"
