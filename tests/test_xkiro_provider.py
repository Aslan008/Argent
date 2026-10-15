import pytest
from unittest.mock import patch, MagicMock

import config
import providers
from providers import XKiroProvider, OllamaProvider
from src.agent.strategy import get_model_strategy, CloudStrategy
from src.agent.critic import parse_critic_model


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestXKiroConfig:
    def test_defaults(self):
        assert config.get_xkiro_url() == "https://api.xkiro.com/v1"
        assert config.get_xkiro_api_key() is None

    def test_set_and_get_key(self):
        config.set_xkiro_api_key("sk-xkiro-abc")
        assert config.get_xkiro_api_key() == "sk-xkiro-abc"

    def test_custom_url(self):
        config.set_xkiro_url("https://proxy.example.com/v1")
        assert config.get_xkiro_url() == "https://proxy.example.com/v1"


class TestXKiroFactory:
    def test_create_xkiro_provider_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "xkiro")
        config.set_xkiro_api_key("sk-xkiro-xyz")
        provider = providers.create_provider()
        assert isinstance(provider, XKiroProvider)
        assert provider.name == "xkiro"
        assert provider._api_key == "sk-xkiro-xyz"
        assert provider._base_url == "https://api.xkiro.com/v1"

    def test_create_xkiro_provider_explicit_name(self):
        config.set_xkiro_api_key("sk-xkiro-explicit")
        provider = providers.create_provider("xkiro")
        assert isinstance(provider, XKiroProvider)
        assert provider.name == "xkiro"
        assert provider._api_key == "sk-xkiro-explicit"


class TestXKiroValidation:
    def test_missing_key_reports_actionable_error(self):
        p = XKiroProvider(api_key=None, base_url="https://api.xkiro.com/v1")
        msg = p.validate_config()
        assert msg and "API key" in msg and "xkiro.com" in msg

    def test_present_key_validates(self):
        p = XKiroProvider(api_key="sk-xkiro-valid", base_url="https://api.xkiro.com/v1")
        assert p.validate_config() is None


class TestXKiroListModels:
    def test_falls_back_to_curated_list_without_network(self):
        p = XKiroProvider(api_key=None, base_url="https://api.xkiro.com/v1")
        models = p.list_models()
        assert "anthropic/claude-3.5-sonnet" in models
        assert "deepseek/deepseek-chat" in models
        assert "openai/gpt-4o" in models

    def test_fetches_models_from_api_endpoint(self):
        p = XKiroProvider(api_key="sk-xkiro-test", base_url="https://api.xkiro.com/v1")
        mock_client = MagicMock()
        mock_model_1 = MagicMock()
        mock_model_1.id = "anthropic/claude-3.7-sonnet"
        mock_model_2 = MagicMock()
        mock_model_2.id = "meta-llama/llama-3.3-70b-instruct"
        mock_client.models.list.return_value = MagicMock(data=[mock_model_1, mock_model_2])

        with patch.object(p, "_get_client", return_value=mock_client):
            models = p.list_models()
            assert models == [
                "anthropic/claude-3.7-sonnet",
                "meta-llama/llama-3.3-70b-instruct",
            ]


class TestXKiroErrorHandling:
    def test_handles_401_invalid_key(self):
        p = XKiroProvider(api_key="sk-xkiro-bad", base_url="https://api.xkiro.com/v1")
        err = MagicMock()
        err.status_code = 401
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Invalid API key" in str(excinfo.value)
        assert "xKiro" in str(excinfo.value)

    def test_handles_402_insufficient_credits(self):
        p = XKiroProvider(api_key="sk-xkiro-empty", base_url="https://api.xkiro.com/v1")
        err = MagicMock()
        err.status_code = 402
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Insufficient credits" in str(excinfo.value)
        assert "xkiro.com" in str(excinfo.value)

    def test_handles_429_rate_limit(self):
        p = XKiroProvider(api_key="sk-xkiro-valid", base_url="https://api.xkiro.com/v1")
        err = MagicMock()
        err.status_code = 429
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Rate limited" in str(excinfo.value)
        assert excinfo.value.retryable is True


class TestXKiroStrategy:
    def test_xkiro_always_uses_cloud_strategy(self):
        for model in ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "deepseek/deepseek-chat"]:
            assert isinstance(get_model_strategy(model, "xkiro"), CloudStrategy)

    def test_cloud_strategy_properties(self):
        s = get_model_strategy("anthropic/claude-3.7-sonnet", "xkiro")
        assert s.supports_native_tools() is True
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is False


class TestXKiroCriticParser:
    def test_critic_xkiro_prefix(self):
        provider, model = parse_critic_model("xkiro:openai/gpt-4o")
        assert provider == "xkiro"
        assert model == "openai/gpt-4o"
