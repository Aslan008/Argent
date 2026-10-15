import pytest
from unittest.mock import patch, MagicMock

import config
import providers
from providers import DeepSeekProvider, _is_openai_reasoning_model
from src.agent.strategy import get_model_strategy, CloudStrategy
from src.agent.critic import parse_critic_model


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    yield store


class TestDeepSeekConfig:
    def test_defaults(self):
        assert config.get_deepseek_url() == "https://api.deepseek.com"
        assert config.get_deepseek_api_key() is None

    def test_set_and_get_key(self):
        config.set_deepseek_api_key("sk-ds-test-key")
        assert config.get_deepseek_api_key() == "sk-ds-test-key"

    def test_custom_url(self):
        config.set_deepseek_url("https://custom.deepseek.proxy/v1")
        assert config.get_deepseek_url() == "https://custom.deepseek.proxy/v1"


class TestDeepSeekFactory:
    def test_create_deepseek_provider_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "deepseek")
        config.set_deepseek_api_key("sk-ds-factory")
        provider = providers.create_provider()
        assert isinstance(provider, DeepSeekProvider)
        assert provider.name == "deepseek"
        assert provider._api_key == "sk-ds-factory"
        assert provider._base_url == "https://api.deepseek.com"

    def test_create_deepseek_provider_explicit_name(self):
        config.set_deepseek_api_key("sk-ds-explicit")
        provider = providers.create_provider("deepseek")
        assert isinstance(provider, DeepSeekProvider)
        assert provider.name == "deepseek"
        assert provider._api_key == "sk-ds-explicit"


class TestDeepSeekValidation:
    def test_missing_key_reports_actionable_error(self):
        p = DeepSeekProvider(api_key=None, base_url="https://api.deepseek.com")
        msg = p.validate_config()
        assert msg and "API key" in msg and "platform.deepseek.com" in msg

    def test_present_key_validates(self):
        p = DeepSeekProvider(api_key="sk-ds-valid", base_url="https://api.deepseek.com")
        assert p.validate_config() is None


class TestDeepSeekListModels:
    def test_falls_back_to_curated_list_without_network(self):
        p = DeepSeekProvider(api_key=None, base_url="https://api.deepseek.com")
        models = p.list_models()
        assert "deepseek-chat" in models
        assert "deepseek-reasoner" in models

    def test_fetches_models_from_api_endpoint(self):
        p = DeepSeekProvider(api_key="sk-ds-test", base_url="https://api.deepseek.com")
        mock_client = MagicMock()
        mock_model_1 = MagicMock()
        mock_model_1.id = "deepseek-chat"
        mock_model_2 = MagicMock()
        mock_model_2.id = "deepseek-reasoner"
        mock_client.models.list.return_value = MagicMock(data=[mock_model_1, mock_model_2])

        with patch.object(p, "_get_client", return_value=mock_client):
            models = p.list_models()
            assert models == ["deepseek-chat", "deepseek-reasoner"]


class TestDeepSeekErrorHandling:
    def test_handles_401_invalid_key(self):
        p = DeepSeekProvider(api_key="sk-ds-bad", base_url="https://api.deepseek.com")
        err = MagicMock()
        err.status_code = 401
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Invalid API key" in str(excinfo.value)
        assert "DeepSeek" in str(excinfo.value)
        assert "platform.deepseek.com" in str(excinfo.value)

    def test_handles_402_insufficient_balance(self):
        p = DeepSeekProvider(api_key="sk-ds-empty", base_url="https://api.deepseek.com")
        err = MagicMock()
        err.status_code = 402
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Insufficient balance" in str(excinfo.value)
        assert "platform.deepseek.com" in str(excinfo.value)

    def test_handles_429_rate_limit(self):
        p = DeepSeekProvider(api_key="sk-ds-valid", base_url="https://api.deepseek.com")
        err = MagicMock()
        err.status_code = 429
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Rate limited" in str(excinfo.value)
        assert excinfo.value.retryable is True

    def test_handles_503_overloaded(self):
        p = DeepSeekProvider(api_key="sk-ds-valid", base_url="https://api.deepseek.com")
        err = MagicMock()
        err.status_code = 503
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "overloaded" in str(excinfo.value)
        assert excinfo.value.retryable is True


class TestDeepSeekStrategy:
    def test_deepseek_always_uses_cloud_strategy(self):
        for model in ["deepseek-chat", "deepseek-reasoner"]:
            assert isinstance(get_model_strategy(model, "deepseek"), CloudStrategy)

    def test_cloud_strategy_properties(self):
        s = get_model_strategy("deepseek-chat", "deepseek")
        assert s.supports_native_tools() is True
        assert s.wants_constrained_decoding() is False
        assert s.wants_objective_anchor() is False


class TestDeepSeekCriticParser:
    def test_critic_deepseek_prefix(self):
        provider, model = parse_critic_model("deepseek:deepseek-reasoner")
        assert provider == "deepseek"
        assert model == "deepseek-reasoner"


class TestDeepSeekReasoning:
    def test_reasoning_model_detection(self):
        assert _is_openai_reasoning_model("deepseek-reasoner") is True
        assert _is_openai_reasoning_model("deepseek/deepseek-reasoner") is True
        assert _is_openai_reasoning_model("deepseek-chat") is False

    def test_reasoning_content_yielded_in_stream(self):
        p = DeepSeekProvider(api_key="sk-ds-test", base_url="https://api.deepseek.com")
        mock_chunk = MagicMock()
        mock_choice = MagicMock()
        mock_choice.delta.content = "Here is the solution."
        mock_choice.delta.reasoning_content = "Thinking through the steps..."
        mock_choice.delta.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = [mock_chunk]

        with patch.object(p, "_get_client", return_value=mock_client):
            stream = p.stream_chat(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "Solve X"}],
            )
            chunks = list(stream)
            assert len(chunks) == 1
            assert chunks[0]["content"] == "Here is the solution."
            assert chunks[0]["thinking"] == "Thinking through the steps..."
