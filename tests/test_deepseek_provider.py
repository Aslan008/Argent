import pytest
from unittest.mock import patch, MagicMock

import argent.config as config
import argent.providers as providers
from argent.providers import DeepSeekProvider, _is_openai_reasoning_model
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
    def test_falls_back_to_current_models_without_network(self):
        p = DeepSeekProvider(api_key=None, base_url="https://api.deepseek.com")
        models = p.list_models()
        assert "deepseek-flash" in models
        assert "deepseek-v4-pro" in models
        assert "deepseek-chat" not in models
        assert "deepseek-reasoner" not in models

    def test_fetches_models_from_api_endpoint(self):
        p = DeepSeekProvider(api_key="sk-ds-test", base_url="https://api.deepseek.com")
        mock_client = MagicMock()
        mock_model_1 = MagicMock()
        mock_model_1.id = "deepseek-flash"
        mock_model_2 = MagicMock()
        mock_model_2.id = "deepseek-v4-pro"
        mock_client.models.list.return_value = MagicMock(data=[mock_model_1, mock_model_2])

        with patch.object(p, "_get_client", return_value=mock_client):
            models = p.list_models()
            assert models == ["deepseek-flash", "deepseek-v4-pro"]


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


def _sent_body(provider, model, monkeypatch, mode, effort, budget=None):
    """The kwargs Argent puts on the wire for one stream request (client mocked)."""
    monkeypatch.setattr(config, "get_thinking_mode", lambda: mode)
    monkeypatch.setattr(config, "get_reasoning_effort", lambda: effort)
    monkeypatch.setattr(config, "get_thinking_budget", lambda: budget)
    client = MagicMock()
    client.chat.completions.create.return_value = []
    with patch.object(provider, "_get_client", return_value=client):
        list(provider.stream_chat(model=model, messages=[{"role": "user", "content": "hi"}]))
    return client.chat.completions.create.call_args.kwargs


class TestDeepSeekThinkingDialect:
    """DeepSeek documents its own thinking fields; the generic ones must not leak in."""

    def _prov(self):
        return DeepSeekProvider(api_key="sk-ds", base_url="https://api.deepseek.com")

    def test_off_sends_the_documented_toggle_not_reasoning_effort_none(self, monkeypatch):
        body = _sent_body(self._prov(), "deepseek-flash", monkeypatch, "off", None)["extra_body"]
        assert body["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in body          # "none" is not a DeepSeek value
        assert "chat_template_kwargs" not in body      # a foreign dialect

    def test_on_sends_enabled_toggle_and_effort(self, monkeypatch):
        body = _sent_body(self._prov(), "deepseek-flash", monkeypatch, "on", "high")["extra_body"]
        assert body["thinking"] == {"type": "enabled"}
        assert body["reasoning_effort"] == "high"

    def test_auto_with_no_effort_sends_no_extra_body(self, monkeypatch):
        kw = _sent_body(self._prov(), "deepseek-flash", monkeypatch, "auto", None)
        assert not kw.get("extra_body")

    def test_max_is_passed_through_to_deepseek(self, monkeypatch):
        body = _sent_body(self._prov(), "deepseek-flash", monkeypatch, "on", "max")["extra_body"]
        assert body["reasoning_effort"] == "max"

    def test_provider_declares_it_needs_reasoning_content(self):
        assert DeepSeekProvider.wants_reasoning_content is True
        assert providers.wants_reasoning_content("deepseek") is True
        assert providers.wants_reasoning_content("openrouter") is False


class TestMaxEffortFolding:
    """`max` is DeepSeek's top level; any other API rejects it, so it folds to high."""

    def test_other_providers_get_high_instead_of_max(self, monkeypatch):
        from argent.providers import ZAIProvider
        p = ZAIProvider(api_key="k", base_url="https://api.z.ai/api/paas/v4/")
        body = _sent_body(p, "glm-5", monkeypatch, "on", "max")["extra_body"]
        assert body["reasoning_effort"] == "high"


class TestReasoningContentPassthrough:
    def test_keep_reasoning_puts_it_in_reasoning_content(self):
        from src.agent.trimmer import clean_messages_for_llm
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "thinking": "I should call a tool",
             "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        out = clean_messages_for_llm(msgs, strip_reasoning=True, keep_reasoning=True)
        assistant = [m for m in out if m.get("role") == "assistant"][0]
        assert assistant["reasoning_content"] == "I should call a tool"
        assert "thinking" not in assistant
        assert "<think>" not in (assistant.get("content") or "")

    def test_without_keep_reasoning_reasoning_is_not_sent(self):
        from src.agent.trimmer import clean_messages_for_llm
        msgs = [{"role": "system", "content": "s"},
                {"role": "assistant", "content": "answer", "thinking": "hidden"}]
        out = clean_messages_for_llm(msgs, strip_reasoning=True)
        assistant = [m for m in out if m.get("role") == "assistant"][0]
        assert "reasoning_content" not in assistant
