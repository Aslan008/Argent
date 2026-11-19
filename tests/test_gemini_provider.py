"""Gemini provider: config, factory, catalogue, error mapping and where it is routed.

Gemini speaks the OpenAI dialect through a compatibility layer, so the tests that
matter are the ones specific to it: the endpoint, the built-in catalogue (the
compat endpoint does not serve /models reliably), the key aliases Google uses, and
the fact that `reasoning_effort` — unlike on most OpenAI-compatible servers — is
actually honoured.
"""
import pytest
from unittest.mock import patch, MagicMock

import argent.config as config
import argent.providers as providers
from argent.providers import GeminiProvider, _EFFORT_PROVIDERS
from src.agent.strategy import get_model_strategy, CloudStrategy
from src.agent.critic import parse_critic_model


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield store


class TestGeminiConfig:
    def test_default_endpoint_is_the_openai_compat_endpoint(self):
        assert config.get_gemini_url() == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_no_key_by_default(self):
        assert config.get_gemini_api_key() is None

    def test_set_and_get_key(self):
        config.set_gemini_api_key("AIza-test-gemini-key")
        assert config.get_gemini_api_key() == "AIza-test-gemini-key"

    def test_key_falls_back_to_google_env_names(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-from-google-env")
        assert config.get_gemini_api_key() == "AIza-from-google-env"

    def test_custom_url(self):
        config.set_gemini_url("https://proxy.example.com/v1beta/openai")
        assert config.get_gemini_url() == "https://proxy.example.com/v1beta/openai"

    def test_google_is_an_alias_for_gemini(self):
        assert config.normalize_provider_name("google") == "gemini"
        assert config.normalize_provider_name("Gemini") == "gemini"
        assert config.normalize_provider_name("google-gemini") == "gemini"


class TestGeminiFactory:
    def test_create_from_config(self, monkeypatch):
        monkeypatch.setattr(config, "get_provider", lambda: "gemini")
        config.set_gemini_api_key("AIza-factory")
        provider = providers.create_provider()
        assert isinstance(provider, GeminiProvider)
        assert provider.name == "gemini"
        assert provider._api_key == "AIza-factory"
        assert provider._base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_create_by_explicit_name(self):
        config.set_gemini_api_key("AIza-explicit")
        provider = providers.create_provider("gemini")
        assert isinstance(provider, GeminiProvider)
        assert provider._api_key == "AIza-explicit"

    def test_google_alias_builds_a_gemini_provider(self):
        provider = providers.create_provider("google")
        assert isinstance(provider, GeminiProvider)


class TestGeminiValidation:
    def test_missing_key_reports_actionable_error(self):
        p = GeminiProvider(api_key=None, base_url=config.DEFAULT_GEMINI_URL)
        msg = p.validate_config()
        assert msg and "API key" in msg and "aistudio.google.com" in msg

    def test_present_key_validates(self):
        p = GeminiProvider(api_key="AIza-valid", base_url=config.DEFAULT_GEMINI_URL)
        assert p.validate_config() is None


class TestGeminiListModels:
    def test_falls_back_to_builtin_catalogue_without_network(self):
        p = GeminiProvider(api_key=None, base_url=config.DEFAULT_GEMINI_URL)
        models = p.list_models()
        assert "gemini-2.5-flash" in models
        assert "gemini-2.5-pro" in models
        assert "gemini-2.0-flash" in models

    def test_fetches_models_from_api_endpoint(self):
        p = GeminiProvider(api_key="AIza-test", base_url=config.DEFAULT_GEMINI_URL)
        client = MagicMock()
        m1, m2 = MagicMock(), MagicMock()
        m1.id, m2.id = "gemini-2.5-pro", "gemini-2.5-flash"
        client.models.list.return_value = MagicMock(data=[m1, m2])
        with patch.object(p, "_get_client", return_value=client):
            assert p.list_models() == ["gemini-2.5-flash", "gemini-2.5-pro"]

    def test_a_typed_model_id_is_kept_as_typed(self):
        # corrects_model_spelling is False, so resolve_model must not hit the network
        # or re-spell an id the person typed (the catalogue over the compat endpoint
        # is not authoritative).
        p = GeminiProvider(api_key="AIza-test", base_url=config.DEFAULT_GEMINI_URL)
        assert p.resolve_model("gemini-9-experimental") == "gemini-9-experimental"

    def test_default_model_is_a_flash_tier(self):
        p = GeminiProvider(api_key="AIza-test", base_url=config.DEFAULT_GEMINI_URL)
        assert p.default_model() == "gemini-2.5-flash"


class TestGeminiErrorHandling:
    def test_401_invalid_key(self):
        p = GeminiProvider(api_key="bad", base_url=config.DEFAULT_GEMINI_URL)
        err = MagicMock()
        err.status_code = 401
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Gemini" in str(excinfo.value)
        assert "aistudio.google.com" in str(excinfo.value)

    def test_429_rate_limit_is_retryable(self):
        p = GeminiProvider(api_key="ok", base_url=config.DEFAULT_GEMINI_URL)
        err = MagicMock()
        err.status_code = 429
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert excinfo.value.retryable is True

    def test_400_names_the_request_problem(self):
        p = GeminiProvider(api_key="ok", base_url=config.DEFAULT_GEMINI_URL)
        err = MagicMock()
        err.status_code = 400
        err.body = {"error": {"message": "Invalid model name"}}
        with pytest.raises(providers.ProviderError) as excinfo:
            p._handle_api_status_error(err)
        assert "Invalid model name" in str(excinfo.value)


class TestGeminiRouting:
    def test_always_uses_cloud_strategy(self):
        for model in ["gemini-2.5-flash", "gemini-3.8-flash", "gemini-2.5-pro"]:
            assert isinstance(get_model_strategy(model, "gemini"), CloudStrategy)

    def test_critic_prefix_recognised(self):
        provider, model = parse_critic_model("gemini:gemini-2.5-pro")
        assert provider == "gemini"
        assert model == "gemini-2.5-pro"

    def test_reasoning_effort_is_sent(self):
        # Gemini's compat layer honours reasoning_effort, so the provider is listed
        # among those that may receive it.
        assert "gemini" in _EFFORT_PROVIDERS


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


class TestGeminiThinkingDialect:
    """The compat layer parses the body against Gemini's schema: a foreign field is a 400.

    Reported: `thinking_mode=auto` + `reasoning_effort=max` (the shipped defaults of the
    person's own config) put `chat_template_kwargs` on the wire and Gemini answered
    `Invalid JSON payload received. Unknown name "chat_template_kwargs"`.
    """

    def _prov(self):
        return GeminiProvider(api_key="AIza-test", base_url=config.DEFAULT_GEMINI_URL)

    def test_reported_payload_no_longer_carries_chat_template_kwargs(self, monkeypatch):
        kw = _sent_body(self._prov(), "gemini-2.5-flash", monkeypatch, "auto", "max")
        body = kw["extra_body"]
        assert "chat_template_kwargs" not in body          # the field that got the 400
        assert body["reasoning_effort"] == "high"          # max -> high, the documented way

    def test_forced_thinking_sends_only_reasoning_effort(self, monkeypatch):
        body = _sent_body(self._prov(), "gemini-2.5-flash", monkeypatch, "on", "medium")["extra_body"]
        assert body == {"reasoning_effort": "medium"}

    def test_off_uses_none_not_enable_thinking(self, monkeypatch):
        body = _sent_body(self._prov(), "gemini-2.5-flash", monkeypatch, "off", None)["extra_body"]
        assert body == {"reasoning_effort": "none"}

    def test_xhigh_folds_to_high(self, monkeypatch):
        body = _sent_body(self._prov(), "gemini-3-pro", monkeypatch, "auto", "xhigh")["extra_body"]
        assert body["reasoning_effort"] == "high"

    def test_auto_without_effort_sends_no_extra_body(self, monkeypatch):
        kw = _sent_body(self._prov(), "gemini-2.5-flash", monkeypatch, "auto", None)
        assert not kw.get("extra_body")

    def test_local_sampling_extensions_and_budget_are_not_sent(self, monkeypatch):
        # These are llama.cpp/vLLM names. Configured or not, Gemini must never see them.
        monkeypatch.setattr(config, "get_top_k", lambda: 40)
        monkeypatch.setattr(config, "get_min_p", lambda: 0.05)
        monkeypatch.setattr(config, "get_repeat_penalty", lambda: 1.1)
        kw = _sent_body(self._prov(), "gemini-2.5-flash", monkeypatch, "auto", "low", budget=2048)
        body = kw.get("extra_body") or {}
        for foreign in ("top_k", "min_p", "repetition_penalty", "reasoning_budget",
                        "chat_template_kwargs"):
            assert foreign not in body, foreign
        assert body == {"reasoning_effort": "low"}

    def test_a_local_server_still_gets_the_local_dialect(self, monkeypatch):
        # The gate is per provider, not a deletion: KoboldCPP keeps the switch it understands.
        from argent.providers import KoboldCPPProvider
        p = KoboldCPPProvider(base_url="http://localhost:5001/v1")
        body = _sent_body(p, "koboldcpp-model", monkeypatch, "off", None)["extra_body"]
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "reasoning_effort" not in body   # a local server is not told this name


class TestGeminiSubagents:
    def test_ready_without_key_is_refused_with_reason(self):
        from src.agent import subagents
        ok, why = subagents.provider_ready("gemini")
        assert ok is False and "API key" in why

    def test_ready_with_key(self):
        from src.agent import subagents
        config.set_gemini_api_key("AIza-sub")
        ok, why = subagents.provider_ready("gemini")
        assert ok is True and why == ""
