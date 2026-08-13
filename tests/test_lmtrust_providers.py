"""Blind-spot tests for providers.py — retry, usage parsing, error helpers,
ProviderError, and create_provider dispatch.

These cover paths that the existing test_provider_errors.py does NOT exercise:
the with_retry loop (success / retry / non-retryable / exhaustion / max_retries=1),
_parse_openai_usage edge cases (None, model_extra cost), _is_transient_stream_error
positive/negative, _friendly_stream_error truncation, _friendly_api_error nested
extraction + upstream + truncation, ProviderError attributes, and create_provider
dispatch for ollama / zai / unknown.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from providers import (
    ProviderError,
    with_retry,
    _parse_openai_usage,
    _is_transient_stream_error,
    _friendly_stream_error,
    _friendly_api_error,
)


# ---------------------------------------------------------------------------
# (1)-(5)  with_retry
# ---------------------------------------------------------------------------

class TestWithRetry:
    def test_succeeds_on_first_call_no_retry(self, monkeypatch):
        """(1) A fn that returns immediately must not sleep or retry."""
        calls = []
        sleep_calls = []
        monkeypatch.setattr("providers.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            calls.append(1)
            return "ok"

        result = with_retry(fn, max_retries=3)
        assert result == "ok"
        assert len(calls) == 1
        assert sleep_calls == []

    def test_retries_on_retryable_error_then_succeeds(self, monkeypatch):
        """(2) A retryable ProviderError (429) is retried, then the next call
        succeeds — sleep must be invoked exactly once."""
        calls = []
        sleep_calls = []
        monkeypatch.setattr("providers.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise ProviderError("429 rate limited", retryable=True)
            return "recovered"

        result = with_retry(fn, max_retries=3)
        assert result == "recovered"
        assert len(calls) == 2
        assert len(sleep_calls) == 1

    def test_raises_on_non_retryable_error_immediately(self, monkeypatch):
        """(3) A non-retryable ProviderError must propagate on the first attempt
        with no sleep."""
        calls = []
        sleep_calls = []
        monkeypatch.setattr("providers.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            calls.append(1)
            raise ProviderError("bad request", retryable=False)

        with pytest.raises(ProviderError) as exc:
            with_retry(fn, max_retries=3)
        assert "bad request" in str(exc.value)
        assert len(calls) == 1
        assert sleep_calls == []

    def test_raises_after_max_retries_exhausted(self, monkeypatch):
        """(4) A retryable error that never recovers must exhaust max_retries
        and then re-raise."""
        calls = []
        sleep_calls = []
        monkeypatch.setattr("providers.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            calls.append(1)
            raise ProviderError("503 always down", retryable=True)

        with pytest.raises(ProviderError) as exc:
            with_retry(fn, max_retries=3)
        assert "503 always down" in str(exc.value)
        # 3 attempts, 2 sleeps (between attempts, not after the last)
        assert len(calls) == 3
        assert len(sleep_calls) == 2

    def test_max_retries_1_does_not_retry(self, monkeypatch):
        """(5) With max_retries=1 a retryable error must still raise immediately
        because there is no second attempt."""
        calls = []
        sleep_calls = []
        monkeypatch.setattr("providers.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            calls.append(1)
            raise ProviderError("429 down", retryable=True)

        with pytest.raises(ProviderError):
            with_retry(fn, max_retries=1)
        assert len(calls) == 1
        assert sleep_calls == []

    def test_generic_exception_with_retryable_code_is_retried(self, monkeypatch):
        """Bonus: a plain Exception whose message contains a retryable code (429)
        is wrapped in a retryable ProviderError and retried."""
        calls = []
        monkeypatch.setattr("providers.time.sleep", lambda d: None)

        def fn():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("HTTP 429 Too Many Requests")
            return "ok"

        result = with_retry(fn, max_retries=3)
        assert result == "ok"
        assert len(calls) == 2

    def test_generic_exception_non_retryable_raises_wrapped(self, monkeypatch):
        """Bonus: a plain Exception without a retryable code is wrapped in a
        non-retryable ProviderError and raised immediately."""
        monkeypatch.setattr("providers.time.sleep", lambda d: None)

        def fn():
            raise ValueError("totally bogus")

        with pytest.raises(ProviderError) as exc:
            with_retry(fn, max_retries=3)
        assert exc.value.retryable is False
        assert "totally bogus" in str(exc.value)


# ---------------------------------------------------------------------------
# (6)-(8)  _parse_openai_usage
# ---------------------------------------------------------------------------

class TestParseOpenAIUsage:
    def test_none_returns_empty_dict(self):
        """(6) None usage → empty dict (no keys at all)."""
        assert _parse_openai_usage(None) == {}

    def test_valid_usage_returns_token_counts(self):
        """(7) A usage object with prompt/completion/total tokens returns a
        dict with those three keys."""
        u = MagicMock()
        u.prompt_tokens = 100
        u.completion_tokens = 50
        u.total_tokens = 150
        u.cost = None
        u.model_extra = None
        d = _parse_openai_usage(u)
        assert d["prompt"] == 100
        assert d["completion"] == 50
        assert d["total"] == 150
        # no cost → key absent
        assert "cost" not in d

    def test_extracts_cost_from_model_extra(self):
        """(8) When cost is absent as a direct attr but present in model_extra,
        it is surfaced and coerced to a non-negative float."""
        u = MagicMock()
        u.prompt_tokens = 10
        u.completion_tokens = 5
        u.total_tokens = 15
        u.cost = None
        u.model_extra = {"cost": "0.0042"}
        d = _parse_openai_usage(u)
        assert d["cost"] == pytest.approx(0.0042)

    def test_cost_direct_attr_takes_precedence(self):
        """Bonus: a direct cost attr wins over model_extra."""
        u = MagicMock()
        u.prompt_tokens = 1
        u.completion_tokens = 1
        u.total_tokens = 2
        u.cost = 0.01
        u.model_extra = {"cost": 0.99}
        d = _parse_openai_usage(u)
        assert d["cost"] == 0.01

    def test_invalid_cost_is_dropped(self):
        """Bonus: a non-numeric cost is silently dropped (no 'cost' key)."""
        u = MagicMock()
        u.prompt_tokens = 1
        u.completion_tokens = 1
        u.total_tokens = 2
        u.cost = "not-a-number"
        u.model_extra = None
        d = _parse_openai_usage(u)
        assert "cost" not in d


# ---------------------------------------------------------------------------
# (9)-(10)  _is_transient_stream_error
# ---------------------------------------------------------------------------

class TestIsTransientStreamError:
    @pytest.mark.parametrize("text", [
        "connection timeout exceeded",
        "upstream idle timeout",
        "server returned 503",
    ])
    def test_matches_transient_hints(self, text):
        """(9) 'timeout', 'idle', '503' (case-insensitive) are transient."""
        assert _is_transient_stream_error(Exception(text)) is True

    def test_does_not_match_success(self):
        """(10) A benign 'success' message is not transient."""
        assert _is_transient_stream_error(Exception("success")) is False


# ---------------------------------------------------------------------------
# (11)  _friendly_stream_error truncation
# ---------------------------------------------------------------------------

class TestFriendlyStreamError:
    def test_truncates_long_messages_to_200_chars(self):
        """(11) A non-transient error longer than 200 chars is truncated to
        197 + '...' (200 total)."""
        long_msg = "x" * 500
        out = _friendly_stream_error("zai", Exception(long_msg))
        # The prefix is "ZAI stream error: " + msg
        body = out.split("stream error: ", 1)[1]
        assert len(body) == 200
        assert body.endswith("...")

    def test_short_message_not_truncated(self):
        """Bonus: a short non-transient message passes through unchanged."""
        out = _friendly_stream_error("zai", Exception("boom"))
        assert "boom" in out
        assert "..." not in out


# ---------------------------------------------------------------------------
# (12)-(14)  _friendly_api_error
# ---------------------------------------------------------------------------

class TestFriendlyApiError:
    def _err(self, status, body=None, message=None):
        e = MagicMock()
        e.status_code = status
        e.body = body
        e.message = message
        return e

    def test_extracts_message_from_nested_body_error(self):
        """(12) body.error.message is surfaced."""
        e = self._err(400, body={"error": {"message": "bad model id"}})
        out = _friendly_api_error("openrouter", e)
        assert "bad model id" in out
        assert "OPENROUTER error (HTTP 400):" in out

    def test_adds_upstream_provider_name(self):
        """(13) body.error.metadata.provider_name is appended as
        '(upstream: <name>)'."""
        e = self._err(500, body={
            "error": {
                "message": "model crashed",
                "metadata": {"provider_name": "Nvidia"},
            },
        })
        out = _friendly_api_error("openrouter", e)
        assert "model crashed" in out
        assert "upstream: Nvidia" in out

    def test_truncates_to_280_chars(self):
        """(14) The message portion is capped at 280 chars (277 + '...')."""
        huge = "y" * 1000
        e = self._err(400, body={"error": {"message": huge}})
        out = _friendly_api_error("openrouter", e)
        body = out.split(": ", 1)[1]
        assert len(body) == 280
        assert body.endswith("...")

    def test_falls_back_to_message_attr_when_no_body(self):
        """Bonus: no body dict → falls back to .message then str(e)."""
        e = self._err(503, body=None, message="Service Unavailable")
        out = _friendly_api_error("zai", e)
        assert "Service Unavailable" in out
        assert "HTTP 503" in out


# ---------------------------------------------------------------------------
# (15)  ProviderError
# ---------------------------------------------------------------------------

class TestProviderError:
    def test_retryable_true(self):
        """(15a) ProviderError carries retryable=True."""
        err = ProviderError("transient", retryable=True)
        assert err.retryable is True
        assert "transient" in str(err)

    def test_retryable_false(self):
        """(15b) ProviderError carries retryable=False (default)."""
        err = ProviderError("permanent")
        assert err.retryable is False

    def test_original_error_preserved(self):
        """Bonus: original_error is kept for diagnostics."""
        cause = RuntimeError("root cause")
        err = ProviderError("wrapped", retryable=False, original_error=cause)
        assert err.original_error is cause


# ---------------------------------------------------------------------------
# (16)-(18)  create_provider
# ---------------------------------------------------------------------------

class TestCreateProvider:
    def test_ollama_returns_ollama_provider(self, monkeypatch):
        """(16) create_provider('ollama') returns an OllamaProvider instance."""
        # OllamaProvider.__init__ imports ollama + httpx + config; stub both.
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.Client = MagicMock()
        monkeypatch.setitem(sys.modules, "ollama", fake_ollama)

        import httpx as _real_httpx
        monkeypatch.setattr("httpx.Timeout", _real_httpx.Timeout)
        monkeypatch.setattr("config.get_stream_read_timeout", lambda: 120.0)

        from providers import create_provider, OllamaProvider
        p = create_provider("ollama")
        assert isinstance(p, OllamaProvider)
        assert p.name == "ollama"

    def test_zai_returns_zai_provider(self, monkeypatch):
        """(17) create_provider('zai') returns a ZAIProvider instance."""
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        monkeypatch.setattr("config.get_zai_api_key", lambda: "test-key")
        monkeypatch.setattr("config.get_zai_endpoint", lambda: "https://api.test.example/v1")

        from providers import create_provider, ZAIProvider
        p = create_provider("zai")
        assert isinstance(p, ZAIProvider)
        assert p.name == "zai"

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        """(18) An unknown provider name raises ValueError."""
        from providers import create_provider
        with pytest.raises(ValueError) as exc:
            create_provider("does-not-exist")
        assert "Unknown provider" in str(exc.value)