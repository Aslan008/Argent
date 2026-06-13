from types import SimpleNamespace

import pytest

from providers import (
    _friendly_api_error, _parse_openai_usage,
    _is_transient_stream_error, _friendly_stream_error,
    ProviderError, OpenRouterProvider,
)


def fake_status_error(status, body=None, message=None):
    return SimpleNamespace(status_code=status, body=body, message=message)


class TestFriendlyApiError:
    def test_extracts_nested_message_and_upstream(self):
        e = fake_status_error(400, body={
            "error": {"message": "Provider returned error",
                      "metadata": {"provider_name": "Nvidia"}},
        })
        out = _friendly_api_error("openrouter", e)
        assert out.startswith("OPENROUTER error (HTTP 400):")
        assert "Provider returned error" in out
        assert "upstream: Nvidia" in out

    def test_collapses_whitespace_and_caps_length(self):
        huge = "x " * 500
        e = fake_status_error(400, body={"error": {"message": huge}})
        out = _friendly_api_error("openrouter", e)
        assert len(out) < 360
        assert "\n" not in out

    def test_falls_back_to_message_attr(self):
        e = fake_status_error(503, body=None, message="Service Unavailable")
        out = _friendly_api_error("zai", e)
        assert "Service Unavailable" in out
        assert "HTTP 503" in out

    def test_no_raw_json_leaks_into_message(self):
        # The whole point: the user must not see the raw nested payload.
        raw = {"error": {"message": "boom", "metadata": {"raw": "{deeply: nested vllm dump}"}}}
        e = fake_status_error(400, body=raw)
        out = _friendly_api_error("openrouter", e)
        assert "deeply" not in out
        assert "boom" in out


class TestTransientStreamErrors:
    @pytest.mark.parametrize("text", [
        "Upstream idle timeout exceeded",
        "Request timed out",
        "503 Service Unavailable",
        "upstream connect error",
        "Provider overloaded",
        "Connection reset by peer",
    ])
    def test_transient_detected(self, text):
        assert _is_transient_stream_error(Exception(text))

    @pytest.mark.parametrize("text", [
        "Invalid request: bad parameter",
        "context length exceeded",
    ])
    def test_non_transient_not_flagged(self, text):
        assert not _is_transient_stream_error(Exception(text))

    def test_friendly_stream_error_adds_hint_for_transient(self):
        msg = _friendly_stream_error("openrouter", Exception("Upstream idle timeout exceeded"))
        assert "OPENROUTER" in msg
        assert "Upstream idle timeout exceeded" in msg
        assert "/model" in msg  # actionable hint present

    def test_friendly_stream_error_plain_for_other(self):
        msg = _friendly_stream_error("zai", Exception("weird parse failure"))
        assert "weird parse failure" in msg
        assert "/model" not in msg


class TestGuardedStream:
    def _provider(self):
        return OpenRouterProvider(api_key="k", base_url="https://openrouter.ai/api/v1")

    def test_passes_chunks_through(self):
        provider = self._provider()
        out = list(provider._guarded_stream(iter(["a", "b", "c"])))
        assert out == ["a", "b", "c"]

    def test_midstream_timeout_becomes_provider_error(self):
        def stream():
            yield "partial"
            raise Exception("Upstream idle timeout exceeded")

        provider = self._provider()
        with pytest.raises(ProviderError) as exc:
            list(provider._guarded_stream(stream()))
        assert exc.value.retryable is True
        assert "/model" in str(exc.value)


class TestParseUsage:
    def test_basic_token_counts(self):
        u = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                            cost=None, model_extra=None)
        assert _parse_openai_usage(u) == {"prompt": 100, "completion": 50, "total": 150}

    def test_cost_from_direct_attr(self):
        u = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                            cost=0.0012, model_extra=None)
        assert _parse_openai_usage(u)["cost"] == 0.0012

    def test_cost_from_model_extra(self):
        u = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                            cost=None, model_extra={"cost": 0.003})
        assert _parse_openai_usage(u)["cost"] == 0.003

    def test_missing_fields_default_to_zero(self):
        u = SimpleNamespace()
        assert _parse_openai_usage(u) == {"prompt": 0, "completion": 0, "total": 0}
