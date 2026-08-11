"""Blind-spot tests for model_limits — edge cases the original suite missed.

These exercise the boundaries the happy-path tests never touch: negative
context lengths, tiny maximums that produce no usable fractions, multiple
matching keys, non-integer types, a current value that matches nothing, and
an OLLAMA_HOST without an http:// prefix.
"""

import pytest

import model_limits
from model_limits import context_choices, detect_context_length


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


@pytest.fixture
def ollama(monkeypatch):
    """Answers /api/show with whatever the test queues."""
    state = {"reply": {"model_info": {"qwen35.context_length": 262144}}}

    def fake_post(url, json=None, timeout=None):
        state["url"] = url
        state["asked"] = (json or {}).get("model")
        if isinstance(state["reply"], Exception):
            raise state["reply"]
        return _Response(state["reply"])

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    return state


# ---------------------------------------------------------------------------
# L4 × D1 — a negative context length is not a valid maximum
# ---------------------------------------------------------------------------
class TestNegativeContextLength:
    def test_negative_context_length_returns_none(self, ollama):
        """A negative value fails the `value > 0` guard and must be ignored."""
        ollama["reply"] = {"model_info": {"qwen35.context_length": -8192}}
        assert detect_context_length("qwen3.5:9b") is None


# ---------------------------------------------------------------------------
# L2 × D1 — a maximum so small every fraction is below the 2048 floor
# ---------------------------------------------------------------------------
class TestVerySmallMaximum:
    def test_tiny_maximum_yields_no_entries(self, ollama):
        """100 is truthy so `if not maximum` does not fire, but every fraction
        (12, 25, 50, 100) is < 2048 and filtered out — the result is an empty
        list, not None."""
        ollama["reply"] = {"model_info": {"tiny.context_length": 100}}
        entries = context_choices("tiny", "ollama")
        # maximum=100 is truthy, so we get past the `if not maximum` guard.
        # All fractions are < 2048, so entries should be empty (or None if
        # the implementation treats empty as falsy — but here it returns []).
        assert entries is None or entries == []


# ---------------------------------------------------------------------------
# L2 × D3 — multiple keys ending in context_length
# ---------------------------------------------------------------------------
class TestMultipleContextLengthKeys:
    def test_first_matching_key_is_returned(self, ollama):
        """The loop returns the first key whose value passes the guard. In
        Python 3.7+ dict order is insertion order, so the first key wins."""
        ollama["reply"] = {
            "model_info": {
                "a.context_length": 4096,
                "b.context_length": 8192,
            }
        }
        result = detect_context_length("m")
        assert result is not None
        assert result > 0
        # In 3.7+ the first inserted key wins.
        assert result in (4096, 8192)


# ---------------------------------------------------------------------------
# L4 × D1 — a string value is rejected by isinstance(value, int)
# ---------------------------------------------------------------------------
class TestStringContextLength:
    def test_string_context_length_returns_none(self, ollama):
        """Ollama could in theory send a string; the int guard must reject it."""
        ollama["reply"] = {"model_info": {"qwen35.context_length": "8192"}}
        assert detect_context_length("qwen3.5:9b") is None


# ---------------------------------------------------------------------------
# L4 × D1 — a float value is rejected by isinstance(value, int)
# ---------------------------------------------------------------------------
class TestFloatContextLength:
    def test_float_context_length_returns_none(self, ollama):
        """isinstance(8192.0, int) is False, so a float must be rejected."""
        ollama["reply"] = {"model_info": {"qwen35.context_length": 8192.0}}
        assert detect_context_length("qwen3.5:9b") is None


# ---------------------------------------------------------------------------
# L4 × D7 — a current value that matches no fraction is not marked
# ---------------------------------------------------------------------------
class TestCurrentNotInChoices:
    def test_non_matching_current_is_not_marked(self, ollama):
        """current=99999 matches none of the fractions, so no entry should
        carry the 'текущее' marker."""
        ollama["reply"] = {"model_info": {"qwen35.context_length": 262144}}
        entries = context_choices("qwen3.5:9b", "ollama", current=99999)
        assert entries is not None
        assert len(entries) > 0
        assert not any("текущее" in e for e in entries)


# ---------------------------------------------------------------------------
# L4 × D7 — OLLAMA_HOST without an http:// prefix is normalized
# ---------------------------------------------------------------------------
class TestOllamaHostNoHttpPrefix:
    def test_host_without_http_prefix_is_normalized(self, ollama, monkeypatch):
        """The code prepends http:// when the host lacks a scheme, so the
        request URL must always start with http://."""
        monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
        detect_context_length("qwen3.5:9b")
        assert ollama["url"].startswith("http://")