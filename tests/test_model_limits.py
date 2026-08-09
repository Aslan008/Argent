"""Offering a context window that means something for THIS model.

It used to be picked from a fixed list — 2048 / 4096 / 8192 — with no relation
to the model in front of you: too small trims the conversation for no reason,
too large overflows, which is exactly what happens when a chat moves from a
cloud model to a local one.

Ollama knows the real number. It is offered as a reference and never applied:
qwen3.5:9b reports 262144, and loading a 9B at 262k asks for memory the machine
does not have. The user still chooses.
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


class TestDetection:
    def test_the_real_maximum_is_read(self, ollama):
        assert detect_context_length("qwen3.5:9b") == 262144
        assert ollama["asked"] == "qwen3.5:9b"

    def test_the_key_is_matched_by_suffix_not_by_architecture(self, ollama):
        """Every architecture names it differently — llama.context_length,
        qwen35.context_length — so the prefix cannot be enumerated."""
        ollama["reply"] = {"model_info": {"gemma3.context_length": 8192}}
        assert detect_context_length("gemma:4b") == 8192

    def test_a_model_without_the_field(self, ollama):
        ollama["reply"] = {"model_info": {"general.architecture": "x"}}
        assert detect_context_length("weird") is None

    def test_a_cloud_provider_is_not_guessed_at(self, monkeypatch):
        """Guessing a window from a model name would be fiction, and a wrong
        one overflows every request."""
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: pytest.fail("must not ask Ollama about a cloud model"))
        assert detect_context_length("gpt-x", "openrouter") is None

    def test_no_model(self, ollama):
        assert detect_context_length("") is None

    def test_a_dead_server_does_not_break_the_menu(self, ollama):
        """The caller is an interactive menu; failing fast beats hanging."""
        ollama["reply"] = OSError("connection refused")
        assert detect_context_length("qwen3.5:9b") is None

    def test_a_nonsense_value_is_ignored(self, ollama):
        ollama["reply"] = {"model_info": {"x.context_length": 0}}
        assert detect_context_length("m") is None


class TestChoices:
    def test_fractions_of_the_real_maximum(self, ollama):
        """8192 means something completely different on a 262k model than on an
        8k one; what matters is how much of the model you are using."""
        entries = context_choices("qwen3.5:9b", "ollama")
        assert [e.split()[0] for e in entries] == ["32768", "65536", "131072", "262144"]
        assert "максимум модели" in entries[-1]

    def test_the_current_value_is_marked(self, ollama):
        entries = context_choices("qwen3.5:9b", "ollama", current=65536)
        assert any("текущее" in e and e.startswith("65536") for e in entries)

    def test_a_small_model_does_not_offer_unusable_sizes(self, ollama):
        """Below ~2k a coding conversation cannot hold one file plus the
        prompt, so offering it would only look like a choice."""
        ollama["reply"] = {"model_info": {"llama.context_length": 8192}}
        entries = context_choices("tiny", "ollama")
        assert all(int(e.split()[0]) >= 2048 for e in entries)

    def test_duplicates_collapse(self, ollama):
        ollama["reply"] = {"model_info": {"llama.context_length": 4096}}
        values = [e.split()[0] for e in context_choices("m", "ollama")]
        assert len(values) == len(set(values))

    def test_none_when_the_limit_is_unknown(self, ollama):
        """The caller then keeps its static list rather than showing nothing."""
        ollama["reply"] = {"model_info": {}}
        assert context_choices("m", "ollama") is None
