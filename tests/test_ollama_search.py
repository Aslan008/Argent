"""Ollama's hosted search as a sixth engine.

Added on evidence, the way Brave was. Measured across four real technical
queries: 20 URLs returned, 4 of them already present in the other five engines
— 80% unique. Union is what lifts recall, and recall is the one thing the
cross-encoder downstream cannot repair.

Two things had to be measured rather than assumed, and both changed the design:
its results carry the WHOLE PAGE (one was 223,466 characters), and of the
operators the model may write it honours only `site:`.
"""

import pytest

from src.research import search as S
from src.research.search import (
    OLLAMA_SNIPPET_CHARS, _ollama_search, active_engines, adapt_query_for_engine,
    engine_labels,
)


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


@pytest.fixture
def api(monkeypatch):
    """Captures the outgoing request; returns whatever the test queues."""
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["url"] = url
        sent["headers"] = headers or {}
        sent["body"] = json or {}
        return _Response(sent.get("reply", {"results": []}))

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("config.get_ollama_api_key", lambda: "test-key")
    return sent


class TestOffByDefault:
    def test_no_key_means_no_request(self, monkeypatch):
        """The keyless zero-setup default must stay untouched."""
        monkeypatch.setattr("config.get_ollama_api_key", lambda: "")
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: pytest.fail("must not call out without a key"))
        assert _ollama_search("anything") == []

    def test_the_engine_joins_only_when_configured(self, monkeypatch):
        monkeypatch.setattr("config.get_brave_api_key", lambda: "")
        monkeypatch.setattr("config.get_ollama_api_key", lambda: "")
        assert "Ollama" not in engine_labels()
        monkeypatch.setattr("config.get_ollama_api_key", lambda: "k")
        assert "Ollama" in engine_labels()

    def test_independent_indexes_run_first(self, monkeypatch):
        """Order decides who survives the result cap, and these two carry what
        the keyless baseline cannot reach."""
        monkeypatch.setattr("config.get_brave_api_key", lambda: "k")
        monkeypatch.setattr("config.get_ollama_api_key", lambda: "k")
        assert engine_labels()[:2] == ["Brave", "Ollama"]


class TestRequest:
    def test_the_key_travels_as_a_bearer_token(self, api):
        _ollama_search("query")
        assert api["headers"]["Authorization"] == "Bearer test-key"
        assert api["url"] == "https://ollama.com/api/web_search"

    def test_the_documented_maximum_is_respected(self, api):
        """The API caps max_results at 10; asking for more is an error, not a
        bigger answer."""
        _ollama_search("q", limit=50)
        assert api["body"]["max_results"] == 10

    def test_an_unset_limit_falls_back_to_the_default(self, api):
        """0 means "not specified" here, as it does across the other engines —
        not "return nothing"."""
        _ollama_search("q", limit=0)
        assert api["body"]["max_results"] == 5

    def test_a_negative_limit_cannot_produce_an_invalid_request(self, api):
        _ollama_search("q", limit=-3)
        assert api["body"]["max_results"] == 1


class TestResults:
    def test_the_shape_matches_the_other_engines(self, api):
        api["reply"] = {"results": [
            {"title": "T", "url": "https://x/1", "content": "текст"}]}
        out = _ollama_search("q")
        assert out == [{"title": "T", "url": "https://x/1",
                        "snippet": "текст", "source": "ollama"}]

    def test_a_whole_page_is_capped(self, api):
        """Real measurement: one result carried 223,466 characters. A handful of
        those would swamp the reranker and any local model's context."""
        api["reply"] = {"results": [
            {"title": "T", "url": "https://x/1", "content": "x" * 223466}]}
        snippet = _ollama_search("q")[0]["snippet"]
        assert len(snippet) <= OLLAMA_SNIPPET_CHARS + 2
        assert snippet.endswith("…")

    def test_short_content_is_left_alone(self, api):
        api["reply"] = {"results": [{"title": "T", "url": "u", "content": "коротко"}]}
        assert _ollama_search("q")[0]["snippet"] == "коротко"

    def test_a_missing_title_does_not_produce_an_empty_line(self, api):
        api["reply"] = {"results": [{"url": "u", "content": "c"}]}
        assert _ollama_search("q")[0]["title"] == "(no title)"

    def test_an_empty_answer(self, api):
        api["reply"] = {"results": []}
        assert _ollama_search("q") == []


class TestOperators:
    def test_site_survives_because_it_is_honoured(self):
        """Probed live: 5/5 results came from the named domain."""
        out = adapt_query_for_engine("addressables site:docs.unity3d.com", _ollama_search)
        assert "site:docs.unity3d.com" in out

    def test_what_it_ignores_is_stripped(self):
        """Probed live: the excluded domain still came back and filetype:pdf
        returned nothing of that type. Passing an ignored operator through is
        worse than removing it — the model reads the answer as if it applied."""
        out = adapt_query_for_engine(
            "addressables site:docs.unity3d.com filetype:pdf -forum", _ollama_search)
        assert out == "addressables site:docs.unity3d.com"

    def test_other_engines_are_unaffected(self):
        q = "addressables filetype:pdf -forum"
        assert adapt_query_for_engine(q, S._ddg_search) == q


class TestFailureIsIsolated:
    def test_an_http_error_does_not_take_the_federation_down(self, monkeypatch):
        """Engine failures are isolated upstream; this asserts the engine raises
        cleanly rather than returning something malformed."""
        monkeypatch.setattr("config.get_ollama_api_key", lambda: "k")
        import requests
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: _Response({}, status=429))
        with pytest.raises(RuntimeError):
            _ollama_search("q")
