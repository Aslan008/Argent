"""Docker-free meta-search: merge/dedup logic + per-engine JSON parsers.

Engines are injected (meta_search) or their network boundary monkeypatched
(_get_json / DDGS), so nothing here touches the network.
"""

from src.research import search as search_mod
from src.research.search import meta_search, _norm_url


def _engine(items):
    """Build a fake engine returning fixed items regardless of query/limit."""
    return lambda query, limit: list(items)


class TestNormUrl:
    def test_trailing_slash_and_case(self):
        assert _norm_url("http://Example.com/Path/") == _norm_url("http://example.com/Path")

    def test_fragment_dropped(self):
        assert _norm_url("http://x.com/a#frag") == "x.com/a"


class TestMetaSearch:
    def test_round_robin_interleaves_sources(self):
        a = _engine([{"url": "http://a/1"}, {"url": "http://a/2"}, {"url": "http://a/3"}])
        b = _engine([{"url": "http://b/1"}, {"url": "http://b/2"}])
        out = meta_search("q", max_results=8, engines=[a, b])
        assert [r["url"] for r in out] == [
            "http://a/1", "http://b/1", "http://a/2", "http://b/2", "http://a/3",
        ]

    def test_dedup_by_normalised_url(self):
        a = _engine([{"url": "http://x.com/p"}])
        b = _engine([{"url": "http://x.com/p/"}])  # same page, trailing slash
        out = meta_search("q", max_results=8, engines=[a, b])
        assert len(out) == 1

    def test_respects_max_results(self):
        a = _engine([{"url": f"http://a/{i}"} for i in range(10)])
        out = meta_search("q", max_results=3, engines=[a])
        assert len(out) == 3

    def test_failing_engine_is_isolated(self):
        def boom(query, limit):
            raise RuntimeError("rate limited")
        good = _engine([{"url": "http://ok/1"}])
        out = meta_search("q", max_results=8, engines=[boom, good])
        assert [r["url"] for r in out] == ["http://ok/1"]

    def test_all_empty_returns_empty(self):
        out = meta_search("q", engines=[_engine([]), _engine([])])
        assert out == []


class TestEngineParsers:
    def test_wikipedia(self, monkeypatch):
        canned = {"query": {"search": [
            {"title": "Python (programming language)",
             "snippet": 'Python is a <span class="m">high-level</span> language'},
        ]}}
        monkeypatch.setattr(search_mod, "_get_json", lambda url, params, timeout=10: canned)
        out = search_mod._wikipedia_search("python", 2)
        assert out[0]["source"] == "wikipedia"
        assert out[0]["url"] == "https://en.wikipedia.org/wiki/Python_%28programming_language%29"
        assert "<span" not in out[0]["snippet"] and "high-level" in out[0]["snippet"]

    def test_stackoverflow(self, monkeypatch):
        canned = {"items": [
            {"title": "How to sort a list in &lt;Python&gt;?",
             "link": "http://so/1", "score": 42, "is_answered": True},
        ]}
        monkeypatch.setattr(search_mod, "_get_json", lambda url, params, timeout=10: canned)
        out = search_mod._stackoverflow_search("sort list", 1)
        assert out[0]["title"] == "How to sort a list in <Python>?"  # HTML-unescaped
        assert out[0]["url"] == "http://so/1"
        assert out[0]["snippet"] == "[42 votes, answered]"
        assert out[0]["source"] == "stackoverflow"

    def test_duckduckgo(self, monkeypatch):
        class FakeDDGS:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def text(self, query, max_results):
                return [{"title": "T", "href": "http://a", "body": "b"}]
        monkeypatch.setattr(search_mod, "DDGS", FakeDDGS)
        out = search_mod._ddg_search("q", 3)
        assert out[0] == {"title": "T", "url": "http://a", "snippet": "b", "source": "duckduckgo"}


class TestSearchWebIntegration:
    def test_formats_merged_results(self, monkeypatch):
        import tools.web_tools as wt
        monkeypatch.setattr("src.research.search.meta_search",
                            lambda query, max_results=5: [
                                {"title": "A", "url": "http://a", "snippet": "sa", "source": "wikipedia"},
                                {"title": "B", "url": "http://b", "snippet": "", "source": "duckduckgo"},
                            ])
        out = wt.search_web("q")
        assert "Search results for: 'q'" in out
        assert "[wikipedia]" in out and "http://a" in out
        assert "SYSTEM REMINDER" in out

    def test_no_results(self, monkeypatch):
        import tools.web_tools as wt
        monkeypatch.setattr("src.research.search.meta_search", lambda query, max_results=5: [])
        assert "No results found" in wt.search_web("q")
