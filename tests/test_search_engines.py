"""New search engines: GitHub issues (keyless) and Brave (optional, keyed).

Network boundaries are monkeypatched, so nothing here touches the internet.
"""

import pytest

import config
from src.research import search as search_mod
from src.research.search import DEFAULT_ENGINES, active_engines


@pytest.fixture
def no_brave_key(monkeypatch):
    monkeypatch.setattr(config, "get_brave_api_key", lambda: "")


@pytest.fixture
def brave_key(monkeypatch):
    monkeypatch.setattr(config, "get_brave_api_key", lambda: "test-key")


class TestGithubEngine:
    def test_parses_issues(self, monkeypatch):
        payload = {"items": [{
            "title": "NullReferenceException in Addressables",
            "html_url": "https://github.com/unity/addressables/issues/42",
            "body": "Repro:\r\n  load asset   async\n\nStack trace here",
            "state": "closed",
            "comments": 17,
            "repository_url": "https://api.github.com/repos/unity/addressables",
        }]}
        monkeypatch.setattr(search_mod, "_get_json", lambda url, params, timeout=10: payload)

        out = search_mod._github_search("NullReferenceException Addressables")
        assert len(out) == 1
        r = out[0]
        assert r["source"] == "github"
        assert r["url"].endswith("/issues/42")
        assert "unity/addressables" in r["snippet"]
        assert "closed" in r["snippet"] and "17 comments" in r["snippet"]
        assert "\r" not in r["snippet"] and "  " not in r["snippet"]   # whitespace collapsed

    def test_asks_for_reaction_sorted_results(self, monkeypatch):
        seen = {}

        def fake_get(url, params, timeout=10):
            seen.update({"url": url, "params": params})
            return {"items": []}

        monkeypatch.setattr(search_mod, "_get_json", fake_get)
        search_mod._github_search("q", limit=4)
        assert "search/issues" in seen["url"]
        assert seen["params"]["sort"] == "reactions"
        assert seen["params"]["per_page"] == 4

    def test_missing_body_is_safe(self, monkeypatch):
        payload = {"items": [{"title": "t", "html_url": "u", "body": None,
                              "state": "open", "comments": 0, "repository_url": None}]}
        monkeypatch.setattr(search_mod, "_get_json", lambda *a, **k: payload)
        assert search_mod._github_search("q")[0]["title"] == "t"

    def test_is_in_the_keyless_baseline(self):
        assert search_mod._github_search in DEFAULT_ENGINES


class TestBraveEngine:
    def test_returns_nothing_without_a_key(self, no_brave_key, monkeypatch):
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("must not call the API"))
        assert search_mod._brave_search("q") == []

    def test_parses_web_and_discussions(self, brave_key, monkeypatch):
        payload = {
            "web": {"results": [
                {"title": "Doc page", "url": "https://docs/x",
                 "description": "a <strong>match</strong> here"},
            ]},
            "discussions": {"results": [
                {"title": "Forum thread", "url": "https://forum/y", "description": "people discuss"},
            ]},
        }

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return payload

        captured = {}

        def fake_get(url, params=None, headers=None, timeout=None):
            captured.update({"url": url, "params": params, "headers": headers})
            return _Resp()

        import requests
        monkeypatch.setattr(requests, "get", fake_get)

        out = search_mod._brave_search("q", limit=5)
        assert captured["headers"]["X-Subscription-Token"] == "test-key"
        assert [r["source"] for r in out] == ["brave", "brave-discussions"]
        assert out[0]["snippet"] == "a match here"      # html stripped

    def test_respects_the_limit(self, brave_key, monkeypatch):
        payload = {"web": {"results": [
            {"title": f"t{i}", "url": f"https://x/{i}", "description": ""} for i in range(10)
        ]}}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return payload

        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
        assert len(search_mod._brave_search("q", limit=3)) == 3


class TestBraveRateLimit:
    """The free tier is ~1 query/second and deep research fires several in a
    row. Without pacing the burst gets 429s, and since engine failures are
    isolated, Brave would vanish from the federation silently."""

    def test_back_to_back_calls_are_paced(self, brave_key, monkeypatch):
        import time as _t
        from src.research import search as s

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return {"web": {"results": []}}

        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
        monkeypatch.setattr(s, "_BRAVE_MIN_INTERVAL", 0.2)
        monkeypatch.setattr(s, "_brave_last_call", 0.0)

        start = _t.monotonic()
        s._brave_search("a")
        s._brave_search("b")
        assert _t.monotonic() - start >= 0.2      # the second call waited

    def test_pacing_does_not_apply_without_a_key(self, no_brave_key, monkeypatch):
        from src.research import search as s
        monkeypatch.setattr(s, "_brave_pace",
                            lambda: pytest.fail("must not pace when disabled"))
        assert s._brave_search("q") == []


class TestActiveEngines:
    def test_keyless_default_is_unchanged(self, no_brave_key):
        assert active_engines() == DEFAULT_ENGINES
        assert search_mod._brave_search not in active_engines()

    def test_key_adds_brave_without_removing_ddg(self, brave_key):
        engines = active_engines()
        # Additive by design: two independent indexes widen recall, and DDG
        # keeps serving when the Brave quota runs out.
        assert search_mod._brave_search in engines
        assert search_mod._ddg_search in engines
        assert engines[0] is search_mod._brave_search

    def test_config_failure_falls_back_to_keyless(self, monkeypatch):
        monkeypatch.setattr(config, "get_brave_api_key",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert active_engines() == DEFAULT_ENGINES


class TestQueryGeneration:
    """Query wording decides what CAN be retrieved — no reranker recovers from
    a query that never surfaced the page. These rules are the actual lever on
    weak local models, so they're pinned."""

    def _prompt(self, monkeypatch):
        import deep_research
        captured = {}
        monkeypatch.setattr(deep_research, "_call_llm_sync",
                            lambda prompt, **kw: captured.setdefault("p", prompt) or '["a"]')
        deep_research._generate_queries("Unity падает с NullReferenceException")
        return captured["p"]

    def test_forces_english_queries(self, monkeypatch):
        prompt = self._prompt(monkeypatch)
        assert "ENGLISH" in prompt

    def test_teaches_verbatim_error_quoting(self, monkeypatch):
        prompt = self._prompt(monkeypatch)
        assert "VERBATIM" in prompt and "double quotes" in prompt
        assert "line numbers" in prompt          # drop the variable parts

    def test_requires_distinct_angles_including_docs_and_issues(self, monkeypatch):
        prompt = self._prompt(monkeypatch)
        assert "site:" in prompt                 # vendor doc targeting
        assert "issue" in prompt.lower()
        assert "do not paraphrase" in prompt.lower()

    def test_objective_is_still_passed_through(self, monkeypatch):
        prompt = self._prompt(monkeypatch)
        assert "NullReferenceException" in prompt

    def test_falls_back_to_the_objective_on_garbage(self, monkeypatch):
        import deep_research
        monkeypatch.setattr(deep_research, "_call_llm_sync", lambda *a, **k: "not json")
        assert deep_research._generate_queries("topic") == ["topic"]


class TestDoctorDiscoverability:
    """A config-only option nobody can find is not a feature — /doctor names
    the active engines and how to add the optional one."""

    def test_lists_keyless_engines_and_hints_at_brave(self, no_brave_key):
        import doctor
        status, detail = doctor._check_web_search()
        assert "DuckDuckGo" in detail and "GitHub" in detail
        assert "brave_api_key" in detail

    def test_no_hint_once_brave_is_active(self, brave_key):
        import doctor
        _, detail = doctor._check_web_search()
        assert "Brave" in detail and "brave_api_key" not in detail


class TestBraveConfig:
    def test_default_is_empty(self, monkeypatch):
        store = {}
        monkeypatch.setattr(config, "load_config", lambda: store)
        assert config.get_brave_api_key() == ""

    def test_set_and_get(self, monkeypatch):
        store = {}
        monkeypatch.setattr(config, "load_config", lambda: store)
        monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
        config.set_brave_api_key("  abc123  ")
        assert config.get_brave_api_key() == "abc123"
