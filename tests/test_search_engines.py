"""New search engines: GitHub issues (keyless) and Brave (optional, keyed).

Network boundaries are monkeypatched, so nothing here touches the internet.
"""

import pytest

import config
from src.research import search as search_mod
from src.research.search import DEFAULT_ENGINES, active_engines, meta_search


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


class TestOperatorAdaptation:
    """One query string reaches five engines that speak different query
    languages. Measured before this existed: a query carrying `site:` returned
    5 results from DuckDuckGo and Brave and ZERO from Wikipedia and GitHub,
    which silently removed them from the federation."""

    def test_google_operators_survive_for_engines_that_honour_them(self):
        q = 'site:forum.unity.com "Addressables" leak -tutorial'
        out = search_mod.adapt_query_for_engine(q, search_mod._ddg_search)
        assert "site:forum.unity.com" in out
        assert '"Addressables"' in out and "-tutorial" in out

    def test_site_is_stripped_for_wikipedia(self):
        out = search_mod.adapt_query_for_engine(
            'site:forum.unity.com "Addressables" leak', search_mod._wikipedia_search)
        assert "site:" not in out
        assert '"Addressables"' in out and "leak" in out     # the real terms stay

    def test_stackoverflow_gets_plain_text_only(self):
        """Its API takes filters as parameters; operators inside q are matched
        as literal text and wreck the search."""
        out = search_mod.adapt_query_for_engine(
            'site:stackoverflow.com filetype:pdf async deadlock -blog',
            search_mod._stackoverflow_search)
        assert out == "async deadlock"

    def test_github_keeps_exclusion_but_drops_site(self):
        out = search_mod.adapt_query_for_engine(
            "site:github.com memory leak -docs", search_mod._github_search)
        assert "site:" not in out and "-docs" in out

    def test_quoted_phrases_always_pass_through(self):
        q = '"exact error text here"'
        for engine in (search_mod._ddg_search, search_mod._wikipedia_search,
                       search_mod._stackoverflow_search, search_mod._github_search):
            assert '"exact error text here"' in search_mod.adapt_query_for_engine(q, engine)

    def test_hyphenated_words_are_not_mistaken_for_exclusion(self):
        out = search_mod.adapt_query("well-known cross-platform issue", set())
        assert out == "well-known cross-platform issue"

    def test_unknown_engine_gets_the_query_unchanged(self):
        assert search_mod.adapt_query_for_engine("site:x.com q", lambda query, limit: []) \
            == "site:x.com q"

    def test_engine_is_skipped_when_nothing_survives(self):
        """A query that is nothing but unusable operators must not be sent as an
        empty string — that would return arbitrary results."""
        calls = []

        def spy(query, limit):
            calls.append(query)
            return []

        spy.__name__ = "_stackoverflow_search"
        meta_search("site:example.com", engines=[spy])
        assert calls == []

    def test_meta_search_adapts_per_engine(self):
        seen = {}

        def make(name):
            def engine(query, limit):
                seen[name] = query
                return []
            engine.__name__ = name
            return engine

        meta_search('site:unity.com "leak" -ads',
                    engines=[make("_ddg_search"), make("_stackoverflow_search")])
        assert "site:unity.com" in seen["_ddg_search"]
        assert seen["_stackoverflow_search"] == '"leak"'


class TestStackExchangeSiteRouting:
    """StackExchange is a network, and the same product splits across sites by
    question type. Measured: "Unity Addressables memory leak" returns 0 on
    stackoverflow and 2 on gamedev, while "Unity shader graph vertex
    displacement" returns 1 on stackoverflow and 0 on gamedev — so picking one
    site loses the other. Both are asked and merged."""

    def test_stackoverflow_is_always_asked(self):
        assert search_mod._se_sites("python asyncio gather")[0] == "stackoverflow"

    @pytest.mark.parametrize("query,expected", [
        ("Unity Addressables memory leak", "gamedev"),
        ("godot collider not triggering", "gamedev"),
        ("integral of e^-x^2 derivative", "math"),
        ("nginx reverse proxy 502", "serverfault"),
        ("ubuntu apt-get broken packages", "askubuntu"),
    ])
    def test_domain_site_is_added(self, query, expected):
        assert search_mod._se_sites(query) == ["stackoverflow", expected]

    def test_generic_query_asks_one_site_only(self):
        """Quota discipline: no cue means no second call."""
        assert search_mod._se_sites("how to reverse a list") == ["stackoverflow"]

    def test_at_most_one_extra_site(self):
        # A query hitting several cue sets must still cost only two calls.
        sites = search_mod._se_sites("unity shader nginx integral ubuntu")
        assert len(sites) == 2

    def test_results_from_both_sites_are_merged_and_labelled(self, monkeypatch):
        def fake_get(url, params, timeout=10):
            return {"items": [{"title": f"from {params['site']}", "link": f"http://{params['site']}",
                               "score": 5, "is_answered": True}]}

        monkeypatch.setattr(search_mod, "_get_json", fake_get)
        out = search_mod._stackoverflow_search("unity shader bug", limit=3)
        assert [r["source"] for r in out] == ["stackexchange:stackoverflow",
                                              "stackexchange:gamedev"]
        assert "gamedev" in out[1]["snippet"]        # the site is part of the signal

    def test_one_site_failing_does_not_lose_the_other(self, monkeypatch):
        def fake_get(url, params, timeout=10):
            if params["site"] == "stackoverflow":
                raise RuntimeError("503")
            return {"items": [{"title": "survivor", "link": "http://x",
                               "score": 1, "is_answered": True}]}

        monkeypatch.setattr(search_mod, "_get_json", fake_get)
        out = search_mod._stackoverflow_search("unity collider bug", limit=3)
        assert [r["title"] for r in out] == ["survivor"]


class TestOperatorTranslation:
    """Where an engine has a native equivalent, translate rather than strip.
    The model writes ONE query for the whole federation, so it cannot express a
    single engine's dialect — only the adapter can."""

    def test_intitle_becomes_githubs_in_title(self):
        out = search_mod.adapt_query_for_engine(
            'intitle:"memory leak" Addressables', search_mod._github_search)
        assert out == "memory leak in:title Addressables"

    def test_intitle_stays_native_on_wikipedia(self):
        out = search_mod.adapt_query_for_engine(
            'intitle:"memory leak" x', search_mod._wikipedia_search)
        assert 'intitle:"memory leak"' in out

    def test_quoted_operator_values_are_not_split(self):
        out = search_mod.adapt_query_for_engine(
            'site:"docs.unity3d.com" leak', search_mod._stackoverflow_search)
        assert "docs.unity3d.com" not in out and out == "leak"

    def test_stackoverflow_lifts_intitle_into_the_title_parameter(self, monkeypatch):
        """Its API filters by parameter; searching for the literal text
        'intitle:' returns nothing."""
        seen = {}

        def fake_get(url, params, timeout=10):
            seen.update(params)
            return {"items": []}

        monkeypatch.setattr(search_mod, "_get_json", fake_get)
        search_mod._stackoverflow_search('intitle:"memory leak" async', limit=3)
        assert seen["title"] == "memory leak"
        assert seen["q"] == "async"
        assert "intitle" not in seen.get("q", "")

    def test_stackoverflow_without_intitle_uses_q_only(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(search_mod, "_get_json",
                            lambda url, params, timeout=10: seen.update(params) or {"items": []})
        search_mod._stackoverflow_search("async deadlock", limit=3)
        assert seen["q"] == "async deadlock" and "title" not in seen

    def test_stackoverflow_skips_an_empty_search(self, monkeypatch):
        monkeypatch.setattr(search_mod, "_get_json",
                            lambda *a, **k: pytest.fail("must not call the API"))
        assert search_mod._stackoverflow_search("   ", limit=3) == []


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
