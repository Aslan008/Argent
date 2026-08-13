"""LMTrust blind-spot tests for deep_research.py.

Tests the PURE helper functions in isolation by monkeypatching the network /
LLM boundaries (``_call_llm_sync``, ``fetch_page``, ``meta_search``) so no real
API call is ever made.

Layers applied (per LMTrust decision tree):
  L0 Smoke, L1 Contract, L2 Boundary, L4 Adversarial, L4b Fallback,
  L9 Negative Space.
Directions: D1 Math/Numeric, D4 Error Handling, D7 API Contract, D8 Security,
D10 Invariant.
"""

import json
import sys
from unittest.mock import MagicMock

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _import_deep_research():
    """Ensure deep_research is importable from the project root."""
    import deep_research as dr
    return dr


# ══════════════════════════════════════════════════════════════════════════════
# 1. _language_rule — L0/L1/L2/L9
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageRule:
    """``_language_rule`` builds the LANGUAGE clause from configured languages."""

    def test_english_only_returns_english_clause(self, monkeypatch):
        import deep_research as dr
        import config
        monkeypatch.setattr(config, "get_search_languages", lambda: ["en"])
        result = dr._language_rule()
        assert "ENGLISH" in result

    def test_multi_language_includes_both_names(self, monkeypatch):
        import deep_research as dr
        import config
        monkeypatch.setattr(config, "get_search_languages", lambda: ["en", "ru"])
        result = dr._language_rule()
        assert "English" in result
        assert "Russian" in result

    def test_single_non_english_uses_local_language(self, monkeypatch):
        import deep_research as dr
        import config
        monkeypatch.setattr(config, "get_search_languages", lambda: ["fr"])
        result = dr._language_rule()
        assert "French" in result
        # 'French' is the non-English name, so it appears in the other_list.
        assert "write the query in French" in result

    def test_all_english_non_singleton_uses_local_language_fallback(self, monkeypatch):
        """When every name is English but langs != ['en'], the 'the local
        language' fallback kicks in because others is empty."""
        import deep_research as dr
        import config
        monkeypatch.setattr(config, "get_search_languages", lambda: ["en", "en"])
        result = dr._language_rule()
        assert "the local language" in result

    def test_unknown_language_code_used_as_name(self, monkeypatch):
        import deep_research as dr
        import config
        monkeypatch.setattr(config, "get_search_languages", lambda: ["xx"])
        result = dr._language_rule()
        # Unknown codes are used verbatim as the name.
        assert "xx" in result


# ══════════════════════════════════════════════════════════════════════════════
# 2. _smart_truncate — L0/L1/L2/L4
# ══════════════════════════════════════════════════════════════════════════════

class TestSmartTruncate:
    """``_smart_truncate`` breaks at sentence boundaries when possible."""

    def test_short_text_returned_unchanged(self):
        import deep_research as dr
        text = "Short text."
        assert dr._smart_truncate(text, 100) == text

    def test_breaks_at_sentence_boundary(self):
        import deep_research as dr
        text = "First sentence here. Second sentence. Third one."
        result = dr._smart_truncate(text, 25)
        # The ". " at index 19 is above 60% of 25 (15), so it breaks there.
        assert result.endswith(".")
        assert len(result) <= 25

    def test_no_sentence_boundary_truncates_at_max(self):
        import deep_research as dr
        text = "a" * 200
        result = dr._smart_truncate(text, 50)
        assert len(result) <= 50
        assert result == "a" * 50

    def test_exclamation_boundary(self):
        import deep_research as dr
        text = "Watch out! " + "x" * 100
        result = dr._smart_truncate(text, 50)
        # Should break at the "! " boundary if it's above 60% of max.
        assert "!" in result

    def test_boundary_below_60_percent_falls_through(self):
        import deep_research as dr
        # Place a ". " very early (below 60% of max_chars) so it should NOT
        # be used as the break point; the function falls through to the next
        # separator and ultimately returns the full max_chars truncation.
        text = "Hi. " + "b" * 200
        max_chars = 50
        result = dr._smart_truncate(text, max_chars)
        # The early ". " at index 3 is below 60% (30), so it must not be used
        # as the break point — the result is the full truncation, NOT "Hi.".
        assert result != "Hi."
        assert len(result) == max_chars


# ══════════════════════════════════════════════════════════════════════════════
# 3. _generate_queries — L0/L1/L4b
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateQueries:
    """``_generate_queries`` parses an LLM JSON array into a list of strings."""

    def test_valid_json_array_returns_list(self, monkeypatch):
        import deep_research as dr
        queries = ["query one", "query two", "query three", "query four", "query five"]
        monkeypatch.setattr(dr, "_call_llm_sync",
                            lambda *a, **k: json.dumps(queries))
        result = dr._generate_queries("test objective")
        assert isinstance(result, list)
        assert all(isinstance(q, str) for q in result)
        assert "query one" in result

    def test_invalid_json_returns_fallback(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "not json at all")
        result = dr._generate_queries("my objective")
        # Fallback is [objective] when parsing fails.
        assert isinstance(result, list)
        assert result == ["my objective"]

    def test_empty_array_returns_fallback(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "[]")
        result = dr._generate_queries("fallback objective")
        # Empty array -> parse_query_list returns [] -> fallback to [objective].
        assert result == ["fallback objective"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. _find_gaps — L0/L1/L2/L4b
# ══════════════════════════════════════════════════════════════════════════════

class TestFindGaps:
    """``_find_gaps`` returns follow-up queries from an LLM JSON array."""

    def test_valid_json_array_returns_list(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync",
                            lambda *a, **k: json.dumps(["gap one", "gap two"]))
        result = dr._find_gaps("objective", "some notes")
        assert isinstance(result, list)
        assert "gap one" in result

    def test_focus_appears_in_prompt(self, monkeypatch):
        import deep_research as dr
        captured = {}

        def fake_llm(prompt, **kw):
            captured["prompt"] = prompt
            return "[]"

        monkeypatch.setattr(dr, "_call_llm_sync", fake_llm)
        dr._find_gaps("objective", "notes", focus="performance")
        assert "performance" in captured["prompt"]

    def test_context_appears_in_prompt(self, monkeypatch):
        import deep_research as dr
        captured = {}

        def fake_llm(prompt, **kw):
            captured["prompt"] = prompt
            return "[]"

        monkeypatch.setattr(dr, "_call_llm_sync", fake_llm)
        dr._find_gaps("objective", "notes", context="already known facts")
        assert "already known facts" in captured["prompt"]

    def test_empty_array_returns_empty_list(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "[]")
        result = dr._find_gaps("objective", "notes")
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 5. _extract_info — L0/L1/L4b
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractInfo:
    """``_extract_info`` returns extracted text or empty string for NOTHING."""

    def test_returns_extracted_text(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync",
                            lambda *a, **k: "Some useful fact [1].")
        result = dr._extract_info("objective", "fragments")
        assert result == "Some useful fact [1]."

    def test_nothing_returns_empty_string(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "NOTHING")
        result = dr._extract_info("objective", "fragments")
        assert result == ""

    def test_quoted_nothing_returns_empty_string(self, monkeypatch):
        import deep_research as dr
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: '"NOTHING"')
        result = dr._extract_info("objective", "fragments")
        assert result == ""

    def test_focus_appears_in_prompt(self, monkeypatch):
        import deep_research as dr
        captured = {}

        def fake_llm(prompt, **kw):
            captured["prompt"] = prompt
            return "some info"

        monkeypatch.setattr(dr, "_call_llm_sync", fake_llm)
        dr._extract_info("objective", "fragments", focus="memory leaks")
        assert "memory leaks" in captured["prompt"]


# ══════════════════════════════════════════════════════════════════════════════
# 6. _fetch_with_retry — L0/L1/L2/L4
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchWithRetry:
    """``_fetch_with_retry`` retries network errors but not HTTP status errors."""

    def test_ok_returns_immediately(self, monkeypatch):
        import deep_research as dr
        import src.research.fetch as fetch_mod
        monkeypatch.setattr(fetch_mod, "fetch_page",
                            lambda url: {"ok": True, "text": "content"})
        result = dr._fetch_with_retry("http://example.com")
        assert result["ok"] is True
        assert result["text"] == "content"

    def test_timeout_retries(self, monkeypatch):
        import deep_research as dr
        import src.research.fetch as fetch_mod
        monkeypatch.setattr(fetch_mod, "fetch_page",
                            lambda url: {"ok": False, "error": "timeout"})
        monkeypatch.setattr("time.sleep", lambda *a: None)
        result = dr._fetch_with_retry("http://example.com", max_retries=2)
        assert result["ok"] is False
        assert result["error"] == "timeout"

    def test_http_404_no_retry(self, monkeypatch):
        import deep_research as dr
        import src.research.fetch as fetch_mod
        call_count = {"n": 0}

        def fake_fetch(url):
            call_count["n"] += 1
            return {"ok": False, "error": "HTTP 404"}

        monkeypatch.setattr(fetch_mod, "fetch_page", fake_fetch)
        monkeypatch.setattr("time.sleep", lambda *a: None)
        result = dr._fetch_with_retry("http://example.com", max_retries=2)
        assert result["ok"] is False
        # HTTP 404 is not retryable — should only be called once.
        assert call_count["n"] == 1

    def test_connection_reset_retries(self, monkeypatch):
        import deep_research as dr
        import src.research.fetch as fetch_mod
        call_count = {"n": 0}

        def fake_fetch(url):
            call_count["n"] += 1
            return {"ok": False, "error": "connection reset"}

        monkeypatch.setattr(fetch_mod, "fetch_page", fake_fetch)
        monkeypatch.setattr("time.sleep", lambda *a: None)
        result = dr._fetch_with_retry("http://example.com", max_retries=2)
        assert result["ok"] is False
        # "connection reset" is retryable — should be called max_retries+1 times.
        assert call_count["n"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# 7. _gather — L0/L1/L2/L4/L9
# ══════════════════════════════════════════════════════════════════════════════

class TestGather:
    """``_gather`` searches, fetches, filters YouTube, dedups URLs, skips short."""

    def test_youtube_urls_filtered_out(self, monkeypatch):
        import deep_research as dr
        import src.research.search as search_mod

        def fake_search(query, max_results=5):
            return [
                {"url": "https://youtube.com/watch?v=abc", "title": "yt"},
                {"url": "https://youtu.be/abc", "title": "yt short"},
                {"url": "https://example.com/page", "title": "real"},
            ]

        monkeypatch.setattr(search_mod, "meta_search", fake_search)

        fetched = []

        def fake_fetch(url, **kw):
            fetched.append(url)
            return {"ok": True, "text": "x" * 500}

        monkeypatch.setattr(dr, "_fetch_with_retry", fake_fetch)

        visited, chunks, source_map = set(), [], {}
        dr._gather(["query"], visited, chunks, source_map)
        assert "https://youtube.com/watch?v=abc" not in fetched
        assert "https://youtu.be/abc" not in fetched
        assert "https://example.com/page" in fetched

    def test_already_visited_urls_not_refetched(self, monkeypatch):
        import deep_research as dr
        import src.research.search as search_mod

        def fake_search(query, max_results=5):
            return [{"url": "https://example.com/visited", "title": "v"}]

        monkeypatch.setattr(search_mod, "meta_search", fake_search)

        fetched = []

        def fake_fetch(url, **kw):
            fetched.append(url)
            return {"ok": True, "text": "x" * 500}

        monkeypatch.setattr(dr, "_fetch_with_retry", fake_fetch)

        visited = {"https://example.com/visited"}
        chunks, source_map = [], {}
        dr._gather(["query"], visited, chunks, source_map)
        assert fetched == []

    def test_short_content_skipped(self, monkeypatch):
        import deep_research as dr
        import src.research.search as search_mod

        def fake_search(query, max_results=5):
            return [{"url": "https://example.com/short", "title": "s"}]

        monkeypatch.setattr(search_mod, "meta_search", fake_search)

        def fake_fetch(url, **kw):
            return {"ok": True, "text": "too short"}  # < 200 chars

        monkeypatch.setattr(dr, "_fetch_with_retry", fake_fetch)

        visited, chunks, source_map = set(), [], {}
        read = dr._gather(["query"], visited, chunks, source_map)
        # Short content is skipped — no chunks added, read count is 0.
        assert chunks == []
        assert read == 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. run_deep_research — L0/L1/L2/L4b
# ══════════════════════════════════════════════════════════════════════════════

class TestRunDeepResearch:
    """``run_deep_research`` orchestrates the loop with clamped rounds."""

    def test_max_rounds_clamped_to_3(self, monkeypatch):
        import deep_research as dr
        calls = {"gather": 0}

        def fake_queries(*a, **k):
            return ["q1"]

        def fake_gather(queries, visited, chunks, source_map, **kw):
            calls["gather"] += 1
            # Only produce chunks on the first gather so gap-filling can run.
            if calls["gather"] == 1:
                chunks.append("some chunk text " * 50)
                source_map["some chunk text " * 50] = "http://example.com"
                return 1
            return 0

        def fake_extract(*a, **k):
            return "extracted notes"

        def fake_gaps(*a, **k):
            return ["gap query"]

        def fake_llm(*a, **k):
            return "final report"

        monkeypatch.setattr(dr, "_generate_queries", fake_queries)
        monkeypatch.setattr(dr, "_gather", fake_gather)
        monkeypatch.setattr(dr, "_extract_info", fake_extract)
        monkeypatch.setattr(dr, "_find_gaps", fake_gaps)
        monkeypatch.setattr(dr, "_call_llm_sync", fake_llm)
        # Stub rerank/dedup/number_sources so _rerank_and_extract works offline.
        monkeypatch.setattr(dr, "rerank", lambda obj, chunks, top_n=15: chunks)
        monkeypatch.setattr(dr, "dedup_chunks", lambda chunks: chunks)
        monkeypatch.setattr(dr, "number_sources",
                            lambda chunks, sm: ("\n".join(chunks), ["[1] http://example.com"]))

        result = dr.run_deep_research("objective", max_rounds=5)
        # max_rounds=5 should be clamped to 3.
        assert "final report" in result or "extracted notes" in result

    def test_max_rounds_zero_clamped_to_1(self, monkeypatch):
        import deep_research as dr
        gather_calls = {"n": 0}

        def fake_queries(*a, **k):
            return ["q1"]

        def fake_gather(queries, visited, chunks, source_map, **kw):
            gather_calls["n"] += 1
            chunks.append("chunk content " * 50)
            source_map["chunk content " * 50] = "http://example.com"
            return 1

        monkeypatch.setattr(dr, "_generate_queries", fake_queries)
        monkeypatch.setattr(dr, "_gather", fake_gather)
        monkeypatch.setattr(dr, "_extract_info", lambda *a, **k: "notes")
        monkeypatch.setattr(dr, "_find_gaps", lambda *a, **k: ["gap"])
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "report")
        monkeypatch.setattr(dr, "rerank", lambda obj, chunks, top_n=15: chunks)
        monkeypatch.setattr(dr, "dedup_chunks", lambda chunks: chunks)
        monkeypatch.setattr(dr, "number_sources",
                            lambda chunks, sm: ("\n".join(chunks), ["[1] http://x.com"]))

        dr.run_deep_research("objective", max_rounds=0)
        # max_rounds=0 clamped to 1 — only the initial gather, no gap-filling.
        assert gather_calls["n"] == 1

    def test_max_rounds_negative_clamped_to_1(self, monkeypatch):
        import deep_research as dr
        gather_calls = {"n": 0}

        def fake_queries(*a, **k):
            return ["q1"]

        def fake_gather(queries, visited, chunks, source_map, **kw):
            gather_calls["n"] += 1
            chunks.append("chunk content " * 50)
            source_map["chunk content " * 50] = "http://example.com"
            return 1

        monkeypatch.setattr(dr, "_generate_queries", fake_queries)
        monkeypatch.setattr(dr, "_gather", fake_gather)
        monkeypatch.setattr(dr, "_extract_info", lambda *a, **k: "notes")
        monkeypatch.setattr(dr, "_find_gaps", lambda *a, **k: ["gap"])
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "report")
        monkeypatch.setattr(dr, "rerank", lambda obj, chunks, top_n=15: chunks)
        monkeypatch.setattr(dr, "dedup_chunks", lambda chunks: chunks)
        monkeypatch.setattr(dr, "number_sources",
                            lambda chunks, sm: ("\n".join(chunks), ["[1] http://x.com"]))

        dr.run_deep_research("objective", max_rounds=-1)
        assert gather_calls["n"] == 1

    def test_no_chunks_returns_failure_message(self, monkeypatch):
        import deep_research as dr

        monkeypatch.setattr(dr, "_generate_queries", lambda *a, **k: ["q1"])
        monkeypatch.setattr(dr, "_gather",
                            lambda *a, **k: 0)  # no chunks gathered
        result = dr.run_deep_research("nonexistent topic")
        assert "failed" in result.lower()

    def test_empty_extracted_notes_stops_gap_filling(self, monkeypatch):
        import deep_research as dr
        gap_calls = {"n": 0}
        gather_calls = {"n": 0}

        def fake_queries(*a, **k):
            return ["q1"]

        def fake_gather(queries, visited, chunks, source_map, **kw):
            gather_calls["n"] += 1
            if gather_calls["n"] == 1:
                chunks.append("chunk content " * 50)
                source_map["chunk content " * 50] = "http://example.com"
                return 1
            return 0

        def fake_extract(*a, **k):
            return ""  # empty notes

        def fake_gaps(*a, **k):
            gap_calls["n"] += 1
            return ["gap query"]

        monkeypatch.setattr(dr, "_generate_queries", fake_queries)
        monkeypatch.setattr(dr, "_gather", fake_gather)
        monkeypatch.setattr(dr, "_extract_info", fake_extract)
        monkeypatch.setattr(dr, "_find_gaps", fake_gaps)
        monkeypatch.setattr(dr, "_call_llm_sync", lambda *a, **k: "report")
        monkeypatch.setattr(dr, "rerank", lambda obj, chunks, top_n=15: chunks)
        monkeypatch.setattr(dr, "dedup_chunks", lambda chunks: chunks)
        monkeypatch.setattr(dr, "number_sources",
                            lambda chunks, sm: ("\n".join(chunks), ["[1] http://x.com"]))

        result = dr.run_deep_research("objective", max_rounds=3)
        # Empty extracted_notes should stop gap-filling — _find_gaps never called.
        assert gap_calls["n"] == 0
        assert "no relevant" in result.lower() or "report" in result.lower()