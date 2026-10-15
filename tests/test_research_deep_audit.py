"""LMTrust Deep Audit and Verification Test Suite for Argent Research and Browser Automation.

Tests verify:
- L1 Contract & L2 Boundary: _norm_url query preservation and UTM filtering
- L3 Property & L5 State: Persistent SourceRegistry in deep_research preventing Citation Drift
- L4 Adversarial & D5 Resource: Stream body truncation and binary MIME rejection in fetch_page/_http_get
- L5 Concurrency & D9 Performance: Deterministic parallel meta_search ordering
- L2 Boundary & L5 Resource: rerank top_n bounds and bi-encoder model caching
- L5 State: SQLite web cache prune_expired
- L6 Integration: Browser upload_file and fill_form contract
"""

import time
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import pytest

from src.research.search import _norm_url, meta_search, _wikipedia_search
from src.research.synthesis import number_sources
from src.research.fetch import _http_get, fetch_page, MAX_BODY_BYTES
from src.research.rerank import rerank, _get_bi_encoder
from src.research.cache import set_cached, get_cached, prune_expired, clear_cache


# ==============================================================================
# 1. URL Normalization & Query Preservation (L1 Contract & L2 Boundary)
# ==============================================================================

class TestNormUrlQueryPreservation:
    """_norm_url must never drop distinguishing query parameters while stripping tracking tags."""

    def test_distinguishes_different_query_values(self):
        u1 = "https://en.wikipedia.org/w/index.php?title=Quantum_computing"
        u2 = "https://en.wikipedia.org/w/index.php?title=General_relativity"
        assert _norm_url(u1) != _norm_url(u2)
        assert _norm_url(u1) == "en.wikipedia.org/w/index.php?title=Quantum_computing"
        assert _norm_url(u2) == "en.wikipedia.org/w/index.php?title=General_relativity"

    def test_strips_utm_and_tracking_parameters(self):
        u = "https://example.com/blog/article?utm_source=twitter&utm_medium=social&utm_campaign=launch"
        assert _norm_url(u) == "example.com/blog/article"

    def test_sorts_query_parameters_for_consistent_dedup(self):
        u1 = "https://example.com/search?b=2&a=1&c=3"
        u2 = "https://example.com/search?c=3&a=1&b=2"
        assert _norm_url(u1) == _norm_url(u2)
        assert _norm_url(u1) == "example.com/search?a=1&b=2&c=3"

    def test_preserves_existing_trailing_slash_and_fragment_contract(self):
        assert _norm_url("http://Example.com/Path/") == _norm_url("http://example.com/Path")
        assert _norm_url("http://x.com/a#frag") == "x.com/a"

    def test_mixed_real_and_tracking_parameters(self):
        u = "https://docs.unity.com/manual?id=42&utm_source=docs&page=2&ref=banner"
        assert _norm_url(u) == "docs.unity.com/manual?id=42&page=2"


# ==============================================================================
# 2. Source Numbering & Citation Drift Prevention (L3 Property & L5 State)
# ==============================================================================

class TestSourceNumberingAndCitationStability:
    """Persistent url_to_num registry must preserve source numbers across rounds."""

    def test_number_sources_preserves_existing_registry(self):
        existing_registry = {"http://a.com": 1, "http://b.com": 2}
        chunks = ["chunk_c", "chunk_a", "chunk_d"]
        source_map = {"chunk_c": "http://c.com", "chunk_a": "http://a.com", "chunk_d": "http://d.com"}

        annotated, sources = number_sources(chunks, source_map, url_to_num=existing_registry)

        # http://a.com must keep index [1]
        assert "[1] chunk_a" in annotated
        # http://c.com and d.com must receive new sequential numbers [3], [4]
        assert "[3] chunk_c" in annotated
        assert "[4] chunk_d" in annotated

        assert "[1] http://a.com" in sources
        assert "[2] http://b.com" in sources
        assert "[3] http://c.com" in sources
        assert "[4] http://d.com" in sources

    def test_number_sources_standalone_backward_compatibility(self):
        chunks = ["c1", "c2"]
        source_map = {"c1": "http://first.com", "c2": "http://second.com"}
        annotated, sources = number_sources(chunks, source_map)
        assert "[1] c1" in annotated
        assert "[2] c2" in annotated
        assert sources == ["[1] http://first.com", "[2] http://second.com"]


# ==============================================================================
# 3. HTTP Stream Protection & OOM Defense (L4 Adversarial & D5 Resource)
# ==============================================================================

class TestHttpStreamProtection:
    """_http_get must reject binary multimedia and truncate huge streams at MAX_BODY_BYTES."""

    def test_early_rejection_of_binary_media(self, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "video/mp4"}
        mock_resp.is_redirect = False

        with patch("src.research.fetch._guard_url"):
            with patch("requests.get", return_value=mock_resp):
                res = _http_get("http://example.com/video.mp4", timeout=5)
                assert res["status"] == 200
                assert res["content"] == b""
                assert res["text"] == ""
                mock_resp.close.assert_called_once()

    def test_truncation_of_stream_exceeding_max_bytes(self):
        chunk_size = 1024 * 1024  # 1 MB
        num_chunks = 15           # 15 MB total
        dummy_chunks = [b"A" * chunk_size for _ in range(num_chunks)]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.is_redirect = False
        mock_resp.iter_content.return_value = iter(dummy_chunks)
        mock_resp.encoding = "utf-8"

        with patch("src.research.fetch._guard_url"):
            with patch("requests.get", return_value=mock_resp):
                res = _http_get("http://example.com/giant.html", timeout=5)
                assert len(res["content"]) <= MAX_BODY_BYTES
                assert len(res["content"]) == 10 * 1024 * 1024
                mock_resp.close.assert_called_once()


# ==============================================================================
# 4. Deterministic Parallel Meta-Search (L5 Concurrency & D9 Performance)
# ==============================================================================

class TestParallelMetaSearch:
    """Parallel meta_search must preserve deterministic round-robin engine order."""

    def test_parallel_engines_preserve_interleaving_order(self):
        def engine_slow(q, limit):
            time.sleep(0.05)
            return [{"url": "http://slow/1", "title": "S1"}, {"url": "http://slow/2", "title": "S2"}]

        def engine_fast(q, limit):
            return [{"url": "http://fast/1", "title": "F1"}, {"url": "http://fast/2", "title": "F2"}]

        # engine_slow is index 0, engine_fast is index 1
        results = meta_search("test", max_results=4, engines=[engine_slow, engine_fast])
        urls = [r["url"] for r in results]
        # Must preserve round-robin: slow[0], fast[0], slow[1], fast[1]
        assert urls == ["http://slow/1", "http://fast/1", "http://slow/2", "http://fast/2"]

    def test_failing_engine_does_not_break_federation(self):
        def engine_error(q, limit):
            raise ConnectionError("network dead")

        def engine_ok(q, limit):
            return [{"url": "http://ok/1", "title": "OK1"}]

        results = meta_search("test", max_results=4, engines=[engine_error, engine_ok])
        assert len(results) == 1
        assert results[0]["url"] == "http://ok/1"

    def test_wikipedia_html_unescape(self):
        fake_api_data = {
            "query": {
                "search": [{
                    "title": "C&amp;C++",
                    "snippet": "Covers &quot;pointers&quot; and memory &amp; registers."
                }]
            }
        }
        with patch("src.research.search._get_json", return_value=fake_api_data):
            res = _wikipedia_search("c programming", limit=1)
            assert len(res) == 1
            assert res[0]["title"] == "C&C++"
            assert res[0]["snippet"] == 'Covers "pointers" and memory & registers.'


# ==============================================================================
# 5. Rerank Boundary Bounds and Caching (L2 Boundary & L5 Resource)
# ==============================================================================

class TestRerankEnhancements:
    """rerank must handle negative or zero top_n and cache SentenceTransformer."""

    def test_top_n_zero_or_negative_returns_empty(self):
        chunks = ["c1", "c2", "c3"]
        assert rerank("q", chunks, top_n=0) == []
        assert rerank("q", chunks, top_n=-1) == []
        assert rerank("q", chunks, top_n=-10) == []

    def test_bi_encoder_singleton_caching(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_inst = MagicMock()
            mock_st.return_value = mock_inst
            # First call
            enc1 = _get_bi_encoder()
            # Second call
            enc2 = _get_bi_encoder()
            assert enc1 is enc2
            # Constructor was called only once
            assert mock_st.call_count == 1


# ==============================================================================
# 6. SQLite Web Cache Pruning (L5 State & D5 Resource)
# ==============================================================================

class TestCachePruning:
    """prune_expired must clean up records older than TTL."""

    def test_prune_expired_removes_old_entries(self, tmp_path, monkeypatch):
        import src.research.cache as cache_mod
        test_db = tmp_path / "test_prune.db"
        monkeypatch.setattr(cache_mod, "_DB_PATH", test_db)

        set_cached("http://fresh.com", False, "fresh content")
        # Artificially set an old entry
        with cache_mod._lock:
            conn = cache_mod._connect()
            old_time = time.time() - 100000
            conn.execute("INSERT INTO pages VALUES (?,?,?,?)", ("http://old.com", 0, "old content", old_time))
            conn.commit()
            conn.close()

        deleted = prune_expired(ttl=3600)
        assert deleted == 1
        assert get_cached("http://old.com", False, ttl=3600) is None
        assert get_cached("http://fresh.com", False, ttl=3600) == "fresh content"


# ==============================================================================
# 7. Browser Automation & Upload File Contract (L1 Contract & L6 Integration)
# ==============================================================================

class TestBrowserEngineFileUpload:
    """browser_engine.upload_file must support file chooser interception and batch upload."""

    @pytest.mark.asyncio
    async def test_upload_file_nonexistent_returns_error(self):
        from browser_engine import browser_engine
        res = await browser_engine.upload_file("C:/nonexistent/file/path.pdf", index=1)
        assert "does not exist" in res

    @pytest.mark.asyncio
    async def test_upload_file_direct_set_input_files(self, tmp_path):
        dummy_file = tmp_path / "test_upload.txt"
        dummy_file.write_text("file content")

        mock_sc = MagicMock()
        mock_page = MagicMock()
        mock_sc.page = mock_page
        mock_el = MagicMock()
        mock_el.count = AsyncMock(return_value=1)
        mock_el.first = mock_el
        mock_el.set_input_files = AsyncMock()

        mock_frame = MagicMock()
        mock_frame.locator = MagicMock(return_value=mock_el)

        from browser_engine import browser_engine
        with patch.object(browser_engine, "_get_session", new=AsyncMock(return_value=mock_sc)):
            with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
                with patch.object(browser_engine, "_format_action_result", new=AsyncMock(return_value="Action success")):
                    browser_engine._element_frames[1] = mock_frame
                    res = await browser_engine.upload_file(str(dummy_file), index=1)
                    assert res == "Action success"
                    mock_el.set_input_files.assert_awaited_once_with([str(dummy_file.resolve())], timeout=3000)
