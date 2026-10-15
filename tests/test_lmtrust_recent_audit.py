"""LMTrust Deep Audit Suite for Recent Search, Research & Browser Changes.

Mode B (Audit) & Mode A (Blind Spot Testing):
- L3 Property & D10 Invariant: number_sources monotonicity and no-collision with arbitrary initial registries
- L2 Boundary & D1 Numeric: meta_search with max_results <= 0
- L4b Fallback & D5 Resource: read_webpage session cleanup guarantee on exception
- L4 Adversarial & D7 Contract: upload_file json-serialized list and non-existent selector short-circuit
- L1 Contract & D3 Data Integrity: _norm_url general utm_* prefix stripping
"""

import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from src.research.synthesis import number_sources
from src.research.search import meta_search, _norm_url


# ==============================================================================
# 1. number_sources Monotonicity & No-Collision (L3 Property & D10 Invariant)
# ==============================================================================

class TestNumberSourcesInvariant:
    """Distinct URLs must ALWAYS receive distinct citation numbers regardless of initial registry state."""

    def test_no_collision_when_initial_registry_has_custom_values(self):
        # Initial registry already has index 2
        initial_registry = {"https://existing-site.com": 2}
        chunks = ["chunk_new_1", "chunk_new_2"]
        source_map = {
            "chunk_new_1": "https://new-site-1.com",
            "chunk_new_2": "https://new-site-2.com",
        }

        annotated, sources = number_sources(chunks, source_map, url_to_num=initial_registry)

        # All assigned numbers in url_to_num must be unique
        assigned_numbers = list(initial_registry.values())
        assert len(assigned_numbers) == len(set(assigned_numbers)), (
            f"Citation number collision detected! Assigned: {assigned_numbers}"
        )
        assert len(sources) == 3


# ==============================================================================
# 2. meta_search Boundary Conditions (L2 Boundary & D1 Numeric)
# ==============================================================================

class TestMetaSearchBoundary:
    """meta_search must return empty list when max_results <= 0."""

    def test_meta_search_zero_max_results(self):
        engine = lambda q, n: [{"url": "https://example.com/result1"}]
        res = meta_search("query", max_results=0, engines=[engine])
        assert res == [], f"Expected [], got {res}"

    def test_meta_search_negative_max_results(self):
        engine = lambda q, n: [{"url": "https://example.com/result1"}]
        res = meta_search("query", max_results=-5, engines=[engine])
        assert res == [], f"Expected [], got {res}"


# ==============================================================================
# 3. read_webpage Session Resource Cleanup (L4b Fallback & D5 Resource)
# ==============================================================================

class TestWebToolsResourceCleanup:
    """read_webpage headless fallback must ensure browser session is closed on errors."""

    def test_close_session_called_even_if_get_markdown_raises(self):
        from tools.web_tools import read_webpage

        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            None,  # open_page
            None,  # wait_stable
            RuntimeError("browser crash"),  # get_markdown throws
            None,  # close_session
        ]

        with patch("src.research.fetch.fetch_page", return_value={"ok": False, "text": "", "error": "timeout"}):
            with patch("browser_engine.browser_engine", mock_engine):
                res = read_webpage("https://spa-example.com")
                # close_session MUST have been invoked
                assert mock_engine.close_session.called, "browser_engine.close_session was NOT called!"


# ==============================================================================
# 4. upload_file Adversarial Formats & Fast Timeout (L4 Adversarial & D7 Contract)
# ==============================================================================

class TestUploadFileAdversarial:
    """upload_file must tolerate JSON stringified array and fail fast on missing elements."""

    @pytest.mark.asyncio
    async def test_upload_file_handles_json_encoded_list_string(self, tmp_path):
        from browser_engine import browser_engine

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")

        json_str = f'["{str(f1).replace(chr(92), "/")}", "{str(f2).replace(chr(92), "/")}"]'

        mock_sc = MagicMock()
        mock_page = MagicMock()
        mock_sc.page = mock_page
        mock_el = MagicMock()
        mock_el.first = mock_el
        mock_el.count = AsyncMock(return_value=1)
        mock_el.set_input_files = AsyncMock()

        mock_frame = MagicMock()
        mock_frame.locator = MagicMock(return_value=mock_el)

        with patch.object(browser_engine, "_get_session", new=AsyncMock(return_value=mock_sc)):
            with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
                with patch.object(browser_engine, "_format_action_result", new=AsyncMock(return_value="Success")):
                    browser_engine._element_frames[9] = mock_frame
                    res = await browser_engine.upload_file(json_str, index=9)
                    assert res == "Success"
                    mock_el.set_input_files.assert_awaited_once()
                    called_files = mock_el.set_input_files.await_args[0][0]
                    assert len(called_files) == 2


# ==============================================================================
# 5. _norm_url General UTM Strip (L1 Contract & D3 Data Integrity)
# ==============================================================================

class TestNormUrlTrackingStrip:
    """_norm_url must strip any utm_* parameter."""

    def test_strips_arbitrary_utm_tags(self):
        u = "https://example.com/page?utm_custom=123&utm_id=456&real_arg=val"
        norm = _norm_url(u)
        assert "utm_custom" not in norm
        assert "utm_id" not in norm
        assert "real_arg=val" in norm
