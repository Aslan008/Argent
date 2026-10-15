import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from browser_engine import browser_engine, _SessionContext
from src.agent.arg_coercion import coerce_and_log


class TestArgCoercionBrowserOpen:
    def test_headless_false_becomes_headed_true(self):
        raw = {"url": "https://example.com", "headless": False}
        coerced = coerce_and_log("browser_open", raw)
        assert coerced.get("headed") is True
        assert "headless" not in coerced

    def test_headless_true_becomes_headed_false(self):
        raw = {"url": "https://example.com", "headless": True}
        coerced = coerce_and_log("browser_open", raw)
        assert coerced.get("headed") is False
        assert "headless" not in coerced

    def test_visible_and_show_aliases(self):
        raw_vis = {"url": "https://example.com", "visible": True}
        coerced_vis = coerce_and_log("browser_open", raw_vis)
        assert coerced_vis.get("headed") is True

        raw_show = {"link": "https://example.com", "show": "yes"}
        coerced_show = coerce_and_log("browser_open", raw_show)
        assert coerced_show.get("url") == "https://example.com"
        assert coerced_show.get("headed") is True


class TestBrowserVisibilityFormatting:
    @pytest.fixture(autouse=True)
    def clean_engine(self):
        browser_engine._sessions.clear()
        browser_engine._last_fallback_notice = None
        yield
        browser_engine._sessions.clear()
        browser_engine._last_fallback_notice = None

    @pytest.mark.asyncio
    async def test_open_page_reports_headless_status(self):
        mock_page = MagicMock()
        mock_page.title = AsyncMock(return_value="My Headless App")
        mock_page.url = "http://localhost:3000"
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)

        sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                browser_engine._headed = False
                res = await browser_engine.open_page("http://localhost:3000", session="test_headless", headed=False)
                assert "Window: HEADLESS (background process — no visible window on screen)" in res
                assert "Mode: isolated (Playwright)" in res

    @pytest.mark.asyncio
    async def test_open_page_reports_visible_status(self):
        mock_page = MagicMock()
        mock_page.title = AsyncMock(return_value="My Visible App")
        mock_page.url = "http://localhost:3000"
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)

        sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                browser_engine._headed = True
                res = await browser_engine.open_page("http://localhost:3000", session="test_visible", headed=True)
                assert "Window: VISIBLE (window is open on user's desktop)" in res

    @pytest.mark.asyncio
    async def test_open_page_includes_fallback_notice_and_clears_it(self):
        mock_page = MagicMock()
        mock_page.title = AsyncMock(return_value="Fallback Page")
        mock_page.url = "http://localhost:3000"
        mock_page.goto = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.is_closed = MagicMock(return_value=False)

        sc = _SessionContext(context=MagicMock(), page=mock_page)

        browser_engine._last_fallback_notice = "Notice: Chrome was busy. Switched to isolated Chromium."
        browser_engine._headed = True

        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.open_page("http://localhost:3000", session="test_notice", headed=True)
                assert "[Notice: Chrome was busy. Switched to isolated Chromium.]" in res
                assert browser_engine._last_fallback_notice is None
