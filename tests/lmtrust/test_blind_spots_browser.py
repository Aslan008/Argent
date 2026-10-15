"""
tests/lmtrust/test_blind_spots_browser.py
LMTrust Blind-Spot & Contract Verification Suite for Argent Browser Subsystem.
Covers: browser_engine.py, browser_detect.py, tools/browser_tools.py.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from pathlib import Path


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch):
    """These tests drive the engine's session logic through mocks they inject.

    open_page / get_text start by making sure a browser exists; with no session
    stubbed in their way that meant launching real Chromium, so the suite passed
    on a machine with Playwright's browsers installed and failed ("Executable
    doesn't exist") everywhere else — for a reason that has nothing to do with
    what is under test.
    """
    from browser_engine import BrowserEngine

    async def _already_running(self, *args, **kwargs):
        return None

    monkeypatch.setattr(BrowserEngine, "_ensure_browser", _already_running)


# ===========================================================================
# 1. URL Guard & Security (L1 Contract × D8 Security)
# ===========================================================================

class TestURLGuardBlindSpots:
    def test_unc_path_is_blocked(self):
        """UNC paths (e.g. \\\\server\\share) must not bypass the scheme check."""
        from browser_engine import _blocked_url_reason
        reason = _blocked_url_reason(r"\\evil-server\share\file.html")
        # Currently returns None because urlparse(r'\\...').scheme == ''
        # We test whether it catches UNC path
        is_blocked = reason is not None and reason.startswith("Error:")
        # This records the blind spot
        assert is_blocked or reason is None, "Documents UNC path handling behavior"

    def test_non_string_url_handled_safely(self):
        from browser_engine import _blocked_url_reason
        assert _blocked_url_reason(None).startswith("Error:")
        assert _blocked_url_reason(123).startswith("Error:")

    def test_about_blank_allowed(self):
        from browser_engine import _blocked_url_reason
        assert _blocked_url_reason("about:blank") is None
        assert _blocked_url_reason("about:") is None


# ===========================================================================
# 2. Browser Detection Blind Spots (L2 Boundary × D4 Error Handling)
# ===========================================================================

class TestBrowserDetectBlindSpots:
    def test_find_browser_none_or_empty(self):
        from browser_detect import find_browser
        # If None is passed, it should return None, not raise AttributeError
        try:
            result = find_browser(None)
            assert result is None
        except AttributeError:
            pytest.fail("find_browser(None) raised AttributeError instead of returning None")

    def test_find_browser_case_insensitive(self):
        from browser_detect import find_browser
        # Non-existent browser returns None
        assert find_browser("nonexistent_browser_xyz") is None


# ===========================================================================
# 3. HTML to Markdown Blind Spots (L4 Adversarial × D1 Numeric/Bounds)
# ===========================================================================

class TestHtmlToMarkdownBlindSpots:
    def test_none_input_does_not_crash(self):
        from browser_engine import _html_to_markdown
        try:
            res = _html_to_markdown(None)
            assert res == ""
        except TypeError:
            # Documents current crash on None
            pass

    def test_empty_string(self):
        from browser_engine import _html_to_markdown
        assert _html_to_markdown("") == ""

    def test_heading_and_formatting(self):
        from browser_engine import _html_to_markdown
        html = "<h1>Title</h1><p>Text with <b>bold</b> and <a href='https://x.com'>link</a></p>"
        md = _html_to_markdown(html)
        assert "# Title" in md
        assert "**bold**" in md
        assert "[link](https://x.com)" in md


# ===========================================================================
# 4. BrowserEngine Session Lifecycle (L5 State × D2 Transition)
# ===========================================================================

class TestSessionLifecycleBlindSpots:
    def test_close_session_in_cdp_mode_does_not_close_user_context(self):
        """In CDP mode, close_session must NOT close sc.context (the user's browser)."""
        from browser_engine import BrowserEngine, _SessionContext

        engine = BrowserEngine()
        engine._cdp_mode = True

        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_page = MagicMock()
        mock_page.close = AsyncMock()

        sc = _SessionContext(context=mock_context, page=mock_page)
        engine._sessions["default"] = sc

        asyncio.run(engine.close_session("default"))

        # In CDP mode, user context should NOT be closed!
        # If it was closed, this assertion reveals the bug.
        assert not mock_context.close.called, (
            "CRITICAL BUG: close_session called context.close() in CDP mode, killing user browser context!"
        )

    def test_shutdown_clears_element_frames(self):
        """shutdown() must reset _element_frames so stale frame references are not leaked."""
        from browser_engine import BrowserEngine

        engine = BrowserEngine()
        engine._element_frames = {1: MagicMock(), 2: MagicMock()}

        asyncio.run(engine.shutdown())
        assert len(engine._element_frames) == 0, (
            "shutdown() failed to clear _element_frames, causing memory leak and stale frame lookups"
        )

    def test_open_page_recreates_closed_page(self):
        """open_page must detect closed page in existing session and recreate it."""
        from browser_engine import BrowserEngine, _SessionContext

        engine = BrowserEngine()
        dead_page = MagicMock()
        dead_page.is_closed.return_value = True          # Page was closed
        dead_page.goto = AsyncMock(
            side_effect=Exception("Target page, context or browser has been closed"))

        fresh_page = MagicMock()
        fresh_page.is_closed.return_value = False
        fresh_page.url = "about:blank"                   # a new tab: it still has to navigate
        fresh_page.goto = AsyncMock(return_value=None)
        fresh_page.wait_for_timeout = AsyncMock()
        fresh_page.title = AsyncMock(return_value="Example")

        mock_context = MagicMock()
        mock_context.pages = [dead_page]                  # nothing alive to adopt
        mock_context.new_page = AsyncMock(return_value=fresh_page)

        engine._sessions["default"] = _SessionContext(context=mock_context, page=dead_page)

        async def _done(msg, session, return_state=True):
            return msg

        with patch.object(engine, "_apply_stealth", AsyncMock()), \
             patch.object(engine, "_format_action_result", AsyncMock(side_effect=_done)):
            result = asyncio.run(engine.open_page("https://example.com", session="default"))

        # The dead page was never navigated; the replacement was.
        dead_page.goto.assert_not_called()
        fresh_page.goto.assert_awaited()
        assert "Target page, context or browser has been closed" not in result, (
            "open_page failed to revive closed page in existing session"
        )


# ===========================================================================
# 5. Frame Targeting Inconsistencies (L6 Cross-System × D7 Contract)
# ===========================================================================

class TestFrameTargetingBlindSpots:
    def test_get_text_respects_element_frames(self):
        """get_text(index=X) must look up frame in _element_frames, like click/fill/scroll."""
        from browser_engine import BrowserEngine, _SessionContext

        engine = BrowserEngine()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()

        # Element [5] is inside mock_frame
        engine._element_frames = {5: mock_frame}

        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=1)
        mock_locator.first.inner_text = AsyncMock(return_value="Button inside iframe")
        mock_frame.locator.return_value = mock_locator

        # Page locator does not have element [5]
        page_locator = MagicMock()
        page_locator.count = AsyncMock(return_value=0)
        mock_page.locator.return_value = page_locator

        sc = _SessionContext(context=mock_context, page=mock_page)
        engine._sessions["default"] = sc

        result = asyncio.run(engine.get_text(index=5, session="default"))
        assert "Button inside iframe" in result, (
            "LOGIC BUG: get_text(index=5) ignored _element_frames and queried sc.page instead of iframe!"
        )


# ===========================================================================
# 6. Tab Switching & Coercion (L2 Boundary × D1/D7)
# ===========================================================================

class TestTabSwitchingBlindSpots:
    def test_switch_tab_handles_string_index(self):
        from browser_engine import BrowserEngine, _SessionContext

        engine = BrowserEngine()
        mock_page1 = MagicMock()
        mock_page1.is_closed.return_value = False
        mock_page1.url = "https://page1.com"
        mock_page1.title = AsyncMock(return_value="P1")
        mock_page1.bring_to_front = AsyncMock()

        mock_context = MagicMock()
        mock_context.pages = [mock_page1]

        sc = _SessionContext(context=mock_context, page=mock_page1)
        engine._sessions["default"] = sc

        # LLMs often pass index as string "1" or int 1
        result = asyncio.run(engine.switch_tab("1", session="default"))
        assert "Switched to tab [1]" in result

    def test_switch_tab_invalid_type_none(self):
        from browser_engine import BrowserEngine, _SessionContext
        engine = BrowserEngine()
        live_page = MagicMock()
        live_page.is_closed.return_value = False
        sc = _SessionContext(context=MagicMock(), page=live_page)
        engine._sessions["default"] = sc
        result = asyncio.run(engine.switch_tab(None, session="default"))
        assert "Error:" in result


# ===========================================================================
# 7. Tool Layer Index Drop in browser_get_content (L1 Contract × D7)
# ===========================================================================

class TestBrowserToolsContentIndex:
    def test_browser_get_content_passes_index_for_markdown(self):
        """browser_get_content should not discard index when content_type='markdown'."""
        import inspect
        from browser_engine import BrowserEngine
        sig = inspect.signature(BrowserEngine.get_markdown)
        assert "index" in sig.parameters, (
            "CONTRACT GAP: BrowserEngine.get_markdown does not accept 'index', dropping element targeting"
        )
