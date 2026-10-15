import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tools.browser_tools import browser_action, browser_inspect, browser_close, browser_close_tab, browser_list_tabs


@pytest.mark.asyncio
async def test_browser_list_tabs_formatting():
    """Verify listing open tabs formats index, active status, URL and title."""
    page1 = MagicMock()
    page1.is_closed.return_value = False
    page1.url = "https://example.com"
    page1.title = AsyncMock(return_value="Example Domain")

    page2 = MagicMock()
    page2.is_closed.return_value = False
    page2.url = "https://google.com"
    page2.title = AsyncMock(return_value="Google Search")

    mock_sc = MagicMock()
    mock_sc.context.pages = [page1, page2]
    mock_sc.page = page1

    with patch("browser_engine.browser_engine._get_session", new_callable=AsyncMock) as mock_get_session:
        mock_get_session.return_value = mock_sc
        from browser_engine import browser_engine

        result = await browser_engine.list_tabs(session="default")
        assert "Open browser tabs (2):" in result
        assert "[1] (Active) https://example.com (\"Example Domain\")" in result
        assert "[2] https://google.com (\"Google Search\")" in result


@pytest.mark.asyncio
async def test_browser_close_tab_by_index():
    """Verify closing a tab by 1-based index closes the page and reassigns active page."""
    page1 = MagicMock()
    page1.is_closed.return_value = False
    page1.url = "https://tab1.com"
    page1.title = AsyncMock(return_value="Tab 1")
    page1.close = AsyncMock()

    page2 = MagicMock()
    page2.is_closed.return_value = False
    page2.url = "https://tab2.com"
    page2.title = AsyncMock(return_value="Tab 2")
    page2.close = AsyncMock()
    page2.bring_to_front = AsyncMock()

    mock_sc = MagicMock()
    mock_sc.context.pages = [page1, page2]
    mock_sc.page = page1

    with patch("browser_engine.browser_engine._get_session", new_callable=AsyncMock) as mock_get_session:
        mock_get_session.return_value = mock_sc
        from browser_engine import browser_engine

        # Close tab 1
        res = await browser_engine.close_tab(index=1, session="default")
        page1.close.assert_awaited_once()
        assert "Closed tab [1]" in res


@pytest.mark.asyncio
async def test_browser_close_tab_by_url_match():
    """Verify closing a tab by URL substring match."""
    page1 = MagicMock()
    page1.is_closed.return_value = False
    page1.url = "https://github.com"
    page1.title = AsyncMock(return_value="GitHub")
    page1.close = AsyncMock()

    page2 = MagicMock()
    page2.is_closed.return_value = False
    page2.url = "chrome-extension://kmgeemlmoiohlehomoligkpelnhjbahl/wake-service-worker.html"
    page2.title = AsyncMock(return_value="Wake Worker")
    page2.close = AsyncMock()

    mock_sc = MagicMock()
    mock_sc.context.pages = [page1, page2]
    mock_sc.page = page1

    with patch("browser_engine.browser_engine._get_session", new_callable=AsyncMock) as mock_get_session:
        mock_get_session.return_value = mock_sc
        from browser_engine import browser_engine

        res = await browser_engine.close_tab(text="wake-service-worker", session="default")
        page2.close.assert_awaited_once()
        page1.close.assert_not_awaited()
        assert "Closed tab [2] matching 'wake-service-worker'" in res


def test_browser_action_close_tab_facade():
    """Verify browser_action(action='close_tab') routes to close_tab."""
    with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}), \
         patch("tools.browser_tools.browser_close_tab", return_value="Closed tab [2]") as mock_ct:
        res = browser_action(action="close_tab", index=2)
        mock_ct.assert_called_once_with(index=2, text=None, session="default")
        assert res == "Closed tab [2]"


def test_browser_inspect_tabs_facade():
    """Verify browser_inspect(mode='tabs') routes to browser_list_tabs."""
    with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}), \
         patch("tools.browser_tools.browser_list_tabs", return_value="Open browser tabs (1): ...") as mock_lt:
        res = browser_inspect(mode="tabs")
        mock_lt.assert_called_once_with(session="default")
        assert "Open browser tabs" in res


def test_browser_close_selective_tab():
    """Verify browser_close(index=2) closes specific tab instead of entire session."""
    with patch("tools.browser_tools.browser_close_tab", return_value="Closed tab [2]") as mock_ct:
        res = browser_close(index=2)
        mock_ct.assert_called_once_with(index=2, text=None, session="default")
        assert res == "Closed tab [2]"
