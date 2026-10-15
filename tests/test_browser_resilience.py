import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from browser_engine import browser_engine, _SessionContext
from tools.browser_tools import (
    browser_state,
    browser_click,
    browser_wait_for,
    browser_get_content,
    _SESSION_POLL_TRACKER,
    _reset_poll_tracker,
)


def create_mock_session(mock_page=None):
    if mock_page is None:
        mock_page = MagicMock()
    mock_page.is_closed.return_value = False
    mock_context = MagicMock()
    return _SessionContext(page=mock_page, context=mock_context)


@pytest.fixture(autouse=True)
def cleanup_browser_state():
    """Reset session poll tracker and browser element metadata before each test."""
    _SESSION_POLL_TRACKER.clear()
    browser_engine._element_metadata.clear()
    browser_engine._element_frames.clear()
    browser_engine._sessions.clear()
    yield
    _SESSION_POLL_TRACKER.clear()
    browser_engine._element_metadata.clear()
    browser_engine._element_frames.clear()
    browser_engine._sessions.clear()


class TestSmartRecovery:
    @pytest.mark.asyncio
    async def test_smart_recovery_by_role_and_name_when_index_shifted(self):
        """When data-argent-idx attribute is missing due to React re-render,

        _resolve_target_locator must recover the element by cached role and name.
        """
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]

        # Missing index element in DOM
        missing_locator = MagicMock()
        missing_locator.count = AsyncMock(return_value=0)
        mock_frame.locator.return_value = missing_locator

        # Recovered element by role and name
        recovered_locator = MagicMock()
        recovered_locator.count = AsyncMock(return_value=1)
        recovered_locator.first = recovered_locator
        recovered_locator.click = AsyncMock()
        recovered_locator.evaluate = AsyncMock(return_value="button")
        mock_frame.get_by_role.return_value = recovered_locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc
        browser_engine._element_frames[256] = mock_frame
        browser_engine._element_metadata[256] = {
            "idx": 256,
            "role": "button",
            "name": "Сохранить",
            "value": "",
            "tag": "button",
            "frame": mock_frame,
        }

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.click(index=256, session="test_session", return_state=False)

        assert "recovered by role 'button' and name 'Сохранить'" in res
        recovered_locator.click.assert_awaited()

    @pytest.mark.asyncio
    async def test_smart_recovery_by_text_fallback(self):
        """When role lookup fails, fallback to get_by_text."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]

        # Missing index element
        missing_locator = MagicMock()
        missing_locator.count = AsyncMock(return_value=0)
        mock_frame.locator.return_value = missing_locator

        # get_by_role returns 0
        role_locator = MagicMock()
        role_locator.count = AsyncMock(return_value=0)
        mock_frame.get_by_role.return_value = role_locator

        # get_by_text succeeds
        text_locator = MagicMock()
        text_locator.count = AsyncMock(return_value=1)
        text_locator.first = text_locator
        text_locator.click = AsyncMock()
        text_locator.evaluate = AsyncMock(return_value="div")
        mock_frame.get_by_text.return_value = text_locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc
        browser_engine._element_frames[101] = mock_frame
        browser_engine._element_metadata[101] = {
            "idx": 101,
            "role": "custom-tab",
            "name": "Вкладка настроек",
            "value": "",
            "tag": "div",
            "frame": mock_frame,
        }

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.click(index=101, session="test_session", return_state=False)

        assert "recovered by text 'Вкладка настроек'" in res
        text_locator.click.assert_awaited()

    @pytest.mark.asyncio
    async def test_smart_recovery_informative_error_when_completely_missing(self):
        """When element is neither at its index nor in DOM by name/text, return actionable guidance."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]

        missing_locator = MagicMock()
        missing_locator.count = AsyncMock(return_value=0)
        mock_frame.locator.return_value = missing_locator
        mock_frame.get_by_role.return_value = missing_locator
        mock_frame.get_by_text.return_value = missing_locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc
        browser_engine._element_frames[42] = mock_frame
        browser_engine._element_metadata[42] = {
            "idx": 42,
            "role": "button",
            "name": "Загрузить видео",
            "value": "",
            "tag": "button",
            "frame": mock_frame,
        }

        res = await browser_engine.click(index=42, session="test_session")
        assert "Error: Element [42] ('Загрузить видео') not found" in res
        assert "The page appears to have re-rendered" in res
        assert "target by text='Загрузить видео'" in res


class TestBrowserWaitFor:
    @pytest.mark.asyncio
    async def test_wait_for_text_success(self):
        """wait_for with text targets get_by_text and calls wait_for."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        locator = MagicMock()
        locator.first = locator
        locator.wait_for = AsyncMock()
        mock_page.get_by_text.return_value = locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.wait_for(text="Видео геймплея", timeout=5.0, session="test_session", return_state=False)

        assert "Element matching text 'Видео геймплея' reached state 'visible'" in res
        locator.wait_for.assert_awaited_once_with(state="visible", timeout=5000)

    @pytest.mark.asyncio
    async def test_wait_for_selector_success(self):
        """wait_for with selector targets locator and calls wait_for."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        locator = MagicMock()
        locator.first = locator
        locator.wait_for = AsyncMock()
        mock_page.locator.return_value = locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.wait_for(selector=".upload-card", state="attached", timeout=10.0, session="test_session", return_state=False)

        assert "Element matching selector '.upload-card' reached state 'attached'" in res
        locator.wait_for.assert_awaited_once_with(state="attached", timeout=10000)

    @pytest.mark.asyncio
    async def test_wait_for_timeout_graceful(self):
        """When Playwright raises TimeoutError, wait_for returns user-friendly timeout string."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        locator = MagicMock()
        locator.first = locator
        locator.wait_for = AsyncMock(side_effect=TimeoutError("Locator.wait_for: Timeout 5000ms exceeded."))
        mock_page.get_by_text.return_value = locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc

        res = await browser_engine.wait_for(text="Медленная кнопка", timeout=5.0, session="test_session")
        assert "Timeout waiting 5.0s for element matching text 'Медленная кнопка' to become visible." in res

    @pytest.mark.asyncio
    async def test_wait_for_validation(self):
        """Validation of empty params and invalid states."""
        res_empty = await browser_engine.wait_for()
        assert "You must specify either 'text', 'selector', or 'expression'" in res_empty

        res_state = await browser_engine.wait_for(text="test", state="flying")
        assert "Error: Invalid state 'flying'" in res_state


class TestContentByTextAndContainerExpansion:
    @pytest.mark.asyncio
    async def test_get_text_by_text_expands_to_parent_card(self):
        """When user/agent queries browser_get_content(text='Видео геймплея'), and the matched

        node is a short header (<60 chars), expand to the parent container/card.
        """
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.frames = []

        # Inner short header node
        header_locator = MagicMock()
        header_locator.count = AsyncMock(return_value=1)
        header_locator.first = header_locator
        header_locator.inner_text = AsyncMock(return_value="Видео геймплея")

        # Parent card container
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.first = card_locator
        card_locator.inner_text = AsyncMock(return_value="Видео геймплея\nЗагрузите видео 16:9 MP4\n[Кнопка: Загрузить]")

        header_locator.locator.return_value = card_locator
        mock_page.get_by_text.return_value = header_locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc

        res = await browser_engine.get_text(text="Видео геймплея", session="test_session")
        assert "Content of element matching text 'Видео геймплея':" in res
        assert "Загрузите видео 16:9 MP4" in res
        assert "[Кнопка: Загрузить]" in res

    @pytest.mark.asyncio
    async def test_get_markdown_by_text(self):
        """get_markdown with text returns markdown of the matching element or expanded container."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.frames = []

        header_locator = MagicMock()
        header_locator.count = AsyncMock(return_value=1)
        header_locator.first = header_locator
        header_locator.inner_text = AsyncMock(return_value="Видео геймплея")

        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.first = card_locator
        card_locator.inner_html = AsyncMock(return_value="<section><h3>Видео геймплея</h3><p>Описание</p></section>")

        header_locator.locator.return_value = card_locator
        mock_page.get_by_text.return_value = header_locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc

        res = await browser_engine.get_markdown(text="Видео геймплея", session="test_session")
        assert "Видео геймплея" in res
        assert "Описание" in res


class TestPollingTracker:
    def test_polling_hint_triggered_after_3_identical_queries(self):
        """Calling browser_state 3 times with the exact same query triggers the [POLLING HINT]."""
        def mock_run(coro):
            if asyncio.iscoroutine(coro):
                coro.close()
            return "[1] button 'Горизонтальное видео'"

        with patch.object(browser_engine, "run", side_effect=mock_run):
            r1 = browser_state(session="test_session", query="Горизонтальное видео")
            assert "[POLLING HINT]" not in r1

            r2 = browser_state(session="test_session", query="Горизонтальное видео")
            assert "[POLLING HINT]" not in r2

            r3 = browser_state(session="test_session", query="Горизонтальное видео")
            assert "[POLLING HINT]" in r3
            assert "Stop blind polling!" in r3
            assert "wait_for" in r3 and "browser_wait_for" not in r3

    def test_polling_hint_resets_on_action(self):
        """Performing an action (like click) resets the poll counter."""
        def mock_run(coro):
            if asyncio.iscoroutine(coro):
                coro.close()
            return "OK"

        with patch.object(browser_engine, "run", side_effect=mock_run):
            browser_state(session="test_session", query="Горизонтальное видео")
            browser_state(session="test_session", query="Горизонтальное видео")

            # Click resets the tracker
            browser_click(index=1, session="test_session")

            # First query after action should NOT have hint
            r_after_action = browser_state(session="test_session", query="Горизонтальное видео")
            assert "[POLLING HINT]" not in r_after_action


class TestResilientClickAndInput:
    @pytest.mark.asyncio
    async def test_click_recovers_from_pointer_interception_via_force(self):
        """When standard click fails due to pointer interception, fallback to force=True."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]

        locator = MagicMock()
        locator.first = locator
        locator.count = AsyncMock(return_value=1)
        locator.evaluate = AsyncMock(return_value="a")

        # First standard click fails with intercepts pointer events; second force=True succeeds
        call_count = 0
        async def mock_click(timeout=None, force=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and not force:
                raise Exception("Locator.click: Timeout 2500ms exceeded. <span ...> subtree intercepts pointer events")
            return None

        locator.click = AsyncMock(side_effect=mock_click)
        mock_frame.locator.return_value = locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc
        browser_engine._element_frames[46] = mock_frame

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.click(index=46, session="test_session", return_state=False)

        assert "Clicked element [46]" in res
        assert call_count == 2  # Level 1 failed, Level 2 succeeded

    @pytest.mark.asyncio
    async def test_click_recovers_via_js_dispatch_when_force_fails(self):
        """When both standard and force click fail (e.g. zero bounding box), fallback to JS DOM dispatch."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]

        locator = MagicMock()
        locator.first = locator
        locator.count = AsyncMock(return_value=1)

        # Both clicks fail
        locator.click = AsyncMock(side_effect=Exception("Locator.click: Element has 0x0 bounding box"))
        # evaluate for tag name returns "div", evaluate for JS dispatch returns True
        async def mock_eval(script, *args):
            if "tagName" in script:
                return "div"
            return True

        locator.evaluate = AsyncMock(side_effect=mock_eval)
        mock_frame.locator.return_value = locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc
        browser_engine._element_frames[99] = mock_frame

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.click(index=99, session="test_session", return_state=False)

        assert "Clicked element [99]" in res

    @pytest.mark.asyncio
    async def test_fill_recovers_when_click_intercepted(self):
        """When focusing/clicking input raises pointer interception, fill still succeeds."""
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]

        locator = MagicMock()
        locator.first = locator
        locator.count = AsyncMock(return_value=1)
        locator.evaluate = AsyncMock(return_value="input")
        # click fails, but fill succeeds
        locator.click = AsyncMock(side_effect=Exception("intercepts pointer events"))
        locator.fill = AsyncMock()

        mock_frame.locator.return_value = locator

        sc = create_mock_session(mock_page)
        browser_engine._sessions["test_session"] = sc
        browser_engine._element_frames[15] = mock_frame

        with patch.object(browser_engine, "_wait_for_page_settle", new=AsyncMock()):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.fill_input(index=15, text="Hello World", session="test_session", return_state=False)

        assert "Filled element [15]" in res
        locator.fill.assert_awaited_once_with("Hello World", timeout=3000)
