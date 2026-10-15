"""
tests/lmtrust/test_lmtrust_browser_visibility_ruthless.py
LMTrust Deep Ruthless Test Suite for Argent Browser Visibility & Fallback Subsystem.

Covers:
  - Layer L0 (Smoke)
  - Layer L1 (Contract & Coordinates)
  - Layer L2 (Boundary Edge Cases: conflicting flags, string booleans)
  - Layer L4 (Adversarial Error Handling: navigation crashes, notice leak prevention)
  - Layer L4b (Fallback: all 4 user->isolated demotion paths + single-shot invariant)
  - Layer L8 (Temporal: headless->headed session transition)
  - Layer L9 (Negative Space: zero false-positive notices, complete alias hygiene)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from browser_engine import browser_engine, _SessionContext
from browser_detect import BrowserInfo
from src.agent.arg_coercion import coerce_and_log, remap_tool_aliases


def create_mock_session(url="http://localhost:3000", title="Test App"):
    mock_page = MagicMock()
    mock_page.title = AsyncMock(return_value=title)
    mock_page.url = url
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    return _SessionContext(context=MagicMock(), page=mock_page)


@pytest.fixture(autouse=True)
def clean_engine_state():
    browser_engine._sessions.clear()
    browser_engine._last_fallback_notice = None
    browser_engine._cdp_mode = False
    browser_engine._headed = False
    yield
    browser_engine._sessions.clear()
    browser_engine._last_fallback_notice = None
    browser_engine._cdp_mode = False
    browser_engine._headed = False


# ===========================================================================
# L0 Smoke: Basic invocation with visibility contract
# ===========================================================================
class TestSmokeVisibility:
    @pytest.mark.lmtrust(layer="L0", direction="D7")
    @pytest.mark.asyncio
    async def test_l0_smoke_visibility_output_format(self):
        sc = create_mock_session()
        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.open_page("http://localhost:3000", session="smoke", headed=False)
                assert "Opened: http://localhost:3000" in res
                assert "Window: HEADLESS (background process — no visible window on screen)" in res


# ===========================================================================
# L1 Contract: Strict output contracts for open_page and navigate
# ===========================================================================
class TestContractVisibility:
    @pytest.mark.lmtrust(layer="L1", direction="D7")
    @pytest.mark.asyncio
    async def test_l1_open_page_headed_contract(self):
        sc = create_mock_session()
        browser_engine._headed = True
        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.open_page("http://localhost:3000", session="c1", headed=True)
                assert "Window: VISIBLE (window is open on user's desktop)" in res
                assert "Mode: isolated (Playwright)" in res

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    @pytest.mark.asyncio
    async def test_l1_open_page_cdp_mode_contract(self):
        sc = create_mock_session()
        browser_engine._cdp_mode = True
        browser_engine._headed = True
        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.open_page("http://localhost:3000", session="c2", headed=True)
                assert "Mode: CDP (user browser)" in res
                assert "Window: VISIBLE (window is open on user's desktop)" in res

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    @pytest.mark.asyncio
    async def test_l1_navigate_visibility_contract(self):
        sc = create_mock_session(url="http://localhost:3000/dashboard")
        browser_engine._sessions["nav_sess"] = sc
        browser_engine._headed = False
        with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
            res = await browser_engine.navigate("http://localhost:3000/dashboard", session="nav_sess")
            assert "Navigated to: http://localhost:3000/dashboard" in res
            assert "Window: HEADLESS (background process — no visible window on screen)" in res


# ===========================================================================
# L2 Boundary: Conflicting flags and adversarial argument coercion
# ===========================================================================
class TestBoundaryAdversarialArgs:
    @pytest.mark.lmtrust(layer="L2", direction="D4")
    def test_l2_conflicting_headless_and_headed_parameters(self):
        """When confused model sends BOTH headless and headed, headless MUST be popped."""
        raw1 = {"url": "https://example.com", "headless": False, "headed": False}
        coerced1 = coerce_and_log("browser_open", raw1)
        # headed=False was explicitly supplied, so headless was dropped
        assert coerced1.get("headed") is False
        assert "headless" not in coerced1

        raw2 = {"url": "https://example.com", "headless": True, "headed": True}
        coerced2 = coerce_and_log("browser_open", raw2)
        assert coerced2.get("headed") is True
        assert "headless" not in coerced2

    @pytest.mark.lmtrust(layer="L2", direction="D4")
    @pytest.mark.parametrize("false_val", ["no", "0", "false", "False", "off", "нет"])
    def test_l2_headless_falsy_strings_become_headed_true(self, false_val):
        raw = {"url": "https://example.com", "headless": false_val}
        coerced = coerce_and_log("browser_open", raw)
        assert coerced.get("headed") is True
        assert "headless" not in coerced

    @pytest.mark.lmtrust(layer="L2", direction="D4")
    @pytest.mark.parametrize("true_val", ["yes", "1", "true", "True", "on", "да"])
    def test_l2_headless_truthy_strings_become_headed_false(self, true_val):
        raw = {"url": "https://example.com", "headless": true_val}
        coerced = coerce_and_log("browser_open", raw)
        assert coerced.get("headed") is False
        assert "headless" not in coerced

    @pytest.mark.lmtrust(layer="L2", direction="D7")
    @pytest.mark.parametrize("alias_key", ["show", "visible", "gui", "display", "window"])
    def test_l2_visibility_aliases_coerced_cleanly(self, alias_key):
        raw = {"url": "https://example.com", alias_key: "yes"}
        coerced = coerce_and_log("browser_open", raw)
        assert coerced.get("headed") is True
        assert alias_key not in coerced


# ===========================================================================
# L4 Adversarial: Navigation crash notice-leak prevention
# ===========================================================================
class TestAdversarialNoticeLifecycle:
    @pytest.mark.lmtrust(layer="L4", direction="D4")
    @pytest.mark.asyncio
    async def test_l4_navigation_crash_consumes_and_resets_fallback_notice(self):
        """If sc.page.goto raises an exception, _last_fallback_notice MUST NOT leak to next call."""
        mock_page = MagicMock()
        mock_page.url = "about:blank"
        mock_page.goto = AsyncMock(side_effect=RuntimeError("DNS_PROBE_FINISHED_NXDOMAIN"))
        mock_page.is_closed = MagicMock(return_value=False)
        sc = _SessionContext(context=MagicMock(), page=mock_page)

        browser_engine._last_fallback_notice = "Notice: Chrome profile locked. Switched to isolated."

        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            err_res = await browser_engine.open_page("http://broken-domain.local", session="err_sess", headed=True)
            assert "Error navigating to http://broken-domain.local" in err_res
            assert "[Notice: Chrome profile locked. Switched to isolated.]" in err_res
            # Notice MUST be reset immediately
            assert browser_engine._last_fallback_notice is None

        # Next call to open_page must NOT have the old notice
        sc2 = create_mock_session()
        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc2)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                ok_res = await browser_engine.open_page("http://localhost:3000", session="clean_sess", headed=True)
                assert "[Notice:" not in ok_res

    @pytest.mark.lmtrust(layer="L4", direction="D5")
    @pytest.mark.asyncio
    async def test_l4_shutdown_cleans_up_stale_notices(self):
        """Calling shutdown() clears _last_fallback_notice."""
        browser_engine._last_fallback_notice = "Stale notice"
        await browser_engine.shutdown()
        assert browser_engine._last_fallback_notice is None


# ===========================================================================
# L4b Fallback: All 4 demotion paths and single-shot notice invariant
# ===========================================================================
class TestFallbackDemotionPaths:
    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    @pytest.mark.asyncio
    async def test_l4b_path1_headless_in_user_mode(self):
        """user mode + headed=False -> fallback to isolated headless with notice."""
        with patch("config.get_browser_mode", return_value="user"):
            with patch.object(browser_engine, "_launch_isolated", new=AsyncMock()) as mock_isolated:
                await browser_engine._connect_user_browser(headed=False)
                mock_isolated.assert_awaited_once_with(False)
                assert "Headless mode requested in 'user' browser mode" in browser_engine._last_fallback_notice

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    @pytest.mark.asyncio
    async def test_l4b_path2_browser_executable_cannot_start(self):
        """user mode + the saved browser cannot be started -> isolated, with a notice.

        (This slot used to test "already open without remote debugging -> profile
        lock notice". Argent now drives its own profile directory, so a running
        personal browser is no obstacle and that fallback is gone; an executable
        that is missing or blocked is the failure that remains.)"""
        fake_info = BrowserInfo(name="Google Chrome", key="chrome", exe_path="C:/Chrome/chrome.exe", user_data_dir="C:/Chrome/User Data")
        with patch("config.get_browser_mode", return_value="user"):
            with patch("config.get_browser_name", return_value="chrome"):
                with patch("browser_detect.find_browser", return_value=fake_info):
                    with patch("subprocess.Popen", side_effect=FileNotFoundError("chrome.exe is gone")):
                        with patch.object(browser_engine, "_launch_isolated", new=AsyncMock()) as mock_isolated:
                            await browser_engine._connect_user_browser(headed=True)
                            mock_isolated.assert_awaited_once_with(True)
                            assert "Could not start Google Chrome" in browser_engine._last_fallback_notice

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    @pytest.mark.asyncio
    async def test_l4b_path3_no_browser_installed(self):
        """user mode + no browser found -> missing browser notice."""
        with patch("config.get_browser_mode", return_value="user"):
            with patch("config.get_browser_name", return_value="chrome"):
                with patch("browser_detect.find_browser", return_value=None):
                    with patch.object(browser_engine, "_launch_isolated", new=AsyncMock()) as mock_isolated:
                        await browser_engine._connect_user_browser(headed=True)
                        mock_isolated.assert_awaited_once_with(True)
                        assert "No supported Chromium-based user browser" in browser_engine._last_fallback_notice

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    @pytest.mark.asyncio
    async def test_l4b_single_shot_notice_does_not_repeat(self):
        """Fallback notice is emitted once and NEVER repeated on subsequent actions."""
        sc = create_mock_session()
        browser_engine._last_fallback_notice = "Notice: Switched to isolated."

        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                # 1st call: contains notice
                res1 = await browser_engine.open_page("http://localhost:3000", session="rep_sess", headed=True)
                assert "[Notice: Switched to isolated.]" in res1

                # 2nd call in same or another session: notice is GONE
                res2 = await browser_engine.open_page("http://localhost:3000/next", session="rep_sess", headed=True)
                assert "[Notice:" not in res2


# ===========================================================================
# L8 Temporal: Headless to Headed session transition
# ===========================================================================
class TestTemporalLifecycle:
    @pytest.mark.lmtrust(layer="L8", direction="D6")
    @pytest.mark.asyncio
    async def test_l8_session_switches_from_headless_to_headed_cleanly(self):
        """Calling open_page with headed=False then headed=True cleanly transitions the engine."""
        sc_headless = create_mock_session()
        sc_headed = create_mock_session()

        async def fake_create_session(session_name, headed=False, **kwargs):
            browser_engine._headed = headed
            return sc_headed if headed else sc_headless

        with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
            with patch.object(browser_engine, "shutdown", new=AsyncMock()) as mock_shutdown:
                with patch.object(browser_engine, "_create_session", new=AsyncMock(side_effect=fake_create_session)):
                    # Step 1: Headless call
                    browser_engine._headed = False
                    res1 = await browser_engine.open_page("http://localhost:3000", session="dyn", headed=False)
                    assert "Window: HEADLESS" in res1
                    # Store session so the next call sees the existing session
                    browser_engine._sessions["dyn"] = sc_headless

                    # Step 2: Headed call (headed state changed: headed=True vs self._headed=False)
                    res2 = await browser_engine.open_page("http://localhost:3000", session="dyn", headed=True)
                    mock_shutdown.assert_awaited()
                    assert "Window: VISIBLE" in res2


# ===========================================================================
# L9 Negative Space: Clean execution produces no spurious notices
# ===========================================================================
class TestNegativeSpace:
    @pytest.mark.lmtrust(layer="L9", direction="D3")
    @pytest.mark.asyncio
    async def test_l9_clean_isolated_run_contains_no_notices(self):
        """In normal isolated mode without fallbacks, no [Notice: ...] should ever appear."""
        sc = create_mock_session()
        with patch.object(browser_engine, "_create_session", new=AsyncMock(return_value=sc)):
            with patch.object(browser_engine, "_format_action_result", new=AsyncMock(side_effect=lambda msg, s, return_state: msg)):
                res = await browser_engine.open_page("http://localhost:3000", session="clean", headed=False)
                assert "[Notice:" not in res
                assert "Window: HEADLESS" in res
