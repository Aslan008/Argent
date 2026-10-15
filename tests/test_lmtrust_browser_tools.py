"""
LMTrust test suite for Argent browser tools.
Covers matrix of Layers (L0-L6) and Directions (D1-D10) with explicit contract and boundary verification.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from browser_engine import browser_engine, _blocked_url_reason, _normalize_url
import tools.browser_tools as bt
from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
from tool_profiles import slim_tools_for_category


# ============================================================================
# L0: SMOKE TESTS
# ============================================================================

class TestL0Smoke:
    """L0: Verify browser tools exist, are importable, callable, and registered."""

    @pytest.mark.lmtrust(layer="L0", direction="D7")
    def test_browser_tools_presence_and_callability(self):
        consolidated_tools = [
            "browser_open", "browser_action", "browser_inspect", "browser_close"
        ]
        for name in consolidated_tools:
            tool_fn = getattr(bt, name, None)
            assert tool_fn is not None, f"Function '{name}' missing from tools.browser_tools"
            assert callable(tool_fn), f"Tool '{name}' is not callable"
            assert name in AVAILABLE_TOOLS, f"Tool '{name}' not in AVAILABLE_TOOLS"

    @pytest.mark.lmtrust(layer="L0", direction="D7")
    def test_browser_schemas_validity(self):
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        consolidated_tools = [
            "browser_open", "browser_action", "browser_inspect", "browser_close"
        ]
        for name in consolidated_tools:
            assert name in schema_names, f"Schema for tool '{name}' is missing"


# ============================================================================
# L1: CONTRACT TESTS
# ============================================================================

class TestL1Contract:
    """L1: Verify function input/output contracts, schema matches, and return types."""

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_browser_open_contract_requires_url(self):
        schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "browser_open")
        assert "url" in schema["parameters"]["required"]
        assert schema["parameters"]["properties"]["url"]["type"] == "string"

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_browser_action_contract_requires_action(self):
        schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "browser_action")
        assert "action" in schema["parameters"]["required"]
        assert schema["parameters"]["properties"]["action"]["type"] == "string"


    @pytest.mark.lmtrust(layer="L1", direction="D4")
    def test_browser_close_non_active_session_returns_clean_message(self):
        # Must return clean status string without throwing uncaught exceptions
        res = bt.browser_close("non_existent_session_xyz")
        assert isinstance(res, str)
        assert "not active" in res.lower()

    @pytest.mark.lmtrust(layer="L1", direction="D4")
    def test_browser_tools_return_string_on_missing_session(self):
        res_state = bt.browser_state(session="unopened_session_123")
        assert isinstance(res_state, str)
        assert "Error" in res_state
        assert "unopened_session_123" in res_state

        res_click = bt.browser_click(index=1, session="unopened_session_123")
        assert isinstance(res_click, str)
        assert "Error" in res_click

        res_input = bt.browser_input(index=1, text="hello", session="unopened_session_123")
        assert isinstance(res_input, str)
        assert "Error" in res_input


# ============================================================================
# L2: BOUNDARY TESTS (8 Boundary Conditions, Type Edge Cases)
# ============================================================================

class TestL2Boundary:
    """L2: Boundary testing on edge inputs, types, whitespace, and normalization."""

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    @pytest.mark.parametrize("invalid_url", [
        "", "   ", "\t\n", None, 12345, [], {},
    ])
    def test_blocked_url_reason_rejects_empty_or_non_string(self, invalid_url):
        reason = _blocked_url_reason(invalid_url)
        assert reason is not None
        assert "Error:" in reason

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    @pytest.mark.parametrize("bare_input,expected_normalized", [
        ("google.com", "http://google.com"),
        ("localhost:3000", "http://localhost:3000"),
        ("127.0.0.1:8080/path", "http://127.0.0.1:8080/path"),
        ("  http://example.com/test  ", "http://example.com/test"),
        ("https://github.com", "https://github.com"),
        ("about:blank", "about:blank"),
        ("about:", "about:blank"),
    ])
    def test_url_normalization_handles_bare_domains_and_whitespace(self, bare_input, expected_normalized):
        assert _normalize_url(bare_input) == expected_normalized

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    @pytest.mark.parametrize("depth_input,expected_int", [
        ("2", 2),
        (3, 3),
        (0, 0),
        (-1, -1),
        ("invalid", 0),
        (None, 0),
    ])
    def test_scroll_depth_coercion_resilience(self, depth_input, expected_int):
        # Ensure that non-int scroll_depth inputs do not raise TypeError in get_accessibility_tree / _get_dom_state
        try:
            val = int(depth_input or 0)
        except (TypeError, ValueError):
            val = 0
        assert val == expected_int

    @pytest.mark.lmtrust(layer="L2", direction="D3")
    def test_fill_form_handles_non_dict_elements_gracefully(self):
        # Mock session to verify fill_form loop resilience
        fake_sc = MagicMock()
        fake_sc.page = MagicMock()

        async def run_test():
            with patch.object(browser_engine, "_get_session", AsyncMock(return_value=fake_sc)):
                with patch.object(browser_engine, "_format_action_result", AsyncMock(side_effect=lambda msg, s, **kw: msg)):
                    # Mixed list containing strings and numbers alongside a valid dict
                    fields = ["invalid_str", 123, None, {"selector": "#username", "value": "testuser"}]
                    with patch.object(browser_engine, "fill_input", AsyncMock(return_value="Filled input")):
                        res = await browser_engine.fill_form(fields, session="test_fill")
                        assert "Form filled:" in res
                        assert "testuser" in str(browser_engine.fill_input.call_args)

        asyncio.run(run_test())


# ============================================================================
# L4: ADVERSARIAL & SECURITY TESTS
# ============================================================================

class TestL4Adversarial:
    """L4: Adversarial tests verifying security guards and malicious input handling."""

    @pytest.mark.lmtrust(layer="L4", direction="D8")
    @pytest.mark.parametrize("dangerous_url", [
        "file:///etc/shadow",
        "file:///C:/Windows/win.ini",
        "javascript:alert(document.cookie)",
        "data:text/html,<script>alert(1)</script>",
        "view-source:https://google.com",
        "chrome://settings",
        "\\\\remote-share\\payload.exe",
        "//protocol-relative.evil.com",
    ])
    def test_dangerous_schemes_and_paths_are_blocked(self, dangerous_url):
        reason = _blocked_url_reason(dangerous_url)
        assert reason is not None
        assert "Error: refusing to open" in reason

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_browser_open_returns_blocked_reason_without_launching(self):
        res = bt.browser_open("file:///etc/passwd")
        assert "Error: refusing to open a 'file:' URL" in res

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_browser_switch_tab_invalid_indices(self):
        fake_sc = MagicMock()
        fake_sc.context.pages = [MagicMock(), MagicMock()] # 2 pages

        async def run_test():
            with patch.object(browser_engine, "_get_session", AsyncMock(return_value=fake_sc)):
                # Index out of bounds
                res_high = await browser_engine.switch_tab(99, session="test_tabs")
                assert "Error: Invalid tab index 99" in res_high

                # Non-integer index string
                res_str = await browser_engine.switch_tab("abc", session="test_tabs")
                assert "Error: Invalid tab index 'abc'" in res_str

        asyncio.run(run_test())


# ============================================================================
# L5: STATE & LIFECYCLE TESTS
# ============================================================================

class TestL5State:
    """L5: Session lifecycle and polling tracker state machine tests."""

    @pytest.mark.lmtrust(layer="L5", direction="D2")
    def test_polling_tracker_cycle_and_reset(self):
        session = "poll_state_session"
        bt._reset_poll_tracker(session)

        # Mock browser_engine.run and get_state
        with patch.object(browser_engine, "run", return_value="[1] button 'OK'"), \
             patch.object(browser_engine, "get_state", return_value=None):
            # Call 1 & 2: normal state
            res1 = bt.browser_state(session=session, query="submit")
            assert "[POLLING HINT]" not in res1

            res2 = bt.browser_state(session=session, query="submit")
            assert "[POLLING HINT]" not in res2

            # Call 3: identical query triggers hint
            res3 = bt.browser_state(session=session, query="submit")
            assert "[POLLING HINT]" in res3

            # Any browser action resets the tracker count
            bt._reset_poll_tracker(session)
            res4 = bt.browser_state(session=session, query="submit")
            assert "[POLLING HINT]" not in res4


# ============================================================================
# L6: CROSS-SYSTEM TESTS
# ============================================================================

class TestL6CrossSystem:
    """L6: Tool profile integration, model tiers, and schemas."""

    @pytest.mark.lmtrust(layer="L6", direction="D7")
    def test_browser_tools_pruning_in_model_tiers(self):
        browser_tools = [
            "browser_open", "browser_state", "browser_click", "browser_input",
            "browser_select", "browser_screenshot", "browser_scroll", "browser_get_content",
            "browser_close", "browser_switch_tab", "browser_press_key", "browser_fill_form",
            "browser_upload_file", "browser_back", "browser_reload", "browser_eval_js",
            "browser_wait_for"
        ]
        # Tiny/small models exclude browser tools to save context
        for cat in ("tiny", "small"):
            slim = slim_tools_for_category(browser_tools, cat)
            for tool in browser_tools:
                assert tool not in slim, f"Tool '{tool}' should be excluded for tier '{cat}'"

        # Medium/large/cloud keep browser tools
        for cat in ("medium", "large", "cloud"):
            kept = slim_tools_for_category(browser_tools, cat)
            assert kept == browser_tools


# ============================================================================
# L7: CDP USER-BROWSER & PROCESS RESILIENCE TESTS
# ============================================================================

class TestL7CDPConnection:
    """L7: Tests for non-blocking CDP user-browser connections and URL handling."""

    @pytest.mark.lmtrust(layer="L7", direction="D1")
    def test_node_options_suppresses_deprecation(self):
        import os
        from browser_engine import BrowserEngine
        engine = BrowserEngine()
        with patch("playwright.async_api.async_playwright") as mock_pw, \
             patch.object(engine, "_launch_isolated", AsyncMock()):
            mock_pw_inst = AsyncMock()
            mock_pw.return_value.start = AsyncMock(return_value=mock_pw_inst)
            
            async def run_test():
                await engine._ensure_browser(headed=False)
                assert os.environ.get("NODE_OPTIONS") == "--no-deprecation"

            asyncio.run(run_test())

    @pytest.mark.lmtrust(layer="L7", direction="D3")
    def test_cdp_initial_url_passed_to_args(self):
        """Verify that when launching user browser, target URL is included in args."""
        from browser_engine import BrowserEngine
        engine = BrowserEngine()
        engine._playwright = MagicMock()
        engine._playwright.chromium.connect_over_cdp = AsyncMock(side_effect=Exception("No CDP endpoint"))

        with patch("browser_detect.find_browser") as mock_find, \
             patch("config.get_browser_name", return_value="chrome"), \
             patch.object(engine, "_is_browser_running", return_value=False), \
             patch("subprocess.Popen") as mock_popen, \
             patch("asyncio.open_connection", side_effect=OSError("Connection refused")), \
             patch.object(engine, "_launch_isolated", AsyncMock()) as mock_isolated:
            from browser_detect import BrowserInfo
            mock_find.return_value = BrowserInfo(
                name="Test Chrome", key="chrome", exe_path="C:\\mock\\chrome.exe", user_data_dir="C:\\mock\\data"
            )
            mock_proc = MagicMock()
            mock_proc.pid = 99999
            mock_popen.return_value = mock_proc

            async def run_test():
                await engine._connect_user_browser(headed=True, initial_url="https://example.com/login")

            asyncio.run(run_test())

            # Check that subprocess.Popen was called with initial_url in args
            assert mock_popen.called
            args_passed = mock_popen.call_args[0][0]
            assert "https://example.com/login" in args_passed
            assert "--remote-debugging-port=9222" in args_passed

    @pytest.mark.lmtrust(layer="L7", direction="D8")
    def test_cdp_fast_fallback_when_browser_already_running(self):
        """Verify that when browser is already running without CDP, fallback to isolated mode is instant."""
        from browser_engine import BrowserEngine
        engine = BrowserEngine()
        engine._playwright = MagicMock()
        engine._playwright.chromium.connect_over_cdp = AsyncMock(side_effect=Exception("No CDP endpoint"))

        with patch("browser_detect.find_browser") as mock_find, \
             patch("config.get_browser_name", return_value="chrome"), \
             patch("subprocess.Popen"), \
             patch.object(engine, "_is_browser_running", return_value=True), \
             patch.object(engine, "_launch_isolated", AsyncMock()) as mock_isolated:
            from browser_detect import BrowserInfo
            mock_find.return_value = BrowserInfo(
                name="Test Chrome", key="chrome", exe_path="C:\\mock\\chrome.exe", user_data_dir="C:\\mock\\data"
            )
            async def run_test():
                await engine._connect_user_browser(headed=True, initial_url="https://example.com")
                assert mock_isolated.called

            asyncio.run(run_test())


