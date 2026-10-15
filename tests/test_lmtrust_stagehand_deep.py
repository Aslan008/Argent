"""
tests/test_lmtrust_stagehand_deep.py
LMTrust Deep Blind Spot & Contract Audit for Stagehand v4 Integration:
- Layer 0: Smoke & Runtime Readiness
- Layer 1: Contract & Parameter Validation (D7)
- Layer 2: Boundary & Extreme Inputs (D1, D3)
- Layer 4: Adversarial Fault Injection & Disconnections (D4, D8)
- Layer 4b: Fallback Degradation (D4)
- Layer 5: State & Singleton Lifecycle (D2, D5)
- Layer 6: Cross-System Coordination with Polling Trackers (D3, D6)
- Layer 7: Compositional Concurrency & Re-entrancy (D6)
- Layer 10: Invariant & Idempotency Proofs (D10)
"""

import pytest
import threading
from unittest.mock import patch, MagicMock, AsyncMock

import tools.stagehand_tools as sht
from tools.stagehand_tools import (
    browser_act,
    browser_extract,
    browser_observe,
    reset_stagehand,
    _get_llm_config,
)
from browser_engine import browser_engine


class TestL0Smoke:
    """Layer 0: Smoke verification of Stagehand endpoints and public symbols."""

    def test_symbols_available_and_callable(self):
        for fn in (browser_act, browser_extract, browser_observe, reset_stagehand):
            assert callable(fn), f"Symbol {fn} is not callable"


class TestL1Contract:
    """Layer 1: Input contract validation and type safety (Parse, don't validate)."""

    def test_browser_act_rejects_empty_instruction(self):
        res_empty = browser_act("")
        assert "Error: You must provide a non-empty instruction" in res_empty
        res_whitespace = browser_act("   \t\n  ")
        assert "Error: You must provide a non-empty instruction" in res_whitespace
        res_none = browser_act(None)
        assert "Error: You must provide a non-empty instruction" in res_none

    def test_browser_extract_rejects_empty_instruction(self):
        res_empty = browser_extract("")
        assert "Error: You must provide a non-empty instruction" in res_empty
        res_whitespace = browser_extract("   ")
        assert "Error: You must provide a non-empty instruction" in res_whitespace
        res_none = browser_extract(None)
        assert "Error: You must provide a non-empty instruction" in res_none

    def test_browser_extract_rejects_invalid_schema_type(self):
        res_str_schema = browser_extract("get articles", schema="invalid_string_schema")
        assert "Error: The 'schema' parameter for browser_extract must be a dictionary" in res_str_schema
        res_list_schema = browser_extract("get articles", schema=["invalid", "list"])
        assert "Error: The 'schema' parameter for browser_extract must be a dictionary" in res_str_schema

    def test_browser_observe_rejects_non_string_instruction(self):
        res_invalid = browser_observe(instruction=12345)
        assert "Error: The 'instruction' parameter for browser_observe must be a string" in res_invalid

    def test_session_must_be_open_before_execution(self):
        reset_stagehand()
        browser_engine.run(browser_engine.shutdown())
        res = browser_act("click submit", session="ghost_tab")
        assert "Session 'ghost_tab' does not exist or browser is not open" in res
        assert "browser_open" in res


class TestL2Boundary:
    """Layer 2: Boundary values, extreme prompt sizes, and Unicode edge cases."""

    def test_huge_prompt_handling_without_crash(self):
        huge_prompt = "click button " * 2000  # ~26KB string
        # Should gracefully fail at session verification without blowing up memory or regex
        res = browser_act(huge_prompt, session="nonexistent")
        assert "Session 'nonexistent' does not exist" in res

    def test_unicode_and_emoji_in_instruction(self):
        cmd = "Нажми кнопку «Войти в систему» 🚀 & выбери 'Русский (RU)'"
        res = browser_act(cmd, session="nonexistent")
        assert "Session 'nonexistent' does not exist" in res

    def test_control_chars_and_null_bytes(self):
        cmd = "type \x00\x01\x1b[31m test \r\n"
        res = browser_act(cmd, session="nonexistent")
        assert "Session 'nonexistent' does not exist" in res


class TestL4Adversarial:
    """Layer 4: Fault injection, connection drops, and simulated CDP errors."""

    def test_cdp_endpoint_missing_when_browser_headless(self, monkeypatch):
        # Simulate active session in browser_engine, but get_cdp_url returning None
        monkeypatch.setattr(browser_engine, "_sessions", {"mock_sess": MagicMock()})
        monkeypatch.setattr(browser_engine, "get_cdp_url", lambda: None)
        res = browser_act("click login", session="mock_sess")
        assert "CDP endpoint is not available" in res

    def test_runtime_exception_during_act_returns_hint(self, monkeypatch):
        monkeypatch.setattr(browser_engine, "_sessions", {"active": MagicMock()})
        monkeypatch.setattr(browser_engine, "get_cdp_url", lambda: "http://127.0.0.1:9222")

        async def fake_get_sh(sess):
            mock_sh = MagicMock()
            mock_sh.act = AsyncMock(side_effect=RuntimeError("CDP connection reset by peer"))
            return mock_sh

        monkeypatch.setattr(sht, "_get_stagehand_instance_async", fake_get_sh)
        res = browser_act("click button", session="active")
        assert "CDP connection reset by peer" in res
        # Whichever branch answers (page state available or not), the hint must
        # send the model to tools that exist.
        assert 'browser_action(action="click"' in res or "browser_inspect" in res
        assert "browser_state" not in res and "browser_click" not in res


class TestL4bFallback:
    """Layer 4b: Graceful degradation when Stagehand package is absent."""

    def test_act_fallback_guidance(self, monkeypatch):
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = browser_act("click sign in")
        assert "Stagehand v4 is not installed" in res
        assert "browser_inspect" in res
        assert "browser_action" in res

    def test_extract_fallback_guidance(self, monkeypatch):
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = browser_extract("get all links")
        assert "Stagehand v4 is not installed" in res
        assert 'browser_inspect(mode="text_content")' in res or 'action="eval_js"' in res

    def test_observe_fallback_guidance(self, monkeypatch):
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = browser_observe("find buttons")
        assert "Stagehand v4 is not installed" in res
        assert "browser_inspect" in res


class TestL5StateLifecycle:
    """Layer 5: Singleton lifecycle, invalidation, and shutdown hooks."""

    def test_singleton_reset_cleans_instance(self):
        sht._STAGEHAND_INSTANCE = MagicMock()
        assert sht._STAGEHAND_INSTANCE is not None
        reset_stagehand()
        assert sht._STAGEHAND_INSTANCE is None

    def test_browser_engine_shutdown_invokes_stagehand_reset(self):
        sht._STAGEHAND_INSTANCE = MagicMock()
        browser_engine.run(browser_engine.shutdown())
        assert sht._STAGEHAND_INSTANCE is None


class TestL6CrossSystem:
    """Layer 6: Coordination between Stagehand actions and browser loop detectors."""

    def test_act_resets_polling_tracker_on_success(self, monkeypatch):
        from tools.browser_tools import _SESSION_POLL_TRACKER, _reset_poll_tracker

        _SESSION_POLL_TRACKER["default"] = {"count": 4, "last_query": "stale"}
        monkeypatch.setattr(browser_engine, "_sessions", {"default": MagicMock()})
        monkeypatch.setattr(browser_engine, "get_cdp_url", lambda: "http://127.0.0.1:9222")

        async def fake_get_sh(sess):
            mock_sh = MagicMock()
            mock_sh.act = AsyncMock(return_value="clicked")
            return mock_sh

        monkeypatch.setattr(sht, "_get_stagehand_instance_async", fake_get_sh)
        res = browser_act("click save button", session="default")
        assert "Stagehand Act Succeeded" in res
        # Polling tracker count must have been reset to 0
        assert _SESSION_POLL_TRACKER["default"]["count"] == 0


class TestL7CompositionalConcurrency:
    """Layer 7: Multi-threaded invocation and re-entrant lock safety."""

    def test_concurrent_reset_stagehand_is_threadsafe(self):
        errors = []

        def worker():
            try:
                for _ in range(50):
                    sht._STAGEHAND_INSTANCE = MagicMock()
                    reset_stagehand()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert sht._STAGEHAND_INSTANCE is None


class TestL10Invariants:
    """Layer 10: Invariant laws and idempotency proofs."""

    def test_reset_stagehand_is_strictly_idempotent(self):
        reset_stagehand()
        reset_stagehand()
        reset_stagehand()
        assert sht._STAGEHAND_INSTANCE is None

    def test_get_llm_config_fallback_always_provides_model(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("config.get_openrouter_api_key", lambda: None)
        cfg = _get_llm_config()
        assert "model" in cfg
        assert cfg["model"] is not None
