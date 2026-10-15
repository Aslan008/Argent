"""Unit and integration tests for Stagehand v4 semantic browser tools."""

import pytest
from unittest.mock import patch, MagicMock

import tools.stagehand_tools as sht
from tools.stagehand_tools import (
    browser_act,
    browser_extract,
    browser_observe,
    reset_stagehand,
    _get_llm_config,
)
from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
from browser_engine import browser_engine


class TestStagehandRegistration:
    """Verify Stagehand integration into consolidated browser tools and exports."""

    def test_tools_registered_in_available_tools(self):
        for name in ("browser_action", "browser_inspect"):
            assert name in AVAILABLE_TOOLS, f"{name} is missing from AVAILABLE_TOOLS"
            assert callable(AVAILABLE_TOOLS[name])

    def test_tools_present_in_tool_schemas(self):
        schema_map = {s["function"]["name"]: s["function"] for s in TOOL_SCHEMAS}
        assert "browser_action" in schema_map
        assert "browser_inspect" in schema_map

        # Verify Stagehand actions are documented in schemas
        act_schema = schema_map["browser_action"]
        assert "act" in act_schema["parameters"]["properties"]["action"]["enum"]
        assert "instruction" in act_schema["parameters"]["properties"]

        inspect_schema = schema_map["browser_inspect"]
        assert "observe" in inspect_schema["parameters"]["properties"]["mode"]["enum"]
        assert "extract" in inspect_schema["parameters"]["properties"]["mode"]["enum"]

    def test_reexported_in_browser_tools(self):
        import tools.browser_tools as bt
        assert hasattr(bt, "browser_act")
        assert hasattr(bt, "browser_extract")
        assert hasattr(bt, "browser_observe")
        assert hasattr(bt, "reset_stagehand")
        assert hasattr(bt, "browser_action")
        assert hasattr(bt, "browser_inspect")


class TestStagehandWithoutBrowser:
    """Verify safe error reporting when no browser session is active."""

    def setup_method(self):
        reset_stagehand()
        browser_engine.run(browser_engine.shutdown())

    def test_browser_act_no_session(self):
        res = browser_act("click on the button", session="nonexistent_session")
        assert "Session 'nonexistent_session' does not exist or browser is not open" in res
        assert "browser_open" in res

    def test_browser_extract_no_session(self):
        res = browser_extract("extract article title", session="nonexistent_session")
        assert "Session 'nonexistent_session' does not exist or browser is not open" in res

    def test_browser_observe_no_session(self):
        res = browser_observe("look for navigation", session="nonexistent_session")
        assert "Session 'nonexistent_session' does not exist or browser is not open" in res


class TestStagehandFallbackWhenUnavailable:
    """Verify graceful fallback messages when stagehand library is not installed."""

    def test_browser_act_fallback(self, monkeypatch):
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = browser_act("click login")
        assert "Stagehand v4 is not installed" in res
        assert "browser_inspect" in res
        assert "browser_action" in res

    def test_browser_extract_fallback(self, monkeypatch):
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = browser_extract("extract table")
        assert "Stagehand v4 is not installed" in res
        assert 'browser_inspect(mode="text_content")' in res or 'action="eval_js"' in res

    def test_browser_observe_fallback(self, monkeypatch):
        monkeypatch.setattr(sht, "STAGEHAND_AVAILABLE", False)
        res = browser_observe()
        assert "Stagehand v4 is not installed" in res
        assert "browser_inspect" in res


class TestStagehandLifecycleAndConfig:
    """Verify lifecycle, singleton reset, and LLM configuration resolution."""

    def test_reset_stagehand_idempotent(self):
        # Should not raise even when no instance is cached
        reset_stagehand()
        assert sht._STAGEHAND_INSTANCE is None
        reset_stagehand()
        assert sht._STAGEHAND_INSTANCE is None

    def test_reset_stagehand_clears_active_instance(self):
        dummy = MagicMock()
        sht._STAGEHAND_INSTANCE = dummy
        reset_stagehand()
        assert sht._STAGEHAND_INSTANCE is None

    def test_llm_config_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-12345")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = _get_llm_config()
        assert cfg["model"] == "openai/gpt-4o"
        assert cfg["model_api_key"] == "sk-test-openai-12345"

    def test_llm_config_anthropic(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-12345")
        cfg = _get_llm_config()
        assert cfg["model"] == "anthropic/claude-3-7-sonnet"
        assert cfg["model_api_key"] == "sk-ant-test-12345"

    def test_llm_config_native_argent_bridge(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr("config.get_openrouter_api_key", lambda: None)
        cfg = _get_llm_config()
        assert "model" in cfg
        # In native fallback, returns ClientLLM instance
        assert isinstance(cfg["model"], sht.ClientLLM)


class TestBrowserEngineCdpIntegration:
    """Verify browser_engine CDP exposure methods."""

    def setup_method(self):
        reset_stagehand()
        browser_engine.run(browser_engine.shutdown())

    def test_get_cdp_url_when_not_running(self):
        # When browser is not launched, get_cdp_url must return None
        assert browser_engine.get_cdp_url() is None

    def test_find_free_port(self):
        port = browser_engine._find_free_port(9222)
        assert isinstance(port, int)
        assert 1024 <= port <= 65535
