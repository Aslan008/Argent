"""
tests/test_browser_loop_prevention.py

Unit tests verifying:
1. browser_eval_js polling tracking, hints, and circuit breaker.
2. browser_wait_for with JS expressions and enabled/disabled states.
3. DOM state rendering with [disabled] attribute for inactive controls.
4. Browser session stability (tab closure recovery, non-blank preference on reload).
5. LoopGuard protection for cloud models.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from tools.browser_tools import (
    browser_eval_js,
    browser_wait_for,
    _reset_poll_tracker,
    _SESSION_EVAL_TRACKER,
    _SESSION_POLL_TRACKER,
)
from browser_engine import BrowserEngine, _SessionContext
from src.agent.loop_guard import LoopGuard
from src.agent.strategy import get_model_strategy, CloudStrategy


# ---------------------------------------------------------------------------
# 1. browser_eval_js Repetition Tracking & Circuit Breaker
# ---------------------------------------------------------------------------

class TestBrowserEvalJsLoopPrevention:
    def setup_method(self):
        _reset_poll_tracker("default")
        _SESSION_EVAL_TRACKER.clear()

    def test_single_eval_returns_normal_result(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = 'JS result: {"status": "ok"}'
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        res = browser_eval_js("document.title")
        assert res == 'JS result: {"status": "ok"}'
        assert "[POLLING HINT]" not in res
        assert "[POLLING HALTED]" not in res

    def test_repeated_eval_injects_hint_at_3(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = 'JS result: {"save": "disabled"}'
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        res1 = browser_eval_js("checkSave()")
        assert "[POLLING HINT]" not in res1

        res2 = browser_eval_js("checkSave()")
        assert "[POLLING HINT]" not in res2

        res3 = browser_eval_js("checkSave()")
        assert "[POLLING HINT]" in res3
        assert 'JS result: {"save": "disabled"}' in res3

    def test_repeated_eval_halts_at_5(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = 'JS result: {"save": "disabled"}'
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        for _ in range(4):
            browser_eval_js("checkSave()")

        res5 = browser_eval_js("checkSave()")
        assert "[POLLING HALTED]" in res5
        assert "Blind polling has been halted" in res5
        assert 'browser_action(action="wait_for"' in res5
        assert "browser_wait_for" not in res5      # that tool no longer exists

    def test_different_expression_resets_counter(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = 'JS result: 0'
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        browser_eval_js("expr1()")
        browser_eval_js("expr1()")

        # Changing expression resets count to 1
        res = browser_eval_js("expr2()")
        assert "[POLLING HINT]" not in res
        assert _SESSION_EVAL_TRACKER["default"]["count"] == 1

    def test_result_change_resets_counter(self, monkeypatch):
        mock_engine = MagicMock()
        results = ['JS result: "loading"', 'JS result: "loading"', 'JS result: "ready"']
        mock_engine.run.side_effect = results
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        browser_eval_js("status()")
        browser_eval_js("status()")
        res3 = browser_eval_js("status()")
        assert "[POLLING HINT]" not in res3
        assert "ready" in res3
        assert _SESSION_EVAL_TRACKER["default"]["count"] == 1

    def test_action_resets_eval_tracker(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = 'JS result: 0'
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        browser_eval_js("expr()")
        browser_eval_js("expr()")
        assert _SESSION_EVAL_TRACKER["default"]["count"] == 2

        _reset_poll_tracker("default")
        assert _SESSION_EVAL_TRACKER["default"]["count"] == 0


# ---------------------------------------------------------------------------
# 2. browser_wait_for Expressions and States
# ---------------------------------------------------------------------------

class TestBrowserWaitForEnhancements:
    @pytest.mark.asyncio
    async def test_wait_for_expression_success(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.wait_for_function = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example")

        sc = _SessionContext(context=MagicMock(), page=mock_page)
        engine._sessions["default"] = sc

        res = await engine.wait_for(expression="!document.querySelector('button')?.disabled", session="default", return_state=False)
        assert "JavaScript condition" in res
        assert "satisfied" in res
        mock_page.wait_for_function.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wait_for_expression_timeout(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.wait_for_function = AsyncMock(side_effect=Exception("Timeout 15000ms exceeded"))

        sc = _SessionContext(context=MagicMock(), page=mock_page)
        engine._sessions["default"] = sc

        res = await engine.wait_for(expression="false", timeout=2.0, session="default", return_state=False)
        assert "Timeout waiting 2.0s" in res

    @pytest.mark.asyncio
    async def test_wait_for_enabled_state(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_locator = MagicMock()
        mock_locator.wait_for = AsyncMock()
        # First check disabled=True, second check disabled=False
        mock_locator.is_disabled = AsyncMock(side_effect=[True, False])
        mock_page.locator.return_value.first = mock_locator
        mock_page.url = "https://example.com"

        sc = _SessionContext(context=MagicMock(), page=mock_page)
        engine._sessions["default"] = sc

        res = await engine.wait_for(selector="button.save", state="enabled", timeout=2.0, session="default", return_state=False)
        assert "reached state 'enabled'" in res

    @pytest.mark.asyncio
    async def test_wait_for_invalid_state(self):
        engine = BrowserEngine()
        sc = _SessionContext(context=MagicMock(), page=MagicMock())
        engine._sessions["default"] = sc

        res = await engine.wait_for(selector="button.save", state="exploded", session="default", return_state=False)
        assert "Error: Invalid state 'exploded'" in res


# ---------------------------------------------------------------------------
# 3. DOM State Rendering of [disabled]
# ---------------------------------------------------------------------------

class TestDomStateDisabledRendering:
    @pytest.mark.asyncio
    async def test_get_dom_state_shows_disabled(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_frame = MagicMock()
        mock_page.frames = [mock_frame]
        mock_page.main_frame = mock_frame
        mock_context = MagicMock()
        mock_context.pages = [mock_page]

        fake_elements = [
            {"idx": 1, "tag": "button", "label": "Save", "disabled": True},
            {"idx": 2, "tag": "button", "label": "Cancel", "disabled": False},
        ]
        mock_frame.evaluate = AsyncMock(return_value={
            "elements": fake_elements,
            "nextIdx": 3,
            "totalFiltered": 0,
            "url": "https://example.com",
            "title": "Test Page",
        })

        sc = _SessionContext(context=mock_context, page=mock_page)
        engine._sessions["default"] = sc

        state = await engine._get_dom_state("default")
        lines = state.splitlines()

        save_line = next(l for l in lines if '[1] button "Save"' in l)
        assert "[disabled]" in save_line

        cancel_line = next(l for l in lines if '[2] button "Cancel"' in l)
        assert "[disabled]" not in cancel_line


# ---------------------------------------------------------------------------
# 4. Session Stability and Page Fallback
# ---------------------------------------------------------------------------

class TestSessionStability:
    @pytest.mark.asyncio
    async def test_closed_page_recovers_alive_content_page(self):
        engine = BrowserEngine()
        mock_context = MagicMock()

        dead_page = MagicMock()
        dead_page.is_closed.return_value = True

        alive_yandex = MagicMock()
        alive_yandex.is_closed.return_value = False
        alive_yandex.url = "https://games.yandex.ru/console"

        mock_context.pages = [dead_page, alive_yandex]

        sc = _SessionContext(context=mock_context, page=dead_page)
        engine._sessions["default"] = sc

        res_sc = await engine._get_session("default")
        assert res_sc.page == alive_yandex
        assert res_sc.page.url == "https://games.yandex.ru/console"

    @pytest.mark.asyncio
    async def test_reload_switches_from_blank_to_content_page(self):
        engine = BrowserEngine()
        mock_context = MagicMock()

        blank_page = MagicMock()
        blank_page.is_closed.return_value = False
        blank_page.url = "about:blank"

        content_page = MagicMock()
        content_page.is_closed.return_value = False
        content_page.url = "https://games.yandex.ru/console"
        content_page.reload = AsyncMock()
        content_page.title = AsyncMock(return_value="Console")

        mock_context.pages = [blank_page, content_page]

        sc = _SessionContext(context=mock_context, page=blank_page)
        engine._sessions["default"] = sc

        msg = await engine.reload("default", return_state=False)
        assert "https://games.yandex.ru/console" in msg
        content_page.reload.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. LoopGuard Protection for Cloud Models
# ---------------------------------------------------------------------------

class TestLoopGuardCloudProtection:
    def test_cloud_strategy_enables_loop_guard(self):
        strat = get_model_strategy("glm-5.3-flash", "zai")
        assert isinstance(strat, CloudStrategy)
        assert strat.is_loop_guard_enabled() is True

    def test_loop_guard_catches_cloud_verbatim_loop(self):
        guard = LoopGuard()
        args = {"expression": "document.querySelector('button.save').disabled"}
        res_str = 'JS result: true'

        # 1st call
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") is None
        # 2nd call -> warn
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") == "warn"
        # 3rd call -> warn
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") == "warn"
        # 4th call -> nudge (sobering alert, doesn't kill the turn!)
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") == "nudge"
        # 5th call -> nudge
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") == "nudge"
        # 6th call -> stop (emergency ceiling)
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") == "stop"

    def test_loop_guard_small_model_strict_stop_at_4(self):
        guard = LoopGuard()
        args = {"command": "npm test"}
        res_str = "failed"

        assert guard.record("run_command", args, res_str, enabled=True, category="small") is None
        assert guard.record("run_command", args, res_str, enabled=True, category="small") == "warn"
        assert guard.record("run_command", args, res_str, enabled=True, category="small") == "warn"
        assert guard.record("run_command", args, res_str, enabled=True, category="small") == "stop"


# ---------------------------------------------------------------------------
# 6. LoopGuard User Control (/loopguard on | off | auto & config)
# ---------------------------------------------------------------------------

class TestLoopGuardUserControl:
    def test_config_get_set_loop_guard_mode(self, monkeypatch):
        import config
        stored = {}
        monkeypatch.setattr(config, "_get", lambda key, default=None: stored.get(key, default))
        monkeypatch.setattr(config, "_set", lambda key, val: stored.__setitem__(key, val))

        assert config.get_loop_guard_mode() == "auto"

        config.set_loop_guard_mode("on")
        assert config.get_loop_guard_mode() == "on"

        config.set_loop_guard_mode("off")
        assert config.get_loop_guard_mode() == "off"

        config.set_loop_guard_mode("AUTO")
        assert config.get_loop_guard_mode() == "auto"

        with pytest.raises(ValueError, match="Invalid loop_guard mode"):
            config.set_loop_guard_mode("unknown_mode")

    def test_loop_guard_respects_user_off_mode(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "get_loop_guard_mode", lambda: "off")

        guard = LoopGuard()
        args = {"expression": "1+1"}
        res_str = "2"

        # Even with 10 identical calls, mode='off' always returns None
        for _ in range(10):
            assert guard.record("browser_eval_js", args, res_str, enabled=True, category="cloud") is None

    def test_loop_guard_protects_moe_with_nudge_and_stop(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "get_loop_guard_mode", lambda: "auto")

        guard = LoopGuard()
        args = {"expression": "1+1"}
        res_str = "2"

        # In auto mode, category='moe' receives warn at 2, nudge at 4, stop at 6
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="moe") is None
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="moe") == "warn"
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="moe") == "warn"
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="moe") == "nudge"
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="moe") == "nudge"
        assert guard.record("browser_eval_js", args, res_str, enabled=True, category="moe") == "stop"

    def test_cli_slash_command_loopguard(self, monkeypatch):
        from src.cli.commands import dispatch_command, CommandContext
        import config

        stored = {}
        monkeypatch.setattr(config, "_get", lambda key, default=None: stored.get(key, default))
        monkeypatch.setattr(config, "_set", lambda key, val: stored.__setitem__(key, val))

        mock_agent = MagicMock()

        # 1. /loopguard on
        ctx_on = CommandContext(agent=mock_agent, user_input="/loopguard on")
        res_on = dispatch_command("/loopguard on", ctx_on)
        assert res_on.handled is True
        assert stored.get("loop_guard_mode") == "on"

        # 2. /loopguard off
        ctx_off = CommandContext(agent=mock_agent, user_input="/loopguard off")
        res_off = dispatch_command("/loopguard off", ctx_off)
        assert res_off.handled is True
        assert stored.get("loop_guard_mode") == "off"

        # 3. /loopguard auto
        ctx_auto = CommandContext(agent=mock_agent, user_input="/loopguard auto")
        res_auto = dispatch_command("/loopguard auto", ctx_auto)
        assert res_auto.handled is True
        assert stored.get("loop_guard_mode") == "auto"

        # 4. /loopguard query (no args)
        ctx_query = CommandContext(agent=mock_agent, user_input="/loopguard")
        res_query = dispatch_command("/loopguard", ctx_query)
        assert res_query.handled is True
        assert res_query.should_continue is True


# ---------------------------------------------------------------------------
# 7. Main Agent System Prompt Browser Protocol
# ---------------------------------------------------------------------------

class TestMainAgentBrowserPrompt:
    def test_browser_protocol_included_when_browser_tools_available(self):
        from agent import ArgentAgent
        agent = ArgentAgent()
        agent.model_name = "glm-5.3-flash"
        prompt = agent.build_system_prompt(available_tools=["browser_open", "browser_action", "wait_heartbeat", "read_file"])
        assert "BROWSER AUTOMATION & WAITING PROTOCOL" in prompt
        assert 'browser_action(action="wait_for"' in prompt
        assert "wait_heartbeat" in prompt
        assert "CANNOT see browser DOM state" in prompt

    def test_browser_protocol_omitted_when_browser_tools_unavailable(self):
        from agent import ArgentAgent
        agent = ArgentAgent()
        agent.model_name = "glm-5.3-flash"
        prompt = agent.build_system_prompt(available_tools=["read_file", "write_file", "run_command"])
        assert "BROWSER AUTOMATION & WAITING PROTOCOL" not in prompt


# ---------------------------------------------------------------------------
# 8. State-Changing Tools Reset Verification
# ---------------------------------------------------------------------------

class TestStateChangingReset:
    def test_browser_wait_for_and_mutations_in_state_changing_tools(self):
        import inspect
        import agent as agent_mod
        src = inspect.getsource(agent_mod.ArgentAgent._run_turn)
        # wait_for is a verb of browser_action now; the agent reads the verb.
        from src.agent.semantic_loop_guard import is_page_changing_browser_action
        assert is_page_changing_browser_action({"action": "wait_for"})
        assert "is_page_changing_browser_action" in src
        assert '"delete_file"' in src
        assert '"create_directory"' in src
        assert '"move_file"' in src

    def test_loop_guard_reset_on_browser_wait_for(self):
        from src.agent.loop_guard import LoopGuard
        guard = LoopGuard()
        guard.record("browser_eval_js", {"expr": "1"}, "ok", category="cloud")
        guard.record("browser_eval_js", {"expr": "1"}, "ok", category="cloud")
        assert len(guard._recent) == 2

        # Reset simulates execution of browser_wait_for
        guard.reset()
        assert len(guard._recent) == 0
        assert guard.record("browser_eval_js", {"expr": "1"}, "ok", category="cloud") is None



