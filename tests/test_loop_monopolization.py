"""
tests/test_loop_monopolization.py

Comprehensive tests for Anti-Doom-Loop and turn monopolization safeguards:
1. Browser eval unproductive probe detector and probing circuit breaker.
2. SemanticLoopGuard browser evaluation tracking and directives.
3. Turn Checkpoint Controller (tool budget limit, graceful pause, active_tools suspension).
4. System prompt anti-reverse-engineering protocol.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import config
from tools.browser_tools import (
    browser_eval_js,
    _reset_poll_tracker,
    _SESSION_EVAL_TRACKER,
)
from src.agent.semantic_loop_guard import (
    SemanticLoopGuard,
    build_semantic_loop_note,
)
from agent import ArgentAgent


# ---------------------------------------------------------------------------
# 1. Browser Probing Circuit Breaker
# ---------------------------------------------------------------------------

class TestBrowserEvalProbingBreaker:
    def setup_method(self):
        _reset_poll_tracker("default")
        _SESSION_EVAL_TRACKER.clear()

    def test_consecutive_varying_void_probes_trigger_halt_at_6(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            "JS result: null",
            "JS result: undefined",
            "Error evaluating JS: TypeError: Cannot read properties of undefined (reading 'store')",
            'JS result: {"ping": 1}',
            'JS result: "x42"',
            "JS result: null",
        ]
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        for i in range(1, 4):
            res = browser_eval_js(f"probe{i}()")
            assert "[PROBING HINT]" not in res
            assert "[PROBING HALTED]" not in res

        # 4th probe triggers hint
        res4 = browser_eval_js("probe4()")
        assert "[PROBING HINT]" in res4
        assert "[PROBING HALTED]" not in res4

        res5 = browser_eval_js("probe5()")
        assert "[PROBING HINT]" in res5

        # 6th probe triggers hard circuit breaker halt
        res6 = browser_eval_js("probe6()")
        assert "[PROBING HALTED]" in res6
        assert "Do NOT attempt to reverse-engineer private React/Redux internals" in res6

    def test_productive_dom_element_resets_void_counter(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            "JS result: null",
            "JS result: null",
            "JS result: null",
            # Real element returned
            'JS result: {"__type": "Element", "tag": "button", "text": "Submit"}',
            "JS result: null",
        ]
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        for i in range(3):
            browser_eval_js(f"expr{i}()")

        assert _SESSION_EVAL_TRACKER["default"]["consecutive_void_probes"] == 3

        # 4th call produces valid element -> resets void probe counter
        res4 = browser_eval_js("document.querySelector('button')")
        assert "[PROBING HINT]" not in res4
        assert _SESSION_EVAL_TRACKER["default"]["consecutive_void_probes"] == 0

        # 5th call starts from 1 again
        res5 = browser_eval_js("expr5()")
        assert "[PROBING HINT]" not in res5
        assert _SESSION_EVAL_TRACKER["default"]["consecutive_void_probes"] == 1

    def test_mutative_js_resets_void_counter(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.side_effect = [
            "JS result: null",
            "JS result: null",
            "JS result: null",
            # Mutative click
            "JS result: null",
            "JS result: null",
        ]
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        for i in range(3):
            browser_eval_js(f"expr{i}()")

        assert _SESSION_EVAL_TRACKER["default"]["consecutive_void_probes"] == 3

        # Mutative click resets counter
        res_mut = browser_eval_js("document.querySelector('#submit').click()")
        assert res_mut == "JS result: null"
        assert _SESSION_EVAL_TRACKER["default"]["consecutive_void_probes"] == 0

        # Next probe starts from 1
        browser_eval_js("exprNext()")
        assert _SESSION_EVAL_TRACKER["default"]["consecutive_void_probes"] == 1


# ---------------------------------------------------------------------------
# 2. SemanticLoopGuard Browser Integration
# ---------------------------------------------------------------------------

class TestSemanticLoopGuardBrowser:
    def test_unproductive_classification(self):
        guard = SemanticLoopGuard()
        # Explicit void / errors / pings
        assert guard._is_unproductive_eval_result("JS result: null") is True
        assert guard._is_unproductive_eval_result("JS result: undefined") is True
        assert guard._is_unproductive_eval_result("Error evaluating JS: ReferenceError: x is not defined") is True
        assert guard._is_unproductive_eval_result('JS result: {"ping": 1}') is True
        assert guard._is_unproductive_eval_result('JS result: "sync works"') is True
        assert guard._is_unproductive_eval_result('JS result: "typeof fetch: function"') is True
        assert guard._is_unproductive_eval_result('JS result: 42') is True
        assert guard._is_unproductive_eval_result('JS result: true') is True

        # Legitimate DOM or structured data
        assert guard._is_unproductive_eval_result('JS result: {"__type": "Element", "tag": "div"}') is False
        assert guard._is_unproductive_eval_result('JS result: {"appTitle": "Super Game", "user": "test_user_999", "coins": 500}') is False

    def test_record_browser_eval_warn_and_stop(self):
        guard = SemanticLoopGuard()

        # 4 unproductive calls: no warning yet
        for i in range(4):
            assert guard.record_browser_eval(f"expr_{i}", "JS result: null") is None

        # 5th unproductive call: warn
        assert guard.record_browser_eval("expr_4", "JS result: null") == "warn"

        # 6th and 7th: warn
        assert guard.record_browser_eval("expr_5", "JS result: null") == "warn"
        assert guard.record_browser_eval("expr_6", "JS result: null") == "warn"

        # 8th unproductive call: stop
        assert guard.record_browser_eval("expr_7", "JS result: null") == "stop"

    def test_reset_browser_eval_clears_history(self):
        guard = SemanticLoopGuard()
        for i in range(5):
            guard.record_browser_eval(f"expr_{i}", "JS result: null")

        assert len(guard._browser_eval_history) == 5
        guard.reset_browser_eval()
        assert len(guard._browser_eval_history) == 0

    def test_build_semantic_loop_note_browser(self):
        warn_note = build_semantic_loop_note("warn", loop_type="browser_probing")
        assert "Unproductive JavaScript inspection loop detected" in warn_note
        assert "STOP probing obfuscated internal closures" in warn_note

        stop_note = build_semantic_loop_note("stop", loop_type="browser_probing")
        assert "Hard loop stop triggered" in stop_note
        assert "Page state is static" in stop_note


# ---------------------------------------------------------------------------
# 3. System Prompt Protocol Verification
# ---------------------------------------------------------------------------

class TestSystemPromptAntiReverseEngineering:
    def test_browser_protocol_has_anti_reverse_engineering_rule(self):
        agent = ArgentAgent()
        agent.model_name = "glm-5.3-flash"
        prompt = agent.build_system_prompt()
        assert "No Reverse-Engineering of SPA Internals" in prompt
        assert "__reactFiber$" in prompt
        assert "standard UI controls" in prompt
