"""
tests/test_lmtrust_browser_eval_js.py

Neuro-symbolic & Blind spot verification tests for browser_eval_js tool:
1. Top-level return statement auto-wrapping
2. Execution timeout guard against infinite loops/unresolved Promises
3. Structured serialization of DOM Elements (replacing useless 'ref: <Node>')
4. Structured serialization of NodeList and HTMLCollection
5. Output truncation for large payloads (preventing context window explosion)
6. return_state integration with action formatting
7. Mutative JS actions bypass polling halts
8. Repeated JS errors report [REPEATED JS ERROR] instead of polling advice
9. Schema validation for tools/schemas.py
10. Concurrency / lock safety of eval tracker
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

from browser_engine import BrowserEngine, _SessionContext
from tools.browser_tools import (
    browser_eval_js,
    _reset_poll_tracker,
    _SESSION_EVAL_TRACKER,
    _MUTATIVE_JS_PATTERN,
)
from tools.schemas import TOOL_SCHEMAS


class TestBrowserEvalJsEngine:
    """Audit tests for BrowserEngine.eval_js logic and edge-cases."""

    @pytest.mark.asyncio
    async def test_empty_expression_returns_error(self):
        engine = BrowserEngine()
        res = await engine.eval_js("")
        assert "Error: 'expression' cannot be empty." in res

        res_spaces = await engine.eval_js("   \n\t  ")
        assert "Error: 'expression' cannot be empty." in res_spaces

    @pytest.mark.asyncio
    async def test_top_level_return_auto_wrapping(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_handle = MagicMock()
        mock_handle.as_element.return_value = None
        mock_handle.evaluate = AsyncMock(return_value=False)
        mock_handle.json_value = AsyncMock(return_value=42)
        mock_handle.dispose = AsyncMock()

        mock_page.evaluate_handle = AsyncMock(return_value=mock_handle)
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)):
            res = await engine.eval_js("const a = 21; return a * 2;")
            assert res == "JS result: 42"

            # Verify that page.evaluate_handle received the wrapped async IIFE
            called_expr = mock_page.evaluate_handle.call_args[0][0]
            assert called_expr.startswith("(async () => {")
            assert "return a * 2;" in called_expr
            assert called_expr.endswith("})()")

    @pytest.mark.asyncio
    async def test_expression_without_return_is_not_wrapped(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_handle = MagicMock()
        mock_handle.as_element.return_value = None
        mock_handle.evaluate = AsyncMock(return_value=False)
        mock_handle.json_value = AsyncMock(return_value="My Page")
        mock_handle.dispose = AsyncMock()

        mock_page.evaluate_handle = AsyncMock(return_value=mock_handle)
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)):
            res = await engine.eval_js("document.title")
            assert res == 'JS result: "My Page"'
            called_expr = mock_page.evaluate_handle.call_args[0][0]
            assert called_expr == "document.title"

    @pytest.mark.asyncio
    async def test_timeout_guard_triggers_on_hang(self):
        engine = BrowserEngine()
        mock_page = MagicMock()

        async def hanging_eval(expr):
            await asyncio.sleep(10)
            return MagicMock()

        mock_page.evaluate_handle = hanging_eval
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)):
            res = await engine.eval_js("new Promise(() => {})", timeout=0.5)
            assert "Execution timed out after 0.5s" in res
            assert "Ensure your script does not contain infinite loops" in res

    @pytest.mark.asyncio
    async def test_dom_element_smart_serialization(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_handle = MagicMock()
        mock_element = MagicMock()

        # as_element returns element handle
        mock_handle.as_element.return_value = mock_element
        mock_handle.dispose = AsyncMock()

        async def el_eval(expr):
            if "tagName" in expr:
                return "button"
            elif "el.id" in expr:
                return "submit-btn"
            elif "className" in expr:
                return "btn btn-primary"
            elif "textContent" in expr:
                return "Submit Order"
            elif "outerHTML" in expr:
                return '<button id="submit-btn" class="btn btn-primary">Submit Order</button>'
            return None

        mock_element.evaluate = AsyncMock(side_effect=el_eval)
        mock_page.evaluate_handle = AsyncMock(return_value=mock_handle)
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)):
            res = await engine.eval_js("document.querySelector('#submit-btn')")
            assert "JS result:" in res
            assert '"__type": "Element"' in res
            assert '"tagName": "button"' in res
            assert '"id": "submit-btn"' in res
            assert '"text": "Submit Order"' in res
            assert mock_handle.dispose.called

    @pytest.mark.asyncio
    async def test_node_collection_smart_serialization(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_handle = MagicMock()
        mock_handle.as_element.return_value = None
        mock_handle.dispose = AsyncMock()

        async def handle_eval(expr):
            if "is_node_collection" in expr or "Array.isArray" in expr:
                return True
            if "Array.from(h)" in expr:
                return [
                    {"__type": "Element", "tagName": "li", "text": "Item 1"},
                    {"__type": "Element", "tagName": "li", "text": "Item 2"}
                ]
            return False

        mock_handle.evaluate = AsyncMock(side_effect=handle_eval)
        mock_page.evaluate_handle = AsyncMock(return_value=mock_handle)
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)):
            res = await engine.eval_js("document.querySelectorAll('li')")
            assert "JS result:" in res
            assert "Item 1" in res
            assert "Item 2" in res
            assert mock_handle.dispose.called

    @pytest.mark.asyncio
    async def test_large_output_truncation(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_handle = MagicMock()
        mock_handle.as_element.return_value = None
        mock_handle.evaluate = AsyncMock(return_value=False)
        # 100,000 character payload
        huge_data = {"data": "X" * 100000}
        mock_handle.json_value = AsyncMock(return_value=huge_data)
        mock_handle.dispose = AsyncMock()

        mock_page.evaluate_handle = AsyncMock(return_value=mock_handle)
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)):
            res = await engine.eval_js("generateHugeData()")
            assert "[Output truncated at 15000 of" in res
            assert len(res) < 20000

    @pytest.mark.asyncio
    async def test_return_state_parameter(self):
        engine = BrowserEngine()
        mock_page = MagicMock()
        mock_handle = MagicMock()
        mock_handle.as_element.return_value = None
        mock_handle.evaluate = AsyncMock(return_value=False)
        mock_handle.json_value = AsyncMock(return_value="done")
        mock_handle.dispose = AsyncMock()

        mock_page.evaluate_handle = AsyncMock(return_value=mock_handle)
        mock_sc = _SessionContext(context=MagicMock(), page=mock_page)

        with patch.object(engine, "_get_session", AsyncMock(return_value=mock_sc)), \
             patch.object(engine, "_format_action_result", AsyncMock(return_value="Formatted with state")) as mock_format:
            res = await engine.eval_js("someAction()", return_state=True)
            assert res == "Formatted with state"
            mock_format.assert_called_once_with("JS result: \"done\"", "default", return_state=True)


class TestBrowserEvalJsToolsWrapper:
    """Audit tests for tools/browser_tools.py browser_eval_js wrapper."""

    def setup_method(self):
        _reset_poll_tracker("default")
        _SESSION_EVAL_TRACKER.clear()

    def test_mutative_actions_bypass_polling_halt(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = "JS result: null"
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        # Scrolling 10 times in a row should NOT trigger [POLLING HALTED]
        for _ in range(10):
            res = browser_eval_js("window.scrollBy(0, 300)")
            assert res == "JS result: null"
            assert "[POLLING HALTED]" not in res
            assert "[POLLING HINT]" not in res

        # Clicking element should NOT trigger [POLLING HALTED]
        for _ in range(6):
            res = browser_eval_js("document.querySelector('button#load-more').click()")
            assert res == "JS result: null"
            assert "[POLLING HALTED]" not in res

    def test_non_mutative_void_still_triggers_polling_halt(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = "JS result: null"
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        # Checking non-existent element 5 times with null result SHOULD trigger circuit breaker
        for _ in range(4):
            browser_eval_js("document.querySelector('#non-existent')")

        res5 = browser_eval_js("document.querySelector('#non-existent')")
        assert "[POLLING HALTED]" in res5

    def test_repeated_js_syntax_or_runtime_errors(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.return_value = "Error evaluating JS: ReferenceError: myUnknownVar is not defined"
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        # Run 1 and 2: returns raw error
        res1 = browser_eval_js("myUnknownVar.test()")
        assert "ReferenceError" in res1
        assert "[REPEATED JS ERROR]" not in res1

        res2 = browser_eval_js("myUnknownVar.test()")
        assert "ReferenceError" in res2
        assert "[REPEATED JS ERROR]" not in res2

        # Run 3: returns specialized repeated error diagnostic
        res3 = browser_eval_js("myUnknownVar.test()")
        assert "[REPEATED JS ERROR]" in res3
        assert "Fix your script syntax" in res3
        # Ensure it does NOT give the misleading browser_wait_for advice
        assert "browser_wait_for" not in res3

    def test_concurrent_sessions_thread_safety(self, monkeypatch):
        mock_engine = MagicMock()
        mock_engine.run.side_effect = lambda coro: "JS result: 123"
        monkeypatch.setattr("tools.browser_tools.browser_engine", mock_engine)

        sessions = [f"session_{i}" for i in range(10)]

        def run_eval(s):
            return browser_eval_js("window.testVal", session=s)

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(run_eval, sessions))

        for r in results:
            assert r == "JS result: 123"

        assert len(_SESSION_EVAL_TRACKER) == 10
        for s in sessions:
            assert _SESSION_EVAL_TRACKER[s]["count"] == 1


class TestBrowserEvalJsSchema:
    """eval_js is an action of browser_action now (browser_eval_js was folded in)."""

    @staticmethod
    def _action_schema():
        tool = next((t for t in TOOL_SCHEMAS if t.get("function", {}).get("name") == "browser_action"), None)
        assert tool is not None, "browser_action must exist in TOOL_SCHEMAS"
        return tool["function"]

    def test_eval_js_is_offered_as_an_action(self):
        enum = self._action_schema()["parameters"]["properties"]["action"]["enum"]
        assert "eval_js" in enum
        assert not any(t.get("function", {}).get("name") == "browser_eval_js" for t in TOOL_SCHEMAS)

    def test_the_schema_says_where_the_javascript_goes(self):
        """The model has to be told: guessing `expression=` or `script=` loses the code."""
        text_desc = self._action_schema()["parameters"]["properties"]["text"]["description"]
        assert "JavaScript" in text_desc and "eval_js" in text_desc

    def test_session_and_return_state_are_still_settable(self):
        props = self._action_schema()["parameters"]["properties"]
        assert props["session"]["type"] == "string"
        assert props["return_state"]["type"] == "boolean"
        assert self._action_schema()["parameters"]["required"] == ["action"]

    @pytest.mark.parametrize("name", ["expression", "script", "js", "javascript", "code"])
    def test_the_names_a_model_reaches_for_reach_the_tool(self, name):
        from src.agent.arg_coercion import coerce_and_log
        args = coerce_and_log("browser_action", {"action": "eval_js", name: "document.title"})
        assert args["text"] == "document.title"
        assert name not in args
