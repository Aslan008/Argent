"""The anti-probing guard has to watch the tool that actually exists.

When the browser tools were folded into `browser_action`, the agent loop kept
asking `func_name == "browser_eval_js"` — a name no call carries any more — so
the semantic guard that stops a model from hammering the page with pointless
JavaScript never ran again, and the list of "state-changing" tools that resets
it still named the old click/input/upload tools. `browser_action` itself sat in
that list for every verb, which would have cleared the guard's memory right
after each eval_js call even once the first problem was fixed.

These tests drive whole turns through the agent with a scripted provider.
"""

import pytest

import agent as agent_module
from agent import ArgentAgent
from src.agent.semantic_loop_guard import (
    browser_eval_expression,
    is_browser_eval_call,
    is_page_changing_browser_action,
)


class TestReadingABrowserActionCall:
    @pytest.mark.parametrize("arguments, expected", [
        ({"action": "eval_js"}, True),
        ({"action": "EVAL_JS"}, True),
        ({"action": " eval "}, True),
        ({"action": "click"}, False),
        ({"action": None}, False),
        ({}, False),
        (None, False),
        ("eval_js", False),
    ])
    def test_which_calls_are_eval_js(self, arguments, expected):
        assert is_browser_eval_call(arguments) is expected

    @pytest.mark.parametrize("arguments, expected", [
        ({"action": "eval_js", "text": "document.title"}, "document.title"),
        ({"action": "eval_js", "value": "1+1"}, "1+1"),
        ({"action": "eval_js", "instruction": "location.href"}, "location.href"),
        ({"action": "eval_js", "text": "", "value": "ignored"}, ""),    # `text` wins, even empty
        ({"action": "eval_js"}, ""),
        (None, ""),
    ])
    def test_where_the_expression_travels(self, arguments, expected):
        """The same precedence browser_action itself uses: text, value, instruction."""
        assert browser_eval_expression(arguments) == expected

    @pytest.mark.parametrize("action", ["click", "type", "input", "select", "upload_file",
                                        "reload", "back", "press_key", "wait_for", "act"])
    def test_these_change_the_page(self, action):
        assert is_page_changing_browser_action({"action": action}) is True

    @pytest.mark.parametrize("action", ["eval_js", "scroll", "switch_tab", "close_tab", "", None])
    def test_these_do_not(self, action):
        assert is_page_changing_browser_action({"action": action}) is False


class _Strategy:
    def wants_constrained_decoding(self): return False
    def supports_native_tools(self): return True
    def wants_objective_anchor(self): return False
    def wants_tool_catalog(self): return False
    def is_loop_guard_enabled(self): return True
    def is_auto_healing_enabled(self): return False


@pytest.fixture
def bare(monkeypatch):
    a = ArgentAgent.__new__(ArgentAgent)
    a.messages = [{"role": "system", "content": "s"}]
    a.model_name = "test-model"
    a.provider = "ollama"
    a.max_context_tokens = 8192
    a.strategy = _Strategy()
    a._native_tools_unsupported = False
    a.loop_guard = agent_module.LoopGuard()
    a.semantic_loop_guard = agent_module.SemanticLoopGuard()
    a._read_cache = {}
    monkeypatch.setattr(ArgentAgent, "_refresh_system_prompt", lambda self, t=None: False)
    monkeypatch.setattr(ArgentAgent, "_trim_history", lambda self: None)
    monkeypatch.setattr(agent_module.memory, "data", {"objective": "x"}, raising=False)
    return a


def _script(monkeypatch, calls, tool_result):
    """A provider that makes `calls` (list of JSON argument strings) in order,
    then answers; every browser_action returns `tool_result(arguments)`."""
    import json
    queue = list(calls)

    class _P:
        def validate_config(self): return None
        def supports_constrained_decoding(self): return False
        def format_tool_result(self, result, call_id=None):
            return {"role": "tool", "content": result, "tool_call_id": call_id or "c1"}

        def stream_chat(self, messages, **kw):
            if queue:
                yield {"tool_call_deltas": [{
                    "index": 0, "id": f"c{len(queue)}",
                    "function_name_delta": "browser_action",
                    "function_arguments_delta": queue.pop(0),
                }]}
            else:
                yield {"content": "stopping"}

    def browser_action(action, text=None, **_):
        return tool_result({"action": action, "text": text})

    monkeypatch.setattr(agent_module, "create_provider", lambda p: _P())
    monkeypatch.setattr(agent_module, "get_available_tools", lambda: {"browser_action": browser_action})
    return json


def _tool_results(agent, prompt="go"):
    return [chunk["result"] for chunk in agent.process_user_input(prompt) if chunk.get("type") == "tool_end"]


class TestTheGuardIsLive:
    def test_repeated_pointless_javascript_is_flagged(self, bare, monkeypatch):
        json = _script(
            monkeypatch,
            [json_args for json_args in
             ['{"action": "eval_js", "text": "probe%d()"}' % i for i in range(7)]],
            lambda args: "JS result: null")
        results = _tool_results(bare)
        flagged = [r for r in results if "[SEMANTIC BROWSER GUARD]" in str(r)]
        assert flagged, "seven null-returning eval_js probes in a row were never flagged"
        assert "browser_action(action=\"click\")" in flagged[0]
        assert "browser_state" not in flagged[0], "the note must name tools that exist"

    def test_a_click_between_probes_resets_the_count(self, bare, monkeypatch):
        calls = ['{"action": "eval_js", "text": "p%d()"}' % i for i in range(4)]
        calls.append('{"action": "click", "text": "Next"}')
        calls += ['{"action": "eval_js", "text": "q%d()"}' % i for i in range(4)]

        def result(args):
            return "JS result: null" if args["action"] == "eval_js" else "Clicked."

        _script(monkeypatch, calls, result)
        results = _tool_results(bare)
        assert not [r for r in results if "[SEMANTIC BROWSER GUARD]" in str(r)], (
            "four probes, a click, four more: the click changed the page, so neither run is a loop")

    def test_a_probe_that_finds_something_is_not_a_loop(self, bare, monkeypatch):
        calls = ['{"action": "eval_js", "text": "d%d()"}' % i for i in range(8)]
        _script(monkeypatch, calls,
                lambda args: 'JS result: {"__type": "Element", "tag": "button", "text": "Submit"}')
        assert not [r for r in _tool_results(bare) if "[SEMANTIC BROWSER GUARD]" in str(r)]

    def test_eval_js_does_not_clear_its_own_history(self, bare, monkeypatch):
        """browser_action was listed as state-changing for EVERY verb, so each
        eval_js call wiped the memory the guard needs to see a pattern."""
        _script(monkeypatch, ['{"action": "eval_js", "text": "p1()"}',
                              '{"action": "eval_js", "text": "p2()"}'],
                lambda args: "JS result: null")
        _tool_results(bare)
        assert len(bare.semantic_loop_guard._browser_eval_history) == 2
