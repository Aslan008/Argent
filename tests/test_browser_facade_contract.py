"""browser_action / browser_inspect must call the functions behind them correctly.

The 17 separate browser tools were folded into `browser_action`, `browser_inspect`
and `browser_close`, which dispatch to the old functions. The existing tests for
the facade replace those functions with plain MagicMocks, and a MagicMock accepts
any arguments — so two calls that could never have worked went unnoticed:

  * action="type" passed `clear_first=True` to browser_input, which has no such
    parameter: typing into a page raised TypeError every single time;
  * action="eval_js" passed `script=` to browser_eval_js, whose parameter is
    `expression`: running JavaScript failed the same way.

These tests substitute autospecced fakes instead. An autospec enforces the real
signature, so an argument that does not exist is a test failure, not a pass.
"""

from unittest.mock import create_autospec

import pytest

import tools.browser_tools as bt
from tools.schemas import TOOL_SCHEMAS

ACTIONS = [
    ("click", {"index": 3}, "browser_click"),
    ("click", {"selector": "#go"}, "browser_click"),
    ("click", {"text": "Save"}, "browser_click"),
    ("type", {"index": 2, "text": "hello"}, "browser_input"),
    ("input", {"selector": "#q", "text": "hello"}, "browser_input"),
    ("select", {"index": 1, "value": "b"}, "browser_select"),
    ("scroll", {"direction": "down", "amount": 300}, "browser_scroll"),
    ("press_key", {"key": "Enter"}, "browser_press_key"),
    ("press", {"key": "Tab"}, "browser_press_key"),
    ("upload_file", {"index": 1, "text": "C:/x.png"}, "browser_upload_file"),
    ("wait_for", {"text": "Done"}, "browser_wait_for"),
    ("back", {}, "browser_back"),
    ("reload", {}, "browser_reload"),
    ("switch_tab", {"index": 2}, "browser_switch_tab"),
    ("close_tab", {"index": 2}, "browser_close_tab"),
    ("eval_js", {"text": "document.title"}, "browser_eval_js"),
    ("act", {"instruction": "click the login button"}, "browser_act"),
]

MODES = [
    ("dom_interactive", {}, "browser_state"),
    ("dom_interactive", {"query": "login"}, "browser_state"),
    ("tabs", {}, "browser_list_tabs"),
    ("screenshot", {}, "browser_screenshot"),
    ("screenshot", {"path": "shot.png"}, "browser_screenshot"),
    ("console_errors", {}, "browser_state"),
    ("text_content", {"selector": "p"}, "browser_get_content"),
    ("html", {"index": 4}, "browser_get_content"),
    ("observe", {"instruction": "find buttons"}, "browser_observe"),
    ("extract", {"instruction": "prices"}, "browser_extract"),
]


@pytest.fixture
def fakes(monkeypatch):
    """Autospecced stand-ins for every function the facade can call."""
    monkeypatch.setattr(bt.browser_engine, "_sessions", {"default": object()}, raising=False)
    names = {n for _, _, n in ACTIONS} | {n for _, _, n in MODES}
    made = {}
    for name in names:
        fake = create_autospec(getattr(bt, name), return_value=f"called:{name}")
        monkeypatch.setattr(bt, name, fake)
        made[name] = fake
    return made


class TestBrowserAction:
    @pytest.mark.parametrize("action, kwargs, target", ACTIONS,
                             ids=[f"{a}-{'-'.join(k) or 'bare'}" for a, k, _ in ACTIONS])
    def test_it_reaches_its_function_with_arguments_that_exist(self, fakes, action, kwargs, target):
        assert bt.browser_action(action=action, **kwargs) == f"called:{target}"
        fakes[target].assert_called_once()

    def test_typing_does_not_pass_arguments_browser_input_lacks(self, fakes):
        bt.browser_action(action="type", index=1, text="hello")
        assert "clear_first" not in fakes["browser_input"].call_args.kwargs

    def test_eval_js_passes_the_expression_under_its_real_name(self, fakes):
        bt.browser_action(action="eval_js", text="1 + 1")
        assert fakes["browser_eval_js"].call_args.kwargs["expression"] == "1 + 1"

    def test_every_action_the_schema_offers_is_implemented(self, fakes):
        schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "browser_action")
        for action in schema["parameters"]["properties"]["action"]["enum"]:
            reply = bt.browser_action(action=action, index=1, text="x", key="Enter",
                                      instruction="x", selector="#a", value="v")
            assert not str(reply).startswith("Error: Unknown"), f"action {action!r} is offered but not handled"

    def test_an_unknown_action_is_refused_with_the_valid_list(self, fakes):
        reply = bt.browser_action(action="teleport")
        assert reply.startswith("Error: Unknown browser action")
        assert "click" in reply


class TestBrowserInspect:
    @pytest.mark.parametrize("mode, kwargs, target", MODES,
                             ids=[f"{m}-{'-'.join(k) or 'bare'}" for m, k, _ in MODES])
    def test_it_reaches_its_function_with_arguments_that_exist(self, fakes, mode, kwargs, target):
        reply = bt.browser_inspect(mode=mode, **kwargs)
        assert reply.startswith("called:") or "Console" in reply or "No console errors" in reply
        fakes[target].assert_called()

    def test_every_mode_the_schema_offers_is_implemented(self, fakes):
        schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == "browser_inspect")
        for mode in schema["parameters"]["properties"]["mode"]["enum"]:
            reply = bt.browser_inspect(mode=mode, instruction="x", selector="#a")
            assert not str(reply).startswith("Error: Unknown"), f"mode {mode!r} is offered but not handled"
