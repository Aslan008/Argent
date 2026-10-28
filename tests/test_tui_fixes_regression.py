import json
import pytest
from unittest.mock import MagicMock, patch
import questionary

from src.bridge.repl_bridge import (ReplBridge, StdoutTrap, patch_questionary_for_tui, AUTO_MAX_IDLE_TURNS)
from src.bridge.session import AgentSession
import argent.tools.misc_tools as misc_tools


def test_stdout_trap_preserves_leading_indentation():
    captured = []
    trap = StdoutTrap(on_output=captured.append)
    trap.start()
    trap.write("   leading indentation\n")
    trap.write("no_indent\n")
    trap.stop()

    # The newline now travels with the fragment: it is what tells the frontend
    # that this write finished its line, so the next one starts a new row instead
    # of continuing it mid-word (see tests/test_tui_output_and_paste.py). The
    # leading indentation — the point of this test — is still intact.
    assert captured == ["   leading indentation\n", "no_indent\n"]


def test_tui_confirm_returns_boolean_or_none(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    bridge = MagicMock(spec=ReplBridge)
    ReplBridge.active_bridge = bridge
    patch_questionary_for_tui()

    # User picks "Да" -> True
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value="Да"):
        assert questionary.confirm("Are you sure?", default=True).ask() is True

    # User picks "Нет" -> False
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value="Нет"):
        assert questionary.confirm("Are you sure?", default=True).ask() is False

    # User presses Esc -> None
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value=None):
        assert questionary.confirm("Are you sure?", default=True).ask() is None


def test_tui_select_returns_mapped_value_or_none(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    bridge = MagicMock(spec=ReplBridge)
    ReplBridge.active_bridge = bridge
    patch_questionary_for_tui()

    choices = [
        {"name": "Option One", "value": "val_1"},
        {"name": "Option Two", "value": "val_2"},
    ]

    # Select Option Two -> returns "val_2"
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value="Option Two"):
        assert questionary.select("Pick option:", choices=choices).ask() == "val_2"

    # User cancels with Esc -> returns None
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value=None):
        assert questionary.select("Pick option:", choices=choices).ask() is None


def test_tui_checkbox_multi_select(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    bridge = MagicMock(spec=ReplBridge)
    ReplBridge.active_bridge = bridge
    patch_questionary_for_tui()

    class MockChoice:
        def __init__(self, title, value, checked):
            self.title = title
            self.value = value
            self.checked = checked

    choices = [
        MockChoice("Tool A", "a", True),
        MockChoice("Tool B", "b", False),
        MockChoice("Tool C", "c", True),
    ]

    # Check that ask_active_select received multi=True and initial_checked
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value=json.dumps(["Tool A", "Tool C"])) as mock_ask:
        selected = questionary.checkbox("Toggle tools:", choices=choices).ask()
        mock_ask.assert_called_once_with(
            "Toggle tools:",
            ["Tool A", "Tool B", "Tool C"],
            None,
            multi=True,
            initial_checked=[True, False, True],
        )
        assert selected == ["a", "c"]

    # On Esc/cancel -> returns None
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value=None):
        assert questionary.checkbox("Toggle tools:", choices=choices).ask() is None


def test_vibe_mode_is_on_until_it_is_switched_off():
    """No turn budget and no deadline: the person switches vibe on or off, that is all."""
    bridge = ReplBridge()

    assert not bridge.vibe_mode
    bridge.set_vibe(True)
    assert bridge.vibe_mode
    assert bridge.session.vibe is True

    bridge.set_vibe(False)
    assert not bridge.vibe_mode
    assert bridge.session.vibe is False


def test_an_auto_turn_is_not_treated_as_a_human_turn(monkeypatch):
    """Регрессия: автоход шёл через `start_turn` как обычный ввод и взводил бюджет
    заново — поэтому vibe крутился бесконечно, повторяя «[Heartbeat завершен]
    Продолжай.» на каждом ходу и ничего не делая."""
    bridge = ReplBridge()
    calls = []
    monkeypatch.setattr(bridge, "start_turn", lambda text, _wake=False: calls.append((text, _wake)))

    bridge._auto_turn("Продолжай.")

    assert calls == [("Продолжай.", True)]


def test_vibe_keeps_going_while_work_gets_done(monkeypatch):
    """A long run of productive turns must never hit a ceiling: that was the whole
    objection to the old budget (20 turns / 30 minutes, whatever the work was)."""
    bridge = ReplBridge()
    bridge.set_vibe(True)
    notices = []
    monkeypatch.setattr(bridge, "_emit", lambda ev: notices.append(ev))
    monkeypatch.setattr(bridge, "_auto_turn", lambda prompt: None)
    bridge.session._last_turn_tools = 5          # every turn did something
    # ... and none of them concluded: the run keeps picking itself up.
    bridge.session._last_turn_end = "tool call limit"

    for _ in range(50):
        bridge._maybe_auto_continue()

    assert bridge.vibe_mode is True, "работающие ходы не должны упираться в лимит"
    assert bridge._idle_turns == 0
    assert not any("остановлено" in n.get("text", "") for n in notices)


def test_a_run_of_turns_that_do_nothing_stops_the_run(monkeypatch):
    """The signal is the agent's own ending: a final answer, or a turn it cut short.

    Anything the work did not conclude on (a limit, an error, a wake it asked for) is
    the only case that continues by itself. The run stops either way — vibe itself does
    not: it is an approval switch the person turns on and off, not a one-shot task.
    """
    bridge = ReplBridge()
    bridge.set_vibe(True)
    notices = []
    monkeypatch.setattr(bridge, "_emit", lambda ev: notices.append(ev))
    monkeypatch.setattr(bridge, "_auto_turn", lambda prompt: None)
    bridge.session._last_turn_tools = 0
    bridge.session._last_turn_end = "final answer"

    bridge._maybe_auto_continue()

    try:
        assert bridge.vibe_mode is True, "vibe — переключатель одобрений, а не одна задача"
        assert bridge.session.vibe is True
        import argent.approval as approval
        assert approval.get_policy() == approval.POLICY_AUTO, "команды должны и дальше одобряться сами"
        assert not any("vibe выключен" in n.get("text", "").lower() for n in notices)
    finally:
        bridge.set_vibe(False)          # do not leak POLICY_AUTO into the rest of the run


def test_session_reply_select_maps_empty_string_to_none():
    session = AgentSession()
    session._pending["sel_1"] = {"event": MagicMock(), "value": "initial"}
    session.reply_select("sel_1", "")
    assert session._pending["sel_1"]["value"] is None

    session.reply_select("sel_1", None)
    assert session._pending["sel_1"]["value"] is None

    session.reply_select("sel_1", "valid_choice")
    assert session._pending["sel_1"]["value"] == "valid_choice"


def test_misc_tools_ask_text_routes_to_tui_select(monkeypatch):
    monkeypatch.setenv("ARGENT_TUI", "1")
    with patch("src.bridge.repl_bridge.ReplBridge.ask_active_select", return_value="user_input_from_tui") as mock_ask:
        val = misc_tools._ask_text("Enter prompt:", default="my_default")
        assert val == "user_input_from_tui"
        mock_ask.assert_called_once_with("Enter prompt:", options=[], default="my_default")
