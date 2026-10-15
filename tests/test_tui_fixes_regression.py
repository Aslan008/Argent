import json
import pytest
from unittest.mock import MagicMock, patch
import questionary

from src.bridge.repl_bridge import ReplBridge, StdoutTrap, patch_questionary_for_tui, AUTO_MAX_TURNS
from src.bridge.session import AgentSession
import tools.misc_tools as misc_tools


def test_stdout_trap_preserves_leading_indentation():
    captured = []
    trap = StdoutTrap(on_output=captured.append)
    trap.start()
    trap.write("   leading indentation\n")
    trap.write("no_indent\n")
    trap.stop()

    assert captured == ["   leading indentation", "no_indent"]


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


def test_vibe_mode_auto_budget():
    bridge = ReplBridge()

    assert not bridge.vibe_mode
    bridge.set_vibe(True)
    assert bridge.vibe_mode
    assert bridge.session.vibe is True
    assert bridge._auto_remaining == AUTO_MAX_TURNS

    # start_turn resets budget
    bridge.start_turn("test prompt")
    assert bridge._auto_remaining == AUTO_MAX_TURNS

    bridge.set_vibe(False)
    assert not bridge.vibe_mode
    assert bridge.session.vibe is False


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
