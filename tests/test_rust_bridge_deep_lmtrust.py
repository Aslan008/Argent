"""Deep LMTrust testing for Argent Rust TUI Bridge and Python integration layer.

Covers:
- L0 Smoke: Rust binary module loading
- L1 Contract: ReplBridge state, command registry, event sinking
- L2 Boundary: Empty inputs, multibyte history strings, whitespace prompts
- L4 Adversarial: StdoutTrap ANSI stripping, non-existent approvals, exception recovery
- L6 Cross-System: End-to-end slash command dispatch through event sink
"""

import sys
import io
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import argent_tui
from src.bridge.repl_bridge import ReplBridge, StdoutTrap
from src.bridge.session import AgentSession


class TestRustTuiSmoke:
    """L0 Smoke: Verifies compiled Rust extension module exports."""

    def test_extension_exports(self):
        assert hasattr(argent_tui, "run_tui"), "argent_tui must export run_tui function"
        assert hasattr(argent_tui, "RustEventSink"), "argent_tui must export RustEventSink class"
        assert callable(argent_tui.run_tui)


class TestBridgeContracts:
    """L1 Contract: Verifies ReplBridge protocol contracts with Rust frontend."""

    def test_get_state_structure(self):
        bridge = ReplBridge()
        state = bridge.get_state()
        assert isinstance(state, dict)
        assert "model" in state
        assert "provider" in state
        assert "turn_counter" in state
        assert "vibe" in state
        assert "cwd" in state
        assert isinstance(state["turn_counter"], int)
        assert isinstance(state["vibe"], bool)

    def test_get_command_completions(self):
        bridge = ReplBridge()
        completions = bridge.get_command_completions()
        assert isinstance(completions, list)
        assert len(completions) > 0

        names = [c["name"] for c in completions]
        assert "/help" in names
        assert "/model" in names
        assert "/clear" in names
        for c in completions:
            assert "name" in c
            assert "help" in c

    def test_event_sink_subscription_and_banner(self):
        bridge = ReplBridge()
        received_events = []

        def mock_sink(event):
            received_events.append(event)

        bridge.set_event_sink(mock_sink)
        # Verify onboarding banner events emitted
        assert len(received_events) >= 2
        # The start screen mirrors the REPL's: a banner rule, then system lines.
        assert any(e.get("type") == "banner" for e in received_events)
        system = [e["text"] for e in received_events if e.get("type") == "system"]
        assert any("Argent Coder" in n for n in system)
        assert any(n.startswith("Провайдер:") for n in system)
        assert any(n.startswith("Модель:") for n in system)


class TestBridgeBoundaries:
    """L2 Boundary: Edge cases for input, history persistence, and file mentions."""

    def test_empty_and_whitespace_start_turn(self):
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))
        events.clear()

        bridge.start_turn("")
        assert len(events) == 1
        assert events[0]["type"] == "done"

        events.clear()
        bridge.start_turn("   \t \n  ")
        assert len(events) == 1
        assert events[0]["type"] == "done"

    def test_history_persistence_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bridge = ReplBridge()

        # 1. Initially empty
        assert bridge.get_history() == []

        # 2. Append entries
        bridge.append_history("first command")
        bridge.append_history("вторая кириллическая команда")
        bridge.append_history("🚀 emoji command")

        # Empty prompt must not be appended
        bridge.append_history("   ")

        history = bridge.get_history()
        assert len(history) == 3
        assert history[0] == "first command"
        assert history[1] == "вторая кириллическая команда"
        assert history[2] == "🚀 emoji command"


class TestBridgeAdversarial:
    """L4 Adversarial: Stdout isolation, ANSI stripping, and error paths."""

    def test_stdout_trap_keeps_colour_but_strips_terminal_control(self):
        """The TUI draws SGR itself, so colours must survive the trap; anything
        that moves the cursor or switches modes must not reach the alt screen."""
        captured = []
        trap = StdoutTrap(on_output=lambda s: captured.append(s))

        with trap:
            print("\x1b[32mGreen Text\x1b[0m and \x1b[1;34mBold Blue\x1b[0m")
            sys.stdout.write("\x1b[31mError text\x1b[0m\n")
            sys.stdout.write("\x1b[2K\x1b[1Aspinner\x1b[?25l\x1b[?25h\n")

        assert len(captured) >= 3
        assert any("\x1b[32mGreen Text\x1b[0m" in c for c in captured)
        assert any("\x1b[1;34mBold Blue\x1b[0m" in c for c in captured)
        assert any("Error text" in c for c in captured)
        control = [c for c in captured if "spinner" in c]
        assert control and control[0] == "spinner"

    def test_stdout_trap_restoration_on_exception(self):
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        trap = StdoutTrap()

        try:
            with trap:
                assert sys.stdout == trap
                raise ValueError("Forced error")
        except ValueError:
            pass

        assert sys.stdout == orig_stdout
        assert sys.stderr == orig_stderr

    def test_unknown_slash_command(self):
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))
        events.clear()

        bridge.start_turn("/non_existent_command_12345")
        import time
        start_wait = time.time()
        while time.time() - start_wait < 2.0 and not any(e.get("type") == "done" for e in events):
            time.sleep(0.02)

        err_events = [e for e in events if e.get("type") == "error"]
        assert len(err_events) == 1
        assert "Неизвестная команда" in err_events[0]["text"]
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1


class TestBridgeCrossSystem:
    """L6 Cross-System: End-to-end slash command dispatch through event sink."""

    def test_help_command_e2e(self):
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))
        events.clear()

        bridge.start_turn("/help")
        # Wait up to 2.0s for the worker thread to finish
        import time
        start_wait = time.time()
        while time.time() - start_wait < 2.0 and not any(e.get("type") == "done" for e in events):
            time.sleep(0.02)

        # /help emits table via stdout, which StdoutTrap captures and forwards as notices
        notices = [e.get("text", "") for e in events if e.get("type") == "notice"]
        assert any("Argent" in n or "Команды" in n or "help" in n for n in notices)

        # Must conclude with a done event
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    def test_successive_slash_commands_release_turn_lock(self):
        """Verifies that slash commands properly reset _turn_running so successive commands are not rejected."""
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))

        # First command
        bridge.start_turn("/sound")
        import time
        start_wait = time.time()
        while time.time() - start_wait < 2.0 and not any(e.get("type") == "done" for e in events):
            time.sleep(0.02)

        assert not bridge.session._turn_running

        # Second command immediately after
        events.clear()
        bridge.start_turn("/vibe")
        assert not bridge.turn_was_rejected()
        start_wait = time.time()
        while time.time() - start_wait < 2.0 and not any(e.get("type") == "done" for e in events):
            time.sleep(0.02)

        assert not bridge.session._turn_running

    def test_ask_select_and_reply_flow(self):
        import threading
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))
        events.clear()

        result = [None]
        def _requester():
            result[0] = bridge.ask_select("Choose provider:", ["ollama", "zai", "openrouter"], default="zai")

        t = threading.Thread(target=_requester)
        t.start()

        # Wait for event emission
        import time
        for _ in range(20):
            if any(e.get("type") == "select_request" for e in events):
                break
            time.sleep(0.05)

        sel_events = [e for e in events if e.get("type") == "select_request"]
        assert len(sel_events) == 1
        sid = sel_events[0]["id"]
        assert sel_events[0]["title"] == "Choose provider:"
        assert sel_events[0]["options"] == ["ollama", "zai", "openrouter"]
        assert sel_events[0]["default"] == "zai"

        # Reply from TUI
        bridge.reply_select(sid, "openrouter")
        t.join(timeout=2.0)
        assert result[0] == "openrouter"

    def test_select_from_list_in_tui_mode(self, monkeypatch):
        import threading
        monkeypatch.setenv("ARGENT_TUI", "1")
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))

        from ui import _select_from_list

        result = [None]
        def _call_select():
            result[0] = _select_from_list("Pick model:", ["m1", "m2", "m3"], default="m2")

        t = threading.Thread(target=_call_select)
        t.start()

        import time
        for _ in range(20):
            if any(e.get("type") == "select_request" for e in events):
                break
            time.sleep(0.05)

        sel_events = [e for e in events if e.get("type") == "select_request"]
        assert len(sel_events) == 1
        sid = sel_events[0]["id"]

        bridge.reply_select(sid, "m3")
        t.join(timeout=2.0)
        assert result[0] == "m3"

    def test_questionary_patching_in_tui(self, monkeypatch):
        import threading
        import questionary
        monkeypatch.setenv("ARGENT_TUI", "1")
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))

        # 1. questionary.confirm
        res_confirm = [None]
        def _ask_confirm():
            res_confirm[0] = questionary.confirm("Delete files?", default=False).ask()

        t1 = threading.Thread(target=_ask_confirm)
        t1.start()

        import time
        for _ in range(20):
            if any(e.get("type") == "select_request" for e in events):
                break
            time.sleep(0.05)

        sel = [e for e in events if e.get("type") == "select_request"][-1]
        assert sel["title"] == "Delete files?"
        assert sel["options"] == ["Да", "Нет"]
        bridge.reply_select(sel["id"], "Да")
        t1.join(timeout=2.0)
        assert res_confirm[0] is True

        # 2. questionary.select
        res_select = [None]
        def _ask_select():
            res_select[0] = questionary.select("Select model:", choices=["glm", "gpt", "claude"]).ask()

        t2 = threading.Thread(target=_ask_select)
        t2.start()

        for _ in range(20):
            if len([e for e in events if e.get("type") == "select_request"]) >= 2:
                break
            time.sleep(0.05)

        sel2 = [e for e in events if e.get("type") == "select_request"][-1]
        assert sel2["title"] == "Select model:"
        assert sel2["options"] == ["glm", "gpt", "claude"]
        bridge.reply_select(sel2["id"], "glm")
        t2.join(timeout=2.0)
        assert res_select[0] == "glm"

    def test_provider_command_no_hang_in_tui(self, monkeypatch):
        import time
        monkeypatch.setenv("ARGENT_TUI", "1")
        bridge = ReplBridge()
        events = []
        bridge.set_event_sink(lambda e: events.append(e))

        # Dispatch /provider command - must NOT block the caller!
        bridge.start_turn("/provider")

        for _ in range(20):
            if any(e.get("type") == "select_request" for e in events):
                break
            time.sleep(0.05)

        sel_events = [e for e in events if e.get("type") == "select_request"]
        assert len(sel_events) == 1
        sid = sel_events[0]["id"]
        assert "провайдера" in sel_events[0]["title"]
        assert "zai" in sel_events[0]["options"]

        # Answering selects the provider
        bridge.reply_select(sid, "zai")

        # Answer any subsequent model picker requests if provider changed
        answered_ids = {sid}
        for _ in range(40):
            if any(e.get("type") == "done" for e in events):
                break
            for e in list(events):
                if e.get("type") == "select_request" and e.get("id") not in answered_ids:
                    answered_ids.add(e["id"])
                    opts = e.get("options", [])
                    ans = opts[0] if opts else "test-model"
                    bridge.reply_select(e["id"], ans)
            time.sleep(0.05)

        assert any(e.get("type") == "done" for e in events)
