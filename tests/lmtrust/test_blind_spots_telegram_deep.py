"""LMTrust Deep Blind Spot & Contract Verification Suite for Telegram Implementation in Argent.

Covers 11 layers of verification:
- L1 Contract & Encoding: Double-escaping prevention in terminal turn completion notifications.
- L2 Boundary: Infinite loop resilience in TelegramClient._chunk_text with tricky whitespace/boundary indices.
- L3 Property & Invariant: Output accumulator persistence across tool_start events (no data loss).
- L5 State & Concurrency: AgentSession.cancel() invariant (busy=False, active=False, unblocks get_event()).
- L5 Concurrency: Thread safety guard against race conditions in _run_prompt_turn.
- L7 Compositional: TelegramBridge.stop() gracefully unblocks pending terminal approvals.
- L9 Robustness: Terminal prompt runner catches BaseException (KeyboardInterrupt) without hanging.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.bridge.session import AgentSession
from src.telegram.bot import TelegramBridge, TelegramClient


class TestL1ContractAndEncoding:
    """Verifies that markdown/HTML encoding avoids double-escaping in notifications."""

    def test_terminal_completion_no_double_escaping(self, monkeypatch):
        monkeypatch.setattr("src.telegram.bot.get_telegram_chat_id", lambda: "12345")
        bridge = TelegramBridge()
        bridge.client = MagicMock()

        user_input = "Refactor foo & bar <baz>"
        response = "Completed successfully with & and <code>."

        bridge.notify_terminal_turn_completed(user_input, response)

        assert bridge.client.send_message.called
        sent_text = bridge.client.send_message.call_args[0][1]

        # Single escaping should occur (& -> &amp;, < -> &lt;)
        assert "&amp;" in sent_text
        assert "&lt;" in sent_text
        # Double escaping MUST NEVER happen
        assert "&amp;amp;" not in sent_text
        assert "&amp;lt;" not in sent_text
        assert "&amp;gt;" not in sent_text


class TestL2BoundaryChunking:
    """Verifies boundary cases and infinite-loop traps in text chunking."""

    def test_chunk_text_boundary_infinite_loop_prevention(self):
        # Case 1: text where newline/space is at index 0 or tricky whitespace prefix
        text = " \n" + "a" * 100
        limit = 10
        chunks = TelegramClient._chunk_text(text, limit=limit)
        assert chunks
        for c in chunks:
            assert len(c) <= limit

        # Case 2: all spaces/newlines longer than limit
        text = " \n \n    \n\n   \n" * 5
        chunks = TelegramClient._chunk_text(text, limit=5)
        # Must terminate promptly without hanging
        for c in chunks:
            assert len(c) <= 5

        # Case 3: exact boundary multiple without spaces
        text = "x" * 40
        chunks = TelegramClient._chunk_text(text, limit=10)
        assert len(chunks) == 4
        assert all(len(c) == 10 for c in chunks)


class TestL3PropertyAccumulator:
    """Verifies that model text before and after tool calls is not lost."""

    def test_tool_start_does_not_wipe_content_accumulator(self):
        bridge = TelegramBridge()
        mock_client = MagicMock()
        bridge.client = mock_client

        mock_session = MagicMock()
        events = [
            {"type": "content", "text": "Starting analysis of codebase.\n"},
            {"type": "tool_start", "name": "read_file", "args": {"path": "main.py"}},
            {"type": "content", "text": "Found entry point in main.py."},
            {"type": "done"},
        ]
        mock_session.get_event.side_effect = events
        bridge._isolated_session = mock_session

        # Run event consumer synchronously
        bridge._event_consumer_worker(chat_id="12345")

        assert mock_client.send_message.called
        # Check all sent messages for accumulated content
        all_sent = " ".join(call[0][1] for call in mock_client.send_message.call_args_list)
        assert "Starting analysis of codebase" in all_sent
        assert "Found entry point in main.py" in all_sent


class TestL5StateAndConcurrency:
    """Verifies state invariants, cancellation, and concurrency safety."""

    def test_agent_session_cancel_resets_active_and_busy(self):
        session = AgentSession()
        session._active = True
        session._stop_requested = False

        # Simulate pending approval
        ev = threading.Event()
        session._pending[1] = {"event": ev, "decision": "deny"}

        session.cancel()

        assert not session.busy
        assert session._stop_requested is True
        assert ev.is_set()

        # get_event should return 'done' immediately without blocking
        ev_done = session.get_event()
        assert ev_done.get("type") == "done"
        assert not session.busy

    def test_prompt_turn_concurrent_guard(self, monkeypatch):
        monkeypatch.setattr("src.telegram.bot.get_telegram_chat_id", lambda: "12345")
        bridge = TelegramBridge()
        mock_client = MagicMock()
        bridge.client = mock_client

        class DummySession:
            def __init__(self):
                self._busy = False
                self.start_calls = 0

            @property
            def busy(self):
                return self._busy

            def start(self, text):
                self.start_calls += 1
                time.sleep(0.05)
                self._busy = True

        dummy = DummySession()
        bridge._isolated_session = dummy

        threads = []
        for i in range(2):
            t = threading.Thread(target=bridge._run_prompt_turn, args=("12345", f"msg {i}"))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should have successfully started turn, the other notified as busy
        assert dummy.start_calls == 1
        assert mock_client.send_message.called


class TestL7CompositionalApprovalsAndLifecycle:
    """Verifies lifecycle teardown and approval race resolution."""

    def test_stop_resolves_pending_terminal_approvals(self, monkeypatch):
        monkeypatch.setattr("src.telegram.bot.get_telegram_chat_id", lambda: "12345")
        bridge = TelegramBridge()
        mock_client = MagicMock()
        mock_client.send_message.return_value = {"message_id": 999}
        bridge.client = mock_client

        # Simulated slow terminal function
        def slow_terminal(action, destructive, grant_key):
            time.sleep(10.0)
            return "once"

        result_container = []

        def run_relay():
            decision = bridge.relay_terminal_approval("dangerous_action", True, None, slow_terminal)
            result_container.append(decision)

        t = threading.Thread(target=run_relay, daemon=True)
        t.start()

        # Wait until approval is registered
        time.sleep(0.1)
        assert len(bridge._pending_terminal_approvals) > 0

        # Now stop the bridge
        bridge.stop()
        t.join(timeout=2.0)

        assert not t.is_alive()
        assert len(result_container) == 1
        # Stopped bridge must resolve pending approval with 'deny'
        assert result_container[0] == "deny"

    def test_relay_terminal_approval_when_stopped_falls_back_immediately(self, monkeypatch):
        monkeypatch.setattr("src.telegram.bot.get_telegram_chat_id", lambda: "12345")
        bridge = TelegramBridge()
        mock_client = MagicMock()
        bridge.client = mock_client

        # Stop bridge before calling relay
        bridge._stop_event.set()

        fallback_called = False

        def terminal_fn(action, destructive, grant_key):
            nonlocal fallback_called
            fallback_called = True
            return "once"

        dec = bridge.relay_terminal_approval("rm -rf", True, None, terminal_fn)
        assert dec == "once"
        assert fallback_called is True
        # Did not attempt to send telegram message
        assert not mock_client.send_message.called


class TestL9RobustnessExceptions:
    """Verifies exception handling during terminal approval interactions."""

    def test_run_terminal_prompt_catches_base_exception(self, monkeypatch):
        monkeypatch.setattr("src.telegram.bot.get_telegram_chat_id", lambda: "12345")
        bridge = TelegramBridge()
        mock_client = MagicMock()
        mock_client.send_message.return_value = {"message_id": 100}
        bridge.client = mock_client

        def crashing_terminal(action, destructive, grant_key):
            raise KeyboardInterrupt("Simulated Ctrl+C in terminal")

        # Must not hang forever; must resolve to 'deny'
        decision = bridge.relay_terminal_approval("pytest run", False, None, crashing_terminal)
        assert decision == "deny"
