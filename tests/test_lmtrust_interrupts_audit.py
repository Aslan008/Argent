"""LMTrust Audit: Blind spot verification tests for Interrupt handling (Ctrl+C, ESC) in Argent."""

import os
import queue
import time
from unittest.mock import MagicMock, patch
import pytest

from agent import ArgentAgent, ArgentSubAgent, _ACTIVE_SUBAGENTS
from src.bridge.session import AgentSession
from src.cli.esc_listener import EscInterruptListener


class TestInterruptAudit:
    def test_run_command_orphan_process_on_keyboardinterrupt(self, monkeypatch):
        """D5 / L4: When run_command receives KeyboardInterrupt while reading output,
        _kill_process_tree MUST be called so child processes do not leak."""
        from tools import command_ops

        killed = []
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.fileno.return_value = 0

        monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
        monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: mock_proc)
        monkeypatch.setattr(command_ops, "_kill_process_tree", lambda proc: killed.append(proc))

        # Simulate KeyboardInterrupt inside line_queue.get
        def fake_get(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr("queue.Queue.get", fake_get)

        with pytest.raises(KeyboardInterrupt):
            command_ops.run_command("python test_script.py")

        assert len(killed) == 1, "Child process tree was orphaned on KeyboardInterrupt!"

    def test_repair_history_tail_user_message_integrity(self):
        """L1 x D3: If a turn was interrupted during model streaming (before any assistant
        response was emitted), repair_history() must either remove the un-responded user
        message or mark it aborted, so consecutive user messages don't violate API contracts."""
        agent = ArgentAgent()
        agent.messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "message that was interrupted mid-stream"},
        ]

        agent.repair_history()

        assert agent.messages[-1]["role"] != "user", (
            "Dangling un-responded user message left at history tail after interruption; "
            "next user message will produce consecutive user turns violating API specs!"
        )
        assert "[Выполнение прервано пользователем" in agent.messages[-1]["content"]

    def test_select_from_list_cancels_on_interrupt(self, monkeypatch):
        """L4b x D4: If user presses Ctrl+C during menu selection, _select_from_list
        MUST return None (cancel) instead of selecting the default/first choice."""
        import ui
        monkeypatch.setattr("questionary.select", lambda *a, **k: MagicMock(ask=lambda: None))
        
        # When questionary returns None, ui._select_from_list falls through to ptk_prompt.
        # If ptk_prompt also gets KeyboardInterrupt:
        def raise_kbi(*a, **k):
            raise KeyboardInterrupt()

        monkeypatch.setattr("prompt_toolkit.prompt", raise_kbi)

        res = ui._select_from_list("Pick one:", ["Choice A", "Choice B"], default="Choice A")
        assert res is None, f"Expected None on cancel, but got default choice: {res!r}"

    def test_session_cancel_propagates_to_agent_and_avoids_double_done(self):
        """L5 x D6: AgentSession.cancel() must set agent._stop_requested and must not duplicate done events."""
        agent = ArgentAgent()
        session = AgentSession(agent)

        session.cancel()

        assert getattr(agent, "_stop_requested", False) is True, (
            "session.cancel() failed to set agent._stop_requested!"
        )

        events = []
        while not session._out.empty():
            events.append(session._out.get_nowait())

        done_count = sum(1 for e in events if e.get("type") == "done")
        assert done_count == 1, f"Expected exactly 1 done event, but got {done_count}"

    def test_esc_listener_pause_and_resume(self):
        """L5 x D2: EscInterruptListener supports pause/resume and context management."""
        agent = ArgentAgent()
        listener = EscInterruptListener(agent)

        assert not listener._running
        assert not listener._paused

        with listener:
            assert listener._running
            assert getattr(agent, "_esc_listener", None) == listener

            listener.pause()
            assert listener._paused is True

            listener.resume()
            assert listener._paused is False

        assert not listener._running

    def test_subagent_cascades_request_stop(self):
        """L6 x D4: Parent agent request_stop cascades to active subagent."""
        parent = ArgentAgent()
        sub = ArgentSubAgent("Coder", "Write a script")

        assert sub not in _ACTIVE_SUBAGENTS
        _ACTIVE_SUBAGENTS.add(sub)
        try:
            assert not sub._stop_requested
            parent.request_stop()
            assert sub._stop_requested is True

            parent.reset_stop()
            assert sub._stop_requested is False
        finally:
            _ACTIVE_SUBAGENTS.discard(sub)

    def test_esc_listener_windows_ungetch_on_regular_key(self, monkeypatch):
        """L4 x D5: If msvcrt fallback is used, non-ESC keystrokes are put back via ungetch."""
        agent = ArgentAgent()
        listener = EscInterruptListener(agent)

        ungetched = []
        # Force fallback by mocking ctypes to None
        with patch.dict("sys.modules", {"ctypes": None}):
            mock_msvcrt = MagicMock()
            mock_msvcrt.kbhit.side_effect = [True, False]
            mock_msvcrt.getch.return_value = b"a"  # Non-ESC key
            mock_msvcrt.ungetch.side_effect = lambda ch: ungetched.append(ch)

            monkeypatch.setattr("src.cli.esc_listener.ctypes", None, raising=False)
            with patch("msvcrt.kbhit", mock_msvcrt.kbhit), \
                 patch("msvcrt.getch", mock_msvcrt.getch), \
                 patch("msvcrt.ungetch", mock_msvcrt.ungetch):
                
                # Run one step of listener
                listener._running = True
                
                # Mock time.sleep to terminate the loop after first check
                def stop_loop(*a):
                    listener._running = False

                with patch("time.sleep", stop_loop):
                    listener._listen_windows()

        assert not getattr(agent, "_stop_requested", False)
        assert ungetched == [b"a"], "Regular key was swallowed instead of ungetch!"
