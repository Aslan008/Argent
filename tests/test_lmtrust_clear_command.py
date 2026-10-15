"""LMTrust Regression & Blind-Spot Test Suite for /clear, /reset, /new commands.

Covers:
- L2 x D4 (Boundary Defense): clear_history on empty messages does not crash (CLR-1)
- L5 x D2 (State Hygiene): _checkpoint_turn_indices and turn flags reset on clear (CLR-2)
- L5 x D2 (Detector Hygiene): semantic_loop_guard reset on clear (CLR-3)
- L4 x D5 (Resource/Leak): vision._pending queue drained on clear (CLR-4)
- L1 x D7 (Contract Freshness): Fresh system prompt generated from scratch (CLR-6)
- L1 x D7 (CLI Parity): /clear, /reset, and /new all execute clean slate (CLR-7)
"""

import pytest
from unittest.mock import MagicMock, patch

from agent import ArgentAgent
from command_handler import handle_slash_command
from memory_manager import memory
import vision


class TestClearCommandRobustness:
    def test_clear_history_empty_messages_does_not_crash(self):
        agent = ArgentAgent()
        agent.messages = []

        # Must not raise IndexError
        agent.clear_history()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert "Argent Coder" in agent.messages[0]["content"]

    def test_clear_history_resets_checkpoints_and_transient_indices(self):
        agent = ArgentAgent()
        agent._checkpoint_turn_indices["dead_sha"] = 42
        agent._turn_checkpoint_done = True
        agent._current_turn_msg_index = 5
        agent._stop_requested = True

        agent.clear_history()

        assert len(agent._checkpoint_turn_indices) == 0
        assert agent._turn_checkpoint_done is False
        assert agent._current_turn_msg_index == 0
        assert agent._stop_requested is False

    def test_clear_history_resets_semantic_loop_guard(self):
        agent = ArgentAgent()
        agent.semantic_loop_guard._revert_count = 5
        agent.semantic_loop_guard._error_ping_pong_count = 3
        agent.semantic_loop_guard._error_signature_history.append("sig")

        agent.clear_history()

        assert agent.semantic_loop_guard._revert_count == 0
        assert agent.semantic_loop_guard._error_ping_pong_count == 0
        assert len(agent.semantic_loop_guard._error_signature_history) == 0

    def test_clear_history_drains_vision_pending_images(self):
        vision._pending.append({"data": "base64img", "media_type": "image/png"})
        assert len(vision._pending) > 0

        agent = ArgentAgent()
        agent.clear_history()

        assert len(vision._pending) == 0

    def test_clear_history_clears_memory_and_session_id(self):
        agent = ArgentAgent()
        agent.session_id = "sess_to_abandon"
        memory.set_objective("Previous objective")

        agent.clear_history()

        assert agent.session_id is None
        assert memory.data["objective"] == ""

    def test_clear_history_generates_fresh_system_prompt(self):
        agent = ArgentAgent()
        agent.messages = [
            {"role": "system", "content": "stale system prompt"},
            {"role": "user", "content": "hello"}
        ]

        agent.clear_history()

        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] != "stale system prompt"
        assert "Argent Coder" in agent.messages[0]["content"]

    def test_command_handler_aliases_clear_reset_new(self, monkeypatch):
        said = []
        monkeypatch.setattr("command_handler.print_system", lambda text: said.append(str(text)))

        agent = ArgentAgent()

        for alias in ("/clear", "/reset", "/new", "  /CLEAR  ", "/RESET"):
            agent.messages.append({"role": "user", "content": "msg"})
            agent.session_id = "some_sess"
            said.clear()

            res = handle_slash_command(alias, agent)
            assert res is None or res is False
            assert len(agent.messages) == 1
            assert agent.session_id is None
            assert any("очищена" in s for s in said)
