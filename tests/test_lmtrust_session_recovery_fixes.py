"""LMTrust Regression & Blind-Spot Test Suite for Session Recovery Fixes.

Covers:
- L1 x D7: System prompt injection preserving user messages in _refresh_system_prompt (BUG-1)
- L4 x D4: Orphaned tool calls healing across history on session restore (BUG-2)
- L6 x D3: Memory manager re-resolution on directory change in restore_session (BUG-3)
- L4b x D4: Corrupted sessions ignored by get_last_session & offer_previous_context (INC-1)
- L5 x D2: Transient state (read cache, loop guard, checkpoints) cleared on restore (POL-1)
- L2 x D4: Robustness against non-dict items in _make_serializable & _first_user_message (POL-3)
- L1 x D7: Image-only user messages supported in _first_user_message (POL-3)
"""

import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

import agent as agent_mod
from agent import ArgentAgent
import main as main_mod
from main import restore_session, offer_previous_context
import session as session_mod
from session import (
    save_session,
    load_session,
    get_last_session,
    _make_serializable,
    _first_user_message,
)
from memory_manager import memory


@pytest.fixture(autouse=True)
def _isolate_sessions_dir(tmp_path, monkeypatch):
    sess_dir = tmp_path / "sessions"
    sess_dir.mkdir()
    index_path = sess_dir / "index.json"
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sess_dir)
    monkeypatch.setattr(session_mod, "INDEX_PATH", index_path)
    monkeypatch.setattr("ui.print_system", lambda *a, **k: None)
    monkeypatch.setattr("ui.print_error", lambda *a, **k: None)
    yield


# ============================================================================
# BUG-1: System Prompt Injection Preserving User Role
# ============================================================================

class TestBug1SystemPromptRolePreservation:
    def test_refresh_system_prompt_does_not_mutate_user_role(self):
        agent = ArgentAgent()
        # Restored session missing initial system message
        agent.messages = [{"role": "user", "content": "Build snake in Python"}]

        replaced = agent._refresh_system_prompt()
        assert replaced is True
        assert len(agent.messages) == 2
        # Index 0 must be the new system prompt
        assert agent.messages[0]["role"] == "system"
        assert "Argent Coder" in agent.messages[0]["content"]
        # Index 1 must remain intact with original user prompt
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[1]["content"] == "Build snake in Python"

    def test_refresh_system_prompt_updates_existing_system_message(self):
        agent = ArgentAgent()
        agent.messages = [
            {"role": "system", "content": "old prompt"},
            {"role": "user", "content": "Hello"}
        ]
        agent._refresh_system_prompt()
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] != "old prompt"
        assert agent.messages[1]["content"] == "Hello"


# ============================================================================
# BUG-2: Repairing Orphaned Tool Calls in History
# ============================================================================

class TestBug2RepairOrphanedToolCalls:
    def test_repair_history_fixes_unclosed_tool_calls_followed_by_user_message(self):
        agent = ArgentAgent()
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "run task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}
                ]
            },
            # A user message entered subsequently in restored chat
            {"role": "user", "content": "What is the status?"}
        ]

        agent.repair_history()

        # Synthetic results must have been inserted right after the assistant message
        # 6 + 1: the trailing user message gets a synthetic assistant reply
        # (strict provider schemas forbid two consecutive user messages).
        assert len(agent.messages) == 7
        assert agent.messages[2]["role"] == "assistant"
        assert agent.messages[3]["role"] == "tool"
        assert agent.messages[3]["tool_call_id"] == "call_1"
        assert agent.messages[4]["role"] == "tool"
        assert agent.messages[4]["tool_call_id"] == "call_2"
        assert agent.messages[5]["role"] == "user"
        assert agent.messages[5]["content"] == "What is the status?"

    def test_restore_session_automatically_repairs_history(self):
        # Save a session that ended mid-tool execution
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "write something"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}]
            }
        ]
        sid = save_session(msgs, metadata={"model": "local-7b"})
        assert sid is not None

        agent = ArgentAgent()
        ok = restore_session(agent, {"id": sid})
        assert ok is True

        # Must have 4 messages: system, user, assistant, and synthetic tool response
        assert len(agent.messages) == 4
        assert agent.messages[-1]["role"] == "tool"
        assert agent.messages[-1]["tool_call_id"] == "call_abc"


# ============================================================================
# BUG-3: Memory Manager Re-resolution on chdir
# ============================================================================

class TestBug3MemoryReResolution:
    def test_restore_session_resets_memory_proxy_on_chdir(self, tmp_path, monkeypatch):
        project_dir = tmp_path / "TargetProject"
        project_dir.mkdir()
        (project_dir / ".argent").mkdir()

        # Initialize memory in current dir
        _ = memory.data

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do work"}
        ]
        sid = save_session(msgs, metadata={"model": "local-7b", "cwd": str(project_dir)})

        class _ConfirmTrue:
            def confirm(self, *a, **k):
                return type("Ans", (), {"ask": lambda s: True})()

        monkeypatch.setattr("src.startup.questionary", _ConfirmTrue())

        agent = ArgentAgent()
        restore_session(agent, {"id": sid})

        # Memory manager must now point to the target project's memory file
        assert os.path.samefile(os.getcwd(), project_dir)
        new_memory_file = memory._memory_file
        assert str(project_dir).lower() in str(new_memory_file).lower()


# ============================================================================
# INC-1: Corrupted Session Ignored by Startup
# ============================================================================

class TestInc1CorruptedSessionIgnored:
    def test_get_last_session_filters_out_corrupted_sessions(self, tmp_path):
        # Create a corrupt session file
        bad = session_mod.SESSIONS_DIR / "bad_session.json.gz"
        bad.write_bytes(b"corrupt non-gzip content")

        # Must return None instead of [corrupted session]
        assert get_last_session() is None

    def test_offer_previous_context_skips_corrupted_session(self, monkeypatch):
        # When only corrupt session is in store
        monkeypatch.setattr("src.startup.get_last_session", lambda: {
            "id": "corrupt", "preview": "[corrupted session]", "cwd": ""
        })
        monkeypatch.setattr("src.startup.carried_over_memory", lambda: [])

        # questionary.select should never be called
        class _FailIfAsked:
            def select(self, *a, **k):
                pytest.fail("Should not offer corrupted session")
        monkeypatch.setattr("src.startup.questionary", _FailIfAsked())

        offer_previous_context(ArgentAgent())


# ============================================================================
# POL-1: Resetting Transient Caches on Session Restore
# ============================================================================

class TestPol1TransientStateReset:
    def test_restore_session_clears_read_cache_and_loop_guard(self):
        msgs = [{"role": "user", "content": "Test transient reset"}]
        sid = save_session(msgs, metadata={"model": "local-7b"})

        agent = ArgentAgent()
        agent._read_cache["some_file.py"] = {"sig": [1, 2], "stored": "content"}
        agent._checkpoint_turn_indices["abc"] = 1
        agent.loop_guard._recent.append("tool")

        restore_session(agent, {"id": sid})

        assert len(agent._read_cache) == 0
        assert len(agent._checkpoint_turn_indices) == 0
        assert len(agent.loop_guard._recent) == 0


# ============================================================================
# POL-3: Robustness Against Non-Dict Items & Image-Only Messages
# ============================================================================

class TestPol3RobustnessAndImages:
    def test_make_serializable_skips_non_dict_gracefully(self):
        cleaned = _make_serializable(["string_item", None, 123, {"role": "user", "content": "Hi"}])
        assert len(cleaned) == 1
        assert cleaned[0]["content"] == "Hi"

    def test_first_user_message_skips_non_dict_gracefully(self):
        assert _first_user_message([None, "invalid", 42]) is None

    def test_first_user_message_handles_image_only(self):
        msgs = [
            {"role": "user", "content": "", "images": [{"data": "base64", "media_type": "image/jpeg"}]}
        ]
        assert _first_user_message(msgs) == "[Изображение]"
