"""Comprehensive test suite for the Universal Rethink system in Argent.

Tests cover:
- Layer 1: Configuration toggles, persistence, and status label formatting
- Layer 2: Universal (non-code-specific) system prompt injection & toggle control
- Layer 3: Streaming tag parsing (<rethink>, <reflection>), event stream emission
- Layer 4: History context trimming and tag sanitization
- Layer 5: CLI command dispatch (/rethink and /think rethink)
- Layer 6: Bridge event normalization
"""

import pytest
from unittest.mock import MagicMock, patch

import config
from src.agent.trimmer import clean_messages_for_llm
from src.bridge.events import to_event


class TestRethinkConfig:
    def setup_method(self):
        self.orig_rethink = config.get_thinking_rethink()
        self.orig_mode = config.get_thinking_mode()
        self.orig_effort = config.get_reasoning_effort()

    def teardown_method(self):
        config.set_thinking_rethink(self.orig_rethink)
        config.set_thinking_mode(self.orig_mode)
        config.set_reasoning_effort(self.orig_effort)

    def test_toggle_rethink_state(self):
        config.set_thinking_rethink(True)
        assert config.get_thinking_rethink() is True

        config.set_thinking_rethink(False)
        assert config.get_thinking_rethink() is False

    def test_display_label_includes_rethink_badge(self):
        config.set_thinking_mode("ON")
        config.set_reasoning_effort("high")

        config.set_thinking_rethink(False)
        lbl_off = config.get_thinking_display_label()
        assert lbl_off == "ON/high"
        assert "+rethink" not in lbl_off

        config.set_thinking_rethink(True)
        lbl_on = config.get_thinking_display_label()
        assert lbl_on == "ON/high+rethink"

    def test_display_label_when_thinking_off(self):
        config.set_thinking_mode("OFF")
        config.set_thinking_rethink(True)
        lbl = config.get_thinking_display_label()
        assert lbl == "OFF+rethink"


class TestRethinkSystemPrompt:
    def setup_method(self):
        self.orig_rethink = config.get_thinking_rethink()

    def teardown_method(self):
        config.set_thinking_rethink(self.orig_rethink)

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_prompt_omitted_when_disabled(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent.provider = "test"
        agent.model_name = "test_model"

        config.set_thinking_rethink(False)
        prompt = agent.build_system_prompt()

        assert "DELIBERATE RETHINK PROTOCOL" not in prompt
        assert "<rethink>" not in prompt

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_prompt_injected_when_enabled(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent.provider = "test"
        agent.model_name = "test_model"

        config.set_thinking_rethink(True)
        prompt = agent.build_system_prompt()

        assert "DELIBERATE RETHINK PROTOCOL" in prompt
        assert "<rethink>...</rethink>" in prompt
        assert "Challenge assumptions" in prompt
        assert "Stress-test" in prompt
        assert "Self-correct" in prompt

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_prompt_is_universal_and_not_code_bound(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent.provider = "test"
        agent.model_name = "test_model"

        config.set_thinking_rethink(True)
        prompt = agent.build_system_prompt()

        # Check that the DELIBERATE RETHINK PROTOCOL section does not restrict itself to code
        start_idx = prompt.find("## DELIBERATE RETHINK PROTOCOL")
        end_idx = prompt.find("## 7. COMMUNICATION")
        rethink_section = prompt[start_idx:end_idx]

        assert "When solving non-trivial problems, formulating plans, or taking actions" in rethink_section
        # Should be compact (< 700 characters)
        assert len(rethink_section.strip()) < 700


class TestRethinkTrimmer:
    def test_strip_closed_rethink_and_reflection_tags(self):
        messages = [
            {
                "role": "assistant",
                "content": "<think>Draft plan</think><rethink>Wait, verify step 2</rethink>Here is the completed plan.",
            },
            {
                "role": "assistant",
                "content": "<reflection>Alternative point of view</reflection>Final synthesis.",
            },
        ]
        cleaned = clean_messages_for_llm(messages, strip_reasoning=True)
        assert cleaned[0]["content"] == "Here is the completed plan."
        assert cleaned[1]["content"] == "Final synthesis."

    def test_strip_unclosed_rethink_tags(self):
        messages = [
            {
                "role": "assistant",
                "content": "Valid answer before.<rethink>Unclosed thinking cut off mid-sentence...",
            }
        ]
        cleaned = clean_messages_for_llm(messages, strip_reasoning=True)
        assert cleaned[0]["content"] == "Valid answer before."


class TestRethinkBridgeEvents:
    def test_normalize_rethink_stream(self):
        event = {"type": "rethink_stream", "content": "Questioning premise"}
        normalized = to_event(event)
        assert normalized["type"] == "thinking"
        assert normalized["text"] == "Questioning premise"
        # The TUI shows a rethink apart from the first-pass reasoning.
        assert normalized["rethink"] is True

    def test_plain_thinking_is_not_flagged_as_rethink(self):
        normalized = to_event({"type": "thinking_stream", "content": "x"})
        assert normalized["type"] == "thinking"
        assert "rethink" not in normalized


class TestRethinkCommands:
    def setup_method(self):
        self.orig_rethink = config.get_thinking_rethink()

    def teardown_method(self):
        config.set_thinking_rethink(self.orig_rethink)

    def test_rethink_command_toggle(self):
        from command_handler import handle_slash_command

        mock_agent = MagicMock()

        # /rethink on
        handle_slash_command("/rethink on", mock_agent)
        assert config.get_thinking_rethink() is True

        # /rethink off
        handle_slash_command("/rethink off", mock_agent)
        assert config.get_thinking_rethink() is False

    def test_think_rethink_subcommand(self):
        from command_handler import handle_slash_command

        mock_agent = MagicMock()

        handle_slash_command("/think rethink on", mock_agent)
        assert config.get_thinking_rethink() is True

        handle_slash_command("/think rethink off", mock_agent)
        assert config.get_thinking_rethink() is False

        # /think rethink without args toggles
        handle_slash_command("/think rethink", mock_agent)
        assert config.get_thinking_rethink() is True

        handle_slash_command("/think rethink", mock_agent)
        assert config.get_thinking_rethink() is False


class TestRethinkStreamingParser:
    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_streaming_rethink_separation(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.model_name = "test_model"
        agent.provider = "test_provider"
        agent.max_history_messages = 20
        agent.max_context_tokens = 8192
        agent._native_tools_unsupported = False
        agent._constrained_unsupported = False
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.strategy.supports_native_tools.return_value = True
        agent.strategy.wants_constrained_decoding.return_value = False
        agent.strategy.wants_objective_anchor.return_value = False
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent._stop_requested = False
        agent.messages = [{"role": "user", "content": "Explain gravity"}]
        agent._read_cache = {}
        agent._checkpoint_turn_indices = {}
        agent._log_turn_end = MagicMock()
        agent._record_turn_start = MagicMock()
        agent._record_turn_end = MagicMock()
        agent._trim_history = MagicMock()
        agent._salvage_continues = 0
        agent._truncate_continues = 0
        agent._last_user_turn = 0
        agent.last_turn_duration_s = 0.0
        agent._read_dedup_note = MagicMock(return_value=None)
        agent.repair_history = MagicMock()
        agent.step_event_callback = None

        # Mock provider streaming response with <think> and <rethink>
        chunks = [
            "<think>Thinking",
            " through the physics</think>",
            "<rethink>Checking ",
            "for counter-theories</rethink>",
            "Gravity is the curvature of spacetime.",
        ]

        mock_provider = MagicMock()
        mock_provider.validate_config.return_value = None
        mock_provider.supports_constrained_decoding.return_value = False
        mock_provider.stream_chat.return_value = (
            {"content": chunk} for chunk in chunks
        )

        with patch("agent.create_provider", return_value=mock_provider), \
             patch("agent.get_available_tools", return_value={}), \
             patch("agent.get_tool_schemas", return_value=[]):

            events = list(agent.process_user_input("Explain gravity"))

            rethink_events = [e for e in events if e.get("type") == "rethink_stream"]
            thinking_events = [e for e in events if e.get("type") == "thinking_stream"]
            content_events = [e for e in events if e.get("type") == "content_stream"]

            rethink_text = "".join(e["content"] for e in rethink_events)
            thinking_text = "".join(e["content"] for e in thinking_events)
            content_text = "".join(e["content"] for e in content_events)

            assert "Checking for counter-theories" in rethink_text
            assert "Thinking through the physics" in thinking_text
            assert "Gravity is the curvature of spacetime." in content_text

            # Assistant message saved to history
            assistant_msg = agent.messages[-1]
            assert assistant_msg["role"] == "assistant"
            assert "Gravity is the curvature of spacetime." in assistant_msg["content"]
            assert "Checking for counter-theories" in assistant_msg["thinking"]
            assert "Thinking through the physics" in assistant_msg["thinking"]
            assert "Checking for counter-theories" in assistant_msg.get("rethink", "")

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_nested_rethink_without_closing_think_does_not_swallow_content(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.model_name = "test_model"
        agent.provider = "test_provider"
        agent.max_history_messages = 20
        agent.max_context_tokens = 8192
        agent._native_tools_unsupported = False
        agent._constrained_unsupported = False
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.strategy.supports_native_tools.return_value = True
        agent.strategy.wants_constrained_decoding.return_value = False
        agent.strategy.wants_objective_anchor.return_value = False
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent._stop_requested = False
        agent.messages = [{"role": "user", "content": "Solve puzzle"}]
        agent._read_cache = {}
        agent._checkpoint_turn_indices = {}
        agent._log_turn_end = MagicMock()
        agent._record_turn_start = MagicMock()
        agent._record_turn_end = MagicMock()
        agent._trim_history = MagicMock()
        agent._salvage_continues = 0
        agent._truncate_continues = 0
        agent._last_user_turn = 0
        agent.last_turn_duration_s = 0.0
        agent._read_dedup_note = MagicMock(return_value=None)
        agent.repair_history = MagicMock()
        agent.step_event_callback = None

        # Model opens <think>, then opens <rethink> without explicitly closing </think> first
        chunks = [
            "<think>Drafting step 1 and step 2.",
            " <rethink>Wait, let me verify step 2 edge case.</rethink>",
            " Here is the final verified answer.",
        ]

        mock_provider = MagicMock()
        mock_provider.validate_config.return_value = None
        mock_provider.supports_constrained_decoding.return_value = False
        mock_provider.stream_chat.return_value = (
            {"content": chunk} for chunk in chunks
        )

        with patch("agent.create_provider", return_value=mock_provider), \
             patch("agent.get_available_tools", return_value={}), \
             patch("agent.get_tool_schemas", return_value=[]):

            events = list(agent.process_user_input("Solve puzzle"))

            content_events = [e for e in events if e.get("type") == "content_stream"]
            rethink_events = [e for e in events if e.get("type") == "rethink_stream"]
            thinking_events = [e for e in events if e.get("type") == "thinking_stream"]

            content_text = "".join(e["content"] for e in content_events)
            rethink_text = "".join(e["content"] for e in rethink_events)
            thinking_text = "".join(e["content"] for e in thinking_events)

            assert "Wait, let me verify step 2 edge case." in rethink_text
            assert "Drafting step 1 and step 2." in thinking_text
            # Final content MUST NOT be swallowed by the thinking stream
            assert "Here is the final verified answer." in content_text

            assistant_msg = agent.messages[-1]
            assert "Here is the final verified answer." in assistant_msg["content"]
            assert "Wait, let me verify step 2 edge case." in assistant_msg["rethink"]

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_rethink_token_splitting_across_chunks(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.model_name = "test_model"
        agent.provider = "test_provider"
        agent.max_history_messages = 20
        agent.max_context_tokens = 8192
        agent._native_tools_unsupported = False
        agent._constrained_unsupported = False
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.strategy.supports_native_tools.return_value = True
        agent.strategy.wants_constrained_decoding.return_value = False
        agent.strategy.wants_objective_anchor.return_value = False
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent._stop_requested = False
        agent.messages = [{"role": "user", "content": "Test split"}]
        agent._read_cache = {}
        agent._checkpoint_turn_indices = {}
        agent._log_turn_end = MagicMock()
        agent._record_turn_start = MagicMock()
        agent._record_turn_end = MagicMock()
        agent._trim_history = MagicMock()
        agent._salvage_continues = 0
        agent._truncate_continues = 0
        agent._last_user_turn = 0
        agent.last_turn_duration_s = 0.0
        agent._read_dedup_note = MagicMock(return_value=None)
        agent.repair_history = MagicMock()
        agent.step_event_callback = None

        # Tags split across token / chunk boundaries
        chunks = [
            "<think>Thinking</think><reth",
            "ink>Reflecting on doubts</re",
            "think>Final answer is ready.",
        ]

        mock_provider = MagicMock()
        mock_provider.validate_config.return_value = None
        mock_provider.supports_constrained_decoding.return_value = False
        mock_provider.stream_chat.return_value = (
            {"content": chunk} for chunk in chunks
        )

        with patch("agent.create_provider", return_value=mock_provider), \
             patch("agent.get_available_tools", return_value={}), \
             patch("agent.get_tool_schemas", return_value=[]):

            events = list(agent.process_user_input("Test split"))

            content_text = "".join(e["content"] for e in events if e.get("type") == "content_stream")
            rethink_text = "".join(e["content"] for e in events if e.get("type") == "rethink_stream")
            thinking_text = "".join(e["content"] for e in events if e.get("type") == "thinking_stream")

            assert rethink_text == "Reflecting on doubts"
            assert thinking_text == "Thinking"
            # Tag fragments like <reth must not leak into content
            assert "<reth" not in content_text
            assert content_text == "Final answer is ready."

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_multi_tag_single_chunk_chronology(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.model_name = "test_model"
        agent.provider = "test_provider"
        agent.max_history_messages = 20
        agent.max_context_tokens = 8192
        agent._native_tools_unsupported = False
        agent._constrained_unsupported = False
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.strategy.supports_native_tools.return_value = True
        agent.strategy.wants_constrained_decoding.return_value = False
        agent.strategy.wants_objective_anchor.return_value = False
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent._stop_requested = False
        agent.messages = [{"role": "user", "content": "Single chunk"}]
        agent._read_cache = {}
        agent._checkpoint_turn_indices = {}
        agent._log_turn_end = MagicMock()
        agent._record_turn_start = MagicMock()
        agent._record_turn_end = MagicMock()
        agent._trim_history = MagicMock()
        agent._salvage_continues = 0
        agent._truncate_continues = 0
        agent._last_user_turn = 0
        agent.last_turn_duration_s = 0.0
        agent._read_dedup_note = MagicMock(return_value=None)
        agent.repair_history = MagicMock()
        agent.step_event_callback = None

        chunks = ["<think>Secret draft</think><rethink>Double check</rethink>Public answer."]

        mock_provider = MagicMock()
        mock_provider.validate_config.return_value = None
        mock_provider.supports_constrained_decoding.return_value = False
        mock_provider.stream_chat.return_value = (
            {"content": chunk} for chunk in chunks
        )

        with patch("agent.create_provider", return_value=mock_provider), \
             patch("agent.get_available_tools", return_value={}), \
             patch("agent.get_tool_schemas", return_value=[]):

            events = list(agent.process_user_input("Single chunk"))

            content_text = "".join(e["content"] for e in events if e.get("type") == "content_stream")
            rethink_text = "".join(e["content"] for e in events if e.get("type") == "rethink_stream")
            thinking_text = "".join(e["content"] for e in events if e.get("type") == "thinking_stream")

            assert thinking_text == "Secret draft"
            assert rethink_text == "Double check"
            assert "<think>" not in content_text
            assert content_text == "Public answer."

    @patch("agent.ArgentAgent.__init__", return_value=None)
    def test_math_less_than_preserved_by_lookahead_buffer(self, mock_init):
        from agent import ArgentAgent

        agent = ArgentAgent()
        agent.model_name = "test_model"
        agent.provider = "test_provider"
        agent.max_history_messages = 20
        agent.max_context_tokens = 8192
        agent._native_tools_unsupported = False
        agent._constrained_unsupported = False
        agent.strategy = MagicMock()
        agent.strategy.should_include_section.return_value = True
        agent.strategy.supports_native_tools.return_value = True
        agent.strategy.wants_constrained_decoding.return_value = False
        agent.strategy.wants_objective_anchor.return_value = False
        agent.tier = "tier_local"
        agent.workspace_dir = "."
        agent._stop_requested = False
        agent.messages = [{"role": "user", "content": "Math test"}]
        agent._read_cache = {}
        agent._checkpoint_turn_indices = {}
        agent._log_turn_end = MagicMock()
        agent._record_turn_start = MagicMock()
        agent._record_turn_end = MagicMock()
        agent._trim_history = MagicMock()
        agent._salvage_continues = 0
        agent._truncate_continues = 0
        agent._last_user_turn = 0
        agent.last_turn_duration_s = 0.0
        agent._read_dedup_note = MagicMock(return_value=None)
        agent.repair_history = MagicMock()
        agent.step_event_callback = None

        chunks = ["if x < ", "y and a > b: return True"]

        mock_provider = MagicMock()
        mock_provider.validate_config.return_value = None
        mock_provider.supports_constrained_decoding.return_value = False
        mock_provider.stream_chat.return_value = (
            {"content": chunk} for chunk in chunks
        )

        with patch("agent.create_provider", return_value=mock_provider), \
             patch("agent.get_available_tools", return_value={}), \
             patch("agent.get_tool_schemas", return_value=[]):

            events = list(agent.process_user_input("Math test"))
            content_text = "".join(e["content"] for e in events if e.get("type") == "content_stream")
            assert content_text == "if x < y and a > b: return True"

    def test_trimmer_reconstructs_separate_think_and_rethink(self):
        messages = [
            {
                "role": "assistant",
                "content": "Completed task.",
                "thinking": "Initial plan.\nVerify safety.",
                "rethink": "Verify safety.",
            }
        ]
        cleaned = clean_messages_for_llm(messages, strip_reasoning=False)
        assert cleaned[0]["content"] == "<think>\nInitial plan.\n</think>\n\n<rethink>\nVerify safety.\n</rethink>\n\nCompleted task."
        # Custom keys removed for API schema compliance
        assert "thinking" not in cleaned[0]
        assert "rethink" not in cleaned[0]
