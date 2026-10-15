"""Unit tests for Argent CLI CommandRegistry and domain handlers."""

import pytest
from src.cli.commands import get_registry, dispatch_command, CommandContext, CommandResult
from src.cli.commands.registry import CommandRegistry
from command_handler import handle_slash_command, known_commands, unknown_command_message


class DummyAgent:
    def __init__(self):
        self.messages = []
        self.session_id = "test_session_id"
        self.model_name = "test-model"

    def clear_history(self):
        self.messages = [{"role": "system", "content": "Argent Coder prompt"}]
        self.session_id = None


def test_registry_contains_core_commands():
    reg = get_registry()
    cmds = reg.get_all_commands()
    assert "/help" in cmds
    assert "/clear" in cmds
    assert "/vibe" in cmds
    assert "/think" in cmds
    assert "/model" in cmds
    assert "/init" in cmds
    assert "/research" in cmds
    assert "/save" in cmds
    assert "/load" in cmds
    assert "/rewind" in cmds
    assert "/diff" in cmds
    assert "/undo" in cmds
    assert len(cmds) >= 40


def test_case_insensitive_dispatch():
    agent = DummyAgent()
    ctx = CommandContext(agent=agent, user_input="/CLEAR")
    res = dispatch_command("/CLEAR", ctx)
    assert res.handled is True
    assert res.should_continue is True
    assert res.turn_counter == 0
    assert len(agent.messages) == 1


def test_init_command_passes_to_agent():
    agent = DummyAgent()
    ctx = CommandContext(agent=agent, user_input="/init")
    res = dispatch_command("/init", ctx)
    assert res.handled is True
    assert res.should_continue is False
    assert res.new_input is not None
    assert "AGENTS.md" in res.new_input
    assert res.active_tools is not None
    assert "analyze_project" in res.active_tools


def test_research_command_validation_and_prompt():
    agent = DummyAgent()
    # Without topic
    ctx_empty = CommandContext(agent=agent, user_input="/research")
    res_empty = dispatch_command("/research", ctx_empty)
    assert res_empty.handled is True
    assert res_empty.should_continue is True
    assert res_empty.new_input is None

    # With topic
    ctx_valid = CommandContext(agent=agent, user_input="/research NextJS SSR")
    res_valid = dispatch_command("/research NextJS SSR", ctx_valid)
    assert res_valid.handled is True
    assert res_valid.should_continue is False
    assert res_valid.new_input is not None
    assert "NextJS SSR" in res_valid.new_input
    assert res_valid.active_tools == ["run_deep_research"]


def test_think_command_modes():
    from config import get_thinking_display_mode
    agent = DummyAgent()

    # think summary
    ctx_sum = CommandContext(agent=agent, user_input="/think summary")
    res_sum = dispatch_command("/think summary", ctx_sum)
    assert res_sum.handled is True
    assert get_thinking_display_mode() == "summary"

    # think full
    ctx_full = CommandContext(agent=agent, user_input="/think full")
    res_full = dispatch_command("/think full", ctx_full)
    assert res_full.handled is True
    assert get_thinking_display_mode() == "full"

    # reset back to summary
    dispatch_command("/think summary", ctx_sum)
    assert get_thinking_display_mode() == "summary"


def test_unknown_command():
    agent = DummyAgent()
    ctx = CommandContext(agent=agent, user_input="/nonexistent_command_xyz")
    res = dispatch_command("/nonexistent_command_xyz", ctx)
    assert res.handled is False


def test_known_commands_integration():
    cmds = known_commands()
    assert "/help" in cmds
    assert "/model" in cmds
    assert "/think" in cmds
    assert "/rewind" in cmds
    assert len(cmds) >= 30


def test_prefix_does_not_collide_with_longer_words():
    """L4×D7: Substring collision test. Commands like /diff must not intercept /difficulty."""
    agent = DummyAgent()
    for false_cmd in ("/difficulty", "/different", "/saved", "/cda", "/undone", "/committee"):
        ctx = CommandContext(agent=agent, user_input=false_cmd)
        res = dispatch_command(false_cmd, ctx)
        assert res.handled is False, f"Command '{false_cmd}' was falsely handled by a prefix command!"


def test_thinking_alias_supported():
    agent = DummyAgent()
    ctx = CommandContext(agent=agent, user_input="/thinking summary")
    res = dispatch_command("/thinking summary", ctx)
    assert res.handled is True


def test_thinking_and_effort_commands_dispatch():
    import config
    agent = DummyAgent()

    # /think on / off / auto
    res = dispatch_command("/think on", CommandContext(agent=agent, user_input="/think on"))
    assert res.handled is True
    assert config.get_thinking_mode() == "on"

    res = dispatch_command("/think off", CommandContext(agent=agent, user_input="/think off"))
    assert res.handled is True
    assert config.get_thinking_mode() == "off"

    res = dispatch_command("/think auto", CommandContext(agent=agent, user_input="/think auto"))
    assert res.handled is True
    assert config.get_thinking_mode() == "auto"

    # /think high / medium / low
    res = dispatch_command("/think high", CommandContext(agent=agent, user_input="/think high"))
    assert res.handled is True
    assert config.get_reasoning_effort() == "high"

    # /effort medium / off
    res = dispatch_command("/effort medium", CommandContext(agent=agent, user_input="/effort medium"))
    assert res.handled is True
    assert config.get_reasoning_effort() == "medium"

    res = dispatch_command("/effort off", CommandContext(agent=agent, user_input="/effort off"))
    assert res.handled is True
    assert config.get_reasoning_effort() is None

    # /think budget
    res = dispatch_command("/think budget 4096", CommandContext(agent=agent, user_input="/think budget 4096"))
    assert res.handled is True
    assert config.get_thinking_budget() == 4096

    res = dispatch_command("/think budget none", CommandContext(agent=agent, user_input="/think budget none"))
    assert res.handled is True
    assert config.get_thinking_budget() is None



def test_duration_and_preview_hardening():
    """L2×D1: Test numerical edge cases (NaN, Inf, None, negative) and None safety."""
    from ui import _format_duration_short, _extract_thought_preview
    assert _format_duration_short(None) == ""
    assert _format_duration_short(-5.0) == ""
    assert _format_duration_short(float("nan")) == ""
    assert _format_duration_short(float("inf")) == ""
    assert _format_duration_short(0.025) == "25ms"
    assert _format_duration_short(2.34) == "2.3s"

    assert _extract_thought_preview(None) == ""
    assert _extract_thought_preview("") == ""
    assert _extract_thought_preview("### Final conclusion: we should proceed") == "Final conclusion: we should proceed"


def test_print_inline_diff_none_safety():
    """L2×D4: Ensure print_inline_diff does not crash when file_path is None or diff is empty."""
    from ui import print_inline_diff
    # None file path
    print_inline_diff(None, "+ added line\n- removed line")
    # Empty diff
    print_inline_diff("some_file.py", "")
    print_inline_diff("some_file.py", None)

