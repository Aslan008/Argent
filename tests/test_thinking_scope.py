"""A helper may think less without making the main agent think less.

Thinking is configured globally, and a sub-agent that runs in the same process must
not change the person's own agent. A scope is the narrower statement: inside this
block, thinking is this level — and providers ask the same getters, so nothing in
providers.py had to change.
"""

import pytest

import argent.config as config


@pytest.fixture(autouse=True)
def no_scope_left_behind():
    yield
    assert config.current_thinking_scope() is None, "scope leaked past its block"


@pytest.fixture
def globals(monkeypatch):
    """Pin the global settings so a test never depends on the real config file."""
    values = {"thinking_mode": "auto", "reasoning_effort": None, "thinking_budget": None}
    monkeypatch.setattr(config, "_get", lambda key, default=None: values.get(key, default))
    return values


def test_a_named_level_maps_to_the_three_knobs():
    assert config.thinking_settings("off") == {"mode": "off", "effort": None, "budget": 0}
    assert config.thinking_settings("low") == {"mode": "on", "effort": "low", "budget": None}
    assert config.thinking_settings("high")["effort"] == "high"
    assert config.thinking_settings("auto") is None
    assert config.thinking_settings("") is None


def test_a_typo_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValueError) as err:
        config.thinking_settings("louw")
    assert "thinking must be one of" in str(err.value)


def test_the_scope_wins_over_the_global_setting(globals):
    assert config.get_thinking_mode() == "auto"
    with config.thinking_scope("low"):
        assert config.get_thinking_mode() == "on"
        assert config.get_reasoning_effort() == "low"
    assert config.get_thinking_mode() == "auto"
    assert config.get_reasoning_effort() is None


def test_off_turns_thinking_off_and_zeroes_the_budget(globals):
    with config.thinking_scope("off"):
        assert config.get_thinking_mode() == "off"
        assert config.get_thinking_budget() == 0


def test_auto_is_a_no_op_even_when_the_global_setting_is_on(globals):
    globals["thinking_mode"] = "on"
    globals["reasoning_effort"] = "high"
    with config.thinking_scope("auto"):
        assert config.get_thinking_mode() == "on"
        assert config.get_reasoning_effort() == "high"


def test_nested_scopes_restore_the_outer_one(globals):
    with config.thinking_scope("low"):
        assert config.get_reasoning_effort() == "low"
        with config.thinking_scope("high"):
            assert config.get_reasoning_effort() == "high"
        assert config.get_reasoning_effort() == "low"
    assert config.get_reasoning_effort() is None


def test_explicit_knobs_win_over_the_named_level(globals):
    with config.thinking_scope("low", budget=2048):
        assert config.get_reasoning_effort() == "low"
        assert config.get_thinking_budget() == 2048


def test_the_global_setting_is_untouched_by_a_scope(globals):
    # The point of a scope: what the person configured is still there afterwards.
    globals["thinking_mode"] = "on"
    with config.thinking_scope("off"):
        assert config.get_thinking_mode() == "off"
    assert config.get_thinking_mode() == "on"
