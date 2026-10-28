"""What a group answer says about itself: the model and the time, nothing else.

The numbers under an answer are for the person who owns the machine. In a group they say
nothing to the people reading — nobody there knows or cares how full a model's context is.
"""

from src.telegram.render import turn_footer

SNAP = {
    "model": "gpt-4o-mini", "total_s": 4.2, "tool_calls": 2,
    "prompt_tokens": 1200, "completion_tokens": 340, "ctx_percent": 18.0, "cost": 0.002,
}


def test_the_private_chat_keeps_every_number():
    footer = turn_footer(SNAP)

    assert "gpt-4o-mini" in footer
    assert "ctx 18%" in footer
    assert "↑1.2k" in footer and "🛠 2" in footer


def test_a_group_footer_is_only_the_model_and_the_time():
    footer = turn_footer(SNAP, compact=True)

    assert "gpt-4o-mini" in footer and "4" in footer
    assert "ctx" not in footer, "context fill means nothing to the people in a group"
    assert "↑" not in footer and "🛠" not in footer and "$" not in footer


def test_a_group_may_hide_the_model_from_everyone_but_the_owner():
    """A group message is read by everyone, so a name for the owner's eyes stays out of it."""
    footer = turn_footer(SNAP, compact=True, with_model=False)

    assert "gpt-4o-mini" not in footer
    assert "4" in footer, "the time is harmless and still useful"


def test_a_compact_footer_without_a_model_does_not_invent_one():
    assert "ctx" not in turn_footer({"total_s": 1.0, "ctx_percent": 50.0}, compact=True)


def test_an_empty_snapshot_has_no_footer():
    assert turn_footer(None) == ""
