"""The reaction tag the model may put at the end of its answer.

The tag is a private agreement between the prompt and `src/telegram/reactions.py`: it is
cut out before the text is sent, so a leak into the chat is a test failure, not a surprise.
"""

import pytest

from src.telegram import reactions


def test_a_tag_becomes_a_reaction_and_leaves_the_text():
    text, emoji = reactions.extract("Норм идея, только тесты добавь.\n[[react:🔥]]")

    assert emoji == "🔥"
    assert text == "Норм идея, только тесты добавь."


def test_no_tag_means_no_reaction():
    text, emoji = reactions.extract("Просто ответ без реакции.")

    assert emoji is None
    assert text == "Просто ответ без реакции."


def test_spaces_and_case_are_tolerated():
    for raw in ("[[ react : 😄 ]]", "[[REACT:😄]]", "  [[react: 😄]]  "):
        text, emoji = reactions.extract(f"ответ {raw}")
        assert emoji == "😄", raw
        assert "[[" not in text, "the tag must never survive into the chat"


def test_the_last_usable_tag_wins_and_all_of_them_are_removed():
    text, emoji = reactions.extract("a [[react:👍]] b [[react:🔥]]")

    assert emoji == "🔥"
    assert text == "a  b"


def test_a_word_instead_of_an_emoji_is_refused_not_sent():
    """Telegram would answer 400 to a text reaction; refusing here keeps the log clean."""
    text, emoji = reactions.extract("готово [[react:happy]]")

    assert emoji is None
    assert text == "готово", "the tag is still removed — a broken tag must not leak either"


def test_a_malformed_tag_is_left_alone():
    text, emoji = reactions.extract("[[react:]] и [[react]]")

    assert emoji is None
    assert "[[react]]" in text


def test_an_empty_answer_stays_empty():
    assert reactions.extract("") == ("", None)
