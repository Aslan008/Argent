# -*- coding: utf-8 -*-
"""Длинное состояние: модель должна получать то, что относится к вопросу, а не первые 70 % и последние 30 %."""

import pytest

from src.leeth2.window import segments, shrink, score_segments


def _long_state(fact: str = "the archive holds the invoice for March as a scanned copy") -> str:
    segs = ["Setting: an operations console for the billing team."]
    segs += [f"Log line {i}: routine task number {i} finished without warnings." for i in range(300)]
    segs.append(f"Note: {fact}.")
    segs += [f"Log line {i}: queue is empty." for i in range(300)]
    return "\n".join(segs)


def test_the_fact_in_the_middle_survives_the_cut():
    """The encoder keeps the head and the tail, so a fact in the middle of a long text is never read (frozen exam: 0.52-0.58,
    the same as guessing)."""
    state = _long_state()
    out = shrink(state, "Does the archive hold the invoice for March?", 4000)
    assert len(out) < len(state) / 5
    assert "invoice for March" in out


def test_the_setting_line_is_always_kept_and_the_order_is_original():
    state = _long_state()
    out = shrink(state, "Does the archive hold the invoice for March?", 4000)
    assert out.startswith("Setting: an operations console")
    kept = out.splitlines()
    positions = [state.splitlines().index(l) for l in kept]
    assert positions == sorted(positions)


def test_a_short_state_is_returned_untouched():
    state = "Setting: one line.\nAnother line with the answer."
    assert shrink(state, "anything", 4000) == state


def test_a_question_with_no_shared_words_still_returns_something_usable():
    """No overlap scores zero everywhere; the result must still be the head, exactly what the encoder kept before — the function
    may improve on the old behaviour but never make it worse."""
    state = _long_state()
    out = shrink(state, "Ззззз qqqq", 2000)
    assert out and len(out) <= 2000 and out.startswith("Setting:")


def test_one_giant_line_is_cut_into_sentences_not_dropped_whole():
    giant = "Head sentence. " + ("Filler sentence number x. " * 500) + "The archive holds the invoice for March."
    segs = segments(giant)
    assert len(segs) > 3
    out = shrink(giant, "Does the archive hold the invoice for March?", 3000)
    assert "invoice for March" in out


def test_rare_words_count_more_than_common_ones():
    segs = ["the task is done", "the task is done", "the task is done", "the invoice for March is archived"]
    scores = score_segments(segs, "Does the archive hold the invoice for March?")
    assert scores[3] > scores[0]
