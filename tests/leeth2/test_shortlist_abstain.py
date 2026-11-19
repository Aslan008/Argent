# -*- coding: utf-8 -*-
"""Резерв места под «ни один из вариантов»: самая непохожая опция должна пережить обрезку списка."""
import pytest

from src.leeth2.retrieval import ShortlistRetriever


def _embed(texts):
    vocab = {}
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in t.lower().split():
            v[vocab.setdefault(w, len(vocab) % 32)] += 1.0
        out.append(v)
    return out


def test_the_least_relevant_option_survives_the_cut():
    """The shortlist keeps what resembles the request most — the hardest distractors — so "none of these fits" (nothing in common)
    would be cut away and the model would have nothing to decline with. Frozen exam: recall 0.722 -> 0.519 without this."""
    r = ShortlistRetriever(_embed)
    options = ["restart postgres cluster", "restart postgres replica", "postgres cluster status", "none of these fits"]
    got = r("restart the postgres cluster", "which tool", options, 2)
    assert "none of these fits" in got and len(got) == 2


def test_it_can_be_switched_off():
    r = ShortlistRetriever(_embed, keep_least_relevant=False)
    options = ["restart postgres cluster", "restart postgres replica", "postgres cluster status", "none of these fits"]
    got = r("restart the postgres cluster", "which tool", options, 2)
    assert "none of these fits" not in got


def test_a_small_list_keeps_every_option():
    r = ShortlistRetriever(_embed)
    options = ["a", "b", "none of them"]
    assert set(r("a", "q", options, 3)) == set(options)
