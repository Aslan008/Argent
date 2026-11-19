# -*- coding: utf-8 -*-
"""Отбор короткого списка опций: эмбеддинги плюс слова, с кэшем векторов."""

import pytest

from src.leeth2.retrieval import ShortlistRetriever


def _hash_embed(texts):
    """A tiny stand-in for a real embedder: same words -> close vectors, so the ranking is predictable without torch."""
    vocab = {}
    out = []
    for t in texts:
        v = [0.0] * 32
        for w in t.lower().split():
            v[vocab.setdefault(w, len(vocab) % 32)] += 1.0
        out.append(v)
    return out


def test_it_returns_at_most_k_and_never_loses_an_option():
    r = ShortlistRetriever(_hash_embed)
    options = [f"option number {i}" for i in range(200)]
    got = r("a request", "which one", options, 16)
    assert len(got) == 16 and set(got) <= set(options)
    assert len(set(got)) == len(got)                      # no repeats


def test_a_short_list_passes_through_whole():
    r = ShortlistRetriever(_hash_embed)
    options = ["alpha", "beta"]
    assert r("q", "q", options, 96) == options or set(r("q", "q", options, 96)) == set(options)


def test_the_vectors_of_options_are_cached_between_calls():
    """A dropdown, a tool catalogue or a category list repeats itself: after the first call only the question is embedded."""
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return _hash_embed(texts)

    r = ShortlistRetriever(embed)
    options = ["Гонконг", "Китай", "Германия", "Япония"]
    r("город", "какой город", options, 2)
    assert sum(len(c) for c in calls) == 1 + len(options)  # one request plus every option, in whatever order
    calls.clear()
    r("город", "какой город", options, 2)
    assert sum(len(c) for c in calls) == 1                 # the request is embedded again, the options are not
    assert r.cache_hits >= len(options)


def test_the_lexical_term_rescues_what_the_embedder_misses():
    """Word overlap alone is a weak retriever but a useful second opinion: a name that appears verbatim must not be dropped."""
    def embed(texts):                                    # embeds nothing useful
        return [[0.0] * 4 for _ in texts]

    r = ShortlistRetriever(embed, semantic_weight=0.0, lexical_weight=1.0)
    options = ["usual noise words here", "a different kind of noise", "PostgreSQL cluster restart", "more noise"]
    got = r("renamed", "how do I restart the PostgreSQL cluster", options, 1)
    assert got == ["PostgreSQL cluster restart"]


def test_a_broken_embedder_does_not_lose_the_call():
    def embed(texts):
        raise RuntimeError("no weights")

    r = ShortlistRetriever(embed)
    options = ["a", "b", "c"]
    with pytest.raises(RuntimeError):
        r("q", "q", options, 2)                          # the caller decides, the retriever does not swallow it silently
