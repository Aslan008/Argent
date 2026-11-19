# -*- coding: utf-8 -*-
"""Окно длинного состояния в рабочем пути: `decide` отдаёт модели выбранные по смыслу части, а не 70/30."""

import pytest

from src.leeth2.runtime import Leeth2Runtime


class _FakeEncoder:
    """Four characters per token is enough to exercise the fitting loop without a model."""
    max_len = 4096

    def tok(self, text):
        return {"input_ids": list(range(max(1, len(text) // 4)))}


def _runtime():
    rt = Leeth2Runtime.__new__(Leeth2Runtime)          # only fit_state is under test here
    rt.encoder = _FakeEncoder()
    return rt


def _long_state():
    segs = ["Setting: a billing console."]
    segs += [f"Log line {i}: routine task {i} finished without warnings." for i in range(600)]
    segs.append("Note: the archive holds the invoice for March as a scanned copy.")
    segs += [f"Log line {i}: queue is empty." for i in range(600)]
    return "\n".join(segs)


def test_a_long_state_is_cut_to_the_budget_and_keeps_the_fact(monkeypatch):
    monkeypatch.delenv("LEETH2_WINDOW", raising=False)
    rt = _runtime()
    state = _long_state()
    out = rt.fit_state(state, "Does the archive hold the invoice for March?")
    assert len(out) < len(state) / 2
    assert "invoice for March" in out
    assert len(rt.encoder.tok(out)["input_ids"]) <= int(rt.encoder.max_len * 0.95)


def test_a_short_state_is_not_touched(monkeypatch):
    monkeypatch.delenv("LEETH2_WINDOW", raising=False)
    rt = _runtime()
    state = "Setting: two lines only.\nThe archive holds the invoice for March."
    assert rt.fit_state(state, "invoice?") == state


def test_the_window_can_be_switched_off(monkeypatch):
    """LEETH2_WINDOW=0 keeps the old behaviour, which the earlier reports were measured under."""
    monkeypatch.setenv("LEETH2_WINDOW", "0")
    rt = _runtime()
    state = _long_state()
    assert rt.fit_state(state, "Does the archive hold the invoice for March?") == state


def test_a_broken_tokenizer_returns_the_state_unchanged(monkeypatch):
    monkeypatch.delenv("LEETH2_WINDOW", raising=False)
    rt = _runtime()

    def boom(_):
        raise RuntimeError("no tokenizer")

    rt.encoder.tok = boom
    state = _long_state()
    assert rt.fit_state(state, "q") == state              # an answer is never lost to a failed window
