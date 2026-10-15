"""Batches are packed by the TRUE encoded length: no batch exceeds its token budget and padding is negligible."""

import random
from pathlib import Path

import numpy as np
import pytest

from .test_train_smoke import TOK_DIR, make_corpus

pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="mmBERT tokenizer not on disk")


def test_exact_lengths_match_direct_encoding_and_packing_respects_the_budget(tmp_path):
    from transformers import AutoTokenizer

    from src.leeth2.datafile import JsonlReader, exact_token_lengths, plan_batches
    from src.leeth2.encoding import Leeth2Encoder

    corpus = tmp_path / "train.jsonl"
    make_corpus(corpus, 600, 5)
    lens1 = exact_token_lengths(corpus, str(TOK_DIR), 256000, 64, workers=1, chunk=100, cache=tmp_path / "l1.npy")
    lens2 = exact_token_lengths(corpus, str(TOK_DIR), 256000, 64, workers=2, chunk=150)
    assert (lens1 == lens2).all()
    assert (tmp_path / "l1.npy").exists() and (exact_token_lengths(corpus, str(TOK_DIR), 256000, 64, cache=tmp_path / "l1.npy") == lens1).all()

    rd = JsonlReader(corpus)
    enc = Leeth2Encoder(AutoTokenizer.from_pretrained(str(TOK_DIR)), 256000, max_len=64)
    for i in (0, 7, 123, 599):
        assert lens1[i] == len(enc.encode(rd.get(i), rng=random.Random(i)))

    budget = 400
    plan = plan_batches(rd.lens, budget, 64, seed=1, world=2, max_len=64, caps=rd.caps, token_lens=lens1)
    flat = [i for b in plan for i in b]
    assert len(flat) == len(set(flat)) and len(plan) % 2 == 0
    padded = real = 0
    for b in plan:
        longest = int(lens1[b].max())
        assert longest * len(b) <= budget or len(b) == 1, "a batch never overshoots its padded-token budget"
        padded += longest * len(b)
        real += int(lens1[b].sum())
    assert real / padded > 0.9, f"length-sorted packing wastes little: {real / padded:.3f}"
