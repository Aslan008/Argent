"""Several equally good options are labelled with a probability list instead of an arbitrary single pick."""

import math
import random

import numpy as np
import pytest
import torch

from src.leeth2 import metrics
from src.leeth2.encoding import Leeth2Encoder, collate
from src.leeth2.losses import choice_loss, compute_losses
from src.leeth2.schema import Sample

from .conftest import VOCAB


def soft_sample(target):
    return Sample("g", "game", "en", "choice", "grid state", "best move", ["up", "down", "left", "right"], target)


def test_schema_accepts_and_rejects_soft_labels():
    soft_sample([0.5, 0.5, 0.0, 0.0]).validate()
    for bad in ([0.5, 0.5, 0.0], [0.7, 0.7, 0.0, 0.0], [-0.1, 1.1, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2]):
        with pytest.raises(ValueError):
            soft_sample(bad).validate()
    assert soft_sample([0.0, 0.25, 0.75, 0.0]).hard_choice() == 2


def test_encoding_permutes_the_soft_label_with_the_options(tok):
    enc = Leeth2Encoder(tok, VOCAB, max_len=64)
    s = soft_sample([0.0, 0.5, 0.0, 0.5])
    for seed in range(20):
        x = enc.encode(s, rng=random.Random(seed))
        shown = [s.options[i] for i in x.perm]
        mass = {shown[i]: p for i, p in enumerate(x.choice_probs)}
        assert mass == {"up": 0.0, "down": 0.5, "left": 0.0, "right": 0.5}
        assert x.choice_probs[x.choice_target] > 0
    b = collate([enc.encode(s), enc.encode(soft_sample(1))])
    assert b["has_soft"].tolist() == [True, False] and b["choice_probs"][0].sum() == pytest.approx(1.0)


def test_loss_of_a_soft_label_is_the_cross_entropy_with_the_distribution():
    logits = torch.tensor([[2.0, 1.0, -1.0, 0.0]])
    q = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    want = -(q * torch.log_softmax(logits, -1)).sum().item()
    got = choice_loss(logits, torch.tensor([0]), q, torch.tensor([True]))[0].item()
    assert got == pytest.approx(want, rel=1e-5)
    # a hard example in the same batch is untouched
    both = choice_loss(logits.repeat(2, 1), torch.tensor([0, 2]), torch.cat([q, torch.zeros(1, 4)]), torch.tensor([True, False]))
    assert both[1].item() == pytest.approx(-math.log(float(torch.softmax(logits, -1)[0, 2])), rel=1e-5)


def test_soft_label_is_minimised_by_splitting_probability_over_the_optimal_set(tok, tiny_model):
    enc = Leeth2Encoder(tok, VOCAB, max_len=64)
    batch = collate([enc.encode(soft_sample([0.5, 0.5, 0.0, 0.0]))] * 4)
    opt = torch.optim.AdamW([p for p in tiny_model.parameters() if p.requires_grad], lr=5e-3)
    tiny_model.train()
    for _ in range(80):
        out = tiny_model(**batch)
        loss, _ = compute_losses(out, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    tiny_model.eval()
    with torch.no_grad():
        p = torch.softmax(tiny_model(**batch)["choice_logits"][0], -1)
    assert p[0] == pytest.approx(0.5, abs=0.08) and p[1] == pytest.approx(0.5, abs=0.08) and p[2] < 0.05


def test_report_counts_a_prediction_on_any_optimal_option_as_correct():
    probs = [np.array([0.1, 0.6, 0.2, 0.1]), np.array([0.7, 0.1, 0.1, 0.1])]
    targets = [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]
    r = metrics.choice_report(probs, targets)
    assert r["acc"] == 0.5
    assert r["nll"] == pytest.approx(np.mean([-(0.5 * math.log(0.1) + 0.5 * math.log(0.6)), -(0.5 * math.log(0.1) + 0.5 * math.log(0.1))]), rel=1e-6)
