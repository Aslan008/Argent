# -*- coding: utf-8 -*-
"""Тесты Фазы D1: контрастные лоссы (contrastive.py).

Чисто тензорные проверки без загрузки весов — быстрый прогон.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leeth.contrastive import (
    choice_contrastive_loss,
    combined_loss,
    info_nce_loss,
    margin_loss,
    noul_contrastive_loss,
)


def test_margin_loss_zero_when_separated():
    pos = torch.tensor([0.9, 0.8])
    neg = torch.tensor([0.1, 0.2])
    loss = margin_loss(pos, neg, margin=0.5)
    assert loss.item() == 0.0  # маржа уже выполнена


def test_margin_loss_positive_when_violated():
    pos = torch.tensor([0.3])
    neg = torch.tensor([0.8])
    loss = margin_loss(pos, neg, margin=0.5)
    assert loss.item() > 0.0  # negative выше positive — штраф


def test_margin_loss_gradient_pushes_apart():
    pos = torch.tensor([0.3], requires_grad=True)
    neg = torch.tensor([0.8], requires_grad=True)
    loss = margin_loss(pos, neg, margin=0.5)
    loss.backward()
    assert pos.grad.item() < 0  # положительный тянется вверх
    assert neg.grad.item() > 0  # отрицательный тянется вниз


def test_margin_loss_shape_mismatch_raises():
    with torch.no_grad():
        try:
            margin_loss(torch.zeros(2), torch.zeros(3))
            assert False, "must raise"
        except ValueError:
            pass


def test_info_nce_prefers_positive():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    pos = torch.nn.functional.normalize(q + 0.1 * torch.randn(4, 16), dim=-1)
    negs = [torch.nn.functional.normalize(torch.randn(4, 16), dim=-1) for _ in range(3)]
    loss = info_nce_loss(q, pos, negs)
    assert torch.isfinite(loss)
    # При близком positive и случайных negatives лосс должен быть мал.
    assert loss.item() < 1.0


def test_info_nce_high_when_positive_is_random():
    torch.manual_seed(1)
    q = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    pos = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)  # unrelated
    negs = [torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)]
    loss = info_nce_loss(q, pos, negs)
    assert loss.item() > 0.5  # unrelated positive -> большой штраф


def test_noul_contrastive():
    pos = torch.tensor([0.95])
    neg = torch.tensor([0.05])
    assert noul_contrastive_loss(pos, neg).item() == 0.0
    assert noul_contrastive_loss(neg, pos).item() > 0.0


def test_choice_contrastive():
    pos_logits = torch.tensor([[5.0, 0.1, 0.1]])
    neg_logits = torch.tensor([[4.0, 3.0, 0.1]])
    pos_t = torch.tensor([0])
    neg_t = torch.tensor([1])
    # Верный класс positive (логит 5) выше неверного у negative (логит 3) на 2 > margin 1.
    assert choice_contrastive_loss(pos_logits, pos_t, neg_logits, neg_t, margin=1.0).item() == 0.0
    # Маржа 3.0 — нарушена (2 < 3) -> штраф.
    assert choice_contrastive_loss(pos_logits, pos_t, neg_logits, neg_t, margin=3.0).item() > 0.0


def test_combined_loss_weighting():
    sup = torch.tensor(1.0)
    cont = torch.tensor(2.0)
    out = combined_loss(sup, cont, 0.4)
    assert torch.isclose(out, torch.tensor(1.8))