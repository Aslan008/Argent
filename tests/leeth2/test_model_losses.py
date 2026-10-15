import math
import random

import numpy as np
import pytest
import torch

from src.leeth2 import metrics
from src.leeth2.encoding import Leeth2Encoder, collate
from src.leeth2.losses import choice_loss, compute_losses, noul_loss, score_loss, span_loss
from src.leeth2.schema import Sample

from .conftest import VOCAB


def samples():
    return [
        Sample("t", "f", "en", "choice", "the cat sat on the mat", "what sat", ["cat", "dog", "bird"], 0),
        Sample("t", "f", "en", "choice", "a dog barked", "what barked", ["cat", "dog"], 1),
        Sample("t", "f", "en", "noul", "it is raining", "is it wet outside", None, 1.0),
        Sample("t", "f", "en", "score", "great film", "how good", None, 4.0, levels=5),
        Sample("t", "f", "en", "extract", "alpha beta gamma", "which greek letter is second", None, [6, 10]),
        Sample("t", "f", "en", "extract", "alpha beta gamma", "who won the match", None, None),
    ]


def batch_of(tok, items, rng=None):
    e = Leeth2Encoder(tok, VOCAB, max_len=128)
    return collate([e.encode(s, rng=rng) for s in items])


def test_forward_shapes_and_masking(tok, tiny_model):
    b = batch_of(tok, samples())
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(**b)
    B, L = b["input_ids"].shape
    assert out["choice_logits"].shape == (B, b["k_max"])
    assert out["noul_logit"].shape == (B,)
    assert out["score_logits"].shape == (B, 10)
    assert out["start_logits"].shape == (B, L)
    # option slots beyond n_opts are masked out of the softmax
    assert out["choice_logits"][1, 2].item() <= -1e8
    p = torch.softmax(out["choice_logits"][1], -1)
    assert p[2].item() == 0.0 and math.isclose(p.sum().item(), 1.0, rel_tol=1e-5)
    # extract logits only exist on state tokens and the read-out token
    assert (out["start_logits"][~b["state_mask"]] <= -1e8).all()


def test_only_the_intended_parameters_train(tiny_model):
    counts = tiny_model.count_parameters()
    assert counts["trainable"] < counts["total"]
    assert not tiny_model.backbone.embeddings.tok_embeddings.weight.requires_grad
    assert tiny_model.marker_embed.weight.requires_grad


def test_one_step_reduces_loss_on_a_tiny_problem(tok, tiny_model):
    torch.manual_seed(0)
    items = samples()
    b = batch_of(tok, items)
    opt = torch.optim.AdamW([p for p in tiny_model.parameters() if p.requires_grad], lr=2e-3)
    tiny_model.train()
    first = None
    for step in range(60):
        out = tiny_model(**b)
        loss, stats = compute_losses(out, b)
        if first is None:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.35, f"memorising 6 examples should be easy: {first} -> {float(loss)}"
    assert set(stats) >= {"loss_choice", "loss_noul", "loss_score", "loss_extract"}
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(**b)
    assert out["choice_logits"][0].argmax().item() == b["choice_target"][0].item()
    assert out["start_logits"][4].argmax().item() == b["start_target"][4].item()
    assert out["start_logits"][5].argmax().item() == b["rd_pos"][5].item(), "no-answer points at [RD]"


def test_every_head_gets_a_gradient_even_when_the_batch_has_one_type(tok, tiny_model):
    """DDP needs this: a head that is absent from a batch must still be in the autograd graph."""
    b = batch_of(tok, [samples()[2]])  # noul only
    out = tiny_model(**b)
    loss, _ = compute_losses(out, b)
    loss.backward()
    for name in ("choice_head", "score_head", "span_head"):
        grads = [p.grad for p in getattr(tiny_model, name).parameters()]
        assert all(g is not None for g in grads), name


def test_option_order_does_not_change_the_prediction(tok, tiny_model):
    """The head looks at each option through its own tokens, so shuffling options only permutes the logits."""
    tiny_model.eval()
    s = Sample("t", "f", "en", "choice", "ctx words here", "which", ["red apple", "blue sky", "green grass", "dark night"], 2)
    e = Leeth2Encoder(tok, VOCAB, max_len=128)
    probs = {}
    for seed in (1, 2, 3):
        x = e.encode(s, rng=random.Random(seed))
        with torch.no_grad():
            out = tiny_model(**collate([x]))
        lg = out["choice_logits"][0, : x.n_opts]
        for shown, orig in enumerate(x.perm):
            probs.setdefault(orig, []).append(lg[shown].item())
    # same option -> (almost) the same logit wherever it was shown; positions differ so allow slack
    spread = max(max(v) - min(v) for v in probs.values())
    assert spread < 1.0


def test_save_and_load_roundtrip(tok, tiny_model, tmp_path):
    from src.leeth2.model import Leeth2Model

    tiny_model.eval()
    b = batch_of(tok, samples())
    with torch.no_grad():
        want = tiny_model(**b)
    tiny_model.save_pretrained(tmp_path)
    loaded = Leeth2Model.from_pretrained(tmp_path)
    loaded.eval()
    with torch.no_grad():
        got = loaded(**b)
    for k in want:
        assert torch.allclose(want[k], got[k], atol=1e-5), k


# -- losses ---------------------------------------------------------------------------


def test_choice_loss_ignores_masked_rows():
    logits = torch.tensor([[2.0, 0.0, -1e9], [0.0, 0.0, 0.0]])
    t = torch.tensor([0, -100])
    l = choice_loss(logits, t)
    assert l[1].item() == 0.0 and l[0].item() == pytest.approx(math.log(1 + math.exp(-2.0)), rel=1e-4)


def test_noul_loss_is_a_proper_scoring_rule():
    t = torch.tensor([0.7])
    losses = [noul_loss(torch.tensor([math.log(p / (1 - p))]), t).item() for p in (0.3, 0.5, 0.7, 0.9)]
    assert min(losses) == losses[2], "minimised at the true probability, not at the extreme"


def test_score_two_hot_matches_fractional_label():
    n = 10
    levels = torch.tensor([5])
    pos = torch.tensor([2.5])
    # logits that put 50/50 on levels 3 and 4 give (almost) zero loss minus entropy of the target
    logits = torch.full((1, n), -30.0)
    logits[0, 2] = 0.0
    logits[0, 3] = 0.0
    l = score_loss(logits, pos, levels).item()
    assert l == pytest.approx(math.log(2), rel=1e-3)
    # levels beyond the asked scale never receive probability
    assert score_loss(torch.zeros(1, n), torch.tensor([4.0]), levels).item() == pytest.approx(math.log(5), rel=1e-4)


def test_span_loss_uses_both_ends():
    s = torch.tensor([[5.0, 0.0, 0.0]])
    e = torch.tensor([[0.0, 0.0, 5.0]])
    good = span_loss(s, e, torch.tensor([0]), torch.tensor([2])).item()
    bad = span_loss(s, e, torch.tensor([2]), torch.tensor([0])).item()
    assert good < 0.1 < bad


# -- metrics --------------------------------------------------------------------------


def test_ece_perfect_and_overconfident():
    assert metrics.ece([0.5] * 4, [1, 0, 1, 0]) == pytest.approx(0.0)
    assert metrics.ece([0.99] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.49, abs=1e-6)


def test_choice_and_noul_reports():
    probs = [np.array([0.7, 0.2, 0.1]), np.array([0.1, 0.8, 0.1])]
    r = metrics.choice_report(probs, [0, 2])
    assert r["acc"] == 0.5 and r["n"] == 2 and r["chance"] == pytest.approx(1 / 3)
    n = metrics.noul_report([0.9, 0.2], [1.0, 0.0])
    assert n["acc"] == 1.0 and n["brier"] == pytest.approx((0.01 + 0.04) / 2)


def test_span_f1():
    assert metrics.span_em_f1("The Eiffel Tower", "eiffel tower") == {"em": 1.0, "f1": 1.0}
    assert metrics.span_em_f1("big red car", "red car")["f1"] == pytest.approx(0.8)
    assert metrics.span_em_f1("", "")["em"] == 1.0 and metrics.span_em_f1("x", "")["f1"] == 0.0


def test_emb16_compression_changes_no_output_bit(tiny_cfg):
    import copy

    import torch

    from src.leeth2.model import Leeth2Config, Leeth2Model, compress_for_inference

    torch.manual_seed(0)
    cfg = Leeth2Config(base_model="tiny", backbone={}, vocab_size=tiny_cfg["vocab_size"], hidden_size=tiny_cfg["hidden_size"])
    from transformers import ModernBertConfig, ModernBertModel

    model = Leeth2Model(cfg, ModernBertModel(ModernBertConfig(**tiny_cfg))).eval()
    # the table as it sits on disk: half precision, widened to fp32 when loaded
    model.backbone.embeddings.tok_embeddings.weight.data = model.backbone.embeddings.tok_embeddings.weight.data.half().float()
    ids = torch.randint(10, tiny_cfg["vocab_size"], (2, 12))
    mask = torch.ones_like(ids)
    squeezed = compress_for_inference(copy.deepcopy(model), "emb16").eval()
    assert squeezed.backbone.embeddings.tok_embeddings.weight.dtype == torch.float16
    with torch.no_grad():
        assert torch.equal(model.encode_states(ids, mask), squeezed.encode_states(ids, mask))
    with pytest.raises(ValueError):
        compress_for_inference(model, "int8")
