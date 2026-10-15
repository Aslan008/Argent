import numpy as np
import pytest
import torch

from src.leeth2 import metrics
from src.leeth2.calibrate import apply_temperatures, fit_temperatures
from src.leeth2.encoding import Leeth2Encoder
from src.leeth2.evaluate import Prediction, headline, predict, report
from src.leeth2.schema import Sample

from .conftest import VOCAB


def mixed():
    out = []
    for i in range(6):
        out.append(Sample("c", "fam", "en", "choice", f"ctx {i} words", "which", ["a", "b", "c"], i % 3))
        out.append(Sample("n", "fam", "en", "noul", f"state {i}", "is it", None, float(i % 2)))
        out.append(Sample("s", "fam", "en", "score", f"text {i}", "how good", None, float(1 + i % 5), levels=5))
        out.append(Sample("e", "fam", "en", "extract", "alpha beta gamma delta", "which", None, [6, 10] if i % 2 == 0 else None))
    return out


def test_predict_returns_one_prediction_per_sample_in_order(tok, tiny_model):
    enc = Leeth2Encoder(tok, VOCAB, max_len=128)
    ss = mixed()
    preds = predict(tiny_model, enc, ss, tokens_per_batch=400, max_batch=5)
    assert len(preds) == len(ss)
    for s, p in zip(ss, preds):
        assert p.ptype == s.ptype
        if s.ptype == "choice":
            assert p.probs.shape == (3,) and p.probs.sum() == pytest.approx(1.0)
        elif s.ptype == "noul":
            assert 0.0 <= p.p_true <= 1.0
        elif s.ptype == "score":
            assert 1.0 <= p.level <= 5.0 and p.level_probs.shape == (5,)
        else:
            assert isinstance(p.text, str)


def test_batching_does_not_change_results(tok, tiny_model):
    enc = Leeth2Encoder(tok, VOCAB, max_len=128)
    ss = mixed()
    a = predict(tiny_model, enc, ss, tokens_per_batch=100000, max_batch=1000)
    b = predict(tiny_model, enc, ss, tokens_per_batch=60, max_batch=2)
    for x, y in zip(a, b):
        if x.ptype == "choice":
            assert np.allclose(x.probs, y.probs, atol=1e-4)
        elif x.ptype == "noul":
            assert x.p_true == pytest.approx(y.p_true, abs=1e-4)
        elif x.ptype == "score":
            assert x.level == pytest.approx(y.level, abs=1e-4)
        else:
            assert x.text == y.text


def test_choice_probabilities_follow_original_option_order(tok, tiny_model):
    """predict() shuffles nothing, but perm bookkeeping must still map shown->original correctly."""
    enc = Leeth2Encoder(tok, VOCAB, max_len=128)
    s = Sample("c", "f", "en", "choice", "ctx", "q", ["x1", "x2", "x3"], 0)
    p = predict(tiny_model, enc, [s])[0]
    assert list(p.raw["perm"]) == [0, 1, 2]
    assert p.probs.argmax() == p.raw["logits_shown"].argmax()


def test_report_groups_by_task_family_and_primitive(tok, tiny_model):
    enc = Leeth2Encoder(tok, VOCAB, max_len=128)
    ss = mixed()
    rep = report(ss, predict(tiny_model, enc, ss))
    assert {"task:c", "task:n", "task:s", "task:e", "family:fam", "ptype:choice", "ptype:extract"} <= set(rep)
    assert rep["task:c"]["choice"]["n"] == 6
    assert rep["task:e"]["extract"]["no_ans"] == 3 and rep["task:e"]["extract"]["has_ans"] == 3
    h = headline(rep)
    assert {"choice_acc", "noul_ece", "score_mae100", "extract_f1"} <= set(h)


def _fake_preds(n, temp_true, rng):
    """Logits that are too confident by a known factor -> the fitted temperature must recover it."""
    ss, ps = [], []
    for _ in range(n):
        k = 4
        true_logits = rng.normal(0, 1.5, size=k)
        probs = metrics.softmax(true_logits)
        y = int(rng.choice(k, p=probs))
        shown = true_logits * temp_true  # overconfident by temp_true
        ss.append(Sample("c", "f", "en", "choice", "s", "q", ["a", "b", "c", "d"], y))
        ps.append(Prediction("choice", raw={"logits_shown": shown, "perm": [0, 1, 2, 3]}))
    return ss, ps


def test_temperature_scaling_recovers_known_overconfidence():
    rng = np.random.default_rng(0)
    ss, ps = _fake_preds(4000, 2.5, rng)
    temps = fit_temperatures(ss, ps)
    assert temps["temp_choice"] == pytest.approx(2.5, rel=0.12)


def test_temperature_scaling_noul_and_score():
    rng = np.random.default_rng(1)
    ss, ps = [], []
    for _ in range(3000):
        z = rng.normal(0, 2)
        y = float(rng.random() < 1 / (1 + np.exp(-z)))
        ss.append(Sample("n", "f", "en", "noul", "s", "q", None, y))
        ps.append(Prediction("noul", raw={"logit": 3.0 * z}))
    temps = fit_temperatures(ss, ps)
    assert temps["temp_noul"] == pytest.approx(3.0, rel=0.15)


def test_apply_temperatures_changes_probabilities_not_the_argmax(tok, tiny_model):
    enc = Leeth2Encoder(tok, VOCAB, max_len=128)
    s = Sample("c", "f", "en", "choice", "ctx", "q", ["x1", "x2", "x3"], 0)
    before = predict(tiny_model, enc, [s])[0]
    apply_temperatures(tiny_model, {"temp_choice": 3.0})
    after = predict(tiny_model, enc, [s])[0]
    assert after.probs.argmax() == before.probs.argmax()
    assert after.probs.max() < before.probs.max() or before.probs.max() == pytest.approx(1 / 3, abs=0.05)
