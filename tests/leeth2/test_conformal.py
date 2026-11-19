"""Thresholds with a guarantee: the promise must hold on fresh data, and a small calibration set must refuse to promise."""

import math
import random

import pytest

from src.leeth2.conformal import binom_cdf, clopper_pearson_upper, risk_threshold, selective_threshold


def test_binomial_cdf_matches_direct_sum():
    n, p = 12, 0.3
    direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(5))
    assert binom_cdf(4, n, p) == pytest.approx(direct, rel=1e-9)


def test_clopper_pearson_known_values():
    # zero events in n trials: the bound is 1 - delta**(1/n) ("rule of three" for delta = 0.05: ~3/n)
    assert clopper_pearson_upper(0, 1000, 0.05) == pytest.approx(1 - 0.05 ** (1 / 1000), rel=1e-4)
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0
    assert clopper_pearson_upper(10, 100, 0.05) > 0.1


def test_a_small_calibration_set_cannot_promise_a_tiny_risk():
    assert risk_threshold([0.1, 0.2, 0.3], alpha=0.003) == math.inf
    tau, cov, _ = selective_threshold([0.9, 0.8], [True, True], alpha=0.01)
    assert tau == math.inf and cov == 0.0


def test_risk_threshold_holds_on_fresh_negatives():
    rng = random.Random(7)
    alpha, delta, violations, trials = 0.02, 0.05, 0, 60
    for _ in range(trials):
        calib = [rng.betavariate(2, 5) for _ in range(2000)]
        tau = risk_threshold(calib, alpha, delta)
        fresh = [rng.betavariate(2, 5) for _ in range(20000)]
        rate = sum(x >= tau for x in fresh) / len(fresh)
        violations += rate > alpha
    assert violations / trials <= delta + 0.05          # the promise fails at most ~delta of the time


def test_risk_threshold_is_not_needlessly_strict():
    rng = random.Random(3)
    calib = [rng.random() for _ in range(5000)]
    tau = risk_threshold(calib, 0.05, 0.05)
    assert 0.93 < tau < 0.96                             # uniform scores: ~0.95 plus a small finite-sample margin


def test_selective_threshold_controls_the_error_among_accepted():
    rng = random.Random(11)
    def draw(n):
        conf = [rng.random() for _ in range(n)]
        return conf, [rng.random() < 0.5 + 0.5 * c for c in conf]        # more confident -> more often right
    alpha, delta, violations = 0.08, 0.05, 0
    for _ in range(40):
        c, ok = draw(3000)
        tau, cov, bound = selective_threshold(c, ok, alpha, delta)
        assert bound <= alpha and 0 < cov < 1
        fc, fok = draw(30000)
        acc = [o for x, o in zip(fc, fok) if x >= tau]
        violations += (1 - sum(acc) / len(acc)) > alpha
    assert violations <= 4


def test_ties_enter_together():
    tau, cov, _ = selective_threshold([0.9] * 200 + [0.5] * 200, [True] * 200 + [False] * 200, alpha=0.1)
    assert tau == 0.9 and cov == 0.5
