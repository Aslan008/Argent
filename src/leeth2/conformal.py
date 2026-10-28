"""Decision thresholds with a guarantee instead of a hand-picked number (Leeth 3 plan, section 4.3).

Two questions a caller asks, both answered from a calibration set that was NOT used to train or to choose the model:

``risk_threshold``       "Act only when the score is at least tau, and act wrongly on at most alpha of the cases that should not
                          be acted on" (Argent: fast-dispatch only when P(tool) >= tau; false dispatches <= alpha).
``selective_threshold``  "Answer only when confident enough that the error rate among the answered cases is <= alpha"
                          (any primitive: accept the decision or hand the case to a bigger model / a person).

Both use exact binomial (Clopper-Pearson) upper bounds, so the statement holds with probability >= 1 - delta over the draw of the
calibration set, for new cases exchangeable with it. When the calibration set is too small to promise alpha, the answer is
``math.inf`` (never act / always defer) - an honest "cannot guarantee", not a guess.

The guarantee is only as good as the calibration set: reusing data that was looked at while fixing the model makes it optimistic.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [_log_comb(n, i) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Smallest p with P(X <= k | p) <= delta: an upper bound on the true rate after k events in n trials, holding w.p. >= 1 - delta."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def risk_threshold(negative_scores: Sequence[float], alpha: float, delta: float = 0.05) -> float:
    """The lowest threshold tau such that, with probability >= 1 - delta, a new case that should NOT be acted on scores >= tau
    with probability <= alpha. ``negative_scores`` are the model's scores on calibration cases that should not be acted on."""
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0, 1)")
    s = sorted(float(x) for x in negative_scores)
    n = len(s)
    # tau = just above the j-th largest negative score lets exactly j negatives through; take the largest j the bound allows
    best = math.inf
    for j in range(n + 1):
        if clopper_pearson_upper(j, n, delta) > alpha:
            break
        best = s[n - j - 1] if j < n else -math.inf      # cases scoring ABOVE this value are acted on
    if best == math.inf:
        return math.inf
    return math.nextafter(best, math.inf) if best != -math.inf else -math.inf


def selective_threshold(confidences: Sequence[float], correct: Sequence[bool], alpha: float, delta: float = 0.05,
                        min_accepted: int = 1) -> Tuple[float, float, float]:
    """The lowest confidence threshold tau such that, w.p. >= 1 - delta, the error rate among cases with confidence >= tau is <= alpha.

    Fixed-sequence testing (Learn-then-Test): candidate thresholds are visited from the most confident down, each one tested once,
    stopping at the first whose bound exceeds alpha - testing in one fixed order needs no multiple-testing correction. The order
    must not depend on the labels, and the sequence must not start where no outcome could pass: with fewer than ``n_min`` accepted
    cases even zero errors cannot certify alpha, so the first candidate is the n_min-th most confident case (n_min depends only on
    alpha and delta). Returns (tau, coverage on the calibration set, the error bound at tau); tau = inf when nothing can be certified."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0, 1)")
    pairs: List[Tuple[float, bool]] = sorted(((float(c), bool(ok)) for c, ok in zip(confidences, correct)), key=lambda x: -x[0])
    n_all = len(pairs)
    n_min = max(1, min_accepted, math.ceil(math.log(delta) / math.log1p(-alpha)))   # zero errors in n_min cases: bound <= alpha
    # candidates on a grid of accepted counts that starts well past n_min: one unlucky error among the first few dozen cases would
    # otherwise stop the whole sequence. The grid depends on n, alpha and delta only (never on the labels), as the method requires
    step = max(1, n_all // 100)
    nxt = max(2 * n_min, n_all // 20)
    tau, cov, bound = math.inf, 0.0, 1.0
    errors = 0
    i = 0
    while i < n_all:
        c = pairs[i][0]
        j = i
        while j < n_all and pairs[j][0] == c:           # ties enter together: a threshold cannot split equal confidences
            errors += not pairs[j][1]
            j += 1
        i = j
        if j < nxt and j < n_all:                       # not a candidate (before the grid start, or between grid points)
            continue
        nxt = j + step
        ub = clopper_pearson_upper(errors, j, delta)
        if ub > alpha:
            break
        tau, cov, bound = c, j / n_all, ub
    return tau, cov, bound
