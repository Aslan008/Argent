"""Post-hoc temperature scaling.

Fitted on a held-out calibration split that the model never trained on: one scalar per primitive,
chosen to minimise negative log-likelihood. It is monotone, so it can never change an argmax —
only how honest the probabilities are.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from . import metrics
from .evaluate import Prediction
from .schema import Sample


def _fit_t(nll_of_logt) -> float:
    r = minimize_scalar(nll_of_logt, bounds=(np.log(0.2), np.log(8.0)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def fit_temperatures(samples: Sequence[Sample], preds: Sequence[Prediction], big_k: int = 64) -> Dict[str, Any]:
    """Return {'temp_choice','temp_noul','temp_score', 'n_*'} fitted on RAW (uncalibrated) predictions."""
    out: Dict[str, Any] = {}

    ch = [(p.raw["logits_shown"], p.raw["perm"].index(s.hard_choice())) for s, p in zip(samples, preds) if s.ptype == "choice"]
    big_k = int(big_k)
    big = [(lg, y) for lg, y in ch if len(lg) > big_k]
    ch = [(lg, y) for lg, y in ch if len(lg) <= big_k] if len(big) >= 50 else ch
    if len(big) >= 50:

        def nll_big(logt: float) -> float:
            t = float(np.exp(logt))
            return float(np.mean([-np.log(max(metrics.softmax(lg / t)[y], 1e-12)) for lg, y in big]))

        out["temp_choice_big"] = _fit_t(nll_big)
    out["n_choice_big"] = len(big)
    if len(ch) >= 50:

        def nll(logt: float) -> float:
            t = float(np.exp(logt))
            return float(np.mean([-np.log(max(metrics.softmax(lg / t)[y], 1e-12)) for lg, y in ch]))

        out["temp_choice"] = _fit_t(nll)
    out["n_choice"] = len(ch)

    nl = [(p.raw["logit"], float(s.target)) for s, p in zip(samples, preds) if s.ptype == "noul"]
    if len(nl) >= 50:
        z = np.array([a for a, _ in nl])
        y = np.array([b for _, b in nl])

        def nll_n(logt: float) -> float:
            t = float(np.exp(logt))
            p = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

        out["temp_noul"] = _fit_t(nll_n)
    out["n_noul"] = len(nl)

    sc = [(p.raw["logits"], float(s.target) - 1.0) for s, p in zip(samples, preds) if s.ptype == "score"]
    if len(sc) >= 50:

        def nll_s(logt: float) -> float:
            t = float(np.exp(logt))
            tot = 0.0
            for lg, pos in sc:
                logp = np.log(np.clip(metrics.softmax(lg / t), 1e-12, 1))
                lo = int(np.floor(pos))
                hi = min(lo + 1, len(lg) - 1)
                w = pos - lo if hi != lo else 0.0
                tot += -((1 - w) * logp[lo] + w * logp[hi])
            return tot / len(sc)

        out["temp_score"] = _fit_t(nll_s)
    out["n_score"] = len(sc)
    return out


def apply_temperatures(model, temps: Dict[str, Any]) -> None:
    for k in ("temp_choice", "temp_choice_big", "temp_noul", "temp_score"):
        if k in temps:
            setattr(model.cfg, k, float(temps[k]))
