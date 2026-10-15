"""Metrics. Calibration matters as much as accuracy for a System-One model: downstream code gates on probabilities."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def ece(conf: Sequence[float], correct: Sequence[float], bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    c = np.asarray(conf, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if c.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(total)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _as_dist(t, k: int) -> np.ndarray:
    if isinstance(t, (list, tuple, np.ndarray)):
        return np.asarray(t, dtype=np.float64)
    d = np.zeros(k)
    d[int(t)] = 1.0
    return d


def choice_report(probs: List[np.ndarray], targets: Sequence) -> Dict[str, float]:
    """probs[i]: distribution over the options of example i. A target is an index or a probability list; with a list,
    a prediction counts as correct when it lands on ANY option the label gives probability to."""
    n = len(targets)
    if n == 0:
        return {"n": 0}
    dists = [_as_dist(t, len(p)) for p, t in zip(probs, targets)]
    top = np.array([p.max() for p in probs])
    hit = np.array([float(d[int(p.argmax())] > 0) for p, d in zip(probs, dists)], dtype=np.float64)
    nll = float(np.mean([-float(np.sum(d * np.log(np.clip(p, 1e-12, 1)))) for p, d in zip(probs, dists)]))
    brier = float(np.mean([float(np.sum((p - d) ** 2)) for p, d in zip(probs, dists)]))
    chance = float(np.mean([1.0 / len(p) for p in probs]))
    return {"n": n, "acc": float(hit.mean()), "chance": chance, "nll": nll, "brier": brier, "ece": ece(top, hit), "conf": float(top.mean())}


def noul_report(p_true: Sequence[float], targets: Sequence[float]) -> Dict[str, float]:
    p = np.clip(np.asarray(p_true, dtype=np.float64), 1e-9, 1 - 1e-9)
    t = np.asarray(targets, dtype=np.float64)
    if p.size == 0:
        return {"n": 0}
    hard = t >= 0.5
    pred = p >= 0.5
    acc = float((pred == hard).mean())
    nll = float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())
    brier = float(((p - t) ** 2).mean())
    conf = np.where(pred, p, 1 - p)
    return {"n": int(p.size), "acc": acc, "nll": nll, "brier": brier, "ece": ece(conf, (pred == hard).astype(float)), "pos_rate": float(hard.mean())}


def score_report(pred_level: Sequence[float], target_level: Sequence[float], levels: Sequence[int]) -> Dict[str, float]:
    """MAE is reported on a 0..100 scale so scales with different numbers of levels are comparable."""
    p = np.asarray(pred_level, dtype=np.float64)
    t = np.asarray(target_level, dtype=np.float64)
    lv = np.asarray(levels, dtype=np.float64)
    if p.size == 0:
        return {"n": 0}
    mae100 = float((np.abs(p - t) / np.maximum(lv - 1, 1) * 100).mean())
    exact = float((np.rint(p) == np.rint(t)).mean())
    within1 = float((np.abs(np.rint(p) - np.rint(t)) <= 1).mean())
    corr = float(np.corrcoef(p, t)[0, 1]) if p.size > 2 and p.std() > 0 and t.std() > 0 else float("nan")
    return {"n": int(p.size), "mae100": mae100, "exact": exact, "within1": within1, "pearson": corr}


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def span_em_f1(pred: str, gold: str) -> Dict[str, float]:
    pn, gn = _norm(pred), _norm(gold)
    if not pn and not gn:
        return {"em": 1.0, "f1": 1.0}
    if not pn or not gn:
        return {"em": 0.0, "f1": 0.0}
    em = float(pn == gn)
    pt, gt = pn.split(), gn.split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0:
        return {"em": em, "f1": 0.0}
    prec, rec = ns / len(pt), ns / len(gt)
    return {"em": em, "f1": 2 * prec * rec / (prec + rec)}


def temperature_nll(logits: np.ndarray, targets: np.ndarray, t: float) -> float:
    p = softmax(logits / t, axis=-1)
    return float(-np.log(np.clip(p[np.arange(len(targets)), targets], 1e-12, 1)).mean())
