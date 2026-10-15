"""Per-primitive losses. All of them are proper scoring rules, so the model is rewarded for honest probabilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

IGNORE = -100
DEFAULT_LAMBDA = {"choice": 1.0, "noul": 1.0, "score": 0.7, "extract": 1.0, "choice_brier": 0.0}


def _wmean(per_example: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (per_example * w).sum() / w.sum().clamp(min=1e-6)


def choice_loss(logits: torch.Tensor, target: torch.Tensor, probs: Optional[torch.Tensor] = None, has_soft: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy over the options of each example. [B,K], [B] -> [B] (0 where target is ignored).
    Examples with a soft label (``probs`` over the options, e.g. several equally good moves) use it instead of the hard index."""
    valid = target != IGNORE
    safe = target.masked_fill(~valid, 0)
    logp = F.log_softmax(logits, dim=-1)
    ce = -logp.gather(1, safe.unsqueeze(1)).squeeze(1)
    if probs is not None and has_soft is not None and bool(has_soft.any()):
        soft = -(probs * logp.masked_fill(probs <= 0, 0.0)).sum(-1)
        ce = torch.where(has_soft, soft, ce)
    return ce * valid.to(ce.dtype)


def choice_brier(logits: torch.Tensor, target: torch.Tensor, probs: Optional[torch.Tensor] = None, has_soft: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Multi-class Brier score of the option distribution, sum_k (p_k - y_k)^2 in [0, 2]. [B,K], [B] -> [B] (0 where ignored).
    Cross-entropy punishes a confident miss without bound, which on a 255-option list is dominated by a few examples; the
    Brier term is bounded and rewards the whole distribution (a second strictly proper rule, the RLCD-style signal)."""
    valid = target != IGNORE
    p = F.softmax(logits, dim=-1)
    y = torch.zeros_like(p)
    y.scatter_(1, target.masked_fill(~valid, 0).unsqueeze(1), 1.0)
    if probs is not None and has_soft is not None and bool(has_soft.any()):
        y = torch.where(has_soft.unsqueeze(1), probs.to(p.dtype), y)
    return ((p - y) ** 2).sum(-1) * valid.to(p.dtype)


def noul_loss(logit: torch.Tensor, target: torch.Tensor, brier: float = 0.25) -> torch.Tensor:
    valid = target >= 0
    t = target.clamp(min=0.0, max=1.0)
    bce = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
    p = torch.sigmoid(logit)
    out = bce + brier * (p - t) ** 2
    return out * valid.to(out.dtype)


def score_loss(logits: torch.Tensor, pos: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Two-hot ordinal cross-entropy: the target position pos in [0, L-1] spreads over its two neighbouring
    levels, so the expected level of a perfect prediction equals the (possibly fractional) label."""
    valid = pos >= 0
    n = logits.size(-1)
    idx = torch.arange(n, device=logits.device).unsqueeze(0)
    lv = levels.clamp(min=2, max=n).unsqueeze(1)
    masked = logits.masked_fill(idx >= lv, -1e9)
    logp = F.log_softmax(masked, dim=-1)
    p = pos.clamp(min=0.0)
    lo = torch.floor(p).clamp(max=n - 1)
    hi = (lo + 1).clamp(max=(lv.squeeze(1) - 1).to(lo.dtype))
    w_hi = (p - lo).clamp(min=0.0, max=1.0)
    w_hi = torch.where(hi == lo, torch.zeros_like(w_hi), w_hi)
    q = torch.zeros_like(logp)
    q.scatter_add_(1, lo.long().unsqueeze(1), (1.0 - w_hi).unsqueeze(1))
    q.scatter_add_(1, hi.long().unsqueeze(1), w_hi.unsqueeze(1))
    ce = -(q * logp).sum(-1)
    return ce * valid.to(ce.dtype)


def span_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_t: torch.Tensor,
    end_t: torch.Tensor,
    start_set: Optional[torch.Tensor] = None,
    end_set: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Start / end cross-entropy. With ``start_set`` / ``end_set`` ([B, M], IGNORE-padded) an example may have several right
    answers (every file path in a log): the target is the uniform distribution over them, so each answer gets ~1/m of the
    probability mass and the decoder (evaluate._decode_spans) can read all of them off the peaks. m = 1 is the plain loss."""
    valid = start_t != IGNORE
    if start_set is None or end_set is None or start_set.size(1) <= 1 or not bool((start_set[:, 1:] != IGNORE).any()):
        s = F.cross_entropy(start_logits, start_t.masked_fill(~valid, 0), reduction="none")
        e = F.cross_entropy(end_logits, end_t.masked_fill(~valid, 0), reduction="none")
        return (s + e) * valid.to(s.dtype)

    def set_nll(logits: torch.Tensor, tset: torch.Tensor) -> torch.Tensor:
        ok = tset != IGNORE
        logp = F.log_softmax(logits, dim=-1)
        got = logp.gather(1, tset.clamp(min=0))
        return -(got * ok.to(got.dtype)).sum(1) / ok.sum(1).clamp(min=1).to(got.dtype)

    return (set_nll(start_logits, start_set) + set_nll(end_logits, end_set)) * valid.to(start_logits.dtype)


def compute_losses(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lam: Optional[Dict[str, float]] = None,
    brier: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    lam = lam or DEFAULT_LAMBDA
    w = batch["weight"]
    parts: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    ct = batch["choice_target"]
    m = ct != IGNORE
    if m.any():
        parts["choice"] = _wmean(choice_loss(out["choice_logits"], ct, batch.get("choice_probs"), batch.get("has_soft"))[m], w[m])
        if lam.get("choice_brier", 0.0) > 0:
            parts["choice_brier"] = _wmean(choice_brier(out["choice_logits"], ct, batch.get("choice_probs"), batch.get("has_soft"))[m], w[m])
        counts["choice"] = int(m.sum())
    nt = batch["noul_target"]
    m = nt >= 0
    if m.any():
        parts["noul"] = _wmean(noul_loss(out["noul_logit"], nt, brier)[m], w[m])
        counts["noul"] = int(m.sum())
    sp = batch["score_pos"]
    m = sp >= 0
    if m.any():
        parts["score"] = _wmean(score_loss(out["score_logits"], sp, batch["levels"])[m], w[m])
        counts["score"] = int(m.sum())
    st = batch["start_target"]
    m = st != IGNORE
    if m.any():
        parts["extract"] = _wmean(span_loss(out["start_logits"], out["end_logits"], st, batch["end_target"], batch.get("start_set"), batch.get("end_set"))[m], w[m]) / 2.0
        counts["extract"] = int(m.sum())

    total = sum(lam.get(k, 1.0) * v for k, v in parts.items()) if parts else out["noul_logit"].sum() * 0.0
    # keep every head in the autograd graph so DDP never sees "unused" parameters
    touch = out["choice_logits"].clamp(-1e4, 1e4).sum() + out["noul_logit"].sum() + out["score_logits"].sum()
    touch = touch + out["start_logits"].clamp(-1e4, 1e4).sum() + out["end_logits"].clamp(-1e4, 1e4).sum()
    total = total + 0.0 * touch
    stats = {f"loss_{k}": float(v.detach()) for k, v in parts.items()}
    stats.update({f"n_{k}": c for k, c in counts.items()})
    return total, stats
