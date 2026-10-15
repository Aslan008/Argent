"""src/leeth/bulk.py -- the same question about many items, answered in batches: filter and categorise lists.

A main model that has to look at 300 bookmarks or 80 posts spends its context and its money on a decision a 300M model makes in
milliseconds.  The split that works:

  * the main model states the criterion ONCE, in words ("posts that complain about delivery", "bookmarks of recipes");
  * Leeth answers it for every item in one batched pass and says how sure it is (calibrated probabilities);
  * the sure answers are acted on; the unsure ones -- a band around 0.5 -- come back to the main model (or to the user).

``accept`` / ``reject`` set that band.  The defaults (0.7 / 0.3) are deliberately wide: a wrong "yes" on a delete is worse than one
more item for the main model to read.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

MAX_ITEM_CHARS = 1200


def _clip(s: Any, n: int = MAX_ITEM_CHARS) -> str:
    t = " ".join(str(s if s is not None else "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def cut_from_examples(pos: Sequence[float], neg: Sequence[float], margin: float = 1.5) -> Dict[str, float]:
    """Decision cuts, in logit units, learnt from the scores of a few labelled examples: {'yes': z, 'no': z} -- an item scoring
    at or above 'yes' is a yes, at or below 'no' a no, between them unsure.

    Why: the yes/no head ranks content questions very well (AUC 0.97-1.00 on posts / bookmarks) but its absolute probabilities are
    conservative: a blatant match sits at 0.6, not 0.95.  Two or three examples of each kind move the cut to where THIS question's
    scores really separate.  The rule is precision-first: a yes needs to score like the weakest positive example."""
    zp = [_logit(p) for p in pos]
    zn = [_logit(p) for p in neg]
    if zp and zn:
        hi, lo = min(zp), max(zn)
        if hi > lo:
            return {"yes": hi, "no": lo}
        # overlapping examples: the question does not separate them cleanly -- a wide unsure band around the midpoint
        mp, mn = sum(zp) / len(zp), sum(zn) / len(zn)
        mid, half = (mp + mn) / 2.0, abs(mp - mn) / 4.0
        return {"yes": mid + half, "no": mid - half}
    if zp:
        hi = min(zp)
        return {"yes": hi, "no": hi - margin}
    if zn:
        lo = max(zn)
        return {"yes": lo + margin, "no": lo}
    raise ValueError("no examples")


def classify_yesno(rt: Any, items: Sequence[Any], criterion: str, accept: float = 0.7, reject: float = 0.3,
                   positives: Optional[Sequence[Any]] = None, negatives: Optional[Sequence[Any]] = None,
                   meta: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Per item: {'i', 'decision': 'yes' | 'no' | 'unsure', 'p': P(criterion holds)}.

    With ``positives`` / ``negatives`` (a few example texts of each kind) the cut comes from their scores (``cut_from_examples``)
    instead of the fixed ``accept`` / ``reject``; ``meta`` (a dict) receives what was used."""
    texts = [_clip(x) for x in items]
    pos = [_clip(x) for x in (positives or []) if str(x).strip()]
    neg = [_clip(x) for x in (negatives or []) if str(x).strip()]
    res = rt.judge_many(pos + neg + texts, criterion.strip())
    probs = [float(r.probability) for r in res]
    pp, pn, pi = probs[: len(pos)], probs[len(pos): len(pos) + len(neg)], probs[len(pos) + len(neg):]
    if pos or neg:
        cut = cut_from_examples(pp, pn)
        lo_z, hi_z = cut["no"], cut["yes"]
        if meta is not None:
            meta.update({"calibrated_by": "examples", "examples": {"positive": len(pos), "negative": len(neg)},
                         "yes_from_p": round(_sigmoid(hi_z), 3), "no_up_to_p": round(_sigmoid(lo_z), 3)})
        decide = lambda p: "yes" if _logit(p) >= hi_z else "no" if _logit(p) <= lo_z else "unsure"  # noqa: E731
    else:
        decide = lambda p: "yes" if p >= accept else "no" if p <= reject else "unsure"  # noqa: E731
    return [{"i": i, "decision": decide(p), "p": round(p, 3)} for i, p in enumerate(pi)]


def classify_categories(rt: Any, items: Sequence[Any], categories: Sequence[str], question: Optional[str] = None, min_conf: float = 0.5) -> List[Dict[str, Any]]:
    """Per item: {'i', 'category' (None when unsure), 'best', 'p', 'runner_up'}."""
    cats = [str(c).strip() for c in categories if str(c).strip()]
    if len(cats) < 2:
        raise ValueError("categories: at least two are needed")
    if len(set(cats)) != len(cats):
        raise ValueError("categories must be distinct")
    texts = [_clip(x) for x in items]
    res = rt.choose_many(texts, (question or "К какой категории относится этот элемент?").strip(), cats)
    out = []
    for i, r in enumerate(res):
        ranked = sorted(r.probabilities.items(), key=lambda kv: -kv[1])
        p = float(ranked[0][1])
        out.append({"i": i, "category": ranked[0][0] if p >= min_conf else None, "best": ranked[0][0], "p": round(p, 3),
                    "runner_up": ranked[1][0] if len(ranked) > 1 else None})
    return out


def summarize(results: List[Dict[str, Any]], items: Sequence[Any], mode: str, detail_limit: int = 60, preview: int = 90) -> Dict[str, Any]:
    """A compact report: index lists for acting, details for the items the caller has to look at."""
    def view(r):
        d = dict(r)
        d["text"] = _clip(items[r["i"]], preview)
        return d

    if mode == "yesno":
        by = {k: [r for r in results if r["decision"] == k] for k in ("yes", "no", "unsure")}
        rep: Dict[str, Any] = {"mode": "yesno", "count": len(results), "counts": {k: len(v) for k, v in by.items()},
                               "yes": [r["i"] for r in by["yes"]], "no": [r["i"] for r in by["no"]],
                               "unsure": [view(r) for r in sorted(by["unsure"], key=lambda r: -r["p"])]}
        if len(results) <= detail_limit:
            rep["details"] = [view(r) for r in results]
        return rep
    cats: Dict[str, List[int]] = {}
    unsure = []
    for r in results:
        if r["category"] is None:
            unsure.append(view(r))
        else:
            cats.setdefault(r["category"], []).append(r["i"])
    rep = {"mode": "categories", "count": len(results), "counts": {k: len(v) for k, v in cats.items()}, "categories": cats, "unsure": unsure}
    if len(results) <= detail_limit:
        rep["details"] = [view(r) for r in results]
    return rep
