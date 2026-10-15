"""Batched inference and per-task reports.

``predict`` is the one inference path shared by evaluation, calibration and the runtime, so a number
measured here is the number users get.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from . import metrics
from .encoding import Encoded, Leeth2Encoder, collate
from .schema import Sample

MAX_SPAN_TOKENS = 192


@dataclass
class Prediction:
    ptype: str
    # choice: probabilities in the ORIGINAL option order
    probs: Optional[np.ndarray] = None
    # noul
    p_true: Optional[float] = None
    # score
    level_probs: Optional[np.ndarray] = None
    level: Optional[float] = None  # expected level in [1, levels]
    # extract
    text: Optional[str] = None
    span: Optional[tuple] = None
    confidence: Optional[float] = None
    no_answer: bool = False
    spans: Optional[List[Dict[str, Any]]] = None  # multi-answer questions (meta["multi"]): every answer, in text order
    # raw logits kept for calibration
    raw: Dict[str, Any] = field(default_factory=dict)


def _best_ends(e: np.ndarray, width: int) -> np.ndarray:
    """For every position i the position of the largest e[j], i <= j < i + width (first one on ties)."""
    n = len(e)
    w = max(1, min(width, n))
    padded = np.concatenate([e, np.full(w - 1, -1e30, dtype=e.dtype)])
    win = np.lib.stride_tricks.sliding_window_view(padded, w)  # win[i, k] = e[i + k]
    return np.arange(n) + win.argmax(axis=1)


TRAILING_PUNCT = ".,;:!?"


def _clean_span(state: str, a: int, b: int, trim_punct: bool = False) -> tuple:
    """Text and character span of tokens [a, b): surrounding whitespace removed; with ``trim_punct`` also the sentence punctuation that the
    tokenizer glued to the last token ("flush_cache()." is one token run), which is never part of an answer in a list of entities."""
    text = state[a:b]
    stripped = text.lstrip()
    a += len(text) - len(stripped)
    stripped = stripped.rstrip()
    if trim_punct:
        stripped = stripped.rstrip(TRAILING_PUNCT)
    return stripped, (a, a + len(stripped))


def _decode_spans(start: np.ndarray, end: np.ndarray, enc: Encoded, state: str, null_bias: float = 0.0, rel: float = 0.25, max_spans: int = 32) -> List[Dict[str, Any]]:
    """Every answer of a multi-answer question. The model is trained (losses.span_loss) to spread the start / end probability
    uniformly over the right answers, so each answer is a peak of about 1/m of the mass: take the starts whose probability is
    within ``rel`` of the highest one, and end each at the best end before the next start. [] = no answer."""
    lo, hi = enc.state_lo, enc.state_hi
    n = hi - lo
    if n <= 0:
        return []
    s_raw, e_raw = start[lo:hi], end[lo:hi]
    ends = _best_ends(e_raw, MAX_SPAN_TOKENS)
    best = float((s_raw + e_raw[ends]).max())
    if float(start[enc.rd_pos] + end[enc.rd_pos]) + null_bias > best:
        return []
    idx = np.r_[lo:hi, enc.rd_pos]
    ps = np.exp(start[idx] - start[idx].max())
    pe = np.exp(end[idx] - end[idx].max())
    ps, pe = ps / ps.sum(), pe / pe.sum()
    p_s, p_e = ps[:-1], pe[:-1]
    starts = np.nonzero(p_s >= rel * p_s.max())[0]
    if starts.size > max_spans:
        starts = np.sort(starts[np.argsort(-p_s[starts])[:max_spans]])
    out: List[Dict[str, Any]] = []
    for k, i in enumerate(starts):
        limit = int(starts[k + 1]) if k + 1 < len(starts) else n
        j_hi = max(int(i) + 1, min(limit, int(i) + MAX_SPAN_TOKENS))
        j = int(i) + int(np.argmax(p_e[int(i) : j_hi]))
        if p_e[j] < rel * p_e.max():
            continue
        text, span = _clean_span(state, enc.state_offsets[int(i)][0], enc.state_offsets[j][1], trim_punct=True)
        if text:
            out.append({"text": text, "span": span, "p": float(p_s[int(i)] * p_e[j])})
    m = max(1, len(out))
    for o in out:
        o["confidence"] = float(min(1.0, o.pop("p") * m * m))
    return out


def _decode_span(start: np.ndarray, end: np.ndarray, enc: Encoded, state: str, null_bias: float = 0.0) -> Dict[str, Any]:
    lo, hi = enc.state_lo, enc.state_hi
    null_s = float(start[enc.rd_pos] + end[enc.rd_pos]) + null_bias
    if hi <= lo:
        return {"text": "", "span": None, "confidence": 1.0, "no_answer": True}
    s = start[lo:hi]
    e = end[lo:hi]
    n = hi - lo
    # best (i<=j<=i+MAX) maximising s_i + e_j
    jbest = _best_ends(e, MAX_SPAN_TOKENS)
    v = s + e[jbest]
    bi = int(v.argmax())
    bj = int(jbest[bi])
    best = float(v[bi])
    # confidence: probability mass the model puts on exactly this start and end among state tokens + null
    ps = np.exp(start[np.r_[lo:hi, enc.rd_pos]] - start[np.r_[lo:hi, enc.rd_pos]].max())
    pe = np.exp(end[np.r_[lo:hi, enc.rd_pos]] - end[np.r_[lo:hi, enc.rd_pos]].max())
    ps, pe = ps / ps.sum(), pe / pe.sum()
    conf = float(ps[bi] * pe[bj])
    if null_s > best:
        return {"text": "", "span": None, "confidence": float(ps[-1] * pe[-1]), "no_answer": True}
    offs = enc.state_offsets
    text, span = _clean_span(state, offs[bi][0], offs[bj][1])
    return {"text": text, "span": span, "confidence": conf, "no_answer": False}


def _chunks(n: int, size: int) -> List[List[int]]:
    """``n`` indices in the fewest contiguous parts of at most ``size`` (their sizes differ by one at most)."""
    parts = -(-n // size)
    base, extra = divmod(n, parts)
    out, lo = [], 0
    for i in range(parts):
        hi = lo + base + (1 if i < extra else 0)
        out.append(list(range(lo, hi)))
        lo = hi
    return out


def predict(
    model: torch.nn.Module,
    encoder: Leeth2Encoder,
    samples: Sequence[Sample],
    device: str = "cpu",
    tokens_per_batch: int = 16000,
    max_batch: int = 128,
    amp: bool = False,
    calibrated: bool = True,
    null_bias: float = 0.0,
) -> List[Prediction]:
    """``_predict_flat`` for everything except a choice with more than ``cfg.split_k`` options (0 = never): that one is answered in two
    rounds -- the list is cut into the fewest parts of at most ``split_k`` options, each part is asked on its own (the model is reliable
    on up to ~128 near-duplicate options and loses accuracy beyond ~160, whatever the training), and the winners of the parts are asked
    against each other.  P(option) = P(its part wins) * P(option | part).  Raw (uncalibrated) predictions are never split: the temperatures
    are fitted on what the network itself says."""
    split_k = int(getattr(model.cfg, "split_k", 0) or 0)
    kw: Dict[str, Any] = dict(device=device, tokens_per_batch=tokens_per_batch, max_batch=max_batch, amp=amp, calibrated=calibrated, null_bias=null_bias)
    big = [i for i, s in enumerate(samples) if split_k > 1 and calibrated and s.ptype == "choice" and len(s.options or []) > split_k]
    if not big:
        return _predict_flat(model, encoder, samples, **kw)
    out: List[Optional[Prediction]] = [None] * len(samples)
    is_big = set(big)
    rest = [i for i in range(len(samples)) if i not in is_big]
    if rest:
        for i, p in zip(rest, _predict_flat(model, encoder, [samples[i] for i in rest], **kw)):
            out[i] = p
    plan, parts = [], []
    for i in big:
        s = samples[i]
        chunks = _chunks(len(s.options), split_k)
        plan.append((i, chunks, len(parts)))
        parts += [dataclasses.replace(s, options=[s.options[j] for j in ch], target=0) for ch in chunks]
    first = _predict_flat(model, encoder, parts, **kw)
    winners, finals = [], []
    for i, chunks, off in plan:
        w = [ch[int(np.argmax(first[off + c].probs))] for c, ch in enumerate(chunks)]
        winners.append(w)
        finals.append(dataclasses.replace(samples[i], options=[samples[i].options[j] for j in w], target=0))
    second = _predict_flat(model, encoder, finals, **kw)
    for (i, chunks, off), q in zip(plan, second):
        n = len(samples[i].options)
        probs = np.zeros(n)
        for c, ch in enumerate(chunks):
            probs[ch] = float(q.probs[c]) * first[off + c].probs
        probs = probs / probs.sum()
        out[i] = Prediction("choice", probs=probs, raw={"logits_shown": np.log(np.maximum(probs, 1e-12)), "perm": list(range(n)), "split": len(chunks)})
    return out  # type: ignore[return-value]


@torch.no_grad()
def _predict_flat(
    model: torch.nn.Module,
    encoder: Leeth2Encoder,
    samples: Sequence[Sample],
    device: str = "cpu",
    tokens_per_batch: int = 16000,
    max_batch: int = 128,
    amp: bool = False,
    calibrated: bool = True,
    null_bias: float = 0.0,
) -> List[Prediction]:
    model.eval()
    encoded = [encoder.encode(s, rng=None, want_offsets=(s.ptype == "extract")) for s in samples]
    order = sorted(range(len(samples)), key=lambda i: len(encoded[i]))
    out: List[Optional[Prediction]] = [None] * len(samples)
    cfg = model.cfg
    i = 0
    while i < len(order):
        j = i
        longest = 0
        while j < len(order) and (j - i) < max_batch:
            longest = max(longest, len(encoded[order[j]]))
            if longest * (j - i + 1) > tokens_per_batch and j > i:
                break
            j += 1
        idx = order[i:j]
        batch = collate([encoded[k] for k in idx], pad_id=encoder.pad)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            res = model(**batch)
        choice = res["choice_logits"].float().cpu().numpy()
        noul = res["noul_logit"].float().cpu().numpy()
        score = res["score_logits"].float().cpu().numpy()
        start = res["start_logits"].float().cpu().numpy()
        end = res["end_logits"].float().cpu().numpy()
        for row, k in enumerate(idx):
            s, e = samples[k], encoded[k]
            if s.ptype == "choice":
                lg = choice[row, : e.n_opts].astype(np.float64)
                t = cfg.temp_choice if calibrated else 1.0
                if calibrated and getattr(cfg, "temp_choice_big", 0.0) > 0 and e.n_opts > getattr(cfg, "big_k", 64):
                    t = cfg.temp_choice_big
                p_shown = metrics.softmax(lg / t)
                p = np.zeros(e.n_opts)
                for shown, orig in enumerate(e.perm):
                    p[orig] = p_shown[shown]
                pr = Prediction("choice", probs=p, raw={"logits_shown": lg, "perm": e.perm})
            elif s.ptype == "noul":
                t = cfg.temp_noul if calibrated else 1.0
                z = float(noul[row]) / t
                pr = Prediction("noul", p_true=float(1.0 / (1.0 + np.exp(-z))), raw={"logit": float(noul[row])})
            elif s.ptype == "score":
                lv = int(s.levels)
                lg = score[row, :lv].astype(np.float64)
                t = cfg.temp_score if calibrated else 1.0
                p = metrics.softmax(lg / t)
                pr = Prediction("score", level_probs=p, level=float((p * np.arange(1, lv + 1)).sum()), raw={"logits": lg})
            elif (s.meta or {}).get("multi"):
                found = _decode_spans(start[row], end[row], e, s.state, null_bias=null_bias)
                top = max(found, key=lambda o: o["confidence"]) if found else None
                pr = Prediction("extract", text=top["text"] if top else "", span=top["span"] if top else None,
                                confidence=top["confidence"] if top else 1.0, no_answer=not found, spans=found)
            else:
                d = _decode_span(start[row], end[row], e, s.state, null_bias=null_bias)
                pr = Prediction("extract", text=d["text"], span=d["span"], confidence=d["confidence"], no_answer=d["no_answer"])
            out[k] = pr
        i = j
    return out  # type: ignore[return-value]


def _gold_text(s: Sample) -> str:
    sp = s.spans()
    return "" if not sp else s.state[sp[0][0] : sp[0][1]]


def _set_scores(pred: Sequence[str], gold: Sequence[str]) -> Dict[str, float]:
    """Set-level precision / recall / F1 of answer texts (normalised like span EM); both empty = a perfect "nothing there"."""
    p = [metrics._norm(x) for x in pred if metrics._norm(x)]
    g = [metrics._norm(x) for x in gold if metrics._norm(x)]
    if not p and not g:
        return {"f1": 1.0, "exact": 1.0}
    if not p or not g:
        return {"f1": 0.0, "exact": 0.0}
    from collections import Counter

    hit = sum((Counter(p) & Counter(g)).values())
    if hit == 0:
        return {"f1": 0.0, "exact": 0.0}
    prec, rec = hit / len(p), hit / len(g)
    return {"f1": 2 * prec * rec / (prec + rec), "exact": float(sorted(p) == sorted(g))}


def report(samples: Sequence[Sample], preds: Sequence[Prediction]) -> Dict[str, Any]:
    """Metrics per task, plus pooled per family and per primitive."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[f"task:{s.task}"].append(i)
        groups[f"family:{s.family}"].append(i)
        groups[f"ptype:{s.ptype}"].append(i)

    def one(idx: List[int]) -> Dict[str, Any]:
        by: Dict[str, List[int]] = defaultdict(list)
        for i in idx:
            by[samples[i].ptype].append(i)
        r: Dict[str, Any] = {}
        if "choice" in by:
            r["choice"] = metrics.choice_report([preds[i].probs for i in by["choice"]], [samples[i].target for i in by["choice"]])
        if "noul" in by:
            r["noul"] = metrics.noul_report([preds[i].p_true for i in by["noul"]], [float(samples[i].target) for i in by["noul"]])
        if "score" in by:
            r["score"] = metrics.score_report(
                [preds[i].level for i in by["score"]], [float(samples[i].target) for i in by["score"]], [int(samples[i].levels) for i in by["score"]]
            )
        if "extract" in by:
            multi = [i for i in by["extract"] if (samples[i].meta or {}).get("multi")]
            if multi:
                sc = [_set_scores([o["text"] for o in (preds[i].spans or [])], [samples[i].state[a:b] for a, b in samples[i].spans()]) for i in multi]
                r["extract_multi"] = {"n": len(multi), "set_f1": float(np.mean([x["f1"] for x in sc])), "set_exact": float(np.mean([x["exact"] for x in sc]))}
            by["extract"] = [i for i in by["extract"] if i not in set(multi)]
        if by.get("extract"):
            ems, f1s, has, noa_hit, noa_n = [], [], 0, 0, 0
            for i in by["extract"]:
                gold = _gold_text(samples[i])
                pred = preds[i].text or ""
                if samples[i].target is None:
                    noa_n += 1
                    noa_hit += int(preds[i].no_answer)
                    continue
                m = metrics.span_em_f1(pred, gold)
                ems.append(m["em"])
                f1s.append(m["f1"])
                has += 1
            r["extract"] = {
                "n": len(by["extract"]),
                "has_ans": has,
                "em": float(np.mean(ems)) if ems else float("nan"),
                "f1": float(np.mean(f1s)) if f1s else float("nan"),
                "no_ans": noa_n,
                "no_ans_acc": (noa_hit / noa_n) if noa_n else float("nan"),
            }
        return r

    return {k: one(v) for k, v in sorted(groups.items())}


def headline(rep: Dict[str, Any]) -> Dict[str, float]:
    """A handful of pooled numbers for log lines."""
    h: Dict[str, float] = {}
    for p in ("choice", "noul", "score", "extract"):
        r = rep.get(f"ptype:{p}", {}).get(p)
        if not r:
            continue
        if p in ("choice", "noul"):
            h[f"{p}_acc"] = r["acc"]
            h[f"{p}_ece"] = r["ece"]
            h[f"{p}_nll"] = r["nll"]
        elif p == "score":
            h["score_mae100"] = r["mae100"]
        else:
            h["extract_f1"] = r["f1"]
    return h
