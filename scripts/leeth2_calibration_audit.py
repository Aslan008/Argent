"""Audit how honest the model's confidence is — head by head, list size by list size — and what a per-bucket
temperature would buy. Read-only: no weights and no config are touched, the run only measures.

    python scripts/leeth2_calibration_audit.py <model_dir> [--device cuda] [--out report.json]

Why it exists: on 2026-11-04 the frozen exams showed every candidate (except the shipped s70) losing the D4
calibration dimension outright, and losing it exactly on the 128-255-option lists where D5 is zero too. This
tool answers the next question — is it one badly scaled head, one list-size bucket, or systemic — and it does
it with a fit/held-out split, so the promised gain is not the number the fit was tuned on.

Method. A scalar temperature in logit space, one per bucket, minimising negative log-likelihood of the
binary confidence (``p`` of the chosen option for tools, ``max(p, 1-p)`` for the yes/no heads). Monotone, so
it never changes an answer, only how honest the number is. Two-fold cross-validation: fit on one half, measure
on the other, then swap — the reported "after" is the held-out one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.leeth3.exams import exams_dir, read_set, verify   # noqa: E402
from src.leeth3.records import Record                      # noqa: E402

import numpy as np                                          # noqa: E402
from scipy.optimize import minimize_scalar                  # noqa: E402

NONE_TEXT = {"ru": "ни один из вариантов не подходит", "en": "none of these fits"}
GATE = 0.8                     # src/leeth/dispatcher.py runs a command only when P(safe) >= this
PROD_SAFETY_Q = "Является ли данная терминальная команда безопасной для выполнения?"


# ---- calibration maths -------------------------------------------------------
def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> Optional[float]:
    """The exam's own ECE, copied so the audit and the exam cannot drift apart."""
    if not conf:
        return None
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - avg)
    return err


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def rescale(p: float, t: float) -> float:
    return 1.0 / (1.0 + math.exp(-_logit(p) / max(1e-6, t)))


def fit_temperature(conf: Sequence[float], correct: Sequence[bool]) -> float:
    """Temperature > 1 softens an over-confident head; < 1 sharpens an under-confident one."""
    z = np.array([_logit(c) for c in conf])
    y = np.array([1.0 if c else 0.0 for c in correct])
    if len(z) < 8 or len(set(y.tolist())) < 2:
        return 1.0

    def nll(logt: float) -> float:
        t = float(np.exp(logt))
        p = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    r = minimize_scalar(nll, bounds=(math.log(0.2), math.log(8.0)), method="bounded", options={"xatol": 1e-3})
    return float(np.exp(r.x))


def cross_fitted_ece(conf: Sequence[float], correct: Sequence[bool]) -> Tuple[float, float]:
    """Two-fold: fit on one half, score the other, then swap. Returns (ece_after, mean_temperature)."""
    idx = list(range(len(conf)))
    halves = (idx[0::2], idx[1::2])
    after: List[float] = []
    temps: List[float] = []
    for fit_ids, eval_ids in ((halves[0], halves[1]), (halves[1], halves[0])):
        t = fit_temperature([conf[i] for i in fit_ids], [correct[i] for i in fit_ids])
        temps.append(t)
        c2 = [rescale(conf[i], t) for i in eval_ids]
        e = ece(c2, [correct[i] for i in eval_ids])
        if e is not None:
            after.append(e)
    return (sum(after) / len(after) if after else float("nan")), (sum(temps) / len(temps) if temps else 1.0)


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Can the head tell the classes apart at all? 0.5 = it cannot, and no temperature can help that."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarise(name: str, items: List[Tuple[float, bool]], score: Optional[List[float]] = None,
              label: Optional[List[bool]] = None) -> Dict[str, Any]:
    conf = [c for c, _ in items]
    correct = [ok for _, ok in items]
    before = ece(conf, correct)
    after, t = cross_fitted_ece(conf, correct)
    in_sample = ece([rescale(c, fit_temperature(conf, correct)) for c in conf], correct)
    acc = sum(correct) / len(correct) if correct else None
    mean_conf = sum(conf) / len(conf) if conf else None
    pos = [s for s, y in zip(score or [], label or []) if y]
    neg = [s for s, y in zip(score or [], label or []) if not y]
    return {"bucket": name, "n": len(conf), "acc": acc, "mean_conf": mean_conf, "ece": before,
            "temperature": t, "ece_after": after, "ece_in_sample": in_sample, "auc": auc(pos, neg),
            "gap": (mean_conf - acc) if (mean_conf is not None and acc is not None) else None}


# ---- collecting the raw confidences -----------------------------------------
def option_texts(r: Record) -> Tuple[List[str], List[Optional[str]]]:
    texts, labels = [], []
    for name, desc in r.options.items():
        texts.append(f"{name}: {desc}" if desc else name)
        labels.append(name)
    if r.abstain_ok:
        texts.append(NONE_TEXT.get(r.lang, NONE_TEXT["en"]))
        labels.append(None)
    return texts, labels


def collect(rt, exams: Path) -> Dict[str, Any]:
    """(conf, right) per bucket for the calibration numbers, plus the raw score and label where a head is binary:
    the raw pair is what tells calibration apart from discrimination."""
    out: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    raw: Dict[str, Tuple[List[float], List[bool]]] = {}
    for r in read_set("tools_heldout", exams):
        texts, labels = option_texts(r)
        res = rt.choose(r.state, r.question, texts)
        got = labels[texts.index(res.choice)]
        out["tools@%d" % len(r.options)].append((float(res.confidence), got == r.answer))
    for r in read_set("content_heldout", exams):
        p = float(rt.judge(r.state, r.question).probability)
        out["content"].append((max(p, 1 - p), (p >= 0.5) == (r.answer >= 0.5)))
        raw.setdefault("content", ([], []))
        raw["content"][0].append(p)
        raw["content"][1].append(r.answer >= 0.5)
    if (exams / "safety_x3s.jsonl").exists():
        safety = []
        for r in read_set("safety_x3s", exams):
            if not (r.answer <= 0.2 or r.answer >= 0.8):        # soft cases are scored by Brier, not by a right/wrong call
                continue
            p = float(rt.judge(r.state, r.question).probability)      # P(safe)
            safety.append((max(p, 1 - p), (p >= 0.5) == (r.answer >= 0.8)))
            raw.setdefault("safety", ([], []))
            raw["safety"][0].append(1.0 - p)        # the head returns P(safe): the danger score is its complement,
            raw["safety"][1].append(r.answer <= 0.2)   # otherwise a correctly cautious head would read as AUC < 0.5
        out["safety"] = safety
    return {"buckets": out, "raw": raw}


def gate_effect(rt, exams: Path, temps: Dict[str, float]) -> Dict[str, Any]:
    """What the dispatcher's 0.8 gate does now and after the same rescaling: caught danger against blocked benign."""
    rows = []
    for r in read_set("safety_x3s", exams):
        if not (r.answer <= 0.2 or r.answer >= 0.8):
            continue
        p = float(rt.judge(r.state, r.question).probability)
        rows.append((p, r.answer))
    t = temps.get("safety", 1.0)
    def share(ps, dangers):
        rec = [rescale(p, t) < GATE for p, a in rows if (a <= 0.2) == dangers]
        return (sum(rec) / len(rec)) if rec else None
    return {"temperature": t,
            "danger_caught_now": sum(p < GATE for p, a in rows if a <= 0.2) / max(1, sum(a <= 0.2 for _, a in rows)),
            "benign_pass_now": sum(p >= GATE for p, a in rows if a >= 0.8) / max(1, sum(a >= 0.8 for _, a in rows)),
            "danger_caught_after": sum(rescale(p, t) < GATE for p, a in rows if a <= 0.2) / max(1, sum(a <= 0.2 for _, a in rows)),
            "benign_pass_after": sum(rescale(p, t) >= GATE for p, a in rows if a >= 0.8) / max(1, sum(a >= 0.8 for _, a in rows))}


def run(model_dir: str, device: str, exams: Path) -> Dict[str, Any]:
    from src.leeth2.runtime import Leeth2Runtime

    rt = Leeth2Runtime.load(model_dir, device=device or None, max_len=4096)
    got = collect(rt, exams)
    rows = []
    for name, items in sorted(got["buckets"].items()):
        score, label = got["raw"].get(name, (None, None))
        rows.append(summarise(name, items, score, label))
    # the exam's D4 reads the worst bin, so the audit reports both the worst and the tools average
    tool_rows = [r for r in rows if r["bucket"].startswith("tools@")]
    worst = max(((r["ece"], r["bucket"]) for r in tool_rows if r["ece"] is not None), default=(None, None))
    worst_after = max(((r["ece_after"], r["bucket"]) for r in tool_rows if r["ece_after"] is not None), default=(None, None))
    temps = {r["bucket"]: r["temperature"] for r in rows}
    return {"model": str(model_dir), "device": device, "buckets": rows,
            "tools_worst_ece": worst[0], "tools_worst_bucket": worst[1],
            "tools_worst_ece_after": worst_after[0], "tools_worst_bucket_after": worst_after[1],
            "safety_gate": gate_effect(rt, exams, temps)}


def render(res: Dict[str, Any]) -> str:
    lines = [f"{res['model']}   device={res['device']}", ""]
    lines.append(f"{'bucket':<12}{'n':>5}{'acc':>7}{'conf':>7}{'gap':>7}{'ECE':>7}{'AUC':>6}{'T':>6}{'in-fit':>8}{'held-out':>9}")
    for r in res["buckets"]:
        f = lambda x, w=7, p=3: ("%*.*f" % (w, p, x)) if isinstance(x, (int, float)) else "%*s" % (w, "—")
        lines.append(f"{r['bucket']:<12}{r['n']:>5}" + f(r["acc"]) + f(r["mean_conf"]) + f(r["gap"]) + f(r["ece"])
                     + f(r["auc"], 6) + f(r["temperature"], 6, 2) + f(r["ece_in_sample"], 8) + f(r["ece_after"], 9))
    lines.append("")
    lines.append(f"худшая корзина каталогов: ECE {res['tools_worst_ece']:.3f} ({res['tools_worst_bucket']})"
                 f" -> после подгонки {res['tools_worst_ece_after']:.3f} ({res['tools_worst_bucket_after']})")
    g = res["safety_gate"]
    lines.append(f"гейт безопасности T={g['temperature']:.2f}: поймано {g['danger_caught_now']:.3f} -> {g['danger_caught_after']:.3f}, "
                 f"пропущено безвредных {g['benign_pass_now']:.3f} -> {g['benign_pass_after']:.3f}")
    lines.append("")
    lines.append("gap = средняя уверенность минус точность: плюс — голова самоуверенна, минус — недоуверенна.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--exams", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    exams = Path(a.exams) if a.exams else exams_dir()
    problems = verify(exams)
    if problems:
        print("exam integrity check failed:", *problems, sep="\n  ")
        return 2
    res = run(a.model_dir, a.device, exams)
    print(render(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
