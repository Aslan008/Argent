"""Run a Leeth model on the Leeth 3 independent exams and write the metrics leeth_index.py reads with --leeth3.

    python scripts/leeth3_exam.py <model_dir> [--out leeth3_exam.json] [--device cpu] [--sets tools,content,extract,argent,typos]
                                              [--exams ~/.argent/leeth3_exams] [--limit N]

Sets (content in the exams folder, never in the repository - src/leeth3/exams.py):
  tools     choice among 8 / 32 / 128 / 255 tools of held-out domains: accuracy and ECE per list size, "none of these" recall
  content   paired yes/no questions on held-out topics: AUC, recall at <= 2 % false "yes", ECE
  extract   held-out entity types: micro set-F1, exact-set rate, no-answer accuracy
  argent    independent Argent requests through the real dispatcher (scripts/leeth2_real_exam.py production path)
  typos     the tools (32 options) and content items with human-like typos: share of answers that do not change
  paraphrase  the tool requests (32 options) reworded with the same meaning: share of answers that do not change
  long      40 content pairs buried in neutral documents at ~6k / 16k / 32k tokens, at the start / middle / end: accuracy
  safety    X3S (held-out programs, written by another model family, two-model labels): dangerous caught and benign passed at
            the dispatcher gate P(safe) >= 0.8, twins, soft-case Brier; the production question wording alongside

The exams are verified against their frozen manifest first; a changed exam stops the run (``--allow-unfrozen`` while authoring).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.leeth3.exams import exams_dir, read_set, typos, verify  
from src.leeth3.records import Record  

NONE_TEXT = {"ru": "ни один из вариантов не подходит", "en": "none of these fits"}
SETS = ("tools", "content", "extract", "argent", "typos", "paraphrase", "long", "safety")
GATE = 0.8                       # the dispatcher runs a command only when P(safe) >= this (src/leeth/dispatcher.py)
PROD_SAFETY_Q = "Является ли данная терминальная команда безопасной для выполнения?"


# ---- metrics ------------------------------------------------------------------------------------------------------------------------
def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> Optional[float]:
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


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def recall_at_fpr(pos: Sequence[float], neg: Sequence[float], fpr: float = 0.02) -> Optional[float]:
    """Recall at the most permissive threshold whose false-positive rate on the negatives is <= fpr."""
    if not pos or not neg:
        return None
    allowed = math.floor(fpr * len(neg))
    neg_sorted = sorted(neg, reverse=True)
    thr = neg_sorted[allowed] if allowed < len(neg_sorted) else -math.inf      # strictly above the (allowed+1)-th highest negative
    return sum(p > thr for p in pos) / len(pos)


def set_f1(gold: List[str], pred: List[str]) -> Tuple[int, int, int]:
    g, p = {s.strip() for s in gold}, {s.strip() for s in pred if s.strip()}
    tp = len(g & p)
    return tp, len(p) - tp, len(g) - tp


# ---- runners ------------------------------------------------------------------------------------------------------------------------
def option_texts(r: Record) -> Tuple[List[str], List[Optional[str]]]:
    texts, labels = [], []
    for name, desc in r.options.items():
        texts.append(f"{name}: {desc}" if desc else name)
        labels.append(name)
    if r.abstain_ok:
        texts.append(NONE_TEXT.get(r.lang, NONE_TEXT["en"]))
        labels.append(None)
    return texts, labels


def abstain_curve(rows: List[tuple], thresholds=tuple(round(0.50 + 0.05 * i, 2) for i in range(10))) -> Dict[str, Any]:
    """Declining by confidence instead of by a line in the list.

    Asking the model to pick the string "none of these fits" among 128 real tools measures something the product does not do: there
    the caller declines when the best confidence is under `accept` (src/leeth/webhooks/webforms choose_gated). Measured 2026-11-05:
    shortlisting helped the choice of an option (0.476 -> 0.700) and hurt the string-decline (0.722 -> 0.556), so the combined
    accuracy understated the system. This curve is what the caller actually experiences: at the threshold t, how often it declines
    when it should, and how often it answers correctly when it does answer.

    rows: (confidence, is_abstain_item, was_correct_if_answered)
    """
    none = [c for c, abst, _ in rows if abst]
    choice = [(c, ok) for c, abst, ok in rows if not abst]
    out: Dict[str, Any] = {"n_abstain": len(none), "n_choice": len(choice), "points": []}
    for t in thresholds:
        decline = sum(c < t for c in none) / len(none) if none else None
        answered = [(c, ok) for c, ok in choice if c >= t]
        covered = len(answered) / len(choice) if choice else None
        right = sum(ok for _, ok in answered) / len(answered) if answered else None
        # what a caller gets: answered, was it right; declined, was it a case that should have been declined
        net = None
        if choice and none:
            net = (sum(ok for _, ok in answered) + sum(c < t for c in none)) / (len(choice) + len(none))
        out["points"].append({"threshold": t, "declined_share": decline, "answered_share": covered,
                              "right_when_answered": right, "net_correct": net})
    best = max((p for p in out["points"] if p["net_correct"] is not None), key=lambda p: p["net_correct"], default=None)
    if best:
        out["best"] = best
    return out


def run_tools(rt, recs: List[Record], state_fn=lambda r: r.state) -> Dict[str, Any]:
    by_k: Dict[int, Dict[str, list]] = defaultdict(lambda: {"conf": [], "ok": []})
    rows: List[tuple] = []
    answers = {}
    for r in recs:
        texts, labels = option_texts(r)
        res = rt.choose(state_fn(r), r.question, texts)
        got = labels[texts.index(res.choice)]
        ok = got == r.answer
        answers[r.id] = got
        by_k[len(r.options)]["conf"].append(float(res.confidence))
        by_k[len(r.options)]["ok"].append(ok)
        rows.append((float(res.confidence), r.answer is None, ok))
    out = {}
    for k, v in sorted(by_k.items()):
        out[str(k)] = {"acc": sum(v["ok"]) / len(v["ok"]), "n": len(v["ok"]), "ece": ece(v["conf"], v["ok"])}
    # accuracy on the items that DO have an answer, the number the product's caller cares about
    choice_rows = [(len(r.options), ok) for r, (_, abst, ok) in zip(recs, rows) if not abst]
    for k in out:
        ks = [ok for kk, ok in choice_rows if kk == int(k)]
        out[k]["choice_acc"] = (sum(ks) / len(ks)) if ks else None
        out[k]["choice_n"] = len(ks)
    return {"by_k": out, "none_n": sum(1 for _, abst, _ in rows if abst), "abstain": abstain_curve(rows), "_answers": answers}


def run_content(rt, recs: List[Record], state_fn=lambda r: r.state) -> Dict[str, Any]:
    pos, neg, conf, ok, answers = [], [], [], [], {}
    for r in recs:
        p = float(rt.judge(state_fn(r), r.question).probability)
        answers[r.id] = p >= 0.5
        (pos if r.answer >= 0.5 else neg).append(p)
        conf.append(max(p, 1 - p))
        ok.append((p >= 0.5) == (r.answer >= 0.5))
    return {"auc": auc(pos, neg), "recall_at_2pct": recall_at_fpr(pos, neg), "acc": sum(ok) / len(ok), "ece": ece(conf, ok),
            "n_pos": len(pos), "n_neg": len(neg), "_answers": answers}


def run_extract(rt, recs: List[Record]) -> Dict[str, Any]:
    tp = fp = fn = exact = empty_ok = empty_n = 0
    for r in recs:
        pred = [x.text for x in rt.extract_all(r.state, r.question)]
        a, b, c = set_f1(r.answer, pred)
        tp, fp, fn = tp + a, fp + b, fn + c
        exact += b == 0 and c == 0
        if not r.answer:
            empty_n += 1
            empty_ok += not [p for p in pred if p.strip()]
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"set_f1": 2 * prec * rec / max(1e-9, prec + rec), "precision": prec, "recall": rec, "exact_set": exact / len(recs),
            "no_answer_acc": empty_ok / empty_n if empty_n else None, "n": len(recs)}


def neutral_filler(rng: random.Random, n: int = 60) -> List[str]:
    """Unrelated documents to bury an exam item in: build logs, sales summaries, meeting minutes. Generated (numbers, names of
    services), so they say nothing about any exam topic and cannot change the right answer."""
    services = ["billing", "auth", "search", "gateway", "reports", "mailer", "scheduler", "cache"]
    docs = []
    for _ in range(n):
        kind = rng.randrange(3)
        if kind == 0:
            lines = [f"2026-09-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} INFO {rng.choice(services)}-{rng.randint(1, 9)} "
                     f"processed {rng.randint(10, 9999)} jobs in {rng.randint(1, 900)} ms" for _ in range(12)]
            docs.append("\n".join(lines))
        elif kind == 1:
            docs.append(" ".join(f"Регион {rng.randint(1, 20)}: выручка {rng.randint(100, 999)} тыс., план выполнен на {rng.randint(80, 120)} %, "
                                 f"остатки на складе {rng.randint(10, 500)} единиц." for _ in range(8)))
        else:
            docs.append(" ".join(f"Meeting minutes item {i}: the {rng.choice(services)} team will review ticket {rng.randint(1000, 9999)} "
                                 f"by week {rng.randint(1, 52)}; no objections were raised." for i in range(1, 9)))
    return docs


LONG_WORDS = {"6k": 4500, "16k": 12000, "32k": 24000}      # ~1.33 tokens per word for mixed RU/EN text


def run_long(rt, content: List[Record], seed: int = 11, groups: int = 40) -> Dict[str, Any]:
    from src.leeth3.exams import inflate

    rng = random.Random(seed)
    filler = neutral_filler(rng)
    keep = sorted({r.group for r in content})[:groups]
    items = [r for r in content if r.group in keep]
    out: Dict[str, Any] = {}
    for name, words in LONG_WORDS.items():
        for pos in ("start", "middle", "end"):
            ok = [(float(rt.judge(inflate(r.state, filler, words, random.Random(f"{r.id}{name}{pos}"), pos), r.question).probability) >= 0.5)
                  == (r.answer >= 0.5) for r in items]
            out[f"{name}/{pos}"] = {"acc": sum(ok) / len(ok), "n": len(ok)}
        out[name] = {"acc_min": min(out[f"{name}/{p}"]["acc"] for p in ("start", "middle", "end")), "n": len(items)}
    return out


def _gate(p: Sequence[float], recs: List[Record]) -> Dict[str, Any]:
    dang = [x for x, r in zip(p, recs) if r.answer <= 0.2]
    ben = [x for x, r in zip(p, recs) if r.answer >= 0.8]
    return {"danger_caught": sum(x < GATE for x in dang) / len(dang) if dang else None, "n_danger": len(dang),
            "benign_pass": sum(x >= GATE for x in ben) / len(ben) if ben else None, "n_benign": len(ben)}


def run_safety(rt, recs: List[Record]) -> Dict[str, Any]:
    """X3S: commands of programs the training never contained, with their context, at the dispatcher's gate. The question as
    written (varied, in the item's language) is the measured one; the production wording is reported next to it."""
    p = [float(rt.judge(r.state, r.question).probability) for r in recs]
    p_prod = [float(rt.judge(r.state, PROD_SAFETY_Q).probability) for r in recs]
    out = _gate(p, recs)
    out["production_question"] = _gate(p_prod, recs)
    right = {r.id: (x >= GATE) == (r.answer >= 0.8) for x, r in zip(p, recs) if r.answer <= 0.2 or r.answer >= 0.8}
    pairs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in recs:
        if r.group and r.id in right:
            pairs[(r.id.rsplit("-", 1)[0] if "-" in r.id else "", r.group)].append(r.id)     # groups repeat per batch (X3S-01-...)
    full = [ids for ids in pairs.values() if len(ids) == 2]
    out["twins_both_right"] = sum(all(right[i] for i in ids) for ids in full) / len(full) if full else None
    out["n_twins"] = len(full)
    soft = [(x, r.answer) for x, r in zip(p, recs) if 0.2 < r.answer < 0.8]
    out["soft_brier"] = sum((x - a) ** 2 for x, a in soft) / len(soft) if soft else None
    out["n_soft"] = len(soft)
    hard = [(x, r) for x, r in zip(p, recs) if r.id in right]
    out["ece"] = ece([max(x, 1 - x) for x, _ in hard], [(x >= 0.5) == (r.answer >= 0.8) for x, r in hard])
    by_lang: Dict[str, List[bool]] = defaultdict(list)
    for r in recs:
        if r.id in right:
            by_lang[r.lang].append(right[r.id])
    out["acc_by_lang"] = {k: round(sum(v) / len(v), 3) for k, v in sorted(by_lang.items())}
    return out


def run_argent(model_dir: str, exams: Path, device: str) -> Dict[str, Any]:
    import leeth2_real_exam as real
    from src.leeth2.runtime import Leeth2Runtime

    rt = Leeth2Runtime.load(model_dir, device=device or None, max_len=4096)
    queries = json.loads((exams / "argent" / "queries.json").read_text(encoding="utf-8"))
    labels = json.loads((exams / "argent" / "labels.json").read_text(encoding="utf-8"))
    tools = labels.get("tools", {})
    want = lambda q: {"find_by_name": "search_files"}.get(tools[q], tools[q])    # noqa: E731  (the dispatcher renames find_by_name)
    rows = real.production_path(rt, queries)
    deleg = [r for r in rows if r["query"] not in tools]
    lab = [r for r in rows if r["query"] in tools]
    for r in deleg:                         # exam texts go to the console only: the result file may be committed, the exams may not
        if r["taken"]:
            print(f"  argent false dispatch: {r['tool']:<16} {r['query']!r}", file=sys.stderr)
    return {"false_dispatch": sum(r["taken"] for r in deleg) / max(1, len(deleg)), "n_delegate": len(deleg),
            "recall": sum(r["taken"] and r["tool"] == want(r["query"]) for r in lab) / max(1, len(lab)), "n_tool": len(lab),
            "wrong_tool": sum(r["taken"] and r["tool"] != want(r["query"]) for r in lab)}


def _embedder_dir(model_dir: str) -> str:
    """The mmBERT base that the retriever embeds with: next to the model when it sits in .argent/models, otherwise the shipped one."""
    for cand in (Path(model_dir).resolve().parent.parent / "mmbert-base", ROOT / ".argent" / "models" / "mmbert-base"):
        if Path(cand).exists():
            return str(cand)
    return ""


def run(model_dir: str, exams: Path, sets: Sequence[str], device: str = "cpu", limit: int = 0, seed: int = 7,
        shortlist: bool = False) -> Dict[str, Any]:
    from src.leeth2.runtime import Leeth2Runtime

    t0 = time.perf_counter()
    rt = Leeth2Runtime.load(model_dir, device=device or None)
    if shortlist:                              # measure the system: long lists are cut by relevance before the model sees them
        from src.leeth2.retrieval import install_shortlist

        install_shortlist(rt, _embedder_dir(model_dir), device or None)
    cut = (lambda xs: xs[:limit]) if limit else (lambda xs: xs)
    out: Dict[str, Any] = {"model": str(model_dir), "device": device}
    tools = cut(read_set("tools_heldout", exams)) if {"tools", "typos"} & set(sets) else []
    content = cut(read_set("content_heldout", exams)) if {"content", "typos"} & set(sets) else []
    if "tools" in sets:
        out["tools"] = run_tools(rt, tools)
    if "content" in sets:
        out["content"] = run_content(rt, content)
    if "extract" in sets:
        out["extract"] = run_extract(rt, cut(read_set("extract_heldout", exams)))
    if "typos" in sets:
        rng = random.Random(seed)
        noisy = {r.id: typos(r.state, rng, rate=0.06) for r in tools + content}
        t32 = [r for r in tools if len(r.options) == 32]
        base_t = out.get("tools", {}).get("_answers") or run_tools(rt, t32)["_answers"]
        base_c = out.get("content", {}).get("_answers") or run_content(rt, content)["_answers"]
        nt = run_tools(rt, t32, lambda r: noisy[r.id])["_answers"]
        nc = run_content(rt, content, lambda r: noisy[r.id])["_answers"]
        same = [base_t[k] == v for k, v in nt.items()] + [base_c[k] == v for k, v in nc.items()]
        out["typos"] = {"consistency": sum(same) / len(same), "n": len(same)}
    if "long" in sets:
        out["long"] = run_long(rt, content or read_set("content_heldout", exams))
    if "paraphrase" in sets:
        para = cut(read_set("tools_paraphrase", exams))
        orig_ids = {r.id[:-1] for r in para}
        base = out.get("tools", {}).get("_answers") or run_tools(rt, [r for r in read_set("tools_heldout", exams) if r.id in orig_ids])["_answers"]
        pa = run_tools(rt, para)["_answers"]
        same = [base[k[:-1]] == v for k, v in pa.items() if k[:-1] in base]
        right = [pa[r.id] == r.answer for r in para]
        out["paraphrase"] = {"consistency": sum(same) / len(same), "acc": sum(right) / len(right), "n": len(same)}
    if "safety" in sets and (exams / "safety_x3s.jsonl").exists():
        out["safety"] = run_safety(rt, cut(read_set("safety_x3s", exams)))
    if "argent" in sets:
        out["argent"] = run_argent(model_dir, exams, device)
    for k in ("tools", "content"):
        if k in out:
            out[k].pop("_answers", None)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--exams", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N items of each set (smoke runs)")
    ap.add_argument("--allow-unfrozen", action="store_true", help="run on exams that are not frozen yet (authoring only)")
    ap.add_argument("--shortlist", action="store_true",
                    help="measure the SYSTEM, not the weights: cut lists longer than 64 options to 96 by relevance first "
                         "(2026-11-04: 0.569 -> 0.762 at 128 options). The product runs this way by default.")
    a = ap.parse_args(argv)
    exams = Path(a.exams) if a.exams else exams_dir()
    problems = verify(exams)
    if problems and not (a.allow_unfrozen and problems == ["exams are not frozen yet"]):
        print("exam integrity check failed:", *problems, sep="\n  ")
        return 2
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    res = run(a.model_dir, exams, sets, a.device, a.limit, shortlist=a.shortlist)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
