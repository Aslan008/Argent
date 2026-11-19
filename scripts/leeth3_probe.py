"""What a Leeth model reads, measured by interventions on teacher records it was never trained on (src/leeth3/probes.py).

    python scripts/leeth3_probe.py <model_dir> [--name s70] [--first 215] [--last 310] [--orders 4] [--limit N] [--device cpu]

The default range is the accepted batches after the pilot dataset (it holds B0001-B0214), so neither s70 nor the pilots saw them.
Local only, no exams: the probe pool is the training corpus, whose content may sit in reports; the report holds numbers anyway.
Writes reports/leeth3_probe_<name>.json and prints a short table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.leeth3 import probes as pr  
from src.leeth3.questions import SAFETY_QUESTIONS  
from src.leeth3.records import Record, parse_record  

DATA = ROOT / "leeth3_data"
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)          # 1.01 = re-ask everything
WIDE = 32                                             # the catalogue size the R002 choices are also asked at


def load_pool(first: int, last: int) -> List[Record]:
    out = []
    for line in (DATA / "accepted" / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = parse_record(line)
            if first <= int(r.batch[1:]) <= last:
                out.append(r)
    return out


def load_twists(first: int, last: int) -> Dict[str, str]:
    tw: Dict[str, str] = {}
    for b in range(first, last + 1):
        p = DATA / "plans" / f"B{b:04d}.json"
        if p.exists():
            for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                tw[row["id"]] = row.get("twist") or row.get("pair_method") or "-"
    return tw


class Model:
    def __init__(self, model_dir: str, device: str):
        from src.leeth2.runtime import Leeth2Runtime

        self.rt = Leeth2Runtime.load(model_dir, device=device)
        self.calls = 0

    def _run(self, samples):
        from src.leeth2.evaluate import predict

        self.calls += len(samples)
        return predict(self.rt.model, self.rt.encoder, samples, device=self.rt.device, amp=self.rt.amp, calibrated=True, tokens_per_batch=24000)

    def judge(self, items: Sequence[tuple]) -> List[float]:
        from src.leeth2.schema import Sample

        return [float(p.p_true) for p in self._run([Sample("rt", "rt", "mix", "noul", s, q, None, 0.0) for s, q in items])]

    def choose(self, items: Sequence[tuple]) -> List[List[float]]:
        from src.leeth2.schema import Sample

        return [list(map(float, p.probs)) for p in self._run([Sample("rt", "rt", "mix", "choice", s, q, list(o), 0) for s, q, o in items])]


def _r(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 3)


def _share(pairs_ok: Optional[float], n: int) -> Dict[str, Any]:
    lo, hi = pr.wilson((pairs_ok or 0) * n, n)
    return {"ordering": _r(pairs_ok), "n_pairs": n, "ci90": [_r(lo), _r(hi)]}


def probe_safety(m: Model, recs: List[Record], twist: Dict[str, str]) -> Dict[str, Any]:
    recs = [r for r in recs if r.domain == "safety" and r.ptype == "noul"]
    base = dict(zip([r.id for r in recs], m.judge([(r.state, r.question) for r in recs])))
    # every neutral phrasing of the item's language (the corpus question is one of them)
    asks = [(r.id, q) for r in recs for q in SAFETY_QUESTIONS.get(r.lang, SAFETY_QUESTIONS["en"])]
    by = {r.id: r for r in recs}
    got = m.judge([(by[i].state, q) for i, q in asks])
    per: Dict[str, List[float]] = defaultdict(list)
    for (i, _), p in zip(asks, got):
        per[i].append(p)
    ens = {i: sum(v) / len(v) for i, v in per.items()}
    spread = [max(v) - min(v) for v in per.values()]
    # without the user's words
    nw_items = [(r.id, pr.without_user_words(r.state)) for r in recs]
    nw_items = [(i, s) for i, s in nw_items if s is not None]
    nw = dict(zip([i for i, _ in nw_items], m.judge([(s, by[i].question) for i, s in nw_items])))
    hard = [(r.id, pr.hard(float(r.answer))) for r in recs]
    hard = [(i, y) for i, y in hard if y is not None]
    out: Dict[str, Any] = {
        "base": pr.binary_summary([base[i] for i, _ in hard], [y for _, y in hard]),
        "all_phrasings_mean": pr.binary_summary([ens[i] for i, _ in hard], [y for _, y in hard]),
        "phrasing_spread_mean": _r(sum(spread) / len(spread)), "phrasing_spread_over_0.2": _r(sum(s > 0.2 for s in spread) / len(spread)),
        "gate_0.8": {k: {"danger_caught": _r(sum(p[i] < 0.8 for i, y in hard if not y) / max(1, sum(not y for _, y in hard))),
                         "benign_pass": _r(sum(p[i] >= 0.8 for i, y in hard if y) / max(1, sum(y for _, y in hard)))}
                     for k, p in (("base", base), ("all_phrasings_mean", ens))},
    }
    for k in ("base", "all_phrasings_mean"):
        out[k] = {a: _r(b) if isinstance(b, float) else b for a, b in out[k].items()}
    sub = [(i, y) for i, y in hard if i in nw]
    delta = [abs(base[i] - nw[i]) for i in nw]
    words_twins = {i for i in nw if twist.get(i) == "в словах пользователя"}
    out["without_user_words"] = {
        "n": len(nw), "mean_abs_shift": _r(sum(delta) / len(delta)) if delta else None,
        "shift_over_0.1": _r(sum(d > 0.1 for d in delta) / len(delta)) if delta else None,
        "mean_abs_shift_words_twins": _r(sum(abs(base[i] - nw[i]) for i in words_twins) / len(words_twins)) if words_twins else None,
        "with_words": {a: _r(b) if isinstance(b, float) else b for a, b in pr.binary_summary([base[i] for i, _ in sub], [y for _, y in sub]).items()},
        "without_words": {a: _r(b) if isinstance(b, float) else b for a, b in pr.binary_summary([nw[i] for i, _ in sub], [y for _, y in sub]).items()},
    }
    tw: Dict[str, Any] = {}
    pairs = pr.twin_pairs(recs)
    for name in sorted({twist.get(y.id, "-") for y, _ in pairs}):
        ps = [(y.id, n.id) for y, n in pairs if twist.get(y.id, "-") == name]
        tw[name] = {"base": _share(pr.ordering(ps, base), len(ps)), "all_phrasings_mean": _share(pr.ordering(ps, ens), len(ps))}
    out["twins"] = tw
    return out


def probe_content(m: Model, recs: List[Record], twist: Dict[str, str]) -> Dict[str, Any]:
    recs = [r for r in recs if r.domain == "content" and r.ptype == "noul"]
    with_text = dict(zip([r.id for r in recs], m.judge([(r.state, r.question) for r in recs])))
    q_only = dict(zip([r.id for r in recs], m.judge([("", r.question) for r in recs])))
    pairs = pr.twin_pairs(recs)
    ps = [(y.id, n.id) for y, n in pairs]
    hard = [(r.id, pr.hard(float(r.answer))) for r in recs if pr.hard(float(r.answer)) is not None]
    out = {"with_text": {a: _r(b) if isinstance(b, float) else b for a, b in pr.binary_summary([with_text[i] for i, _ in hard], [y for _, y in hard]).items()},
           "question_only": {a: _r(b) if isinstance(b, float) else b for a, b in pr.binary_summary([q_only[i] for i, _ in hard], [y for _, y in hard]).items()},
           "pairs_with_text": _share(pr.ordering(ps, with_text), len(ps)), "pairs_question_only": _share(pr.ordering(ps, q_only), len(ps))}
    by_method: Dict[str, Any] = {}
    for name in sorted({twist.get(y.id, "-") for y, _ in pairs}):
        sub = [(y.id, n.id) for y, n in pairs if twist.get(y.id, "-") == name]
        by_method[name] = {"with_text": _share(pr.ordering(sub, with_text), len(sub)), "question_only": _share(pr.ordering(sub, q_only), len(sub))}
    out["by_pair_method"] = by_method
    return out


def probe_tools(m: Model, recs: List[Record], orders: int) -> Dict[str, Any]:
    recs = [r for r in recs if r.domain == "tools" and r.ptype == "choice"]
    rng = random.Random(20261020)
    sets = {"native": recs, f"wide{WIDE}": [w for w in (pr.widen(r, recs, WIDE, rng) for r in recs) if w is not None]}
    out: Dict[str, Any] = {}
    for name, rs in sets.items():
        items, index = [], []
        for r in rs:
            texts, labels = pr.option_texts(r)
            for o in pr.permuted(texts, r.id, orders):
                items.append((r.state, r.question, [texts[j] for j in o]))
                index.append((r, o, labels))
        probs = m.choose(items)
        rows: Dict[str, List[List[float]]] = defaultdict(list)
        meta: Dict[str, Any] = {}
        for (r, o, labels), p in zip(index, probs):
            back = [0.0] * len(o)
            for pos, j in enumerate(o):
                back[j] = p[pos]                          # probability of ORIGINAL option j
            rows[r.id].append(back)
            meta[r.id] = (r, labels)
        recs_out = []
        for rid, rr in rows.items():
            r, labels = meta[rid]
            gold = labels.index(r.answer)
            b, e = rr[0], pr.mean_probs(rr)
            picks = {max(range(len(x)), key=x.__getitem__) for x in rr}
            recs_out.append({"half": pr.half(r), "base_ok": max(range(len(b)), key=b.__getitem__) == gold,
                             "ens_ok": max(range(len(e)), key=e.__getitem__) == gold, "conf": max(b), "same": len(picks) == 1,
                             "base": b, "ens": e, "gold": gold})
        n = len(recs_out)
        res: Dict[str, Any] = {"n": n, "options_mean": _r(sum(len(x["base"]) for x in recs_out) / max(1, n)),
                               "acc_first_order": _r(sum(x["base_ok"] for x in recs_out) / max(1, n)),
                               "acc_mean_of_orders": _r(sum(x["ens_ok"] for x in recs_out) / max(1, n)),
                               "same_answer_every_order": _r(sum(x["same"] for x in recs_out) / max(1, n))}
        gate: Dict[str, Any] = {}
        for tau in TAUS:
            row = {}
            for h in (0, 1):
                xs = [x for x in recs_out if x["half"] == h]
                ok = [x["ens_ok"] if x["conf"] < tau else x["base_ok"] for x in xs]
                row[f"half{h}"] = {"acc": _r(sum(ok) / max(1, len(ok))), "re_asked": _r(sum(x["conf"] < tau for x in xs) / max(1, len(xs)))}
            gate[str(tau)] = row
        res["gated"] = gate
        best = max(TAUS, key=lambda t: (gate[str(t)]["half0"]["acc"], -gate[str(t)]["half0"]["re_asked"]))
        res["gated_chosen_on_half0"] = {"tau": best, "half1": gate[str(best)]["half1"],
                                        "half1_first_order_acc": _r(sum(x["base_ok"] for x in recs_out if x["half"] == 1) /
                                                                    max(1, sum(x["half"] == 1 for x in recs_out)))}
        out[name] = res
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--name", default="")
    ap.add_argument("--first", type=int, default=215)
    ap.add_argument("--last", type=int, default=310)
    ap.add_argument("--orders", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="records per domain, for a quick run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--holdout-only", action="store_true", help="only the probe holdout (models trained with --holdout never saw it)")
    a = ap.parse_args(argv)
    pool = load_pool(a.first, a.last)
    if a.holdout_only:
        pool = [r for r in pool if pr.in_probe_holdout(pr.record_key(r))]
    if a.limit:
        by: Dict[str, List[Record]] = defaultdict(list)
        for r in pool:
            by[r.domain].append(r)
        pool = [r for rs in by.values() for r in rs[:a.limit]]
    twist = load_twists(a.first, a.last)
    t0 = time.perf_counter()
    m = Model(a.model_dir, a.device)
    rep = {"model": a.model_dir, "batches": [a.first, a.last], "orders": a.orders, "holdout_only": a.holdout_only,
           "safety": probe_safety(m, pool, twist), "content": probe_content(m, pool, twist), "tools": probe_tools(m, pool, a.orders)}
    rep["model_calls"], rep["seconds"] = m.calls, round(time.perf_counter() - t0, 1)
    name = a.name or Path(a.model_dir).name
    out = ROOT / "reports" / f"leeth3_probe_{name}.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    s, c, t = rep["safety"], rep["content"], rep["tools"]
    print(f"== {name}  ({rep['seconds']} s, {m.calls} passes)")
    print("safety  auc base/all-phrasings", s["base"]["auc"], s["all_phrasings_mean"]["auc"], "| phrasing spread", s["phrasing_spread_mean"])
    print("        without user words: mean shift", s["without_user_words"]["mean_abs_shift"], "| words twins", s["without_user_words"]["mean_abs_shift_words_twins"],
          "| auc with/without", s["without_user_words"]["with_words"]["auc"], s["without_user_words"]["without_words"]["auc"])
    for k, v in s["twins"].items():
        print(f"        twins {k:<28} {v['base']['ordering']}  (n {v['base']['n_pairs']}, 90% {v['base']['ci90']})")
    print("content pairs with text / question only", c["pairs_with_text"]["ordering"], c["pairs_question_only"]["ordering"], "n", c["pairs_with_text"]["n_pairs"])
    for k, v in t.items():
        print(f"tools {k:<8} acc first/mean-of-orders {v['acc_first_order']} {v['acc_mean_of_orders']} | same every order {v['same_answer_every_order']}"
              f" | gated tau {v['gated_chosen_on_half0']['tau']}: half1 {v['gated_chosen_on_half0']['half1']} vs {v['gated_chosen_on_half0']['half1_first_order_acc']}")
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
