"""Compare two Leeth 2 training reports (final_test.json) task by task: what a new model gained, and above all what it LOST.

    python scripts/leeth2_compare_reports.py old/final_test.json new/final_test.json [--min-n 100] [--tol 0.02]

Every task present in both reports is compared on its headline metric (choice/noul accuracy, score MAE and extract F1);
the table lists the regressions larger than ``--tol`` first.  Task splits are seeded and shared between runs that use the same
build, so the same samples are scored; when the mix differs, treat small differences as noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# (ptype, metric, higher_is_better)
METRICS: List[Tuple[str, str, bool]] = [("choice", "acc", True), ("noul", "acc", True), ("score", "mae100", False), ("extract", "f1", True)]


def load(path: str) -> Dict[str, Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(old: Dict, new: Dict, min_n: int) -> List[Tuple[str, str, str, float, float, float, int]]:
    out = []
    for key in sorted(set(old) & set(new)):
        if not key.startswith("task:"):
            continue
        for ptype, metric, higher in METRICS:
            a, b = old[key].get(ptype), new[key].get(ptype)
            if not a or not b or a.get("n", 0) < min_n or b.get("n", 0) < min_n or metric not in a or metric not in b:
                continue
            if any(v != v for v in (a[metric], b[metric])):                 # NaN: a task with nothing to score
                continue
            delta = (b[metric] - a[metric]) * (1 if higher else -1)       # positive = the new model is better
            scale = 1.0 if higher else 0.01                                # mae100 is on a 0..100 scale: 1 point = 0.01
            out.append((key[5:], ptype, metric, a[metric], b[metric], delta * scale if not higher else delta, min(a["n"], b["n"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    rs = rows(load(a.old), load(a.new), a.min_n)
    worse = sorted([r for r in rs if r[5] < -a.tol], key=lambda r: r[5])
    better = sorted([r for r in rs if r[5] > a.tol], key=lambda r: -r[5])
    print(f"{len(rs)} comparable (task, primitive) pairs; worse by more than {a.tol}: {len(worse)}; better by more than {a.tol}: {len(better)}")
    for title, group in (("REGRESSIONS", worse[:40]), ("GAINS", better[:25])):
        print(f"\n{title}")
        for task, ptype, metric, x, y, d, n in group:
            print(f"  {task:<28} {ptype:<8} {metric:<7} {x:8.3f} -> {y:8.3f}   {d:+.3f}  (n={n})")
    for ptype, metric, higher in METRICS:
        v = [r for r in rs if r[1] == ptype]
        if v:
            print(f"mean over {ptype} tasks ({metric}): {sum(r[3] for r in v) / len(v):.3f} -> {sum(r[4] for r in v) / len(v):.3f}")


if __name__ == "__main__":
    main()
