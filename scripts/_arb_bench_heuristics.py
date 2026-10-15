"""Micro-benchmark: physics vs vsids heuristics on the arbitrator's real CNF workload.

Both heuristics must agree on every verdict (correctness cross-check); latency
and conflict counts show whether the author's ChronoPhysics earns its keep on
this class of tiny structured instances.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.leeth.arbitrator as arb_mod
from src.leeth.arbitrator import CDCLArbitrator

CASES = [
    ("SAT read_file", dict(tool="read_file", need_tool=True, param_text="config.py", query="прочитай config.py")),
    ("SAT calculate", dict(tool="calculate", need_tool=True, param_text="512 / 8", query="сколько будет 512 / 8")),
    ("SAT list_dir", dict(tool="list_directory", need_tool=True, param_text="src", query="покажи содержимое папки src")),
    ("SAT chat", dict(tool="NO_TOOL", need_tool=False, param_text="", query="продолжай")),
    ("SAT unfamiliar", dict(tool="UNFAMILIAR", need_tool=True, param_text="Haskell-модуль", query="отрефактори этот Haskell-модуль")),
    ("UNSAT destructive", dict(tool="run_command", need_tool=True, param_text="rm -rf /", query="rm -rf /")),
    ("UNSAT need mismatch", dict(tool="read_file", need_tool=False, param_text="config.py", query="прочитай config.py")),
    ("UNSAT empty param", dict(tool="read_file", need_tool=True, param_text="", query="прочитай")),
    ("UNSAT wrong param type", dict(tool="calculate", need_tool=True, param_text="config.py", query="прочитай config.py")),
]

WARMUP, ITERS = 5, 50


def bench(heuristic):
    arb = CDCLArbitrator(heuristic=heuristic)
    arb._log_shadow = lambda *a, **kw: None  # no file I/O in timing loop
    results = {}
    for label, kwargs in CASES:
        for _ in range(WARMUP):
            arb.arbitrate(**kwargs)
        lats, confs, last = [], [], None
        for _ in range(ITERS):
            r = arb.arbitrate(**kwargs)
            lats.append(r.latency_ms)
            confs.append(r.conflicts)
            last = r
        lats.sort()
        results[label] = {
            "sat": last.sat, "core": last.core,
            "mean": statistics.mean(lats), "p50": lats[len(lats) // 2],
            "p95": lats[max(0, int(len(lats) * 0.95) - 1)],
            "conflicts": statistics.mean(confs),
        }
    return results


out = {}
for h in ("physics", "vsids"):
    out[h] = bench(h)
    print(f"=== {h} ===")
    for label, s in out[h].items():
        print(f"  {label:22s} {'SAT ' if s['sat'] else 'UNSAT'} "
              f"mean={s['mean']:6.2f}ms p50={s['p50']:6.2f} p95={s['p95']:6.2f} "
              f"conflicts={s['conflicts']:5.1f} core={s['core'] or '-'}")

mismatch = [l for l in out["physics"] if out["physics"][l]["sat"] != out["vsids"][l]["sat"]]
print(f"\nverdict mismatches physics vs vsids: {mismatch or 'none'}")

p_unsat = [out["physics"][l] for l in out["physics"] if not out["physics"][l]["sat"]]
print(f"physics UNSAT core samples: {[s['core'] for s in p_unsat][:2]}")