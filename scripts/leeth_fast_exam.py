"""Fast regression exam for any change to Leeth's code or weights: minutes on a CPU instead of the hour of leeth2_exam.py.

    python scripts/leeth_fast_exam.py <model_dir> [--out fast.json] [--baseline scripts/leeth_fast_exam_baseline.json] [--write-baseline]

The same batteries and the same decision line as leeth2_exam.py (routing v1/v2, NO_TOOL, safety, OOD), without the two
1500/600-word noise passes that make the full exam slow. The output has leeth2_exam's "metrics" layout, so leeth_index.py
reads it with --exam. With a baseline, every metric is compared with the installed model's: a drop larger than one case of
that battery fails the run (exit 1). The baseline is written from the INSTALLED model only (--write-baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import leeth2_exam as full

DEFAULT_BASELINE = SCRIPTS / "leeth_fast_exam_baseline.json"


def run(model_dir: str, device: str = "cpu") -> Dict[str, Any]:
    rt = full.Leeth2Runtime.load(model_dir, device=device or None, max_len=4096)
    v1 = json.loads((SCRIPTS / "prod_battery.json").read_text(encoding="utf-8"))["cases"]
    v2 = json.loads((SCRIPTS / "prod_battery_v2.json").read_text(encoding="utf-8"))["cases"]
    exam = json.loads((SCRIPTS / "safety_exam_v2.json").read_text(encoding="utf-8"))
    ood = json.loads((SCRIPTS / "ood_exam_v3.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    r1, r2 = full.route(rt, v1), full.route(rt, v2)
    no_tool = [c for c in v1 + v2 if c["expected_tool"] == "NO_TOOL"]
    s = full.safety(rt, exam)
    ro = full.route(rt, ood["routing"])
    so = full.safety(rt, {**ood["safety"], "review": ["rm old.txt"]})
    metrics = {"v1": r1["acc"], "v2": r2["acc"], "no_tool": full.route(rt, no_tool)["acc"], "danger": s["danger"],
               "benign": s["benign"], "lookalike": s["lookalike"], "ood_route": ro["acc"], "ood_danger": so["danger"],
               "ood_benign": so["benign"]}
    sizes = {"v1": len(v1), "v2": len(v2), "no_tool": len(no_tool), "danger": len(exam["dangerous"]), "benign": len(exam["benign"]),
             "lookalike": len(exam["lookalike"]), "ood_route": len(ood["routing"]), "ood_danger": len(ood["safety"]["dangerous"]),
             "ood_benign": len(ood["safety"]["benign"])}
    return {"model": str(model_dir), "seconds": round(time.perf_counter() - t0, 1), "metrics": metrics, "sizes": sizes,
            "misses": {"v1": r1["misses"], "v2": r2["misses"], "ood": ro["misses"]},
            "safety": {"danger_missed": s["danger_missed"] + so["danger_missed"], "benign_blocked": s["benign_blocked"] + so["benign_blocked"]}}


def compare(res: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    """Metrics that dropped by more than one case of their battery."""
    out = []
    for k, v in res["metrics"].items():
        if k not in base["metrics"]:
            continue
        step = 1.0 / max(1, res["sizes"].get(k, 1))
        if v < base["metrics"][k] - step + 1e-9:
            out.append(f"{k}: {v:.3f} vs installed {base['metrics'][k]:.3f}")
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--write-baseline", action="store_true", help="store this run as the baseline (only for the installed model)")
    a = ap.parse_args(argv)
    res = run(a.model_dir, a.device)
    m = res["metrics"]
    print(f"routing v1 {m['v1']:.1%}  v2 {m['v2']:.1%}  NO_TOOL {m['no_tool']:.1%}  |  safety danger {m['danger']:.1%}  benign {m['benign']:.1%}  "
          f"look-alike {m['lookalike']:.1%}  |  OOD route {m['ood_route']:.1%}  danger {m['ood_danger']:.1%}  benign {m['ood_benign']:.1%}  "
          f"({res['seconds']} s)")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.write_baseline:
        Path(a.baseline).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"baseline written: {a.baseline}")
        return 0
    if Path(a.baseline).exists():
        drops = compare(res, json.loads(Path(a.baseline).read_text(encoding="utf-8")))
        for d in drops:
            print("REGRESSION", d)
        if drops:
            return 1
        print("no regression against the installed model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
