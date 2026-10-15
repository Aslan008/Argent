"""Temperature for long option lists, fitted on a library the model has never seen.

    python scripts/leeth21_calibrate.py SRC_DIR OUT_DIR [--device cuda] [--n 120]

A model trained on lists of 6-12 options is over-confident on lists of 64-255, and one trained on 255 near-duplicates is the other way round
(round 1 of Leeth 2.1: ECE 0.21 at 255 options with the trained temperature, 0.08 with one fitted on a separate draw).  The choice head has
a second temperature for that (``temp_choice_big``, used for lists longer than ``big_k``); this script fits it.

The library is a third verb x object vocabulary, disjoint from the one the model trained on (data/synth/bigtools.py) and from the exam's
(scripts/leeth21_exam.py), so the number is a measure of "a list of tools the model has not met", which is what an agent attached to a new
server presents.  Half of the draws fit the temperature, the other half report ECE before and after, so the file says what the fit bought.

OUT_DIR is a copy of SRC_DIR with ``temp_choice_big`` and ``big_k`` written into leeth2_config.json; ``calibration_big.json`` lands beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proj"))

from src.leeth2 import metrics  # noqa: E402
from src.leeth2.calibrate import _fit_t  # noqa: E402
from src.leeth2.evaluate import predict  # noqa: E402
from src.leeth2.runtime import Leeth2Runtime  # noqa: E402
from src.leeth2.schema import Sample  # noqa: E402

BIG_K = 64
VERBS = {
    "attach": ["add as an attachment to", "clip onto", "append a file to", "link a file with"], "compress": ["shrink", "make smaller", "zip up", "pack tightly"],
    "decrypt": ["unscramble", "make readable again", "unlock the contents of", "decode"], "import": ["bring in", "load from outside", "pull in", "ingest"],
    "rate": ["give a score to", "grade", "evaluate with stars", "mark out of five"], "subscribe": ["follow", "sign me up for updates from", "keep me posted on", "watch"],
    "transcribe": ["write down what is said in", "turn into text", "produce subtitles for", "type out"], "anonymize": ["strip personal details from", "scrub names out of", "make anonymous", "mask identities in"],
    "snapshot": ["take a point-in-time copy of", "capture the state of", "freeze a copy of", "photograph the state of"], "unlock": ["let me back into", "release the lock on", "open up access to", "lift the restriction on"],
    "split": ["break into parts", "divide", "cut in two", "separate into pieces"], "reconcile": ["match up against the bank", "square away", "make the books agree for", "cross-check the totals of"],
}
OBJECTS = [
    "ledger", "lease", "playbook", "notebook", "pipeline", "inventory", "roster", "grant", "patent", "petition", "manifest", "catalogue", "dataset", "glossary", "newsletter",
    "fleet vehicle", "tenant", "portfolio", "prescription", "warranty", "transcript", "tutorial", "vault", "dns zone",
]
SUFFIX = ["", "", " from the spring term", " for the Oslo branch", " that Priya sent over", " for the audit"]
TEMPLATES = ["{syn} the {obj}{suf}", "can you {syn} the {obj}{suf}?", "{syn} my {obj}{suf} please", "I want to {syn} this {obj}{suf}"]
QUESTION = "Which tool should be called?"


def library() -> List[Tuple[str, str]]:
    return [(f"{v}_{o.replace(' ', '_')}", f"{v.capitalize()} a {o}") for v in VERBS for o in OBJECTS]


def draws(K: int, n: int, seed: int) -> List[Sample]:
    lib = library()
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ti = rng.randrange(len(lib))
        name, _ = lib[ti]
        verb, obj = name.split("_", 1)[0], name.split("_", 1)[1]
        near = [j for j, (nm, _) in enumerate(lib) if j != ti and (nm.split("_", 1)[1] == obj or nm.split("_", 1)[0] == verb)]
        rng.shuffle(near)
        rest = [j for j in range(len(lib)) if j != ti and j not in near]
        rng.shuffle(rest)
        ids = [ti] + (near[: (K - 1) // 2] + rest)[: K - 1]
        rng.shuffle(ids)
        req = rng.choice(TEMPLATES).format(syn=rng.choice(VERBS[verb]), obj=obj.replace("_", " "), suf=rng.choice(SUFFIX))
        out.append(Sample("calib_big", "tools", "en", "choice", f"User request: “{req}”", QUESTION, [f"{lib[j][0]} — {lib[j][1]}" for j in ids], ids.index(ti)))
    return out


def raw(rt: Leeth2Runtime, ss: Sequence[Sample]):
    preds = predict(rt.model, rt.encoder, list(ss), device=rt.device, amp=rt.amp, calibrated=False, tokens_per_batch=24000)
    return [(p.raw["logits_shown"], p.raw["perm"].index(s.hard_choice())) for s, p in zip(ss, preds)]


def ece_at(rows, t: float) -> float:
    tops, hits = [], []
    for lg, y in rows:
        q = metrics.softmax(lg / t)
        tops.append(float(q.max()))
        hits.append(float(int(q.argmax()) == y))
    return round(float(metrics.ece(tops, hits)), 4)


def nll_at(rows, t: float) -> float:
    return float(np.mean([-np.log(max(metrics.softmax(lg / t)[y], 1e-12)) for lg, y in rows]))


def fit_big_temperature(rt: Leeth2Runtime, n: int) -> Dict[str, Any]:
    t_small = float(rt.model.cfg.temp_choice)
    fit_rows, check_rows = [], {}
    for K in (96, 160, 255):
        fit_rows += raw(rt, draws(K, n, 5100 + K))
    for K in (16, 64, 128, 255):
        check_rows[K] = raw(rt, draws(K, max(60, n // 2), 8100 + K))
    # logits_shown are before any temperature: the fitted value is the divisor the runtime applies to lists longer than BIG_K
    t_big = _fit_t(lambda lt: nll_at(fit_rows, float(np.exp(lt))))
    report: Dict[str, Any] = {"temp_choice": round(t_small, 4), "temp_choice_big": round(float(t_big), 4), "big_k": BIG_K, "n_fit": len(fit_rows), "by_k": {}}
    for K, rows in check_rows.items():
        t_used = t_big if K > BIG_K else t_small
        report["by_k"][str(K)] = {"acc": round(float(np.mean([int(np.argmax(lg) == y) for lg, y in rows])), 4), "n": len(rows),
                                  "ece_before": ece_at(rows, t_small), "ece_after": ece_at(rows, t_used),
                                  "nll_before": round(nll_at(rows, t_small), 4), "nll_after": round(nll_at(rows, t_used), 4)}
    return report


def write_calibrated(src: Path, out: Path, report: Dict[str, Any]) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            if f.name == "leeth2_config.json":
                continue
            try:
                os.symlink(f.resolve(), out / f.name)
            except OSError:
                shutil.copy2(f, out / f.name)
        elif f.is_dir():
            shutil.copytree(f, out / f.name)
    cfg = json.loads((src / "leeth2_config.json").read_text(encoding="utf-8"))
    cfg["temp_choice_big"] = report["temp_choice_big"]
    cfg["big_k"] = report["big_k"]
    (out / "leeth2_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "calibration_big.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=120)
    a = ap.parse_args()
    rt = Leeth2Runtime.load(a.src, device=a.device, max_len=4096)
    report = fit_big_temperature(rt, a.n)
    del rt
    write_calibrated(Path(a.src), Path(a.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
