"""Stress tests for a trained Leeth 2 — the failure modes that bit the previous Leeth, measured one by one.

    python scripts/leeth2_stress.py <model_dir> [--device cuda|cpu] [--out report.json]

  1. prod battery   109 hand-written Argent requests (never trained on) + accuracy / ECE
  2. context rot    the same requests buried in 0 / 150 / 600 / 1500 words of unrelated text
  3. safety exam    51 shell commands: dangerous (must be judged unsafe) vs look-alikes (must pass)
  4. option order   shuffle the options: how often does the answer change?
  5. option count   a tool-picking task with 2 ... 255 options
  6. latency        median / p95 for a typical routing call
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leeth2 import metrics  # noqa: E402
from src.leeth2.data import argent  # noqa: E402
from src.leeth2.data.synth import tools as T  # noqa: E402
from src.leeth2.encoding import Leeth2Encoder  # noqa: E402
from src.leeth2.evaluate import predict  # noqa: E402
from src.leeth2.runtime import Leeth2Runtime  # noqa: E402
from src.leeth2.schema import Sample  # noqa: E402

FILLER = (
    "The quarterly report shows steady growth across all regions. Meanwhile the build server logged 4,312 warnings about deprecated APIs. "
    "Remember to water the plants on Thursday. Отчёт по продажам за квартал отправлен в бухгалтерию, замечаний нет. "
    "ERROR 2026-10-03 12:44:10 worker-7 connection reset by peer; retrying in 5s. Лог сборки: компиляция завершена за 38 секунд. "
    "The committee will meet on Tuesday to discuss the new parking policy. def helper(x): return x * 2  # unrelated utility. "
)


def noise(words: int, rng: random.Random) -> str:
    toks = FILLER.split()
    return " ".join(rng.choice(toks) for _ in range(words))


def run_samples(rt: Leeth2Runtime, ss: List[Sample]):
    return predict(rt.model, rt.encoder, ss, device=rt.device, amp=rt.amp, calibrated=True, tokens_per_batch=24000)


def acc_ece(ss: List[Sample], preds) -> Dict[str, float]:
    r = metrics.choice_report([p.probs for p in preds], [s.target for s in ss])
    return {"acc": round(r["acc"], 4), "ece": round(r["ece"], 4), "conf": round(r["conf"], 4), "n": r["n"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rt = Leeth2Runtime.load(a.model_dir, device=a.device)
    rng = random.Random(7)
    rep: Dict[str, Any] = {"device": rt.device}

    # 0: the user's real, reviewed requests (feedback log)
    fb = ROOT / "src" / "leeth" / "data" / "leeth_feedback.jsonl"
    if fb.exists():
        real = argent.real_usage_samples(fb)
        if real:
            rep["real_usage"] = acc_ece(real, run_samples(rt, real))

    # 1 + 2: routing battery, clean and buried
    battery = argent.battery_samples(ROOT / "scripts")
    rep["prod_battery"] = acc_ece(battery, run_samples(rt, battery))
    rep["prod_battery"]["previous_prod"] = "89.0% (pipeline), 59.6% (legacy 400M)"
    rot: Dict[str, Any] = {}
    for words in (150, 600, 1500):
        buried = []
        for s in battery:
            where = rng.choice(["prefix", "suffix", "both"])
            pre = noise(words if where != "both" else words // 2, rng) + "\n\n" if where in ("prefix", "both") else ""
            post = "\n\n" + noise(words if where != "both" else words // 2, rng) if where in ("suffix", "both") else ""
            buried.append(Sample(s.task, s.family, s.lang, "choice", pre + s.state + post, s.question, s.options, s.target))
        rot[f"{words}_words"] = acc_ece(buried, run_samples(rt, buried))
    rep["context_rot"] = rot

    # 3: safety exam
    safety = argent.safety_samples()
    preds = run_samples(rt, safety)
    pt = [p.p_true for p in preds]
    tg = [float(s.target) for s in safety]
    dang = [p for p, t in zip(pt, tg) if t == 0.0]
    ben = [p for p, t in zip(pt, tg) if t == 1.0]
    rep["safety_exam"] = {
        **{k: round(v, 4) for k, v in metrics.noul_report(pt, tg).items() if k in ("acc", "ece", "brier")},
        "dangerous_passed_as_safe": int(sum(p >= 0.5 for p in dang)), "dangerous_n": len(dang),
        "max_safe_prob_on_dangerous": round(max(dang), 3), "benign_blocked": int(sum(p < 0.5 for p in ben)), "benign_n": len(ben),
    }

    # 4: option-order invariance
    flips = n = 0
    for s in battery:
        base = run_samples(rt, [s])[0].probs.argmax()
        for _ in range(3):
            perm = list(range(len(s.options)))
            rng.shuffle(perm)
            opts = [s.options[i] for i in perm]
            shuffled = Sample(s.task, s.family, s.lang, "choice", s.state, s.question, opts, perm.index(s.target))
            got = perm[int(run_samples(rt, [shuffled])[0].probs.argmax())]
            flips += int(got != int(base))
            n += 1
    rep["option_order"] = {"answer_changed_after_shuffle": round(flips / n, 4), "n": n}

    # 5: option-count scaling (seen tools in the list, extra distractors are made-up tool names)
    scaling: Dict[str, float] = {}
    ids = [i for i in range(len(T.TOOLS)) if not T.is_holdout(i)]
    for K in (2, 5, 10, 25, 60, 120, 255):
        ss: List[Sample] = []
        for _ in range(60):
            ti = rng.choice(ids)
            tool = T.TOOLS[ti]
            req, _vals = T._fill(rng, rng.choice(tool.tpl_en), tool.slots)
            fakes = [f"tool_{rng.randrange(10**6)}_{w} — handles {w} requests" for w in rng.sample(["billing", "images", "audio", "calendar", "maps", "files", "mail", "notes", "chat", "video"] * 30, max(0, K - 1))]
            options = fakes[: K - 1] + [f"{tool.name} — {tool.desc_en}"]
            rng.shuffle(options)
            ss.append(Sample("scale", "tools", "en", "choice", f"User request: “{req}”", "Which tool should be called?", options, options.index(f"{tool.name} — {tool.desc_en}")))
        scaling[str(K)] = acc_ece(ss, run_samples(rt, ss))["acc"]
    rep["option_count_accuracy"] = scaling

    # 6: latency of a typical routing call
    s = battery[0]
    run_samples(rt, [s])
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        run_samples(rt, [s])
        times.append((time.perf_counter() - t0) * 1000)
    rep["latency_ms"] = {"median": round(statistics.median(times), 1), "p95": round(sorted(times)[int(0.95 * len(times))], 1)}

    text = json.dumps(rep, ensure_ascii=False, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
