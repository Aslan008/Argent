# -*- coding: utf-8 -*-
"""scripts/run_prod_battery.py — Фаза B3: замер базовой линии на прод-батарее.

Зачем: фиксируем честную точку отсчёта качества choice-роутинга на реальных
кейсах (не синтетике). Все последующие фазы сравниваются с этой линией —
так видно, улучшает ли дообучение дело или только синтетику.

Запускает dispatcher.triage по всем кейсам (быстрый инференс, НЕ обучение),
считает accuracy + ECE + Brier, пишет baseline-артефакт.

Запуск: python scripts/run_prod_battery.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leeth.dispatcher import dispatcher

BATTERY_PATH = Path("scripts/prod_battery.json")
BASELINE_PATH = Path(".argent/artifacts/prod_battery_baseline.md")


def main() -> None:
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]
    queries = [c["query"] for c in cases]

    t0 = time.perf_counter()
    infos = dispatcher.triage(queries)
    wall = time.perf_counter() - t0

    correct = 0
    per_tool: dict = {}
    confs: list = []
    corrs: list = []
    misses: list = []

    for case, info in zip(cases, infos):
        expected = case["expected_tool"]
        # The MODEL's answer = raw choice-head output (gates may delegate
        # afterwards by design — that is pipeline behavior, not model error).
        raw = (info.get("tool") or {}).get("choice") or "DELEGATE"
        ok = raw == expected
        correct += ok
        bucket = per_tool.setdefault(expected, {"n": 0, "ok": 0})
        bucket["n"] += 1
        bucket["ok"] += ok
        conf = (info.get("tool") or {}).get("confidence")
        if conf is not None:
            confs.append(float(conf))
            corrs.append(1.0 if ok else 0.0)
        else:
            confs.append(0.0)
            corrs.append(1.0 if expected in ("UNFAMILIAR", "NO_TOOL") else 0.0)
        if not ok:
            dispatch = info.get("dispatch")
            pipeline = (dispatch or {}).get("tool") if isinstance(dispatch, dict) else (
                getattr(dispatch, "tool_name", None) if dispatch else None)
            misses.append({
                "query": case["query"], "expected": expected, "raw": raw,
                "pipeline": pipeline or "DELEGATE",
                "skip": info.get("skip_reason"),
            })

    n = len(cases)
    acc = correct / n
    # ECE over dispatched confidence (delegations counted as conf=0)
    n_bins = 10
    ece = 0.0
    brier = 0.0
    if confs:
        bins: list = [[] for _ in range(n_bins)]
        for c, o in zip(confs, corrs):
            bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
        for b in bins:
            if b:
                avg_c = sum(c for c, _ in b) / len(b)
                acc_b = sum(o for _, o in b) / len(b)
                ece += len(b) / len(confs) * abs(acc_b - avg_c)
        brier = sum((c - o) ** 2 for c, o in zip(confs, corrs)) / len(confs)

    lines = [
        "# Prod battery baseline",
        "",
        f"- Cases: {n}",
        f"- **Accuracy: {acc:.1%}** ({correct}/{n})",
        f"- ECE (dispatched): {ece:.4f}",
        f"- Brier: {brier:.4f}",
        f"- Wall time: {wall:.1f}s ({wall / n * 1000:.0f} ms/case)",
        "",
        "## Per-tool accuracy",
        "",
        "| Expected tool | n | ok |",
        "|---|---|---|",
    ]
    for tool, b in sorted(per_tool.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {tool} | {b['n']} | {b['ok']} ({b['ok'] / b['n']:.0%}) |")
    lines += ["", "## Misses", ""]
    for m in misses:
        lines.append(f"- {m['query']!r}: expected **{m['expected']}**, raw **{m['raw']}**, pipeline **{m['pipeline']}** (skip: {m['skip']})")

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Accuracy: {acc:.1%} ({correct}/{n}) | ECE: {ece:.4f} | Brier: {brier:.4f}")
    print(f"Baseline written -> {BASELINE_PATH}")
    print(f"Misses: {len(misses)} (see {BASELINE_PATH})")


if __name__ == "__main__":
    main()