"""Turn a Leeth 2 evaluation (final_test.json from the Kaggle run) into a readable comparison with Jev and with the old Leeth.

    python scripts/leeth2_report.py <run_dir> [--out report.md]

Honesty rules baked in: every row says whether the model was trained on that task's training split (in-domain),
never saw the task at all (held-out), or the data is evaluation-only. Jev numbers are the vendor's / third-party
published figures and use a different protocol (zero-shot, 122 languages for Belebele...), so they are context, not a verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# (label, [task ids], primitive, metric, Jev figure or None, status)
ROWS: List[Tuple[str, List[str], str, str, Optional[float], str]] = [
    ("IMDB sentiment", ["imdb"], "choice", "acc", 96.5, "in-domain"),
    ("SST-2 sentiment", ["sst2"], "choice", "acc", 96.4, "in-domain"),
    ("Language ID (20 languages)", ["langid"], "choice", "acc", 99.6, "in-domain"),
    ("AG News topics", ["ag_news"], "choice", "acc", None, "in-domain"),
    ("Banking77 intents (77 options)", ["banking77"], "choice", "acc", 87.0, "HELD-OUT"),
    ("CLINC150 intents (150 options)", ["clinc150"], "choice", "acc", None, "HELD-OUT"),
    ("Emotion (6 classes)", ["emotion"], "choice", "acc", 58.5, "HELD-OUT"),
    ("DBpedia-14 topics", ["dbpedia14"], "choice", "acc", None, "HELD-OUT"),
    ("MNLI", ["mnli"], "choice", "acc", None, "in-domain"),
    ("XNLI Russian", ["xnli_ru"], "choice", "acc", None, "in-domain"),
    ("ANLI round 3 (adversarial)", ["anli"], "choice", "acc", None, "eval-only"),
    ("ARC Easy+Challenge", ["arc_easy", "arc_challenge"], "choice", "acc", 98.8, "in-domain train split"),
    ("HellaSwag", ["hellaswag"], "choice", "acc", 95.5, "in-domain train split"),
    ("MMLU (4 options)", ["mmlu_test"], "choice", "acc", 91.8, "knowledge: only aux_train seen"),
    ("MMLU-Pro (10 options)", ["mmlu_pro"], "choice", "acc", None, "eval-only"),
    ("Belebele Russian", ["belebele_ru"], "choice", "acc", None, "eval-only"),
    ("Belebele English", ["belebele_en"], "choice", "acc", 86.7, "eval-only (Jev: mean of 122 languages)"),
    ("XStoryCloze Russian", ["xstory_ru"], "choice", "acc", None, "HELD-OUT"),
    ("Toxic-Chat (toxic?)", ["toxic_chat"], "noul", "acc", None, "eval-only"),
    ("STS-B similarity (Pearson)", ["stsb"], "score", "pearson", None, "in-domain"),
    ("SummEval ratings (Pearson)", ["summeval"], "score", "pearson", 0.56, "eval-only (Jev: 0.27-0.56)"),
    ("SQuAD 2.0 span F1", ["squad2"], "extract", "f1", None, "in-domain"),
    ("XQuAD Russian span F1", ["xquad_ru"], "extract", "f1", None, "HELD-OUT"),
    ("Argent prod battery (109 hand-written requests)", ["prod_battery"], "choice", "acc", None, "HELD-OUT; previous prod: 89.0% pipeline / 59.6% legacy model"),
    ("Games: snake/maze/treasure/tic-tac-toe", ["game_snake", "game_maze", "game_treasure", "game_ttt"], "choice", "acc", None, "procedural"),
    ("NPC: follow a written policy", ["npc_policy"], "choice", "acc", None, "procedural"),
    ("NPC: in-character reply", ["npc_persona"], "choice", "acc", None, "procedural"),
    ("Sorting: files/tickets/ordering", ["sort_files", "sort_tickets", "sort_order"], "choice", "acc", None, "procedural"),
    ("Browser: pick the element", ["browser_element"], "choice", "acc", None, "procedural"),
    ("Tools: pick the tool (seen tools)", ["tool_pick"], "choice", "acc", None, "procedural"),
    ("Tools: pick the tool (UNSEEN tools)", ["tool_pick_unseen"], "choice", "acc", None, "HELD-OUT tools"),
]


def pool(rep: Dict[str, Any], tasks: List[str], ptype: str, metric: str) -> Tuple[Optional[float], int]:
    num, den = 0.0, 0
    for t in tasks:
        r = rep.get(f"task:{t}", {}).get(ptype)
        if not r or not r.get("n"):
            continue
        n = r["n"] if metric != "f1" else r.get("has_ans", r["n"])
        v = r.get(metric)
        if v is None or v != v:
            continue
        num += float(v) * n
        den += n
    return (num / den if den else None), den


def calibration(rep: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for p in ("choice", "noul"):
        r = rep.get(f"ptype:{p}", {}).get(p)
        if r:
            out[f"{p}_ece"] = r["ece"]
            out[f"{p}_brier"] = r["brier"]
            out[f"{p}_acc"] = r["acc"]
    return out


def render(rep: Dict[str, Any], run: Dict[str, Any]) -> str:
    lines = ["# Leeth 2 — results", ""]
    lines.append(f"Run: {run.get('steps', '?')} steps, {run.get('seconds', 0) // 60} min on Kaggle; temperatures {run.get('temperatures', {})}")
    lines += ["", "| Task | Leeth 2 | n | Jev (published) | Status |", "|---|---|---|---|---|"]
    for label, tasks, ptype, metric, jev, status in ROWS:
        v, n = pool(rep, tasks, ptype, metric)
        if v is None:
            continue
        ours = f"{v * 100:.1f}%" if metric in ("acc", "f1") else f"{v:.3f}"
        j = "" if jev is None else (f"{jev:.1f}%" if metric in ("acc", "f1") else f"{jev:.2f}")
        lines.append(f"| {label} | **{ours}** | {n} | {j} | {status} |")
    cal = calibration(rep)
    if cal:
        lines += ["", "## Calibration (all tasks pooled)", "", "| | accuracy | ECE | Brier |", "|---|---|---|---|"]
        for p in ("choice", "noul"):
            if f"{p}_ece" in cal:
                lines.append(f"| {p} | {cal[f'{p}_acc'] * 100:.1f}% | {cal[f'{p}_ece']:.3f} | {cal[f'{p}_brier']:.3f} |")
        lines.append("")
        lines.append("Jev reports a pooled Choice ECE of 0.028.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--file", default="final_test.json")
    a = ap.parse_args()
    d = Path(a.run_dir)
    rep = json.loads((d / a.file).read_text(encoding="utf-8"))
    run = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else {}
    md = render(rep, run)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
