"""The install gates for a Leeth 2 candidate -- written down BEFORE the candidates' results existed, so they cannot drift to fit them.

    python scripts/leeth2_gate.py --exam exam_final.json \\
        --real real_final.json --real-incumbent real_installed_v31.json \\
        --arena arena_report.json --arena-model final \\
        --games final_report.json --games-baseline merge_report.json

Four sections, each a list of PASS / FAIL lines; the verdict is the AND of all of them, and a section whose input was not given is a FAIL
("not measured"): a model is installed only when every gate was actually measured.

  exam    the synthetic exams of scripts/leeth2_exam.py: every gate there (routing, safety, look-alikes, context rot, out-of-distribution)
  real    the real-traffic exam: false-dispatch rate <= 3.6 % (what the installed v3.1 achieves), the "adjusted" count and tool recall
          no worse than the installed model's, no wrong tool taken
  arena   the same samples answered by the best of Laya's checkpoints: the floor the owner set ("Laya is the minimum, better where you
          can, not necessarily everywhere") -- at most two task groups significantly below Laya, none more than 0.10 below, better on
          average and on most groups, at least as well calibrated
  games   the closed-loop games and sites against the previous best (merge-v2) on the original environments (no significant regression,
          no harm on the purchase/consent sites), above random on every new and every never-trained environment, RTS not worse
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

Gate = Tuple[bool, str, str]

# ---- the thresholds (change them only with a written reason, never after looking at a candidate) -----------------------------------
FALSE_DISPATCH_MAX = 0.036
ARENA_MAX_WORSE_GROUPS = 2
ARENA_MAX_DEFICIT = 0.10
ARENA_MIN_MEAN_GAIN = 0.02
ARENA_MIN_WINS = 5
ARENA_Z = 1.64                                  # one-sided 5 %: "significantly below"
GAME_Z = 2.0
GAME_MAX_REGRESSIONS = 1
GAME_MAX_DROP = 0.15
HARM_MAX = 0.02
RTS_MEAN_DROP = 0.06
RTS_BOT_DROP = 0.25
MASSIVE13 = ["ru", "de", "fr", "es", "pt", "it", "pl", "tr", "ja", "zh", "ko", "nl", "sv"]
XNLI14 = ["ru", "de", "fr", "es", "zh", "ar", "tr", "vi", "th", "hi", "bg", "el", "sw", "ur"]
ARENA_GROUPS: Dict[str, List[str]] = {
    "massive_en": ["massive_en"], "massive_13": [f"massive_{c}" for c in MASSIVE13], "xnli_en": ["xnli_en"], "xnli_14": [f"xnli_{c}" for c in XNLI14],
    "banking77": ["banking77_hf"], "sst5": ["sst5"], "clinc150": ["clinc150"], "mnli": ["mnli"], "sst2": ["sst2"],
}
RATE_KEYS = ("success", "solved", "reached", "won")


def read_json(path: Any) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def se_diff(p1: float, n1: float, p2: float, n2: float) -> float:
    """Standard error of the difference of two proportions (clipped away from 0/1 so a perfect score still has an error bar)."""
    p1, p2 = min(max(p1, 0.02), 0.98), min(max(p2, 0.02), 0.98)
    return math.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))


# ---- 1. synthetic exams ---------------------------------------------------------------------------------------------------------------
def exam_gates(exam_path: Optional[str]) -> List[Gate]:
    if not exam_path:
        return [(False, "exam", "not measured")]
    tree = ast.parse((ROOT / "scripts" / "leeth2_exam.py").read_text(encoding="utf-8"))
    gates = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GATES" for t in n.targets))
    metrics = read_json(exam_path)["metrics"]
    return [(metrics[key] >= floor, f"exam: {name}", f"got {metrics[key]:.3f}") for name, (key, floor) in gates.items()]


# ---- 2. real traffic ------------------------------------------------------------------------------------------------------------------
def _defensible() -> Callable[[str, str], bool]:
    spec = importlib.util.spec_from_file_location("leeth2_real_exam_for_gate", ROOT / "scripts" / "leeth2_real_exam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.defensible


def real_stats(path: str, labels: Dict[str, Any], defensible: Callable[[str, str], bool]) -> Dict[str, float]:
    data = read_json(path)
    tools: Dict[str, str] = labels.get("tools", {})
    skip = set(labels.get("skip", []))
    bare = [r for r in data["bare"] if r["query"] not in skip]
    delegate = [r for r in bare if r["query"] not in tools]
    labelled = [r for r in bare if r["query"] in tools]
    taken = [r for r in delegate if r["taken"]]
    right = [r for r in labelled if r["taken"] and r["tool"] == tools[r["query"]]]
    wrong = [r for r in labelled if r["taken"] and r["tool"] != tools[r["query"]]]
    n = max(1, len(delegate))
    stats = {"n": len(delegate), "false": len(taken), "rate": len(taken) / n,
             "adjusted": sum(1 for r in taken if not defensible(r["query"], r["tool"])),
             "recall": len(right) / max(1, len(labelled)), "wrong": len(wrong),
             "ctx_rate": sum(1 for r in data.get("ctx", []) if r["taken"]) / max(1, len(data.get("ctx", [])))}
    if "production" in data:                                     # the dispatcher's own triage: the model's gates plus its regex guards and argument checks
        prod = [r for r in data["production"] if r["query"] not in skip]
        p_delegate = [r for r in prod if r["query"] not in tools]
        want = lambda q: {"find_by_name": "search_files"}.get(tools[q], tools[q])         # noqa: E731  (the dispatcher renames find_by_name)
        p_labelled = [r for r in prod if r["query"] in tools]
        stats.update(prod_false=sum(1 for r in p_delegate if r["taken"]), prod_n=len(p_delegate),
                     prod_adjusted=sum(1 for r in p_delegate if r["taken"] and not defensible(r["query"], r["tool"])),
                     prod_recall=sum(1 for r in p_labelled if r["taken"] and r["tool"] == want(r["query"])) / max(1, len(p_labelled)),
                     prod_wrong=sum(1 for r in p_labelled if r["taken"] and r["tool"] != want(r["query"])))
    return stats


def real_gates(cand: Optional[str], incumbent: Optional[str], labels_path: Optional[str] = None,
               defensible: Optional[Callable[[str, str], bool]] = None) -> List[Gate]:
    if not cand or not incumbent:
        return [(False, "real", "not measured (need the candidate's and the installed model's real-exam JSON)")]
    labels = read_json(labels_path or ROOT / "scripts" / "real_exam_v4.json")
    defensible = defensible or _defensible()
    c, i = real_stats(cand, labels, defensible), real_stats(incumbent, labels, defensible)
    lines: List[Gate] = [
        (c["rate"] <= FALSE_DISPATCH_MAX, f"real: false-dispatch rate <= {FALSE_DISPATCH_MAX:.1%}", f"{c['rate']:.2%} ({c['false']}/{c['n']}); installed {i['rate']:.2%}"),
        (c["adjusted"] <= i["adjusted"], "real: adjusted false dispatches not above the installed model's", f"{c['adjusted']} vs {i['adjusted']}"),
        (c["recall"] >= i["recall"], "real: tool recall not below the installed model's", f"{c['recall']:.1%} vs {i['recall']:.1%}"),
        (c["wrong"] <= i["wrong"], "real: no more wrong tools than the installed model", f"{c['wrong']} vs {i['wrong']}"),
        (True, "real: with session context (information only)", f"{c['ctx_rate']:.2%} vs installed {i['ctx_rate']:.2%}"),
    ]
    if "prod_false" in c and "prod_false" in i:                  # what the owner meets in Argent: never worse than the installed model there either
        lines += [
            (c["prod_false"] <= i["prod_false"], "real: dispatcher-path false dispatches not above the installed model's", f"{c['prod_false']} vs {i['prod_false']} of {c['prod_n']}"),
            (c["prod_adjusted"] <= i["prod_adjusted"], "real: dispatcher-path adjusted false dispatches not above the installed model's", f"{c['prod_adjusted']} vs {i['prod_adjusted']}"),
            (c["prod_recall"] >= i["prod_recall"] - 1e-9, "real: dispatcher-path tool recall not below the installed model's", f"{c['prod_recall']:.1%} vs {i['prod_recall']:.1%}"),
            (c["prod_wrong"] <= i["prod_wrong"], "real: dispatcher-path wrong tools not above the installed model's", f"{c['prod_wrong']} vs {i['prod_wrong']}"),
        ]
    else:
        lines.append((False, "real: dispatcher-path numbers", "missing from the candidate's or the installed model's real-exam JSON (re-run the real exam)"))
    return lines


# ---- 3. the arena ---------------------------------------------------------------------------------------------------------------------
def _score(rep: Dict[str, Any], tasks: List[str], key: str) -> Optional[Tuple[float, int]]:
    num = den = 0.0
    for t in tasks:
        r = rep.get(t)
        if isinstance(r, dict) and key in r and r[key] == r[key]:
            n = r.get("n_common", r.get("n", 0)) if key == "acc_common" else r.get("n", 0)
            num, den = num + r[key] * n, den + n
    return (num / den, int(den)) if den else None


def _mean_ece(rep: Dict[str, Any]) -> Optional[float]:
    v = [r["ece"] for t, r in rep.items() if not t.startswith("_") and isinstance(r, dict) and r.get("ece") == r.get("ece") and "ece" in r]
    return sum(v) / len(v) if v else None


def arena_gates(path: Optional[str], model: str) -> List[Gate]:
    if not path:
        return [(False, "arena", "not measured")]
    rep = read_json(path)
    ours = rep["ours"].get(model)
    laya = {k: v for k, v in rep.get("laya", {}).items() if isinstance(v, dict) and "error" not in v and k != "laya_version"}
    if ours is None:
        return [(False, "arena", f"the report has no model {model!r}: {sorted(rep['ours'])}")]
    if not laya:
        return [(False, "arena", f"Laya did not run: {rep.get('laya')}")]
    rows, worse, deficits, gains, wins = [], 0, [], [], 0
    for group, tasks in ARENA_GROUPS.items():
        mine = _score(ours, tasks, "acc_common")
        theirs = [(name, s) for name, r in laya.items() if (s := _score(r, tasks, "acc")) is not None]
        if mine is None or not theirs:
            continue
        name, (best, n_l) = max(theirs, key=lambda x: x[1][0])
        diff = mine[0] - best
        z = diff / se_diff(mine[0], mine[1], best, n_l)
        worse += z < -ARENA_Z
        deficits.append(-diff)
        gains.append(diff)
        wins += diff > 0.01 + 1e-9
        rows.append(f"{group} {mine[0]:.3f} vs {best:.3f} ({name}) {diff:+.3f}")
    if not rows:
        return [(False, "arena", "no task group is present in both the model's and Laya's results")]
    lines: List[Gate] = [
        (worse <= ARENA_MAX_WORSE_GROUPS, f"arena: groups significantly below Laya <= {ARENA_MAX_WORSE_GROUPS}", f"{worse} of {len(rows)}"),
        (max(deficits) <= ARENA_MAX_DEFICIT, f"arena: no group more than {ARENA_MAX_DEFICIT} below Laya", f"worst deficit {max(deficits):+.3f}"),
        (sum(gains) / len(gains) >= ARENA_MIN_MEAN_GAIN, f"arena: mean over groups at least {ARENA_MIN_MEAN_GAIN} above Laya", f"{sum(gains) / len(gains):+.3f}"),
        (wins >= min(ARENA_MIN_WINS, len(rows)), f"arena: ahead of Laya on >= {ARENA_MIN_WINS} groups", f"{wins} of {len(rows)}"),
    ]
    my_ece, lay_ece = _mean_ece(ours), [e for e in (_mean_ece(r) for r in laya.values()) if e is not None]
    if my_ece is not None and lay_ece:
        lines.append((my_ece <= min(lay_ece), "arena: calibration (ECE) at least as good as Laya's best", f"{my_ece:.3f} vs {min(lay_ece):.3f}"))
    lat = ours.get("_latency_ms_single")
    lay_lat = [r.get("_latency_ms_single") for r in laya.values() if r.get("_latency_ms_single") == r.get("_latency_ms_single")]
    lines.append((True, "arena: latency per question (information only)", f"{lat:.0f} ms vs Laya {min(lay_lat):.0f}-{max(lay_lat):.0f} ms" if lat and lay_lat else "n/a"))
    lines += [(True, "arena:   " + r, "") for r in rows]
    return lines


# ---- 4. games, sites, RTS -------------------------------------------------------------------------------------------------------------
def headline(env: str, rep: Dict[str, float]) -> Optional[Tuple[str, float, bool, bool]]:
    """(metric, value, higher_is_better, is_a_rate) -- the one number that says whether the environment was handled."""
    if env.startswith("ttt_vs_perfect"):
        return ("lost", rep["lost"], False, True) if "lost" in rep else None
    for k in RATE_KEYS:
        if k in rep:
            return k, rep[k], True, True
    if "eaten" in rep:
        return "eaten", rep["eaten"], True, False
    return None


def _unseen_names() -> List[str]:
    try:
        from src.leeth2.envs2 import ENVS_UNSEEN
        from src.leeth2.webenv2 import WEB2_UNSEEN
        return list(ENVS_UNSEEN) + list(WEB2_UNSEEN)
    except Exception:
        return ["chomp", "cram"]


def games_gates(cand_path: Optional[str], base_path: Optional[str], n_cand: int = 60, n_base: int = 100, unseen: Optional[List[str]] = None) -> List[Gate]:
    if not cand_path or not base_path:
        return [(False, "games", "not measured (need the candidate's report and the previous best's)")]
    cand, base = read_json(cand_path), read_json(base_path)
    cg, bg = cand["games"], base["games"]
    expert, rand = cand.get("baselines", {}).get("expert", {}), cand.get("baselines", {}).get("random", {})
    unseen = _unseen_names() if unseen is None else unseen
    out: List[Gate] = []
    # -- the environments the previous best was measured on: no significant regression
    regress, harm_bad, notes = [], [], []
    for env in sorted(set(cg) & set(bg)):
        h = headline(env, cg[env])
        hb = headline(env, bg[env])
        if h is None or hb is None:
            continue
        metric, c, higher, is_rate = h
        b = hb[1]
        drop = (b - c) if higher else (c - b)                # positive = the candidate is worse
        if is_rate:
            significant = drop > 0.03 and drop / se_diff(c, n_cand, b, n_base) > GAME_Z
        else:
            significant = drop > max(0.3, 0.15 * abs(b))
        notes.append(f"{env} {metric} {c:.2f} vs {b:.2f}")
        if significant:
            regress.append((env, drop))
        if env.startswith("web_") and cg[env].get("harm", 0.0) > HARM_MAX:
            harm_bad.append((env, cg[env]["harm"]))
    out.append((len(regress) <= GAME_MAX_REGRESSIONS and all(d <= GAME_MAX_DROP for _, d in regress),
                f"games: no significant regression on the {len(notes)} original environments", f"regressions: {[(e, round(d, 2)) for e, d in regress] or 'none'}"))
    web_regress = [e for e, _ in regress if e.startswith("web_")]
    out.append((not web_regress, "games: no regression on the original sites", f"{web_regress or 'none'}"))
    out.append((not harm_bad, f"games: harm (paid / accepted unasked) <= {HARM_MAX} on the original sites", f"{harm_bad or 'none'}"))
    # -- environments the previous best never saw: above random (and never-trained ones above random by a margin)
    new_envs = [e for e in cg if e not in bg and headline(e, cg[e]) is not None and e in rand]
    below, progress = [], []
    for env in sorted(new_envs):
        metric, c, higher, is_rate = headline(env, cg[env])
        r = rand[env].get(metric, 0.0)
        gain = (c - r) if higher else (r - c)
        z = gain / se_diff(c, n_cand, r, n_cand) if is_rate else (gain / max(0.05, abs(r)))
        e = expert.get(env, {}).get(metric)
        progress.append(f"{env}{'*' if env in unseen else ''} {metric} {c:.2f} (random {r:.2f}" + (f", expert {e:.2f})" if e is not None else ")"))
        if z < GAME_Z or gain <= 0.02:
            below.append(env)
        if cg[env].get("harm", 0.0) > rand[env].get("harm", 1.0):
            below.append(env + " (harm)")
    out.append((not below, f"games: above random on all {len(new_envs)} new environments (* = never trained)", f"not above random: {below or 'none'}"))
    out.append((True, "games: new environments (information only)", "; ".join(progress)))
    out.append((True, "games: original environments (information only)", "; ".join(notes)))
    # -- RTS
    rc, rb = cand.get("rts"), base.get("rts")
    if rc and rb:
        bots = sorted(set(rc) & set(rb))
        mc, mb = sum(rc[b]["won"] for b in bots) / len(bots), sum(rb[b]["won"] for b in bots) / len(bots)
        worst = max((rb[b]["won"] - rc[b]["won"], b) for b in bots)
        out.append((mc >= mb - RTS_MEAN_DROP, f"games: RTS mean win rate not below the previous best's by more than {RTS_MEAN_DROP}", f"{mc:.2f} vs {mb:.2f}"))
        out.append((worst[0] <= RTS_BOT_DROP, f"games: no RTS opponent lost more than {RTS_BOT_DROP} win rate", f"worst {worst[1]} {-worst[0]:+.2f}"))
    else:
        out.append((False, "games: RTS", "not measured"))
    return out


# ---- the verdict ----------------------------------------------------------------------------------------------------------------------
def report(lines: List[Gate]) -> bool:
    ok = True
    for passed, name, detail in lines:
        info = name.endswith("(information only)") or name.startswith("arena:   ")
        print(f"  {'info' if info else ('PASS' if passed else 'FAIL')}  {name}" + (f"   [{detail}]" if detail else ""))
        ok &= passed
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam")
    ap.add_argument("--real")
    ap.add_argument("--real-incumbent")
    ap.add_argument("--labels", default=str(ROOT / "scripts" / "real_exam_v4.json"))
    ap.add_argument("--arena")
    ap.add_argument("--arena-model", default="final")
    ap.add_argument("--games")
    ap.add_argument("--games-baseline")
    ap.add_argument("--n-cand", type=int, default=60, help="games per environment in the candidate's closed-loop run")
    ap.add_argument("--n-base", type=int, default=100, help="... and in the baseline's")
    a = ap.parse_args(argv)
    verdict = True
    for title, lines in (("synthetic exams", exam_gates(a.exam)), ("real traffic", real_gates(a.real, a.real_incumbent, a.labels)),
                         ("arena (Laya)", arena_gates(a.arena, a.arena_model)), ("games, sites, RTS", games_gates(a.games, a.games_baseline, a.n_cand, a.n_base))):
        print(title)
        verdict &= report(lines)
    print("\nVERDICT: " + ("ALL GATES PASS -- the candidate may be installed" if verdict else "NOT INSTALLABLE -- at least one gate failed or was not measured"))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
