"""Measure how well a Leeth 2 model PLAYS: closed-loop games plus per-step accuracy on the synthetic game / browser / NPC tasks.

    from leeth2.gameeval import evaluate_model
    report = evaluate_model("path/to/final", games=200)

Runs on the GPU box (Kaggle) in a couple of minutes per model; the numbers come from the same ``predict`` the runtime uses.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .data import synth as SY
from .data.common import stable_hash
from .data.synth import browser, games, npc, npc2, tools
from .envs import ENVS, ENVS_SPACED, expert_policy, random_policy, rollout, summarize
from .envs2 import ENVS_MORE, ENVS_UNSEEN
from .evaluate import predict
from .webenv import WEB_ENVS
from .webenv2 import WEB2_ENVS, WEB2_UNSEEN
from .schema import Sample

STEP_TASKS = {
    "snake": games.gen_snake, "maze": games.gen_maze, "ttt": games.gen_ttt, "treasure": games.gen_treasure,
    "npc_policy": npc.gen_policy, "npc_persona": npc.gen_persona, "npc_policy2": npc2.gen_policy2, "npc_policy2_unseen": npc2.gen_policy2_unseen,
    "browser_click": browser.gen_click, "browser_risk": browser.gen_risk, "toolpick": tools.gen_toolpick,
}
from .data.synth import play as _play  # noqa: E402

for _group in (ENVS_MORE, ENVS_UNSEEN, WEB2_ENVS, WEB2_UNSEEN):
    STEP_TASKS.update({"step_" + _name: _play.position_gen(_make) for _name, _make in _group.items()})


def model_policy(rt, tokens_per_batch: int = 24000):
    def policy(samples: List[Sample]) -> List[int]:
        preds = predict(rt.model, rt.encoder, samples, device=rt.device, amp=rt.amp, calibrated=True, tokens_per_batch=tokens_per_batch)
        return [int(np.argmax(p.probs)) for p in preds]

    return policy


def step_accuracy(rt, n: int = 1200, seed: int = 20261009) -> Dict[str, Dict[str, float]]:
    """Share of positions where the model's top option is one of the expert's optimal options (choice samples only)."""
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in STEP_TASKS.items():
        samples = [s for s in SY.generate(fn, n, seed + (stable_hash("gameeval", name) & 0xFFFF) + SY.ROLE_SEED["test"]) if s.ptype == "choice"]
        if not samples:
            continue
        picks = model_policy(rt)(samples)
        ok = []
        for s, c in zip(samples, picks):
            t = s.target
            ok.append(float(t[c] > 0) if isinstance(t, (list, tuple)) else float(c == int(t)))
        by_k = {}
        for s, o in zip(samples, ok):
            by_k.setdefault(len(s.options or []), []).append(o)
        out[name] = {"acc": float(np.mean(ok)), "n": float(len(ok)), "mean_options": float(np.mean([len(s.options or []) for s in samples]))}
    return out


def play_games(rt, games_per_env: int = 200, seed: int = 7, extended: bool = True) -> Dict[str, Dict[str, float]]:
    policy = model_policy(rt)
    out = {name: summarize(rollout(make, games_per_env, policy, seed=seed)) for name, make in ENVS.items()}
    out.update({name + "_sp": summarize(rollout(make, games_per_env, policy, seed=seed)) for name, make in ENVS_SPACED.items()})   # one token per grid cell
    out.update({name: summarize(rollout(make, games_per_env, policy, seed=seed)) for name, make in WEB_ENVS.items()})            # multi-step browser tasks
    if extended:
        for group in (ENVS_MORE, ENVS_UNSEEN, WEB2_ENVS, WEB2_UNSEEN):                                                         # more games, generated sites, never-seen ones
            out.update({name: summarize(rollout(make, games_per_env, policy, seed=seed)) for name, make in group.items()})
    return out


def baselines(games_per_env: int = 200, seed: int = 7, extended: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
    envs = {**ENVS, **WEB_ENVS}
    if extended:
        envs.update({**ENVS_MORE, **ENVS_UNSEEN, **WEB2_ENVS, **WEB2_UNSEEN})
    return {
        "expert": {n: summarize(rollout(m, games_per_env, expert_policy, seed=seed)) for n, m in envs.items()},
        "random": {n: summarize(rollout(m, games_per_env, random_policy(1), seed=seed)) for n, m in envs.items()},
    }


class Player:
    """What ``predict`` needs, without the runtime module (that one imports the whole Argent package, which Kaggle does not have)."""

    def __init__(self, model, encoder, device: str):
        self.model, self.encoder, self.device = model.eval(), encoder, device
        self.amp = device.startswith("cuda")


def load_player(path: str | Path, device: Optional[str] = None, max_len: int = 1024) -> Player:
    import torch
    from transformers import AutoTokenizer

    from .encoding import Leeth2Encoder
    from .model import Leeth2Model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Leeth2Model.from_pretrained(Path(path), device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    tok = AutoTokenizer.from_pretrained(str(path))
    return Player(model, Leeth2Encoder(tok, model.cfg.vocab_size, max_len=max_len, hard_max_len=8192), device)


def evaluate_model(path: str | Path, games_per_env: int = 200, step_n: int = 1200, device: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    rt = load_player(path, device)
    rep = {"model": str(path), "steps": step_accuracy(rt, step_n), "games": play_games(rt, games_per_env)}
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def table(reports: Dict[str, Dict[str, Any]], base: Dict[str, Any]) -> str:
    """One text table: games (the headline numbers) then per-step accuracy."""
    names = list(reports)
    lines = []
    head = f"{'metric':<34}" + "".join(f"{n:>11}" for n in names) + f"{'expert':>10}{'random':>10}"
    lines += [head, "-" * len(head)]
    rows = [("snake", "eaten", "snake: foods eaten"), ("snake", "died", "snake: died"), ("maze", "solved", "maze: solved"), ("treasure", "reached", "treasure: reached"),
            ("treasure", "died", "treasure: died in trap"), ("ttt_vs_random", "lost", "ttt vs random: lost"), ("ttt_vs_random", "won", "ttt vs random: won"),
            ("ttt_vs_perfect", "lost", "ttt vs perfect: LOST"), ("ttt_vs_perfect", "drawn", "ttt vs perfect: drawn")]
    for suffix in ("", "_sp"):
        for env, key, label in rows:
            if env + suffix not in reports[names[0]]["games"]:
                continue
            cells = "".join(f"{reports[n]['games'][env + suffix].get(key, float('nan')):>11.3f}" for n in names)
            lines.append(f"{label + (' [1 token/cell]' if suffix else ''):<34}{cells}{base['expert'][env].get(key, float('nan')):>10.3f}{base['random'][env].get(key, float('nan')):>10.3f}")
    for env in WEB_ENVS:
        if env in reports[names[0]]["games"]:
            for key, label in (("success", "success"), ("harm", "HARM (paid/accepted unasked)")):
                cells = "".join(f"{reports[n]['games'][env].get(key, float('nan')):>11.3f}" for n in names)
                lines.append(f"{env + ': ' + label:<34}{cells}{base['expert'][env].get(key, float('nan')):>10.3f}{base['random'][env].get(key, float('nan')):>10.3f}")
    lines.append("")
    for env in list(ENVS_MORE) + list(ENVS_UNSEEN) + list(WEB2_ENVS) + list(WEB2_UNSEEN):
        if env not in reports[names[0]]["games"]:
            continue
        key = next(k for k in ("success", "solved", "reached", "won") if k in reports[names[0]]["games"][env])
        cells = "".join(f"{reports[n]['games'][env].get(key, float('nan')):>11.3f}" for n in names)
        tag = " [UNSEEN]" if env in ENVS_UNSEEN or env in WEB2_UNSEEN else ""
        lines.append(f"{env + ': ' + key + tag:<34}{cells}{base['expert'][env].get(key, float('nan')):>10.3f}{base['random'][env].get(key, float('nan')):>10.3f}")
        if "harm" in reports[names[0]]["games"][env]:
            cells = "".join(f"{reports[n]['games'][env].get('harm', float('nan')):>11.3f}" for n in names)
            lines.append(f"{env + ': HARM':<34}{cells}{base['expert'][env].get('harm', float('nan')):>10.3f}{base['random'][env].get('harm', float('nan')):>10.3f}")
    lines.append("")
    for env in list(ENVS) + [e + "_sp" for e in ENVS] + list(WEB_ENVS):
        if env not in reports[names[0]]["games"]:
            continue
        cells = "".join(f"{reports[n]['games'][env]['optimal_rate']:>11.3f}" for n in names)
        lines.append(f"{'optimal move rate: ' + env:<34}{cells}")
    lines.append("")
    for task in STEP_TASKS:
        if all(task in reports[n]["steps"] for n in names):
            cells = "".join(f"{reports[n]['steps'][task]['acc']:>11.3f}" for n in names)
            lines.append(f"{'per-step acc: ' + task:<34}{cells}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path pairs")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", default="game_eval.json")
    a = ap.parse_args(argv)
    reports = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        reports[name] = evaluate_model(path, a.games, a.steps)
        print(f"[{name}] done in {reports[name]['seconds']} s", flush=True)
    base = baselines(a.games)
    Path(a.out).write_text(json.dumps({"reports": reports, "baselines": base}, indent=1), encoding="utf-8")
    print(table(reports, base))


if __name__ == "__main__":
    main()
