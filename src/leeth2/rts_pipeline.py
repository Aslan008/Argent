"""MiniRTS data rounds: teacher imitation (round 0), DAgger rollouts of the model (later rounds), evaluation splits.

All functions are plain Python + a policy callable, so the Kaggle kernel only wires them to training:

    rows = teacher_rows(games=1200, workers=4)                    # round 0
    rows = dagger_rows(policy, games=260, beta=0.3)               # round r >= 1, positions the MODEL reaches, labelled by the teacher
    benchmark(policy)                                             # win rate vs the scripted bots
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .rts_bots import BOTS
from .rts_play import Policy, benchmark, play_many, summarize
from .schema import Sample

OPPONENTS = ("random", "worker_rush", "light_rush", "heavy_boom", "ranged_wall")

#: round 0: who plays whom while the teacher's decisions are recorded (noise widens the set of positions seen)
TEACHER_MATCHUPS: Sequence[Tuple[str, str]] = (
    [("noisy:0.0", f"bot:{b}") for b in OPPONENTS] + [("noisy:0.15", f"bot:{b}") for b in OPPONENTS] + [("noisy:0.3", f"bot:{b}") for b in ("worker_rush", "light_rush", "heavy_boom", "ranged_wall")]
    + [("teacher", "teacher"), ("noisy:0.15", "teacher"), ("noisy:0.3", "noisy:0.15")]
)


def _teacher_chunk(args) -> List[str]:
    games, seats, seed = args
    samples, _ = play_many(games, seats, record=True, seed=seed)
    return [s.to_json() for s in samples]


def teacher_rows(games: int = 1200, workers: int = 4, seed: int = 0, per_job: int = 12) -> List[str]:
    """JSON rows of teacher decisions from ``games`` games spread over the matchups (parallel over processes)."""
    jobs = []
    k = 0
    while sum(j[0] for j in jobs) < games:
        seats = TEACHER_MATCHUPS[k % len(TEACHER_MATCHUPS)]
        jobs.append((per_job, seats, seed * 1000 + k))
        k += 1
    rows: List[str] = []
    if workers <= 1:
        for j in jobs:
            rows += _teacher_chunk(j)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for part in ex.map(_teacher_chunk, jobs):
                rows += part
    return rows


def dagger_rows(policy: Policy, games: int = 260, beta: float = 0.3, seed: int = 100, sample_rate: float = 0.45) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Let the MODEL play (vs every scripted bot, the teacher and itself), record the positions it reaches labelled by the teacher."""
    per = max(2, games // (len(OPPONENTS) + 2))
    rows: List[str] = []
    results: Dict[str, Dict[str, float]] = {}
    for i, opp in enumerate(list(OPPONENTS) + ["teacher", "model"]):
        spec = opp if opp in ("teacher", "model") else f"bot:{opp}"
        samples, res = play_many(per, ("model", spec), policy, record=True, seed=seed + i, sample_rate=sample_rate, beta=beta)
        rows += [s.to_json() for s in samples]
        results[opp] = summarize(res)
    return rows, results


def eval_rows(games: int = 40, seed: int = 777) -> List[str]:
    """Held-out positions for dev / calib / test (teacher games on unseen seeds)."""
    samples, _ = play_many(games, ("noisy:0.15", "teacher"), record=True, seed=seed, sample_rate=0.5)
    return [s.to_json() for s in samples]


def write_rows(rows: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def mix_round(rts_rows: Sequence[str], keep: int, rng: random.Random) -> List[str]:
    """At most ``keep`` rows from the aggregate; the newest round is always kept whole, older ones are thinned (DAgger's aggregate, bounded)."""
    rows = list(rts_rows)
    if len(rows) <= keep:
        return rows
    rng.shuffle(rows)
    return rows[:keep]
