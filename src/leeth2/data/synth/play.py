"""Game data from PLAYED games instead of random positions.

The older generators (games.py) draw one random position per sample. Closed-loop evaluation (envs.py) showed what that misses:
a model that is right on random positions still dies in snake within a few moves, because the positions a policy meets while
playing — long snakes, mid-maze corners, second-player tic-tac-toe boards — are not the random ones. Here an episode is
played by the perfect expert with a fraction of random moves mixed in (so the data also covers the mistakes a learner makes
and shows the expert's correction), and every visited position becomes a sample with the expert's optimal set as a soft target.

One token per grid cell (``spaced``): a BPE tokenizer glues runs of '#' and '.' together, which destroys the column alignment
a model needs to see "the cell above me is a wall".
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ...envs import Env, MazeEnv, SnakeEnv, TreasureEnv, TttEnv, _targets
from ...envs2 import ENVS_MORE, ENVS_UNSEEN
from ...webenv import WebEnv
from ...webenv2 import TRAINED_FAMILIES as WEB2_FAMILIES, WebEnv2
from ...schema import Sample
from . import ROLE_SEED, generate

#: (probability that a move is random, probability that a visited position is kept)
EPSILON = (0.0, 0.1, 0.25, 0.4)
KEEP = 0.55


def _episode(make: Callable[[random.Random], Env], rng: random.Random, max_decisions: int = 80) -> List[Sample]:
    env = make(rng)
    eps = rng.choice(EPSILON)
    out: List[Sample] = []
    for _ in range(max_decisions):
        if env.done:
            break
        s = env.decision(rng)
        t = _targets(s)
        if rng.random() < KEEP:
            out.append(s)
        env.apply(rng.randrange(len(s.options or [0])) if rng.random() < eps else rng.choice(t))
    return out


def gen_snake(rng): return _episode(lambda r: SnakeEnv(r, max_size=11, spaced=True), rng)
def gen_maze(rng): return _episode(lambda r: MazeEnv(r, spaced=True), rng)
def gen_treasure(rng): return _episode(lambda r: TreasureEnv(r, spaced=True), rng)
def gen_ttt(rng): return _episode(lambda r: TttEnv(r, opponent=r.choice(["random", "perfect", "perfect"]), spaced=True), rng)


def _web(task):
    return lambda rng: _episode(lambda r: WebEnv(r, task), rng, max_decisions=16)


def _uniform(s: Sample) -> bool:
    """Every option equally good (a lost game of Nim, the first move on an empty board): the label says nothing."""
    t = s.target
    return isinstance(t, (list, tuple)) and len(set(round(float(x), 6) for x in t)) == 1


def _more(key: str, max_decisions: int = 60):
    make = ENVS_MORE[key]

    def gen(rng):
        return [s for s in _episode(make, rng, max_decisions) if not _uniform(s)]

    return gen


#: the games added for universality (envs2.py): each has a different kind of move and states its own rules
GENERATORS_MORE: List[Tuple[str, Callable, int]] = [
    ("play_connect_random", _more("connect_vs_random"), 7000), ("play_connect_search", _more("connect_vs_search"), 5000),
    ("play_sokoban", _more("sokoban"), 5000), ("play_slide", _more("slide"), 4500),
    ("play_nim_random", _more("nim_vs_random"), 9000), ("play_nim_perfect", _more("nim_vs_perfect"), 6000),
    ("play_keydoor", _more("keydoor"), 4500), ("play_lights", _more("lights"), 6500),
    ("play_hanoi", _more("hanoi"), 3500), ("play_terrain", _more("terrain"), 5000),
]

def _web2(family: str):
    return lambda rng: [s for s in _episode(lambda r: WebEnv2(r, family), rng, max_decisions=18) if not _uniform(s)]


#: procedurally generated sites (webenv2.py): random wording, layout and language; "booking" is deliberately absent (held out)
GENERATORS_WEB2: List[Tuple[str, Callable, int]] = [
    ("play_web2_login", _web2("login"), 6500), ("play_web2_search", _web2("search"), 5500), ("play_web2_listing", _web2("listing"), 8000),
    ("play_web2_form", _web2("form"), 4500), ("play_web2_checkout", _web2("checkout"), 5500), ("play_web2_settings", _web2("settings"), 8000),
    ("play_web2_compose", _web2("compose"), 6500), ("play_web2_rows", _web2("rows"), 9000), ("play_web2_nav", _web2("nav"), 6500),
]


def position_gen(make: Callable[[random.Random], Env], max_lead: int = 6):
    """A generator of ONE position of a game: play a few expert moves, then show the position (for per-step accuracy)."""

    def gen(rng):
        env = make(rng)
        for _ in range(rng.randint(0, max_lead)):
            if env.done:
                break
            s = env.decision(rng)
            env.apply(rng.choice(_targets(s)))
        if env.done:
            return None
        s = env.decision(rng)
        return None if _uniform(s) else s

    return gen


#: (name, function, training episodes)
GENERATORS: List[Tuple[str, Callable, int]] = [
    ("play_snake", gen_snake, 12000), ("play_maze", gen_maze, 40000), ("play_treasure", gen_treasure, 30000), ("play_ttt", gen_ttt, 30000),
    ("play_web_login", _web("login"), 7000), ("play_web_search", _web("search"), 7000), ("play_web_form", _web("form"), 9000), ("play_web_upload", _web("upload"), 6000), ("play_web_purchase", _web("purchase"), 7000),
]
EVAL_EPISODES = {"dev": 150, "calib": 250, "test": 400}


def _registry() -> Dict[str, Callable]:
    return {name: fn for name, fn, _ in GENERATORS + GENERATORS_MORE + GENERATORS_WEB2}


def _job(args) -> List[Tuple[str, str]]:
    """One (generator, split) of the build, runnable in a worker process: [(dedupe key, json row)]."""
    name, calls, seed = args
    out = []
    for s in generate(_registry()[name], calls, seed):
        out.append((f"{s.state}|{s.question}|{json.dumps(s.options)}", s.to_json()))
    return out


def build(out_dir: Path, seed: int = 9100, scale: float = 1.0, generators: Optional[List[Tuple[str, Callable, int]]] = None,
          workers: int = 1, eval_scale: float = 1.0) -> Dict[str, int]:
    """train/dev/calib/test.jsonl of played-game samples; an evaluation split never repeats a training position.

    ``workers`` > 1 spreads the (generator, split) jobs over processes (fork: the generators are closures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gens = GENERATORS if generators is None else generators
    taken: set = set()
    counts: Dict[str, int] = {}
    pool = None
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))
    try:
        for role in ("test", "calib", "dev", "train"):
            jobs = []
            for i, (name, fn, n_train) in enumerate(gens):
                calls = max(40, int(n_train * scale)) if role == "train" else max(20, int(EVAL_EPISODES[role] * eval_scale))
                jobs.append((name, calls, seed + 97 * i + ROLE_SEED[role]))
            parts = list(pool.map(_job, jobs)) if pool else [_job(j) for j in jobs]
            rows, used = [], set()
            for part in parts:
                for k, row in part:
                    if k in taken or k in used:
                        continue
                    used.add(k)
                    rows.append(row)
            if role != "train":
                taken |= used
            random.Random(seed).shuffle(rows)
            (out_dir / f"{role}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
            counts[role] = len(rows)
    finally:
        if pool:
            pool.shutdown()
    return counts


def with_routing(play_dir: Path, routing_dir: Path, out_dir: Path, routing_rows: int = 200000, seed: int = 9101) -> Dict[str, int]:
    """Played-game rows + a random slice of the production-state routing/safety rows (so the game fine-tune does not unlearn Argent)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counts = {}
    for role in ("train", "dev", "calib", "test"):
        rows = [l for l in (play_dir / f"{role}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        src = [l for l in (routing_dir / f"{role}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        rng.shuffle(src)
        rows += src[: routing_rows if role == "train" else max(1500, routing_rows // 40)]
        rng.shuffle(rows)
        (out_dir / f"{role}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        counts[role] = len(rows)
    return counts
