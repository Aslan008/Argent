"""Playing MiniRTS in bulk: collecting imitation data from the teacher and measuring a model against the scripted bots.

    samples, results = play_many(games=64, seats=("model", "bot:worker_rush"), policy=model_policy, record=True)

* ``seats`` — what drives each player: ``"teacher"``, ``"bot:<name>"``, ``"noisy:<eps>"`` (teacher with a share of random orders),
  or ``"model"`` (answers come from ``policy``).
* ``record`` — every decision of a recorded seat becomes a ``Sample``: the position the seat actually met (whoever chose the
  move), labelled with the TEACHER's order for that position. With teacher or noisy seats this is plain imitation; with a
  ``model`` seat it is DAgger — the model's own mistakes lead to positions the teacher then labels.
* Games run in lock-step, so a model answers one batch per tick across all games (GPU friendly).

Pure Python apart from the policy callable.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .rts import MAX_TICKS, RTS
from .rts_bots import BOTS, teacher
from .schema import Sample

Policy = Callable[[List[Sample]], List[int]]


def _seat_driver(spec: str, seed: int):
    if spec == "teacher":
        return teacher
    if spec.startswith("bot:"):
        return BOTS[spec[4:]]()
    if spec.startswith("noisy:"):
        eps, rng = float(spec[6:]), random.Random(seed)
        return lambda game, e, opts: rng.choice(opts)[1] if rng.random() < eps else teacher(game, e, opts)
    if spec == "model":
        return None
    raise ValueError(f"unknown seat {spec!r}")


def _sample(game: RTS, owner: int, eid: int, texts: List[str], orders: List[Tuple], label_order: Tuple, state: str, question: str) -> Sample:
    idx = orders.index(label_order) if label_order in orders else next((i for i, o in enumerate(orders) if o[0] == "hold"), 0)
    return Sample("rts_order", "game", "en", "choice", state, question, texts, idx, meta={"tick": game.tick, "kind": game.entities[eid].kind})


def play_many(
    games: int,
    seats: Sequence[str],
    policy: Optional[Policy] = None,
    record: bool = True,
    seed: int = 0,
    sizes: Sequence[int] = (12, 14, 16),
    swap_seats: bool = True,
    sample_rate: float = 1.0,
    beta: float = 0.0,
) -> Tuple[List[Sample], List[Dict[str, float]]]:
    """Play ``games`` games in lock-step. Returns (samples, results); a result is {won, lost, drawn, ticks, ...} for seat 0 of ``seats``
    (seat order is swapped on every second game when ``swap_seats`` so neither side of the map is favoured).
    ``beta``: share of a ``model`` seat's decisions executed by the teacher instead (DAgger's mixing: early rounds lean on the expert)."""
    rng = random.Random(seed)
    runs = []
    for i in range(games):
        swapped = swap_seats and i % 2 == 1
        order = tuple(reversed(seats)) if swapped else tuple(seats)
        game = RTS(seed * 100003 + i, rng.choice(list(sizes)))
        drivers = [_seat_driver(s, seed * 7 + i * 2 + k) for k, s in enumerate(order)]
        runs.append({"game": game, "order": order, "drivers": drivers, "swapped": swapped})
    samples: List[Sample] = []
    rec_rng = random.Random(seed + 1)
    for _ in range(MAX_TICKS + 1):
        live = [r for r in runs if r["game"].winner is None]
        if not live:
            break
        batch, meta = [], []
        for r in live:
            g: RTS = r["game"]
            for owner in (0, 1):
                for eid in g.pending(owner):
                    e = g.entities.get(eid)
                    if e is None:
                        continue
                    state, question, texts, orders = g.describe(owner, eid)
                    driver = r["drivers"][owner]
                    if driver is None:                      # the model decides
                        batch.append(Sample("rts_order", "game", "en", "choice", state, question, texts, 0))
                        meta.append((r, owner, eid, texts, orders, state, question))
                        continue
                    opts = list(zip(texts, orders))
                    chosen = driver(g, e, opts)
                    if chosen not in orders:
                        chosen = ("hold",)
                    if record and order_recorded(r, owner, seats) and rec_rng.random() < sample_rate:
                        samples.append(_sample(g, owner, eid, texts, orders, teacher(g, e, opts), state, question))
                    g.set_order(eid, chosen)
        if batch:
            picks = policy(batch)
            for (r, owner, eid, texts, orders, state, question), pick in zip(meta, picks):
                g: RTS = r["game"]
                if eid not in g.entities:
                    continue
                if record and rec_rng.random() < sample_rate:
                    opts = list(zip(texts, orders))
                    samples.append(_sample(g, owner, eid, texts, orders, teacher(g, g.entities[eid], opts), state, question))
                if beta and rec_rng.random() < beta:
                    pick_order = teacher(g, g.entities[eid], list(zip(texts, orders)))
                    g.set_order(eid, pick_order if pick_order in orders else orders[int(pick)])
                else:
                    g.set_order(eid, orders[int(pick)])
        for r in live:
            r["game"].step()
    results = []
    for r in runs:
        g = r["game"]
        w = g.winner if g.winner is not None else g.material_leader()
        seat0 = 1 if r["swapped"] else 0
        results.append({
            "won": float(w == seat0), "lost": float(w == 1 - seat0), "drawn": float(w == -1), "ticks": float(g.tick),
            "kills": float(g.stats[seat0]["kills"]), "lost_units": float(g.stats[seat0]["lost"]), "mined": float(g.stats[seat0]["mined"]),
        })
    return samples, results


def order_recorded(run: Dict, owner: int, seats: Sequence[str]) -> bool:
    """Teacher / noisy / model seats are recorded; plain scripted-bot seats (opponents) are not."""
    spec = run["order"][owner]
    return not spec.startswith("bot:")


def summarize(results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in results for k in r})
    return {k: sum(r.get(k, 0.0) for r in results) / max(1, len(results)) for k in keys}


def benchmark(policy: Optional[Policy], seats_vs: Sequence[str] = ("random", "worker_rush", "light_rush", "heavy_boom", "ranged_wall"), games: int = 40, seed: int = 11) -> Dict[str, Dict[str, float]]:
    """Win rate of a model policy (or, with policy=None, of the teacher) against each scripted opponent."""
    out = {}
    for name in seats_vs:
        _, res = play_many(games, ("model" if policy else "teacher", f"bot:{name}"), policy, record=False, seed=seed)
        out[name] = summarize(res)
    return out
