"""Scripted MiniRTS players: the TEACHER Leeth 2 imitates, and the opponents it is measured against.

Every bot answers one question — "which of these legal orders for this entity?" — and must return one of the offered
orders, so a bot, the teacher and the model play through exactly the same interface (``RTS.options`` / ``RTS.set_order``).
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from .rts import CARRY_MAX, MAX_TICKS, RTS, SOLDIERS, STATS, Entity

Order = Tuple
Options = List[Tuple[str, Order]]
Bot = Callable[[RTS, Entity, Options], Order]


def _orders(opts: Options) -> List[Order]:
    return [o for _, o in opts]


def _first(opts: Options, kind: str) -> Optional[Order]:
    return next((o for _, o in opts if o[0] == kind), None)


def _value(units) -> float:
    return sum(STATS[u.kind]["cost"] * (0.5 + 0.5 * u.hp / STATS[u.kind]["hp"]) for u in units if u.kind in SOLDIERS)


def _memory(game: RTS) -> Dict:
    return game.__dict__.setdefault("_bot_memory", {})


# ------------------------------------------------------------------------------------------------ simple opponents
def random_bot(seed: int = 0) -> Bot:
    rng = random.Random(seed)
    return lambda game, e, opts: rng.choice(opts)[1]


def _economy(game: RTS, e: Entity, opts: Options, workers_wanted: int, build_barracks: bool = True) -> Optional[Order]:
    """The common worker / base behaviour: harvest, return, build one barracks, train workers up to a number."""
    me = e.owner
    workers = sorted(game.mine(me, "worker"), key=lambda w: w.id)
    if e.kind == "base":
        return ("train", "worker") if len(workers) < workers_wanted and ("train", "worker") in _orders(opts) else ("hold",)
    if e.kind != "worker":
        return None
    have_barracks = bool(game.mine(me, "barracks")) or any(w.building == "barracks" for w in workers)
    builder = next((w for w in workers if w.carry == 0 and not w.building), None)
    if build_barracks and not have_barracks and len(workers) >= 3 and builder is not None and builder.id == e.id:
        b = _first(opts, "build")
        if b:
            return b
    if e.carry >= CARRY_MAX or (e.carry > 0 and not [o for o in _orders(opts) if o[0] == "harvest"]):
        return ("return",) if ("return",) in _orders(opts) else ("hold",)
    harvest = [o for o in _orders(opts) if o[0] == "harvest"]
    if harvest:
        rank = [w.id for w in workers].index(e.id) if e.id in [w.id for w in workers] else 0
        return harvest[rank % len(harvest)]
    return ("return",) if e.carry > 0 and ("return",) in _orders(opts) else ("hold",)


def worker_rush(game: RTS, e: Entity, opts: Options) -> Order:
    if e.kind == "base":
        return ("train", "worker") if ("train", "worker") in _orders(opts) else ("hold",)
    attack = _first(opts, "attack")
    return attack or ("move", game._staging(e.owner)) if e.kind == "worker" else ("hold",)


def _rush(kind: str, minimum: int) -> Bot:
    def bot(game: RTS, e: Entity, opts: Options) -> Order:
        if e.kind == "barracks":
            return ("train", kind) if ("train", kind) in _orders(opts) else ("hold",)
        if e.kind in ("worker", "base"):
            return _economy(game, e, opts, workers_wanted=3) or ("hold",)
        mem = _memory(game)
        key = ("go", e.owner)
        if len([u for u in game.units(e.owner) if u.kind in SOLDIERS]) >= minimum:
            mem[key] = True
        if mem.get(key):
            return _first(opts, "attack") or next((o for _, o in opts if o[0] == "move" and o[1] == game._staging(e.owner)), ("hold",))
        return next((o for _, o in opts if o[0] == "move" and o[1] == game._rally(e.owner)), ("hold",))
    return bot


light_rush = _rush("light", 2)
heavy_boom = _rush("heavy", 4)
ranged_wall = _rush("ranged", 5)


# ------------------------------------------------------------------------------------------------ the teacher
def _choose_unit(game: RTS, me: int, opts: Options) -> Order:
    foe_units = game.units(1 - me)
    count = lambda k: len([u for u in foe_units if u.kind == k])      # noqa: E731
    mine = {k: len([u for u in game.units(me) if u.kind == k]) for k in SOLDIERS}
    # rock-paper-scissors: light (fast) catches ranged, heavy (tough) beats light, ranged (reach) shreds heavy
    scores = {"light": 1.0 + 1.2 * count("ranged") - 0.8 * count("heavy"),
              "heavy": 0.9 + 1.2 * count("light") - 0.8 * count("ranged"),
              "ranged": 0.9 + 1.2 * count("heavy") - 0.8 * count("light")}
    for k in SOLDIERS:
        scores[k] -= 0.25 * mine[k]            # keep the army mixed
    options = [k for k in SOLDIERS if ("train", k) in _orders(opts)]
    if not options:
        return ("hold",)
    best = max(options, key=lambda k: (scores[k], -SOLDIERS.index(k)))
    return ("train", best)


#: teacher knobs, found by random search against the four scripted opponents and CONFIRMED on 120 games per opponent with fresh seeds
#: (old knobs min_army=2 ratio=1.0 force_tick=380 workers=5 : worker_rush .89  light_rush .74  heavy_boom .38  ranged_wall .59  -> mean .65;
#:  these:                                                     worker_rush .90  light_rush .76  heavy_boom .58  ranged_wall .69  -> mean .73).
#: A teacher that strikes early with a small army (it beat the slow "boom" openings the old one lost to) is also easier to imitate.
TEACHER = dict(min_army=1, ratio=0.6, force_tick=200, workers=4)


def teacher(game: RTS, e: Entity, opts: Options) -> Order:
    """A solid all-round player: economy first, a barracks at once, an army mixed against the enemy's, everyone fights near
    home, attack once clearly stronger (or late). It is the label source, so it is written for clarity, not cleverness."""
    me = e.owner
    base = game._anchor(me)
    attacks = [o for o in _orders(opts) if o[0] == "attack"]
    if base is not None and e.kind in ("worker", "light", "heavy", "ranged"):
        # defence: anything hostile close to home is attacked first (workers too, while the army is small)
        my_soldiers = [u for u in game.units(me) if u.kind in SOLDIERS]
        raiders = [o for o in attacks if game.entities[o[1]].kind != "base" and game.dist(game.entities[o[1]].pos, base.pos) <= 6]
        if raiders and (e.kind != "worker" or len(my_soldiers) < 3):
            return raiders[0]
    if e.kind in ("worker", "base"):
        wanted = TEACHER["workers"] if game.tick < 300 else 3
        return _economy(game, e, opts, workers_wanted=wanted) or ("hold",)
    if e.kind == "barracks":
        return _choose_unit(game, me, opts)

    my_army = [u for u in game.units(me) if u.kind in SOLDIERS]
    foe_army = [u for u in game.units(1 - me) if u.kind in SOLDIERS]
    ratio = (_value(my_army) + 0.1) / (_value(foe_army) + 0.1)
    # a function of what is on the board (no hidden "attack mode" flag): the model can only imitate what it can see
    attacking = len(my_army) >= TEACHER["min_army"] and (ratio >= TEACHER["ratio"] or game.tick >= TEACHER["force_tick"])
    close = [o for o in attacks if game.dist(e.pos, game.entities[o[1]].pos) <= 5]
    if close and not attacking and ratio >= 0.5:      # already in a fight: finish it rather than turn your back (no oscillation)
        return close[0]
    if attacking:
        units_first = [o for o in attacks if game.entities[o[1]].kind != "base"] or attacks
        return (units_first or [("move", game._staging(me))])[0]
    rally = ("move", game._rally(me))
    return rally if rally in _orders(opts) and game.dist(e.pos, rally[1]) > 1 else ("hold",)


BOTS: Dict[str, Callable[[], Bot]] = {
    "random": lambda: random_bot(1),
    "worker_rush": lambda: worker_rush,
    "light_rush": lambda: light_rush,
    "heavy_boom": lambda: heavy_boom,
    "ranged_wall": lambda: ranged_wall,
    "teacher": lambda: teacher,
}


# ------------------------------------------------------------------------------------------------ running a match
def play(seed: int, bots: Tuple[Bot, Bot], size: int = 14, max_ticks: int = MAX_TICKS) -> Tuple[int, RTS]:
    game = RTS(seed, size)
    while game.winner is None and game.tick < max_ticks:
        for owner in (0, 1):
            for eid in game.pending(owner):
                e = game.entities.get(eid)
                if e is None:
                    continue
                opts = game.options(eid)
                order = bots[owner](game, e, opts)
                if order not in _orders(opts):
                    order = ("hold",)
                game.set_order(eid, order)
        game.step()
    if game.winner is None:
        game.winner = game.material_leader()
    return game.winner, game


def duel(a: str, b: str, seeds: int = 20, size: int = 14) -> Dict[str, float]:
    """Win rate of bot ``a`` against bot ``b``, each seat played equally often (the map is symmetric but order of play is not)."""
    wins = losses = draws = 0
    for s in range(seeds):
        for swap in (False, True):
            bots = (BOTS[b](), BOTS[a]()) if swap else (BOTS[a](), BOTS[b]())
            w, _ = play(s, bots, size)
            mine = 1 if swap else 0
            if w == -1:
                draws += 1
            elif w == mine:
                wins += 1
            else:
                losses += 1
    n = wins + losses + draws
    return {"win": wins / n, "loss": losses / n, "draw": draws / n}
