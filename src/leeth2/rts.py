"""MiniRTS: a small real-time strategy game for training and testing Leeth 2 as a player.

Two players, a grid with walls and mineral patches, workers that harvest and build, a base and barracks that train units
(worker / light / heavy / ranged). A game is decided by destroying the other side's buildings and army.

How a model plays it (the design that makes an RTS feasible for a text policy):

* Decisions are ORDERS, not key presses: "attack enemy light#31", "harvest patch#2", "build barracks at (4,5)",
  "train heavy", "move to enemy base". A pathfinder walks the unit; an order stays in force until it is done, becomes
  invalid, or 12 ticks pass. A 600-tick game is then ~1000 decisions, not 20000.
* One decision per idle entity. The question shows that entity and its candidate targets sorted by distance (with distances,
  hit points and "who wins this fight" hints) next to a compact picture of the whole position, in coordinates and lists
  — no ASCII art, whose columns a tokenizer cannot keep aligned.
* ``Options`` are the legal orders of that entity (at most ~20). The same option list is what the scripted teacher chooses
  from, so a label is simply "the option the teacher picks", and any controller (teacher, scripted bots, the model) plays
  through ``decide`` / ``apply``.

Pure Python, deterministic given the seed, no torch.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Pos = Tuple[int, int]
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]            # up, right, down, left (y grows downwards)
DIR_NAMES = ["up", "right", "down", "left"]

STATS = {  # hp, attack, range, move period (ticks per step), cost, train time
    "worker": dict(hp=3, atk=1, rng=1, move=1, cost=1, train=3),
    "light": dict(hp=4, atk=2, rng=1, move=1, cost=2, train=4),
    "heavy": dict(hp=8, atk=3, rng=1, move=2, cost=3, train=6),
    "ranged": dict(hp=2, atk=2, rng=2, move=1, cost=2, train=5),
    "base": dict(hp=40, atk=0, rng=0, move=0, cost=0, train=0),
    "barracks": dict(hp=16, atk=0, rng=0, move=0, cost=5, train=0),
}
UNITS = ("worker", "light", "heavy", "ranged")
SOLDIERS = ("light", "heavy", "ranged")
BARRACKS_BUILD_TIME = 8
ATTACK_COOLDOWN = 2
CARRY_MAX = 4
ORDER_LIFETIME = 12
MAX_TICKS = 600


@dataclass
class Entity:
    id: int
    kind: str
    owner: int
    pos: Pos
    hp: int
    carry: int = 0
    cooldown: int = 0
    order: Optional[Tuple] = None              # ("attack", id) ("move", (x, y)) ("harvest", patch_id) ("return",) ("build", (x, y)) ("train", kind) ("hold",)
    order_tick: int = -99
    busy: int = 0                              # ticks left of a train / build in progress
    building: Optional[str] = None             # what a base/barracks is producing, or what a worker is constructing
    move_clock: int = 0

    @property
    def is_building(self) -> bool:
        return self.kind in ("base", "barracks")

    @property
    def max_hp(self) -> int:
        return STATS[self.kind]["hp"]


@dataclass
class Patch:
    id: int
    pos: Pos
    amount: int


class RTS:
    def __init__(self, seed: int = 0, size: int = 14):
        self.rng = random.Random(seed)
        self.W = self.H = size
        self.tick = 0
        self.entities: Dict[int, Entity] = {}
        self.patches: Dict[int, Patch] = {}
        self.walls: set = set()
        self.resources = [6, 6]
        self._next = 1
        self.winner: Optional[int] = None
        self.stats = [{"kills": 0, "lost": 0, "mined": 0} for _ in range(2)]
        self._generate()

    # ------------------------------------------------------------------------------------------ map
    def _mirror(self, p: Pos) -> Pos:
        return (self.W - 1 - p[0], self.H - 1 - p[1])

    def _spawn(self, kind: str, owner: int, pos: Pos) -> Entity:
        e = Entity(self._next, kind, owner, pos, STATS[kind]["hp"])
        self.entities[e.id] = e
        self._next += 1
        return e

    def _generate(self) -> None:
        r, W, H = self.rng, self.W, self.H
        corner = (r.randint(1, 2), r.randint(1, 2))
        self._spawn("base", 0, corner)
        self._spawn("base", 1, self._mirror(corner))
        self._spawn("worker", 0, (corner[0] + 1, corner[1]))
        self._spawn("worker", 1, self._mirror((corner[0] + 1, corner[1])))
        reserved = {corner, self._mirror(corner), (corner[0] + 1, corner[1]), self._mirror((corner[0] + 1, corner[1]))}
        for _ in range(r.randint(1, 2)):          # a patch next to each base, mirrored
            for _try in range(30):
                p = (corner[0] + r.randint(-1, 3), corner[1] + r.randint(-1, 3))
                if 0 <= p[0] < W and 0 <= p[1] < H and p not in reserved and abs(p[0] - corner[0]) + abs(p[1] - corner[1]) >= 2:
                    q = self._mirror(p)
                    if q not in reserved and q != p:
                        reserved |= {p, q}
                        for pos in (p, q):
                            self.patches[len(self.patches) + 1] = Patch(len(self.patches) + 1, pos, 40)
                        break
        mid = (W // 2 - (1 if W % 2 == 0 else 0), H // 2 - (1 if H % 2 == 0 else 0))
        if mid not in reserved and self._mirror(mid) not in reserved and r.random() < 0.7:     # contested patch(es) in the middle, mirrored
            for pos in ({mid, self._mirror(mid)}):
                reserved.add(pos)
                self.patches[len(self.patches) + 1] = Patch(len(self.patches) + 1, pos, 50)
        for _ in range(r.randint(2, 5)):          # wall segments, point-symmetric so neither side is favoured
            x, y = r.randint(2, W - 3), r.randint(2, H - 3)
            dx, dy = r.choice([(1, 0), (0, 1)])
            for k in range(r.randint(1, 3)):
                p = (x + dx * k, y + dy * k)
                q = self._mirror(p)
                if all(0 <= c[0] < W and 0 <= c[1] < H and c not in reserved and self._near_start(c, corner) is False for c in (p, q)):
                    self.walls |= {p, q}
        # a map must stay connected between the two bases; if walls cut it, drop them
        if self._path_len(corner, self._mirror(corner)) is None:
            self.walls.clear()

    def _near_start(self, c: Pos, corner: Pos) -> bool:
        return min(abs(c[0] - corner[0]) + abs(c[1] - corner[1]), abs(c[0] - self._mirror(corner)[0]) + abs(c[1] - self._mirror(corner)[1])) <= 2

    # ------------------------------------------------------------------------------------------ queries
    def occupied(self) -> Dict[Pos, int]:
        return {e.pos: e.id for e in self.entities.values()}

    def mine(self, owner: int, kind: Optional[str] = None) -> List[Entity]:
        return [e for e in self.entities.values() if e.owner == owner and (kind is None or e.kind == kind)]

    def units(self, owner: int) -> List[Entity]:
        return [e for e in self.entities.values() if e.owner == owner and not e.is_building]

    def base(self, owner: int) -> Optional[Entity]:
        b = self.mine(owner, "base")
        return b[0] if b else None

    @staticmethod
    def dist(a: Pos, b: Pos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _blocked(self, extra: Optional[set] = None) -> set:
        blocked = set(self.walls) | {p.pos for p in self.patches.values() if p.amount > 0} | set(self.occupied())
        return blocked | (extra or set())

    def _path_len(self, a: Pos, b: Pos) -> Optional[int]:
        blocked = set(self.walls)
        seen, q = {a}, deque([(a, 0)])
        while q:
            cur, d = q.popleft()
            if cur == b:
                return d
            for dx, dy in DIRS:
                n = (cur[0] + dx, cur[1] + dy)
                if 0 <= n[0] < self.W and 0 <= n[1] < self.H and n not in seen and n not in blocked:
                    seen.add(n)
                    q.append((n, d + 1))
        return None

    def next_step(self, src: Pos, goal: Pos, adjacent: bool = False, ignore_units: bool = False, within: int = 1) -> Optional[Pos]:
        """First step of a shortest path from src to goal (or to any cell within ``within`` of it when ``adjacent``). None if none."""
        blocked = set(self.walls) | {p.pos for p in self.patches.values() if p.amount > 0}
        if not ignore_units:
            blocked |= {pos for pos in self.occupied() if pos != src}
        if not adjacent:
            blocked.discard(goal)
        seen, q = {src: None}, deque([src])
        target = None
        while q:
            cur = q.popleft()
            if (adjacent and self.dist(cur, goal) <= within and cur != goal) or (not adjacent and cur == goal):
                target = cur
                break
            for dx, dy in DIRS:
                n = (cur[0] + dx, cur[1] + dy)
                if 0 <= n[0] < self.W and 0 <= n[1] < self.H and n not in seen and n not in blocked:
                    seen[n] = cur
                    q.append(n)
        if target is None or target == src:
            return None
        while seen[target] != src:
            target = seen[target]
        return target

    def path_dist(self, a: Pos, goal: Pos, adjacent: bool = True) -> Optional[int]:
        """Walking distance (units ignored), to the goal or next to it."""
        blocked = set(self.walls) | {p.pos for p in self.patches.values() if p.amount > 0}
        blocked.discard(goal)
        seen, q = {a}, deque([(a, 0)])
        while q:
            cur, d = q.popleft()
            if (adjacent and self.dist(cur, goal) <= 1) or cur == goal:
                return d
            for dx, dy in DIRS:
                n = (cur[0] + dx, cur[1] + dy)
                if 0 <= n[0] < self.W and 0 <= n[1] < self.H and n not in seen and n not in blocked:
                    seen.add(n)
                    q.append((n, d + 1))
        return None

    # ------------------------------------------------------------------------------------------ orders
    def needs_decision(self, e: Entity) -> bool:
        if e.busy > 0 and e.is_building:
            return False
        if e.kind == "worker" and e.building:
            return False
        if e.order is None or self.tick - e.order_tick >= ORDER_LIFETIME:
            return True
        return not self._valid(e, e.order)

    def _valid(self, e: Entity, o: Tuple) -> bool:
        k = o[0]
        if k == "attack":
            t = self.entities.get(o[1])
            return t is not None and t.owner != e.owner
        if k == "harvest":
            p = self.patches.get(o[1])
            return p is not None and p.amount > 0 and e.carry < CARRY_MAX
        if k == "return":
            return e.carry > 0 and self.base(e.owner) is not None
        if k == "build":
            return e.building is None and self.resources[e.owner] >= STATS["barracks"]["cost"] and o[1] not in self.occupied() and o[1] not in self.walls
        if k == "train":
            return e.is_building and e.busy == 0
        return k in ("move", "hold")

    def pending(self, owner: int) -> List[int]:
        return [e.id for e in sorted(self.entities.values(), key=lambda x: x.id) if e.owner == owner and self.needs_decision(e)]

    def set_order(self, eid: int, order: Tuple) -> None:
        e = self.entities[eid]
        e.order, e.order_tick = order, self.tick

    # ------------------------------------------------------------------------------------------ simulation
    def step(self) -> None:
        if self.winner is not None:
            return
        for e in sorted(list(self.entities.values()), key=lambda x: x.id):
            if e.id not in self.entities:
                continue
            if e.cooldown > 0:
                e.cooldown -= 1
            self._act(e)
        self.tick += 1
        self._check_end()

    def _act(self, e: Entity) -> None:
        if e.is_building:
            if e.busy > 0:
                e.busy -= 1
                if e.busy == 0 and e.building:
                    self._finish_training(e)
            elif e.order and e.order[0] == "train" and e.busy == 0:
                kind = e.order[1]
                cost = STATS[kind]["cost"]
                legal = (e.kind == "base" and kind == "worker") or (e.kind == "barracks" and kind in SOLDIERS)
                if legal and self.resources[e.owner] >= cost:
                    self.resources[e.owner] -= cost
                    e.busy, e.building = STATS[kind]["train"], kind
                e.order = None
            return
        if e.kind == "worker" and e.building:           # constructing a barracks: the worker stands still until it is done
            e.busy -= 1
            if e.busy <= 0:
                self._finish_building(e)
            return
        o = e.order
        if not o:
            return
        if not self._valid(e, o):
            e.order = None
            return
        if o[0] == "hold":
            self._auto_attack(e)
        elif o[0] == "move":
            self._walk(e, o[1], adjacent=False)
            if e.pos == o[1]:
                e.order = None
        elif o[0] == "attack":
            t = self.entities[o[1]]
            if self.dist(e.pos, t.pos) <= STATS[e.kind]["rng"]:
                self._hit(e, t)
            else:
                self._walk(e, t.pos, adjacent=True, within=STATS[e.kind]["rng"])
        elif o[0] == "harvest":
            p = self.patches[o[1]]
            if self.dist(e.pos, p.pos) <= 1:
                if self.tick % 2 == 0:
                    e.carry += 1
                    p.amount -= 1
                    self.stats[e.owner]["mined"] += 1
                if e.carry >= CARRY_MAX:
                    e.order = ("return",)
            else:
                self._walk(e, p.pos, adjacent=True)
        elif o[0] == "return":
            b = self.base(e.owner)
            if self.dist(e.pos, b.pos) <= 1:
                self.resources[e.owner] += e.carry
                e.carry = 0
                e.order = None
            else:
                self._walk(e, b.pos, adjacent=True)
        elif o[0] == "build":
            if self.dist(e.pos, o[1]) <= 1 and o[1] not in self.occupied() and o[1] not in self.walls:
                if self.resources[e.owner] >= STATS["barracks"]["cost"]:
                    self.resources[e.owner] -= STATS["barracks"]["cost"]
                    e.building, e.busy, e.order = "barracks", BARRACKS_BUILD_TIME, ("buildat", o[1])
            else:
                self._walk(e, o[1], adjacent=True)

    def _walk(self, e: Entity, goal: Pos, adjacent: bool, within: int = 1) -> None:
        sp = STATS[e.kind]["move"]
        if sp == 0:
            return
        e.move_clock += 1
        if e.move_clock < sp:
            return
        e.move_clock = 0
        nxt = self.next_step(e.pos, goal, adjacent=adjacent, within=within)
        if nxt is not None and nxt not in self.occupied():
            e.pos = nxt

    def _hit(self, a: Entity, t: Entity) -> None:
        if a.cooldown > 0:
            return
        a.cooldown = ATTACK_COOLDOWN
        t.hp -= STATS[a.kind]["atk"]
        if t.hp <= 0:
            self.stats[a.owner]["kills"] += 1
            self.stats[t.owner]["lost"] += 1
            del self.entities[t.id]

    def _auto_attack(self, e: Entity) -> None:
        if STATS[e.kind]["atk"] == 0:
            return
        rng = STATS[e.kind]["rng"]
        foes = [t for t in self.entities.values() if t.owner != e.owner and self.dist(e.pos, t.pos) <= rng]
        if foes:
            self._hit(e, min(foes, key=lambda t: (t.hp, t.id)))

    def _finish_training(self, b: Entity) -> None:
        kind, b.building = b.building, None
        for dx, dy in DIRS + [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            p = (b.pos[0] + dx, b.pos[1] + dy)
            if 0 <= p[0] < self.W and 0 <= p[1] < self.H and p not in self.walls and p not in self.occupied() and all(pt.pos != p for pt in self.patches.values()):
                self._spawn(kind, b.owner, p)
                return
        self.resources[b.owner] += STATS[kind]["cost"]       # no room: refund

    def _finish_building(self, w: Entity) -> None:
        pos = w.order[1] if w.order and w.order[0] == "buildat" else None
        w.building, w.order = None, None
        if pos and pos not in self.occupied() and pos not in self.walls:
            self._spawn("barracks", w.owner, pos)
        else:
            self.resources[w.owner] += STATS["barracks"]["cost"]

    def _check_end(self) -> None:
        """A side is out when it has no buildings and no soldiers (workers alone cannot train anything)."""
        out = [not self.mine(o, "base") and not self.mine(o, "barracks") and not [u for u in self.units(o) if u.kind in SOLDIERS] for o in (0, 1)]
        if out[0] and out[1]:
            self.winner = -1
        elif out[0] or out[1]:
            self.winner = 1 if out[0] else 0
        elif self.tick >= MAX_TICKS:
            self.winner = self.material_leader()

    def material(self, owner: int) -> int:
        return sum(e.hp + (4 if e.is_building else 0) + STATS[e.kind]["cost"] for e in self.entities.values() if e.owner == owner)

    def material_leader(self) -> int:
        a, b = self.material(0), self.material(1)
        return 0 if a > b else 1 if b > a else -1          # -1: draw

    # ------------------------------------------------------------------------------------------ options for a decision
    def options(self, eid: int, k_targets: int = 4) -> List[Tuple[str, Tuple]]:
        """[(text, order)] for the entity: every legal order, targets limited to the nearest few."""
        e = self.entities[eid]
        out: List[Tuple[str, Tuple]] = []
        foes = sorted((t for t in self.entities.values() if t.owner != e.owner), key=lambda t: (self.dist(e.pos, t.pos), t.id))
        base = self.base(e.owner)
        if e.is_building:
            if e.kind == "base":
                if self.resources[e.owner] >= STATS["worker"]["cost"]:
                    out.append(("train worker", ("train", "worker")))
            else:
                for kind in SOLDIERS:
                    if self.resources[e.owner] >= STATS[kind]["cost"]:
                        out.append((f"train {kind}", ("train", kind)))
            out.append(("wait", ("hold",)))
            return out
        if STATS[e.kind]["atk"] > 0:
            for t in foes[:k_targets]:
                out.append((f"attack enemy {t.kind}#{t.id}", ("attack", t.id)))
        if e.kind == "worker":
            for pt in sorted((p for p in self.patches.values() if p.amount > 0), key=lambda p: (self.dist(e.pos, p.pos), p.id))[:3]:
                if e.carry < CARRY_MAX:
                    out.append((f"harvest patch#{pt.id}", ("harvest", pt.id)))
            if e.carry > 0 and base is not None:
                out.append(("return resources to base", ("return",)))
            if self.resources[e.owner] >= STATS["barracks"]["cost"] and base is not None:
                for cell in self._build_cells(e):
                    out.append((f"build barracks at ({cell[0]},{cell[1]})", ("build", cell)))
        if self._anchor(e.owner) is not None:
            out.append(("move to own base", ("move", self._rally(e.owner))))
        eb = self._anchor(1 - e.owner)
        if eb is not None and STATS[e.kind]["atk"] > 0:
            out.append(("move to enemy base", ("move", self._staging(e.owner))))
        out.append(("hold position", ("hold",)))
        return out

    def _anchor(self, owner: int) -> Optional[Entity]:
        """The owner's base, else any building, else None."""
        return self.base(owner) or next(iter(self.mine(owner, "barracks")), None)

    def _rally(self, owner: int) -> Pos:
        b = self._anchor(owner)
        if b is None:
            return (self.W // 2, self.H // 2)
        step = (1, 1) if b.pos[0] < self.W // 2 else (-1, -1)
        for k in (2, 1, 0, 3):
            p = (b.pos[0] + step[0] * k, b.pos[1] + step[1] * k)
            if 0 <= p[0] < self.W and 0 <= p[1] < self.H and p not in self.walls:
                return p
        return b.pos

    def _staging(self, owner: int) -> Pos:
        b = self._anchor(1 - owner)
        if b is None:
            return (self.W // 2, self.H // 2)
        step = (-1, -1) if b.pos[0] > self.W // 2 else (1, 1)
        for k in (2, 3, 1):
            p = (b.pos[0] + step[0] * k, b.pos[1] + step[1] * k)
            if 0 <= p[0] < self.W and 0 <= p[1] < self.H and p not in self.walls:
                return p
        return b.pos

    def _build_cells(self, e: Entity) -> List[Pos]:
        base = self.base(e.owner)
        occ = self.occupied()
        cells = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                p = (base.pos[0] + dx, base.pos[1] + dy)
                if 2 <= abs(dx) + abs(dy) <= 3 and 0 <= p[0] < self.W and 0 <= p[1] < self.H and p not in self.walls and p not in occ and all(pt.pos != p for pt in self.patches.values()):
                    cells.append(p)
        cells.sort(key=lambda p: (self.dist(p, e.pos), p))
        return cells[:2]

    # ------------------------------------------------------------------------------------------ text
    def describe(self, owner: int, eid: int) -> Tuple[str, str, List[str], List[Tuple]]:
        """(state, question, option texts, orders) for one decision.

        Compact on purpose (~250 tokens): the walls are left out — orders are pathfound, so a wall never changes what to order —
        and only the units CLOSEST to the deciding entity are listed; the full army sizes are in the 'Army' line."""
        e = self.entities[eid]
        opts = self.options(eid)
        me, foe = owner, 1 - owner

        def line(x: Entity) -> str:
            extra = f" carry {x.carry}" if x.carry else ""
            busy = f" training {x.building}" if x.is_building and x.busy and x.building else ""
            return f"{x.kind}#{x.id} ({x.pos[0]},{x.pos[1]}) hp {x.hp}/{x.max_hp}{extra}{busy}"

        near = lambda xs, k: sorted(xs, key=lambda x: (self.dist(e.pos, x.pos), x.id))[:k]      # noqa: E731
        own_b = [line(x) for x in sorted(self.mine(me), key=lambda x: x.id) if x.is_building]
        foe_b = [line(x) for x in sorted(self.mine(foe), key=lambda x: x.id) if x.is_building]
        own_u = [line(x) for x in near([u for u in self.units(me) if u.id != e.id], 6)]
        foe_u = [line(x) for x in near(self.units(foe), 5)]
        count = lambda o, k: len([x for x in self.units(o) if x.kind == k])      # noqa: E731
        comp = lambda o: ", ".join(f"{count(o, k)} {k}" for k in UNITS if count(o, k)) or "no units"      # noqa: E731
        patches = "; ".join(f"patch#{p.id} ({p.pos[0]},{p.pos[1]}) {p.amount} left" for p in near_patches(self, e)) or "none left"
        state = (
            f"MiniRTS {self.W}x{self.H}, tick {self.tick}/{MAX_TICKS}. You are player {me}. Resources: you {self.resources[me]}, enemy {self.resources[foe]}.\n"
            f"Army: you {comp(me)}; enemy {comp(foe)}.\n"
            f"Your buildings: {'; '.join(own_b) or 'none'}\nEnemy buildings: {'; '.join(foe_b) or 'none'}\n"
            f"Your nearest units: {'; '.join(own_u) or 'none'}\nNearest enemy units: {'; '.join(foe_u) or 'none'}\n"
            f"Minerals: {patches}\n"
            f"Deciding for: {line(e)}"
        )
        return state, "What should this unit do next?" if not e.is_building else "What should this building do next?", [t for t, _ in opts], [o for _, o in opts]


def near_patches(game: "RTS", e: Entity, k: int = 3) -> List[Patch]:
    return sorted((p for p in game.patches.values() if p.amount > 0), key=lambda p: (game.dist(e.pos, p.pos), p.id))[:k]
