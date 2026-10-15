"""Playable versions of the game tasks Leeth 2 is trained on, for CLOSED-LOOP evaluation.

Per-step accuracy ("is this move optimal?") hides what matters when a model plays: a policy that is 95 % right per step
still dies in a snake game after a few dozen moves, and positions the training data never produced (the second player in
tic-tac-toe) are exactly where it slips. An ``Env`` therefore

  * shows the CURRENT position as a ``Sample`` (same text and option formats the training generators use, with a fresh
    wording each move, soft target = the perfect expert's optimal set),
  * applies the move the policy chose and moves on, until the game ends or a step cap is reached.

``rollout`` plays many games in lock-step so a model can answer all of their positions in one batched forward pass.
Pure Python, no torch: the expert and a random policy run anywhere, so the environments themselves are tested locally.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .data.synth import games as G
from .schema import Sample

Cell = Tuple[int, int]
Policy = Callable[[List[Sample]], List[int]]


class Env:
    name = "env"
    done = False
    spaced = False        # True: every grid cell is its own token ("# . # ."), so columns line up for the tokenizer

    def _join(self, cells) -> str:
        return (" " if self.spaced else "").join(cells)

    def _sym(self, ch: str) -> str:
        """A cell symbol that survives spacing: a blank cell would vanish between the separators."""
        return "-" if (self.spaced and ch == " ") else ch

    def decision(self, rng: random.Random) -> Sample:        # the position to decide on
        raise NotImplementedError

    def apply(self, option: int) -> None:                    # play option ``option`` of the last decision
        raise NotImplementedError

    def result(self) -> Dict[str, float]:
        raise NotImplementedError


def _targets(sample: Sample) -> List[int]:
    t = sample.target
    return [i for i, p in enumerate(t) if p > 0] if isinstance(t, (list, tuple)) else [int(t)]


# ------------------------------------------------------------------------------------------------ snake
class SnakeEnv(Env):
    name = "snake"

    def __init__(self, rng: random.Random, max_size: int = 9, spaced: bool = False):
        self.spaced = spaced
        game = None
        while game is None:
            game = G._random_snake_game(rng, max_size=max_size)
        self.H, self.W, body, self.food = game
        self.body: List[Cell] = list(body)
        self.sym = tuple(self._sym(x) for x in rng.choice([("H", "F", "B", "."), ("@", "*", "o", "."), ("S", "$", "s", "_"), ("^", "+", "#", " ")]))
        self.lang = rng.choice(["ru", "en"])
        self.style = rng.choice(["ru", "en", "arrow", "compass"]) if self.lang == "en" else rng.choice(["ru", "arrow", "en"])
        self.cap = 3 * self.H * self.W
        self.steps = self.eaten = 0
        self.dead = False
        self._names: Dict[str, str] = {}
        self._options: List[str] = []

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.dead or self.steps >= self.cap

    def decision(self, rng: random.Random) -> Sample:
        head, body, bs = self.body[0], self.body, set(self.body)
        opt = G.snake_optimal(head, self.food, body, self.H, self.W)
        sym = self.sym
        rows = [self._join(sym[0] if (r, c) == head else sym[1] if (r, c) == self.food else sym[2] if (r, c) in bs else sym[3] for c in range(self.W)) for r in range(self.H)]
        if self.lang == "ru":
            legend = f"Змейка {self.H}x{self.W}. Символы: {sym[0]} — голова, {sym[1]} — еда, {sym[2]} — тело, {sym[3]!r} — пусто."
            q = rng.choice(["Какой ход лучше всего?", "Куда двигаться, чтобы добраться до еды и не погибнуть?", "Выберите оптимальный ход."])
        else:
            legend = f"Snake game {self.H}x{self.W}. Symbols: {sym[0]} = head, {sym[1]} = food, {sym[2]} = body, {sym[3]!r} = empty."
            q = rng.choice(["Which move is best?", "Where should the snake go to reach the food without dying?", "Choose the optimal move."])
        _, self._names = G._moves_text(rng, self.style)
        s = None
        while s is None:      # _finish returns None when no move is legal or two names collide: re-draw the wording
            if not any(opt.values()):
                opt = {d: True for d in G.DIR_ORDER}      # trapped: every move is fatal, any is as good as another
            s = G._finish("game_snake", self.lang, legend + "\n" + "\n".join(rows), q, opt, self._names, rng)
            if s is None:
                _, self._names = G._moves_text(rng, self.style)
        self._options = list(s.options or [])
        return s

    def apply(self, option: int) -> None:
        d = next(k for k, v in self._names.items() if v == self._options[option])
        dr, dc = G.DIRS[d]
        head = self.body[0]
        nxt = (head[0] + dr, head[1] + dc)
        self.steps += 1
        if not (0 <= nxt[0] < self.H and 0 <= nxt[1] < self.W) or nxt in set(self.body):
            self.dead = True
            return
        if nxt == self.food:
            self.body.insert(0, nxt)
            self.eaten += 1
            free = [(r, c) for r in range(self.H) for c in range(self.W) if (r, c) not in set(self.body)]
            if not free:
                self.steps = self.cap          # the board is full: perfect game
                return
            self.food = random.Random(self.steps * 7919 + self.eaten).choice(free)
        else:
            self.body.insert(0, nxt)
            self.body.pop()

    def result(self) -> Dict[str, float]:
        return {"eaten": float(self.eaten), "died": float(self.dead), "steps": float(self.steps)}


# ------------------------------------------------------------------------------------------------ maze
class MazeEnv(Env):
    name = "maze"

    def __init__(self, rng: random.Random, spaced: bool = False):
        self.spaced = spaced
        cw, ch = rng.choice([(3, 3), (4, 4), (5, 5), (3, 4), (4, 3), (4, 5), (5, 4), (6, 5), (5, 6)])
        grid = G._carve_maze(cw, ch, rng)
        self.H, self.W = len(grid), len(grid[0])
        free = [(r, c) for r in range(self.H) for c in range(self.W) if grid[r][c] == "."]
        while True:
            self.pos, self.goal = rng.sample(free, 2)
            self.walls = {(r, c) for r in range(self.H) for c in range(self.W) if grid[r][c] == "#"}
            self.shortest = G._bfs_dist(self.pos, self.goal, self.walls, self.H, self.W)
            if self.shortest is not None and self.shortest >= 2:
                break
        self.wall_c, self.free_c = (self._sym(x) for x in rng.choice([("#", "."), ("█", " "), ("X", "_")]))
        self.lang = rng.choice(["ru", "en"])
        self.style = rng.choice(["ru", "arrow"]) if self.lang == "ru" else rng.choice(["en", "arrow", "compass"])
        self.cap = 3 * self.shortest + 6
        self.steps = self.bumps = 0
        self.solved = False
        self._names: Dict[str, str] = {}
        self._options: List[str] = []

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.solved or self.steps >= self.cap

    def decision(self, rng: random.Random) -> Sample:
        total = G._bfs_dist(self.pos, self.goal, self.walls, self.H, self.W)
        opt = {}
        for n, (dr, dc) in G.DIRS.items():
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            ok = 0 <= nxt[0] < self.H and 0 <= nxt[1] < self.W and nxt not in self.walls
            opt[n] = bool(ok and G._bfs_dist(nxt, self.goal, self.walls, self.H, self.W) == total - 1)
        rows = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                row.append("S" if (r, c) == self.pos else "E" if (r, c) == self.goal else self.wall_c if (r, c) in self.walls else self.free_c)
            rows.append(self._join(row))
        if self.lang == "ru":
            legend = f"Лабиринт {self.H}x{self.W}. S — вы, E — выход, {self.wall_c} — стена, {self.free_c!r} — проход."
            q = rng.choice(["Какой шаг ведёт к выходу кратчайшим путём?", "Куда сделать шаг?"])
        else:
            legend = f"Maze {self.H}x{self.W}. S = you, E = exit, {self.wall_c} = wall, {self.free_c!r} = path."
            q = rng.choice(["Which step leads to the exit along a shortest path?", "Where should you step?"])
        s = None
        while s is None:
            _, self._names = G._moves_text(rng, self.style)
            s = G._finish("game_maze", self.lang, legend + "\n" + "\n".join(rows), q, opt, self._names, rng)
        self._options = list(s.options or [])
        return s

    def apply(self, option: int) -> None:
        d = next(k for k, v in self._names.items() if v == self._options[option])
        dr, dc = G.DIRS[d]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        self.steps += 1
        if 0 <= nxt[0] < self.H and 0 <= nxt[1] < self.W and nxt not in self.walls:
            self.pos = nxt
            self.solved = nxt == self.goal
        else:
            self.bumps += 1

    def result(self) -> Dict[str, float]:
        return {"solved": float(self.solved), "steps": float(self.steps), "optimal": float(self.shortest), "bumps": float(self.bumps)}


# ------------------------------------------------------------------------------------------------ treasure
class TreasureEnv(Env):
    name = "treasure"

    def __init__(self, rng: random.Random, spaced: bool = False):
        self.spaced = spaced
        while True:
            self.H, self.W = rng.randint(5, 10), rng.randint(5, 10)
            cells = [(r, c) for r in range(self.H) for c in range(self.W)]
            rng.shuffle(cells)
            self.pos = cells.pop()
            self.chests = [cells.pop() for _ in range(rng.randint(1, 3))]
            self.traps = {cells.pop() for _ in range(rng.randint(2, max(3, self.H * self.W // 8)))}
            self.walls = {cells.pop() for _ in range(rng.randint(0, self.H * self.W // 6))}
            blocked = self.traps | self.walls
            ds = [G._bfs_dist(self.pos, ch, blocked, self.H, self.W) for ch in self.chests]
            ds = [d for d in ds if d is not None]
            if ds:
                self.shortest = min(ds)
                break
        self.lang = rng.choice(["ru", "en"])
        self.style = rng.choice(["ru", "arrow"]) if self.lang == "ru" else rng.choice(["en", "arrow", "compass"])
        self.cap = 3 * self.shortest + 6
        self.steps = 0
        self.reached = self.dead = False
        self._names: Dict[str, str] = {}
        self._options: List[str] = []

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.reached or self.dead or self.steps >= self.cap

    def decision(self, rng: random.Random) -> Sample:
        blocked = self.traps | self.walls
        dists = [(d, ch) for ch in self.chests for d in [G._bfs_dist(self.pos, ch, blocked, self.H, self.W)] if d is not None]
        dmin = min(d for d, _ in dists)
        opt = {}
        for n, (dr, dc) in G.DIRS.items():
            nxt = (self.pos[0] + dr, self.pos[1] + dc)
            ok = 0 <= nxt[0] < self.H and 0 <= nxt[1] < self.W and nxt not in blocked
            opt[n] = bool(ok and any(G._bfs_dist(nxt, ch, blocked, self.H, self.W) == dmin - 1 for _, ch in dists))
        rows = [self._join("P" if (r, c) == self.pos else "T" if (r, c) in self.chests else "^" if (r, c) in self.traps else "#" if (r, c) in self.walls else "." for c in range(self.W)) for r in range(self.H)]
        if self.lang == "ru":
            state = f"Карта {self.H}x{self.W}: P — вы, T — сундук, ^ — ловушка (смертельна), # — стена, '.' — пусто.\n" + "\n".join(rows)
            q = rng.choice(["Куда идти, чтобы быстрее добраться до ближайшего сундука?", "Выберите лучший шаг."])
        else:
            state = f"Map {self.H}x{self.W}: P = you, T = treasure chest, ^ = trap (deadly), # = wall, '.' = empty.\n" + "\n".join(rows)
            q = rng.choice(["Where should you go to reach the nearest chest fastest?", "Choose the best step."])
        s = None
        while s is None:
            _, self._names = G._moves_text(rng, self.style)
            s = G._finish("game_treasure", self.lang, state, q, opt, self._names, rng)
        self._options = list(s.options or [])
        return s

    def apply(self, option: int) -> None:
        d = next(k for k, v in self._names.items() if v == self._options[option])
        dr, dc = G.DIRS[d]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        self.steps += 1
        if not (0 <= nxt[0] < self.H and 0 <= nxt[1] < self.W) or nxt in self.walls:
            return
        self.pos = nxt
        self.dead = nxt in self.traps
        self.reached = nxt in self.chests

    def result(self) -> Dict[str, float]:
        return {"reached": float(self.reached), "died": float(self.dead), "steps": float(self.steps), "optimal": float(self.shortest)}


# ------------------------------------------------------------------------------------------------ tic-tac-toe
class TttEnv(Env):
    """A full game: X always moves first, so the model meets BOTH kinds of position (equal marks, and one more X than O)."""

    name = "ttt"

    def __init__(self, rng: random.Random, me: Optional[str] = None, opponent: str = "random", spaced: bool = False):
        self.spaced = spaced
        self.me = me or rng.choice(["X", "O"])
        self.opponent = opponent
        self.b: Tuple[str, ...] = tuple("." for _ in range(9))
        self.turn = "X"
        self.lang = rng.choice(["ru", "en"])
        self.style = rng.choice(["rc", "num", "word"])
        self._rng = random.Random(rng.random())
        self._empty: List[int] = []
        self.moves = 0
        self._opponent_moves()

    @property
    def done(self) -> bool:  # type: ignore[override]
        return G._winner(self.b) is not None or all(x != "." for x in self.b)

    def _place(self, i: int) -> None:
        self.b = self.b[:i] + (self.turn,) + self.b[i + 1:]
        self.turn = "O" if self.turn == "X" else "X"

    def _opponent_moves(self) -> None:
        """Play the opponent's moves until it is the model's turn or the game is over."""
        while not self.done and self.turn != self.me:
            empty = [i for i in range(9) if self.b[i] == "."]
            self._place(self._rng.choice(empty) if self.opponent == "random" else self._rng.choice(G._ttt_best(self.b, self.turn)))

    def decision(self, rng: random.Random) -> Sample:
        me = self.me
        best = G._ttt_best(self.b, me)
        empty = [i for i in range(9) if self.b[i] == "."]
        lang, style = self.lang, self.style

        def cell(i: int) -> str:
            r, c = divmod(i, 3)
            if style == "rc":
                return f"({r + 1},{c + 1})" if lang == "en" else f"({r + 1};{c + 1})"
            if style == "num":
                return f"cell {i + 1}" if lang == "en" else f"клетка {i + 1}"
            rows = ["top", "middle", "bottom"] if lang == "en" else ["верх", "середина", "низ"]
            cols = ["left", "center", "right"] if lang == "en" else ["лево", "центр", "право"]
            return f"{rows[r]}-{cols[c]}"

        board = "\n".join(self._join(self.b[r * 3: r * 3 + 3]) for r in range(3))
        if lang == "ru":
            state = f"Крестики-нолики 3x3. Вы играете за {me}. Клетки: X, O, '.' — пусто. Ваш ход.\n{board}\nНумерация клеток: слева направо, сверху вниз, от 1 до 9."
            q = rng.choice(["Какой ход оптимален?", "Куда поставить свой знак?"])
        else:
            state = f"Tic-tac-toe 3x3. You play {me}. Cells: X, O, '.' = empty. Your move.\n{board}\nCells are numbered left to right, top to bottom, 1 to 9."
            q = rng.choice(["Which move is optimal?", "Where should you place your mark?"])
        order = list(empty)
        rng.shuffle(order)
        self._empty = order
        k = sum(1 for i in order if i in best)
        probs = [(1.0 / k) if i in best else 0.0 for i in order]
        if len(order) < 2:        # a single legal move: the generators never ask this, the policy cannot go wrong
            return Sample("game_ttt", "game", lang, "choice", state, q, [cell(order[0]), "—"], [1.0, 0.0])
        return Sample("game_ttt", "game", lang, "choice", state, q, [cell(i) for i in order], probs if k > 1 else probs.index(1.0))

    def apply(self, option: int) -> None:
        if option >= len(self._empty):
            return
        self._place(self._empty[option])
        self.moves += 1
        self._opponent_moves()

    def result(self) -> Dict[str, float]:
        w = G._winner(self.b)
        return {"won": float(w == self.me), "lost": float(w is not None and w != self.me), "drawn": float(w is None)}


ENVS: Dict[str, Callable[[random.Random], Env]] = {
    "snake": SnakeEnv,
    "maze": MazeEnv,
    "treasure": TreasureEnv,
    "ttt_vs_random": lambda rng: TttEnv(rng, opponent="random"),
    "ttt_vs_perfect": lambda rng: TttEnv(rng, opponent="perfect"),
}
#: the same games with one token per grid cell
ENVS_SPACED: Dict[str, Callable[[random.Random], Env]] = {
    "snake": lambda rng: SnakeEnv(rng, spaced=True),
    "maze": lambda rng: MazeEnv(rng, spaced=True),
    "treasure": lambda rng: TreasureEnv(rng, spaced=True),
    "ttt_vs_random": lambda rng: TttEnv(rng, opponent="random", spaced=True),
    "ttt_vs_perfect": lambda rng: TttEnv(rng, opponent="perfect", spaced=True),
}


# ------------------------------------------------------------------------------------------------ playing
def expert_policy(samples: List[Sample]) -> List[int]:
    return [_targets(s)[0] for s in samples]


def random_policy(seed: int = 0) -> Policy:
    rng = random.Random(seed)
    return lambda samples: [rng.randrange(len(s.options or [0])) for s in samples]


def rollout(make: Callable[[random.Random], Env], n: int, policy: Policy, seed: int = 0, max_decisions: int = 400) -> List[Dict[str, float]]:
    """Play ``n`` games in lock-step; each tick the policy answers every unfinished game's position in one call.
    Every move is also scored against the expert's optimal set (``optimal_rate``), so one run gives both numbers."""
    rng = random.Random(seed)
    envs = [make(random.Random(rng.random())) for _ in range(n)]
    hits = [0] * n
    asked = [0] * n
    for _ in range(max_decisions):
        live = [i for i, e in enumerate(envs) if not e.done]
        if not live:
            break
        samples = [envs[i].decision(rng) for i in live]
        choices = policy(samples)
        for i, s, c in zip(live, samples, choices):
            asked[i] += 1
            hits[i] += int(c in _targets(s))
            envs[i].apply(c)
    out = []
    for e, h, a in zip(envs, hits, asked):
        r = e.result()
        r["optimal_rate"] = h / max(1, a)
        r["decisions"] = float(a)
        out.append(r)
    return out


def summarize(results: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in results for k in r})
    return {k: sum(r.get(k, 0.0) for r in results) / max(1, len(results)) for k in keys}


def collect(make: Callable[[random.Random], Env], n: int, policy: Policy, beta: float = 0.0, seed: int = 0, keep: float = 0.5, max_decisions: int = 400) -> Tuple[List[Sample], Dict[str, float]]:
    """DAgger rollouts: the POLICY plays ``n`` games (with probability ``beta`` a move is the expert's instead), every visited position is
    kept with probability ``keep`` as a training sample whose target is the expert's optimal set. Returns (samples, summary of the games)."""
    rng = random.Random(seed)
    envs = [make(random.Random(rng.random())) for _ in range(n)]
    out: List[Sample] = []
    for _ in range(max_decisions):
        live = [e for e in envs if not e.done]
        if not live:
            break
        samples = [e.decision(rng) for e in live]
        choices = policy(samples)
        for e, s, c in zip(live, samples, choices):
            if rng.random() < keep:
                out.append(s)
            e.apply(rng.choice(_targets(s)) if beta and rng.random() < beta else c)
    return out, summarize([e.result() for e in envs])
