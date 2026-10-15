"""More games for Leeth 2 -- chosen so that "play a game" cannot be learned as "know these four games".

Each game below has a different kind of move (drop in a column, push a box, slide a tile, take from a pile, press a light, move a
disc, walk with keys, walk the cheapest way) and an EXACT expert (full search, a distance table or linear algebra over GF(2)), so
every label is the true optimal set, never a guess.  And every position states the RULES in its own text ("Rules: ..."): a model
that learns to read them can sit down at a game it has never seen.  Two games, Chomp and Cram, exist only to test that: they are
never in the training data.

Same contract as ``envs.py``: ``decision(rng)`` shows the CURRENT position as a ``Sample`` (soft target = the optimal set),
``apply(option)`` plays the chosen option, ``result()`` summarises the game.  Pure Python, deterministic given the rng.
"""

from __future__ import annotations

import random
from collections import deque
from functools import lru_cache
from itertools import product
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .envs import Env
from .schema import Sample

TRAINED = ("connect", "sokoban", "slide", "nim", "keydoor", "lights", "hanoi", "terrain")
EXAM_ONLY = ("chomp", "cram")


# ------------------------------------------------------------------------------------------------ shared plumbing
class _Game(Env):
    """What every game here shares: language, one-token-per-cell grids, and turning "these actions, these are good" into a Sample."""

    task = "game_x"
    spaced = True

    def __init__(self, rng: random.Random, spaced: bool = True):
        self.spaced = spaced
        self.lang = rng.choice(["ru", "en"])
        self._actions: List[Any] = []
        self.steps = 0

    def L(self, ru: str, en: str) -> str:
        return ru if self.lang == "ru" else en

    def _sample(self, rng: random.Random, state: str, question: str, actions: Sequence[Any], texts: Sequence[str], good: Set[int]) -> Sample:
        """``good`` indexes ``actions``.  Options are shuffled; duplicate wordings get a numeric suffix; a lone legal move gets a '—' companion."""
        idx = list(range(len(actions)))
        rng.shuffle(idx)
        acts = [actions[i] for i in idx]
        txt = [texts[i] for i in idx]
        flags = [i in good for i in idx]
        seen: Dict[str, int] = {}
        for k, t in enumerate(txt):
            seen[t] = seen.get(t, 0) + 1
            if seen[t] > 1:
                txt[k] = f"{t} ({seen[t]})"
        if len(acts) == 1:
            acts.append(None)
            txt.append("—")
            flags.append(False)
        if not any(flags):
            flags = [a is not None for a in acts]               # nothing is better than anything else: any real move is as good as another
        k = sum(flags)
        probs = [(1.0 / k) if f else 0.0 for f in flags]
        self._actions = acts
        return Sample(self.task, "game", self.lang, "choice", state, question, txt, probs if k > 1 else probs.index(1.0))

    def _chosen(self, option: int) -> Any:
        return self._actions[option] if 0 <= option < len(self._actions) else None

    def grid(self, rows: Sequence[Sequence[str]]) -> str:
        return "\n".join(self._join(r) for r in rows)


def _bfs_reverse(nodes: Dict[Any, List[Any]], goals: Sequence[Any]) -> Dict[Any, int]:
    """Distance to the nearest goal for every node of a directed graph given as {node: [successors]}."""
    rev: Dict[Any, List[Any]] = {}
    for u, vs in nodes.items():
        for v in vs:
            rev.setdefault(v, []).append(u)
    dist = {g: 0 for g in goals}
    q = deque(goals)
    while q:
        v = q.popleft()
        for u in rev.get(v, ()):
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


# ------------------------------------------------------------------------------------------------ connect-K
@lru_cache(maxsize=None)
def _windows(R: int, C: int, K: int) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    out = []
    for r, c in product(range(R), range(C)):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = tuple((r + i * dr, c + i * dc) for i in range(K))
            if all(0 <= rr < R and 0 <= cc < C for rr, cc in cells):
                out.append(cells)
    return tuple(out)


def _c_legal(board: Tuple[str, ...]) -> List[int]:
    return [c for c, ch in enumerate(board[0]) if ch == "."]


def _c_drop(board: Tuple[str, ...], c: int, p: str) -> Tuple[Tuple[str, ...], int]:
    r = max(i for i, row in enumerate(board) if row[c] == ".")
    row = board[r][:c] + p + board[r][c + 1:]
    return board[:r] + (row,) + board[r + 1:], r


def _c_wins(board: Tuple[str, ...], K: int, p: str) -> bool:
    R, C = len(board), len(board[0])
    return any(all(board[r][c] == p for r, c in w) for w in _windows(R, C, K))


def _c_eval(board: Tuple[str, ...], K: int, me: str) -> int:
    R, C = len(board), len(board[0])
    other = "O" if me == "X" else "X"
    score = 0
    for w in _windows(R, C, K):
        mine = sum(board[r][c] == me for r, c in w)
        theirs = sum(board[r][c] == other for r, c in w)
        if mine and theirs:
            continue
        if mine:
            score += (1, 6, 40, 300)[min(mine, 4) - 1] if mine < K else 0
        elif theirs:
            score -= (1, 6, 40, 300)[min(theirs, 4) - 1] if theirs < K else 0
    mid = C // 2
    score += sum(2 for r in range(R) if board[r][mid] == me) - sum(2 for r in range(R) if board[r][mid] == other)
    return score


@lru_cache(maxsize=400000)
def _c_negamax(board: Tuple[str, ...], K: int, p: str, depth: int, alpha: int, beta: int) -> int:
    """Value for ``p`` to move (higher is better for p). A win is worth more the sooner it comes."""
    other = "O" if p == "X" else "X"
    moves = _c_legal(board)
    if not moves:
        return 0
    if depth == 0:
        return _c_eval(board, K, p)
    C = len(board[0])
    best = -10 ** 6
    for c in sorted(moves, key=lambda x: abs(x - C // 2)):
        nb, _ = _c_drop(board, c, p)
        if _c_wins(nb, K, p):
            v = 10000 + depth
        else:
            v = -_c_negamax(nb, K, other, depth - 1, -beta, -alpha)
        if v > best:
            best = v
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best


def connect_values(board: Tuple[str, ...], K: int, p: str, depth: int) -> Dict[int, int]:
    """Exact search value of every legal column for ``p`` to move (no pruning at the root, so ties are real ties)."""
    other = "O" if p == "X" else "X"
    out: Dict[int, int] = {}
    for c in _c_legal(board):
        nb, _ = _c_drop(board, c, p)
        out[c] = 10000 + depth if _c_wins(nb, K, p) else -_c_negamax(nb, K, other, depth - 1, -10 ** 7, 10 ** 7)
    return out


class ConnectEnv(_Game):
    """Connect-3 / Connect-4 on boards of several sizes. X moves first; the model plays either side."""

    name = "connect"
    task = "game_connect"
    SIZES = [(4, 5, 3), (4, 4, 3), (5, 5, 3), (5, 5, 4), (5, 6, 4), (6, 7, 4)]

    def __init__(self, rng: random.Random, opponent: str = "random", spaced: bool = True):
        super().__init__(rng, spaced)
        self.R, self.C, self.K = rng.choice(self.SIZES)
        self.depth = 6 if self.K == 3 and self.C <= 5 else (4 if self.C <= 6 else 3)
        self.me = rng.choice(["X", "O"])
        self.opponent = opponent
        self.board: Tuple[str, ...] = tuple("." * self.C for _ in range(self.R))
        self.turn = "X"
        self.style = rng.choice(["col", "drop", "num"])
        self._rng = random.Random(rng.random())
        self.moves = 0
        self._opponent_moves()

    @property
    def done(self) -> bool:  # type: ignore[override]
        return _c_wins(self.board, self.K, "X") or _c_wins(self.board, self.K, "O") or not _c_legal(self.board)

    def _play(self, c: int) -> None:
        self.board, _ = _c_drop(self.board, c, self.turn)
        self.turn = "O" if self.turn == "X" else "X"

    def _opponent_moves(self) -> None:
        while not self.done and self.turn != self.me:
            legal = _c_legal(self.board)
            if self.opponent == "random":
                c = self._rng.choice(legal)
            else:                                                       # "search": win now, block, else a shallow search
                vals = connect_values(self.board, self.K, self.turn, 2)
                top = max(vals.values())
                c = self._rng.choice([x for x, v in vals.items() if v == top])
            self._play(c)

    def decision(self, rng: random.Random) -> Sample:
        vals = connect_values(self.board, self.K, self.me, self.depth)
        top = max(vals.values())
        cols = sorted(vals)
        good = {i for i, c in enumerate(cols) if vals[c] == top}
        word = {"col": self.L("колонка", "column"), "drop": self.L("бросить в колонку", "drop in column"), "num": self.L("столбец №", "column #")}[self.style]
        texts = [f"{word} {c + 1}" if self.style != "num" else f"{word}{c + 1}" for c in cols]
        rows = self.grid([list(r) for r in self.board])
        nums = " ".join(str(i + 1) for i in range(self.C)) if self.spaced else "".join(str(i + 1) for i in range(self.C))
        rules = self.L(f"Правила: игроки по очереди бросают фишку в колонку, она падает в самую нижнюю свободную клетку. Выигрывает тот, кто первым выстроит {self.K} своих в ряд (по горизонтали, вертикали или диагонали). X ходит первым.",
                       f"Rules: players take turns dropping a disc into a column; it falls to the lowest free cell. The first to line up {self.K} of their own in a row (horizontally, vertically or diagonally) wins. X moves first.")
        state = (self.L(f"Connect-{self.K}, поле {self.R}x{self.C}. Вы играете за {self.me}. '.' — пусто. Ваш ход.", f"Connect-{self.K} on a {self.R}x{self.C} board. You play {self.me}. '.' = empty. Your move.")
                 + f"\n{rules}\n{rows}\n{nums}\n" + self.L("Колонки пронумерованы слева направо с 1.", "Columns are numbered from 1, left to right."))
        q = rng.choice([self.L("Какой ход лучший?", "Which move is best?"), self.L("В какую колонку бросить фишку?", "Which column should you drop into?")])
        return self._sample(rng, state, q, cols, texts, good)

    def apply(self, option: int) -> None:
        c = self._chosen(option)
        self.steps += 1
        if c is None or c not in _c_legal(self.board):
            return
        self._play(c)
        self.moves += 1
        self._opponent_moves()

    def result(self) -> Dict[str, float]:
        won, lost = _c_wins(self.board, self.K, self.me), _c_wins(self.board, self.K, "O" if self.me == "X" else "X")
        return {"won": float(won), "lost": float(lost), "drawn": float(not won and not lost)}


# ------------------------------------------------------------------------------------------------ sokoban
_SK_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_SK_ORDER = ["up", "down", "left", "right"]


def _sk_step(walls: FrozenSet[Tuple[int, int]], boxes: FrozenSet[Tuple[int, int]], pl: Tuple[int, int], d: str):
    dr, dc = _SK_DIRS[d]
    nxt = (pl[0] + dr, pl[1] + dc)
    if nxt in walls:
        return boxes, pl
    if nxt in boxes:
        beyond = (nxt[0] + dr, nxt[1] + dc)
        if beyond in walls or beyond in boxes:
            return boxes, pl
        return (boxes - {nxt}) | {beyond}, nxt
    return boxes, nxt


class SokobanEnv(_Game):
    name = "sokoban"
    task = "game_sokoban"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        for _ in range(400):
            if self._make_level(rng):
                break
        else:                                                           # cannot happen with these sizes; a trivial level keeps the contract
            self._trivial()
        self.style = rng.choice(["ru", "arrow"]) if self.lang == "ru" else rng.choice(["en", "arrow", "compass"])
        self.cap = 3 * self.optimal + 8
        self.solved = self.stuck = False
        self._names: Dict[str, str] = {}

    def _make_level(self, rng: random.Random) -> bool:
        h, w = rng.randint(3, 5), rng.randint(3, 5)
        n = rng.choice([1, 1, 2, 2, 3]) if h * w >= 16 else rng.choice([1, 2])
        cells = [(r, c) for r in range(1, h + 1) for c in range(1, w + 1)]
        rng.shuffle(cells)
        self.H, self.W = h + 2, w + 2
        wall = {(r, c) for r in range(self.H) for c in range(self.W) if r in (0, self.H - 1) or c in (0, self.W - 1)}
        inner = rng.randint(0, 3)
        if len(cells) < 2 * n + 1 + inner:
            return False
        wall |= {cells.pop() for _ in range(inner)}
        self.goals = frozenset(cells.pop() for _ in range(n))
        boxes = frozenset(cells.pop() for _ in range(n))
        if boxes == self.goals:
            return False
        self.walls = frozenset(wall)
        self.boxes = boxes
        self.pl = cells.pop()
        return self._solve()

    def _trivial(self) -> None:
        self.H = self.W = 5
        self.walls = frozenset((r, c) for r in range(5) for c in range(5) if r in (0, 4) or c in (0, 4))
        self.pl, self.boxes, self.goals = (1, 1), frozenset({(2, 2)}), frozenset({(2, 3)})
        assert self._solve()

    def _solve(self) -> bool:
        """The whole reachable state graph (a few thousand states) and every state's distance to a solved one."""
        start = (self.pl, self.boxes)
        graph: Dict[Any, List[Any]] = {}
        q = deque([start])
        graph[start] = []
        while q:
            s = q.popleft()
            pl, boxes = s
            for d in _SK_ORDER:
                nb, np_ = _sk_step(self.walls, boxes, pl, d)
                t = (np_, nb)
                graph[s].append(t)
                if t not in graph:
                    graph[t] = []
                    q.append(t)
            if len(graph) > 30000:
                return False
        goals = [s for s in graph if s[1] == self.goals]
        dist = _bfs_reverse(graph, goals)
        if start not in dist or not (3 <= dist[start] <= 40):
            return False
        self.graph, self.dist, self.optimal = graph, dist, dist[start]
        return True

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.solved or self.stuck or self.steps >= self.cap

    def decision(self, rng: random.Random) -> Sample:
        s = (self.pl, self.boxes)
        d0 = self.dist[s]
        acts, texts, good = [], [], set()
        succ = dict(zip(_SK_ORDER, self.graph[s]))
        _, self._names = _moves(rng, self.style)
        for i, d in enumerate(_SK_ORDER):
            acts.append(d)
            texts.append(self._names[d])
            if self.dist.get(succ[d], 10 ** 9) == d0 - 1:
                good.add(i)
        rows = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                p = (r, c)
                row.append("#" if p in self.walls else "+" if p == self.pl and p in self.goals else "@" if p == self.pl else "*" if p in self.boxes and p in self.goals
                           else "$" if p in self.boxes else "." if p in self.goals else "-")
            rows.append(row)
        rules = self.L("Правила: вы толкаете ящики, шагая в их сторону; ящик нельзя тянуть, а толкать можно только в свободную клетку (не в стену и не в другой ящик). Цель — поставить все ящики на цели.",
                       "Rules: you push boxes by walking into them; a box cannot be pulled, and can only be pushed into a free cell (not a wall or another box). The goal is to put every box on a target.")
        legend = self.L("Сокобан. @ — вы, $ — ящик, '.' — цель, * — ящик на цели, + — вы на цели, # — стена, '-' — пол.", "Sokoban. @ = you, $ = box, '.' = target, * = box on a target, + = you on a target, # = wall, '-' = floor.")
        state = f"{legend}\n{rules}\n{self.grid(rows)}"
        q = rng.choice([self.L("Какой шаг приближает к решению?", "Which step brings you closer to solving it?"), self.L("Куда сделать шаг?", "Where should you step?")])
        return self._sample(rng, state, q, acts, texts, good)

    def apply(self, option: int) -> None:
        d = self._chosen(option)
        self.steps += 1
        if d is None:
            return
        self.boxes, self.pl = _sk_step(self.walls, self.boxes, self.pl, d)
        s = (self.pl, self.boxes)
        self.solved = self.boxes == self.goals
        self.stuck = s not in self.dist

    def result(self) -> Dict[str, float]:
        return {"solved": float(self.solved), "steps": float(self.steps), "optimal": float(self.optimal), "stuck": float(self.stuck)}


def _moves(rng: random.Random, style: str) -> Tuple[List[str], Dict[str, str]]:
    from .data.synth import games as G

    return G._moves_text(rng, style)


# ------------------------------------------------------------------------------------------------ sliding puzzle
@lru_cache(maxsize=None)
def _slide_table(R: int, C: int) -> Dict[Tuple[int, ...], int]:
    goal = tuple(list(range(1, R * C)) + [0])
    dist = {goal: 0}
    q = deque([goal])
    while q:
        s = q.popleft()
        z = s.index(0)
        zr, zc = divmod(z, C)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = zr + dr, zc + dc
            if 0 <= r < R and 0 <= c < C:
                t = list(s)
                t[z], t[r * C + c] = t[r * C + c], 0
                t = tuple(t)
                if t not in dist:
                    dist[t] = dist[s] + 1
                    q.append(t)
    return dist


@lru_cache(maxsize=None)
def _slide_by_distance(R: int, C: int) -> Dict[int, List[Tuple[int, ...]]]:
    out: Dict[int, List[Tuple[int, ...]]] = {}
    for s, d in _slide_table(R, C).items():
        out.setdefault(d, []).append(s)
    return out


class SlideEnv(_Game):
    name = "slide"
    task = "game_slide"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        self.R, self.C = rng.choice([(2, 3), (3, 3), (3, 3)])
        table = _slide_table(self.R, self.C)
        by = _slide_by_distance(self.R, self.C)
        hi = max(by)
        d = rng.randint(3, min(hi, 24 if self.R * self.C == 9 else 15))
        self.s = rng.choice(by[d])
        self.optimal = d
        self.table = table
        self.style = rng.choice(["tile", "tile", "blank"])
        self.cap = 3 * d + 8
        self.steps = 0
        self.names = rng.choice(["ru", "arrow"]) if self.lang == "ru" else rng.choice(["en", "arrow", "compass"])

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.table[self.s] == 0 or self.steps >= self.cap

    def _neighbours(self) -> List[Tuple[Tuple[int, ...], int, str]]:
        z = self.s.index(0)
        zr, zc = divmod(z, self.C)
        out = []
        for name, (dr, dc) in (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))):
            r, c = zr + dr, zc + dc
            if 0 <= r < self.R and 0 <= c < self.C:
                t = list(self.s)
                tile = t[r * self.C + c]
                t[z], t[r * self.C + c] = tile, 0
                out.append((tuple(t), tile, name))
        return out

    def decision(self, rng: random.Random) -> Sample:
        d0 = self.table[self.s]
        nb = self._neighbours()
        acts = [t for t, _, _ in nb]
        if self.style == "tile":
            texts = [self.L(f"сдвинуть плитку {tile}", f"slide tile {tile}") for _, tile, _ in nb]
        else:
            _, names = _moves(rng, self.names)
            texts = [self.L("пустое место ", "blank ") + names[d] for _, _, d in nb]
        good = {i for i, t in enumerate(acts) if self.table[t] == d0 - 1}
        rows = [["_" if v == 0 else str(v) for v in self.s[r * self.C:(r + 1) * self.C]] for r in range(self.R)]
        goal = [["_" if v == 0 else str(v) for v in (list(range(1, self.R * self.C)) + [0])[r * self.C:(r + 1) * self.C]] for r in range(self.R)]
        rules = self.L("Правила: плитку, соседнюю с пустым местом '_', можно сдвинуть на него. Нужно собрать плитки по порядку, пустое место — в конце.",
                       "Rules: a tile next to the blank '_' can slide into it. Put the tiles in order, with the blank last.")
        state = (self.L(f"Пятнашки {self.R}x{self.C}.", f"Sliding puzzle {self.R}x{self.C}.") + f"\n{rules}\n" + self.L("Сейчас:", "Now:") + f"\n{self.grid(rows)}\n"
                 + self.L("Цель:", "Goal:") + f"\n{self.grid(goal)}")
        q = rng.choice([self.L("Какой ход быстрее всего ведёт к цели?", "Which move leads to the goal fastest?"), self.L("Что сдвинуть?", "What should move?")])
        return self._sample(rng, state, q, acts, texts, good)

    def apply(self, option: int) -> None:
        t = self._chosen(option)
        self.steps += 1
        if t is not None:
            self.s = t

    def result(self) -> Dict[str, float]:
        return {"solved": float(self.table[self.s] == 0), "steps": float(self.steps), "optimal": float(self.optimal)}


# ------------------------------------------------------------------------------------------------ nim
@lru_cache(maxsize=None)
def _nim_wins(piles: Tuple[int, ...], misere: bool) -> bool:
    """True if the player to move wins (exact search; piles are sorted, zeros dropped)."""
    if not piles:
        return misere                  # no objects left: in normal play the previous player took the last one and the mover lost; misère reverses it
    for i, p in enumerate(piles):
        for k in range(1, p + 1):
            nxt = tuple(sorted(x for j, x in enumerate(piles) if j != i) + ([p - k] if p - k else []))
            nxt = tuple(sorted(nxt))
            if not _nim_wins(nxt, misere):
                return True
    return False


class NimEnv(_Game):
    name = "nim"
    task = "game_nim"

    def __init__(self, rng: random.Random, opponent: str = "random", spaced: bool = True):
        super().__init__(rng, spaced)
        self.piles = [rng.randint(1, 9) for _ in range(rng.randint(2, 4))]
        self.misere = rng.random() < 0.25
        self.me_first = rng.random() < 0.5
        self.opponent = opponent
        self.turn = 0 if self.me_first else 1
        self.last = None
        self._rng = random.Random(rng.random())
        self.style = rng.choice(["take", "remove"])
        self.steps = 0
        self._opponent_moves()

    @property
    def done(self) -> bool:  # type: ignore[override]
        return sum(self.piles) == 0

    def _moves(self) -> List[Tuple[int, int]]:
        return [(i, k) for i, p in enumerate(self.piles) for k in range(1, p + 1)]

    def _take(self, i: int, k: int) -> None:
        self.piles[i] -= k
        self.last = self.turn
        self.turn = 1 - self.turn

    def _good(self, mv: Tuple[int, int]) -> bool:
        i, k = mv
        nxt = tuple(sorted(x for x in (p - k if j == i else p for j, p in enumerate(self.piles)) if x))
        return not _nim_wins(nxt, self.misere)

    def _opponent_moves(self) -> None:
        while not self.done and self.turn == 1:
            mvs = self._moves()
            if self.opponent == "perfect":
                best = [m for m in mvs if self._good(m)] or mvs
                self._take(*self._rng.choice(best))
            else:
                self._take(*self._rng.choice(mvs))

    def decision(self, rng: random.Random) -> Sample:
        mvs = self._moves()
        good = {i for i, m in enumerate(mvs) if self._good(m)}
        verb = self.L("взять", "take") if self.style == "take" else self.L("убрать", "remove")
        texts = [self.L(f"{verb} {k} из кучи {i + 1}", f"{verb} {k} from pile {i + 1}") for i, k in mvs]
        rules = (self.L("Правила: игроки по очереди берут любое количество предметов (минимум 1) из ОДНОЙ кучи. ", "Rules: players take turns removing any number of objects (at least 1) from ONE pile. ")
                 + (self.L("Проигрывает тот, кто взял последний предмет.", "Whoever takes the last object LOSES.") if self.misere
                    else self.L("Выигрывает тот, кто взял последний предмет.", "Whoever takes the last object WINS.")))
        state = (self.L("Ним. Кучи: ", "Nim. Piles: ") + ", ".join(f"{i + 1}: {p}" for i, p in enumerate(self.piles) if p) + self.L(". Ваш ход.", ". Your move.") + f"\n{rules}")
        state += "\n" + self.L("Пустые кучи не показаны, но их номера сохраняются.", "Empty piles are not shown; the numbers stay as they were.") if 0 in self.piles else ""
        q = rng.choice([self.L("Какой ход выигрывает?", "Which move wins?"), self.L("Что взять?", "What should you take?")])
        return self._sample(rng, state, q, mvs, texts, good)

    def apply(self, option: int) -> None:
        mv = self._chosen(option)
        self.steps += 1
        if mv is None:
            return
        self._take(*mv)
        self._opponent_moves()

    def result(self) -> Dict[str, float]:
        if not self.done:
            return {"won": 0.0, "lost": 0.0, "drawn": 1.0}
        took_last_is_me = self.last == 0
        won = (not took_last_is_me) if self.misere else took_last_is_me
        return {"won": float(won), "lost": float(not won), "drawn": 0.0}


# ------------------------------------------------------------------------------------------------ keys and doors
class KeyDoorEnv(_Game):
    name = "keydoor"
    task = "game_keydoor"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        for _ in range(500):
            if self._make(rng):
                break
        else:
            raise RuntimeError("could not build a key/door level")
        self.style = rng.choice(["ru", "arrow"]) if self.lang == "ru" else rng.choice(["en", "arrow", "compass"])
        self.cap = 3 * self.optimal + 8
        self.reached = False
        self.keys: FrozenSet[str] = frozenset()

    def _make(self, rng: random.Random) -> bool:
        self.H, self.W = rng.randint(5, 7), rng.randint(5, 7)
        cells = [(r, c) for r in range(self.H) for c in range(self.W)]
        rng.shuffle(cells)
        self.pos, self.goal = cells.pop(), cells.pop()
        pairs = rng.choice([1, 1, 2])
        letters = rng.sample("abc", pairs)
        self.key_at = {cells.pop(): ch for ch in letters}
        self.door_at = {cells.pop(): ch.upper() for ch in letters for _ in range(rng.randint(1, 2))}
        self.walls = {cells.pop() for _ in range(rng.randint(self.H * self.W // 8, self.H * self.W // 4))}
        self.start_pos = self.pos
        # the graph over (cell, keys held)
        masks = [frozenset(s) for s in self._subsets(letters)]
        graph: Dict[Any, List[Any]] = {}
        for r, c in product(range(self.H), range(self.W)):
            if (r, c) in self.walls:
                continue
            for m in masks:
                succ = []
                for dr, dc in _SK_DIRS.values():
                    n = (r + dr, c + dc)
                    if not (0 <= n[0] < self.H and 0 <= n[1] < self.W) or n in self.walls:
                        succ.append(((r, c), m))
                        continue
                    if n in self.door_at and self.door_at[n].lower() not in m:
                        succ.append(((r, c), m))
                        continue
                    succ.append((n, m | ({self.key_at[n]} if n in self.key_at else set())))
                graph[((r, c), m)] = [(p, frozenset(k)) for p, k in succ]
        goals = [s for s in graph if s[0] == self.goal]
        dist = _bfs_reverse(graph, goals)
        s0 = (self.pos, frozenset())
        if s0 not in dist or dist[s0] < 5:
            return False
        # it must NEED a key: without keys (doors as walls) the exit is not reachable by the same length
        blocked = self.walls | set(self.door_at)
        from .data.synth import games as G

        plain = G._bfs_dist(self.pos, self.goal, blocked, self.H, self.W)
        if plain is not None and plain <= dist[s0] and rng.random() < 0.8:
            return False
        self.graph, self.dist, self.optimal = graph, dist, dist[s0]
        return True

    @staticmethod
    def _subsets(letters: Sequence[str]) -> List[Tuple[str, ...]]:
        out: List[Tuple[str, ...]] = [()]
        for ch in letters:
            out += [t + (ch,) for t in out]
        return out

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.reached or self.steps >= self.cap

    def decision(self, rng: random.Random) -> Sample:
        s = (self.pos, self.keys)
        d0 = self.dist[s]
        succ = dict(zip(_SK_DIRS, self.graph[s]))
        _, names = _moves(rng, self.style)
        acts, texts, good = [], [], set()
        for i, d in enumerate(_SK_ORDER):
            acts.append(d)
            texts.append(names[d])
            if self.dist.get(succ[d], 10 ** 9) == d0 - 1:
                good.add(i)
        rows = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                p = (r, c)
                row.append("P" if p == self.pos else "G" if p == self.goal else "#" if p in self.walls else self.door_at[p] if p in self.door_at
                           else self.key_at[p] if p in self.key_at and self.key_at[p] not in self.keys else "-")
            rows.append(row)
        held = ", ".join(sorted(self.keys)) or self.L("нет", "none")
        rules = self.L("Правила: шаг на ключ (строчная буква) подбирает его; дверь (заглавная буква) проходима, только если у вас есть ключ с той же буквой. Стены непроходимы. Цель — добраться до G.",
                       "Rules: stepping on a key (lowercase letter) picks it up; a door (capital letter) can only be entered if you hold the key with the same letter. Walls are solid. Reach G.")
        state = (self.L("Ключи и двери. P — вы, G — цель, # — стена, '-' — пол.", "Keys and doors. P = you, G = goal, # = wall, '-' = floor.")
                 + f"\n{rules}\n" + self.L(f"У вас ключи: {held}.", f"Keys you hold: {held}.") + f"\n{self.grid(rows)}")
        q = rng.choice([self.L("Какой шаг ведёт к цели кратчайшим путём?", "Which step leads to the goal by a shortest path?"), self.L("Куда сделать шаг?", "Where should you step?")])
        return self._sample(rng, state, q, acts, texts, good)

    def apply(self, option: int) -> None:
        d = self._chosen(option)
        self.steps += 1
        if d is None:
            return
        nxt = self.graph[(self.pos, self.keys)][_SK_ORDER.index(d)]
        self.pos, self.keys = nxt
        self.reached = self.pos == self.goal

    def result(self) -> Dict[str, float]:
        return {"reached": float(self.reached), "steps": float(self.steps), "optimal": float(self.optimal)}


# ------------------------------------------------------------------------------------------------ lights out
def _lights_solutions(n: int, state: Tuple[int, ...]) -> List[int]:
    """All press-sets (bitmasks over the n*n cells) that switch every light off, via Gaussian elimination over GF(2)."""
    m = n * n
    rows = []
    for i in range(m):
        r, c = divmod(i, n)
        mask = 0
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                mask |= 1 << (rr * n + cc)
        rows.append(mask | (state[i] << m))                     # press j toggles cell i  <=>  bit j of row i
    pivots: List[int] = []
    rk = 0
    for col in range(m):
        piv = next((i for i in range(rk, m) if rows[i] >> col & 1), None)
        if piv is None:
            continue
        rows[rk], rows[piv] = rows[piv], rows[rk]
        for i in range(m):
            if i != rk and rows[i] >> col & 1:
                rows[i] ^= rows[rk]
        pivots.append(col)
        rk += 1
    if any(rows[i] >> m & 1 and not (rows[i] & ((1 << m) - 1)) for i in range(rk, m)):
        return []                                              # no solution
    free = [c for c in range(m) if c not in pivots]
    sols = []
    for assign in range(1 << len(free)):
        x = 0
        for k, c in enumerate(free):
            if assign >> k & 1:
                x |= 1 << c
        for k, col in enumerate(pivots):
            row = rows[k]
            val = row >> m & 1
            for c in free:
                if row >> c & 1 and x >> c & 1:
                    val ^= 1
            if val:
                x |= 1 << col
        sols.append(x)
    return sols


class LightsEnv(_Game):
    name = "lights"
    task = "game_lights"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        self.n = rng.choice([3, 3, 4, 4, 5])
        while True:
            state = [0] * (self.n * self.n)
            for _ in range(rng.randint(2, 2 + self.n * 2)):
                self._toggle(state, rng.randrange(self.n * self.n))
            if any(state):
                break
        self.state = tuple(state)
        self.optimal = min(bin(x).count("1") for x in _lights_solutions(self.n, self.state))
        self.cap = 2 * self.optimal + 4
        self.steps = 0
        self.on_c, self.off_c = rng.choice([("O", "."), ("*", "-"), ("1", "0")])
        self.cellstyle = rng.choice(["rc", "num"])

    def _toggle(self, state: List[int], i: int) -> None:
        r, c = divmod(i, self.n)
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.n and 0 <= cc < self.n:
                state[rr * self.n + cc] ^= 1

    @property
    def done(self) -> bool:  # type: ignore[override]
        return not any(self.state) or self.steps >= self.cap

    def _cell(self, i: int) -> str:
        r, c = divmod(i, self.n)
        return f"({r + 1},{c + 1})" if self.cellstyle == "rc" else self.L(f"клетка {i + 1}", f"cell {i + 1}")

    def decision(self, rng: random.Random) -> Sample:
        sols = _lights_solutions(self.n, self.state)
        best = min(bin(x).count("1") for x in sols)
        good_cells = set()
        for x in sols:
            if bin(x).count("1") == best:
                good_cells |= {i for i in range(self.n * self.n) if x >> i & 1}
        acts = list(range(self.n * self.n))
        texts = [self.L("нажать ", "press ") + self._cell(i) for i in acts]
        rows = [[self.on_c if self.state[r * self.n + c] else self.off_c for c in range(self.n)] for r in range(self.n)]
        rules = self.L(f"Правила: нажатие на клетку переключает её и четырёх соседей (вверх, вниз, влево, вправо, если они есть). Нужно погасить все огни ({self.on_c} — горит, {self.off_c} — погашено).",
                       f"Rules: pressing a cell toggles it and its four neighbours (up, down, left, right, where they exist). Switch every light off ({self.on_c} = on, {self.off_c} = off).")
        numbering = self.L("Клетки нумеруются слева направо, сверху вниз с 1.", "Cells are numbered left to right, top to bottom, from 1.") if self.cellstyle == "num" else \
            self.L("Координаты (строка,столбец) считаются с 1.", "Coordinates (row,column) start at 1.")
        state = self.L(f"Погасить свет, поле {self.n}x{self.n}.", f"Lights out, {self.n}x{self.n} board.") + f"\n{rules}\n{self.grid(rows)}\n{numbering}"
        q = rng.choice([self.L("Какую клетку нажать, чтобы быстрее всего погасить все огни?", "Which cell should you press to switch everything off in the fewest presses?"), self.L("Что нажать?", "What should you press?")])
        return self._sample(rng, state, q, acts, texts, good_cells)

    def apply(self, option: int) -> None:
        i = self._chosen(option)
        self.steps += 1
        if i is None:
            return
        s = list(self.state)
        self._toggle(s, i)
        self.state = tuple(s)

    def result(self) -> Dict[str, float]:
        return {"solved": float(not any(self.state)), "steps": float(self.steps), "optimal": float(self.optimal)}


# ------------------------------------------------------------------------------------------------ towers of hanoi
@lru_cache(maxsize=None)
def _hanoi_table(n: int) -> Dict[Tuple[int, ...], int]:
    """State = peg of each disc (disc 0 is the smallest). Distance to 'everything on peg 2'."""
    def moves(s: Tuple[int, ...]):
        for src in range(3):
            top = next((d for d in range(n) if s[d] == src), None)
            if top is None:
                continue
            for dst in range(3):
                if dst == src:
                    continue
                if any(s[d] == dst for d in range(top)):
                    continue
                yield top, src, dst, s[:top] + (dst,) + s[top + 1:]

    goal = tuple([2] * n)
    dist = {goal: 0}
    q = deque([goal])
    while q:
        s = q.popleft()
        for _, _, _, t in moves(s):
            if t not in dist:
                dist[t] = dist[s] + 1
                q.append(t)
    return dist


class HanoiEnv(_Game):
    name = "hanoi"
    task = "game_hanoi"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        self.n = rng.randint(2, 5)
        self.table = _hanoi_table(self.n)
        hi = max(self.table.values())
        while True:
            s = tuple(rng.randrange(3) for _ in range(self.n))
            if 1 <= self.table[s] and rng.random() < (self.table[s] + 1) / (hi + 1) + 0.15:
                break
        self.s = s
        self.optimal = self.table[s]
        self.cap = 3 * self.optimal + 6
        self.steps = 0
        self.pegs = rng.choice([("A", "B", "C"), ("1", "2", "3"), ("left", "middle", "right")])
        if self.lang == "ru" and self.pegs[0] == "left":
            self.pegs = ("левый", "средний", "правый")

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.table[self.s] == 0 or self.steps >= self.cap

    def _legal(self) -> List[Tuple[int, int, int, Tuple[int, ...]]]:
        out = []
        s, n = self.s, self.n
        for src in range(3):
            top = next((d for d in range(n) if s[d] == src), None)
            if top is None:
                continue
            for dst in range(3):
                if dst != src and not any(s[d] == dst for d in range(top)):
                    out.append((top, src, dst, s[:top] + (dst,) + s[top + 1:]))
        return out

    def decision(self, rng: random.Random) -> Sample:
        d0 = self.table[self.s]
        legal = self._legal()
        texts = [self.L(f"диск {top + 1}: {self.pegs[src]} → {self.pegs[dst]}", f"disc {top + 1}: {self.pegs[src]} → {self.pegs[dst]}") for top, src, dst, _ in legal]
        good = {i for i, (_, _, _, t) in enumerate(legal) if self.table[t] == d0 - 1}
        peg_lines = []
        for p in range(3):
            discs = [d + 1 for d in range(self.n - 1, -1, -1) if self.s[d] == p]
            peg_lines.append(f"{self.pegs[p]}: " + (" ".join(map(str, discs)) if discs else self.L("пусто", "empty")))
        rules = self.L("Правила: за ход переносится один верхний диск на другой стержень; большой диск нельзя класть на меньший. Диск 1 — самый маленький. Цель — перенести все диски на последний стержень.",
                       "Rules: each move carries one top disc to another peg; a larger disc may never sit on a smaller one. Disc 1 is the smallest. The goal is to move every disc to the last peg.")
        state = (self.L(f"Ханойские башни, {self.n} диска(ов). Стержни перечислены от дна к верху (слева — нижний диск).", f"Towers of Hanoi, {self.n} disc(s). Pegs are listed bottom to top (the left number is the bottom disc).")
                 + f"\n{rules}\n" + "\n".join(peg_lines))
        q = rng.choice([self.L("Какой ход приближает к цели быстрее всего?", "Which move gets closest to the goal fastest?"), self.L("Какой диск и куда перенести?", "Which disc goes where?")])
        return self._sample(rng, state, q, legal, texts, good)

    def apply(self, option: int) -> None:
        m = self._chosen(option)
        self.steps += 1
        if m is not None:
            self.s = m[3]

    def result(self) -> Dict[str, float]:
        return {"solved": float(self.table[self.s] == 0), "steps": float(self.steps), "optimal": float(self.optimal)}


# ------------------------------------------------------------------------------------------------ cheapest way over terrain
import heapq  # noqa: E402

TERRAIN = {".": 1, ",": 2, "~": 4}


class TerrainEnv(_Game):
    name = "terrain"
    task = "game_terrain"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        while True:
            self.H, self.W = rng.randint(5, 9), rng.randint(5, 9)
            self.cost: Dict[Tuple[int, int], int] = {}
            self.walls: Set[Tuple[int, int]] = set()
            for r, c in product(range(self.H), range(self.W)):
                roll = rng.random()
                if roll < 0.12:
                    self.walls.add((r, c))
                else:
                    self.cost[(r, c)] = 1 if roll < 0.5 else 2 if roll < 0.78 else 4
            cells = list(self.cost)
            self.pos, self.goal = rng.sample(cells, 2)
            self.dist = self._dijkstra()
            if self.pos in self.dist and self.dist[self.pos] >= 4:
                break
        self.optimal_cost = self.dist[self.pos]
        self.spent = 0
        self.steps = 0
        self.cap = 3 * (abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])) + 12
        self.reached = False
        self.style = rng.choice(["ru", "arrow"]) if self.lang == "ru" else rng.choice(["en", "arrow", "compass"])

    def _dijkstra(self) -> Dict[Tuple[int, int], int]:
        """Cost to reach the goal from every cell; entering a cell costs that cell's price."""
        dist = {self.goal: 0}
        pq = [(0, self.goal)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for dr, dc in _SK_DIRS.values():
                v = (u[0] + dr, u[1] + dc)               # v -> u: the walker enters u, paying u's price
                if v in self.cost:
                    nd = d + self.cost[u]
                    if nd < dist.get(v, 10 ** 9):
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
        return dist

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.reached or self.steps >= self.cap

    def decision(self, rng: random.Random) -> Sample:
        here = self.dist[self.pos]
        _, names = _moves(rng, self.style)
        acts, texts, good = [], [], set()
        for i, d in enumerate(_SK_ORDER):
            dr, dc = _SK_DIRS[d]
            n = (self.pos[0] + dr, self.pos[1] + dc)
            acts.append(d)
            texts.append(names[d])
            if n in self.cost and n in self.dist and self.cost[n] + self.dist[n] == here:
                good.add(i)
        rows = [["P" if (r, c) == self.pos else "G" if (r, c) == self.goal else "#" if (r, c) in self.walls else ("." if self.cost[(r, c)] == 1 else "," if self.cost[(r, c)] == 2 else "~")
                 for c in range(self.W)] for r in range(self.H)]
        rules = self.L("Правила: шаг стоит столько, сколько стоит клетка, в которую вы входите: '.' — дорога (1), ',' — трава (2), '~' — грязь (4). '#' — стена. Нужно дойти до G с наименьшей суммарной стоимостью.",
                       "Rules: a step costs what the cell you enter costs: '.' = road (1), ',' = grass (2), '~' = mud (4). '#' is a wall. Reach G with the lowest total cost.")
        state = self.L("Местность. P — вы, G — цель.", "Terrain. P = you, G = goal.") + f"\n{rules}\n{self.grid(rows)}"
        q = rng.choice([self.L("Какой шаг лежит на самом дешёвом пути?", "Which step lies on the cheapest path?"), self.L("Куда сделать шаг?", "Where should you step?")])
        return self._sample(rng, state, q, acts, texts, good)

    def apply(self, option: int) -> None:
        d = self._chosen(option)
        self.steps += 1
        if d is None:
            return
        dr, dc = _SK_DIRS[d]
        n = (self.pos[0] + dr, self.pos[1] + dc)
        if n in self.cost:
            self.pos = n
            self.spent += self.cost[n]
            self.reached = n == self.goal

    def result(self) -> Dict[str, float]:
        return {"reached": float(self.reached), "steps": float(self.steps), "cost": float(self.spent), "optimal_cost": float(self.optimal_cost),
                "cost_ratio": float(self.spent / max(1, self.optimal_cost)) if self.reached else 0.0}


# ------------------------------------------------------------------------------------------------ exam-only games (never trained)
@lru_cache(maxsize=None)
def _chomp_wins(rows: Tuple[int, ...]) -> bool:
    """rows = bar length of each row, non-increasing. The player who must eat the poison square (top-left) loses."""
    if rows == (1,):
        return False
    for r in range(len(rows)):
        for c in range(rows[r]):
            if (r, c) == (0, 0):
                continue
            nxt = tuple(min(x, c) if i >= r else x for i, x in enumerate(rows))
            nxt = tuple(x for x in nxt if x > 0)
            if not _chomp_wins(nxt):
                return True
    return False


class ChompEnv(_Game):
    name = "chomp"
    task = "game_chomp"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        self.R, self.C = rng.randint(2, 4), rng.randint(2, 5)
        self.rows: Tuple[int, ...] = tuple([self.C] * self.R)
        self.turn = 0 if rng.random() < 0.5 else 1
        self._rng = random.Random(rng.random())
        self.steps = 0
        self._opp()

    @property
    def done(self) -> bool:  # type: ignore[override]
        return self.rows == (1,)                       # only the poisoned square is left: whoever has to move eats it

    def _moves(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(len(self.rows)) for c in range(self.rows[r]) if (r, c) != (0, 0)]

    def _bite(self, r: int, c: int) -> Tuple[int, ...]:
        nxt = tuple(min(x, c) if i >= r else x for i, x in enumerate(self.rows))
        return tuple(x for x in nxt if x > 0)

    def _opp(self) -> None:
        if self.done or self.turn == 0:
            return
        mv = self._rng.choice(self._moves())
        self.rows = self._bite(*mv)
        self.turn = 0

    def decision(self, rng: random.Random) -> Sample:
        mvs = self._moves()
        good = set()
        for i, (r, c) in enumerate(mvs):
            if (r, c) == (0, 0):
                continue
            nxt = tuple(x for x in tuple(min(x, c) if j >= r else x for j, x in enumerate(self.rows)) if x > 0)
            if not _chomp_wins(nxt):
                good.add(i)
        texts = [self.L(f"откусить ({r + 1},{c + 1})", f"bite ({r + 1},{c + 1})") for r, c in mvs]
        have = lambda r: self.rows[r] if r < len(self.rows) else 0      # noqa: E731
        grid = [["X" if (r, c) == (0, 0) else "o" for c in range(have(r))] + ["-"] * (self.C - have(r)) for r in range(self.R)]
        rules = self.L("Правила: плитка шоколада; клетка X — отравлена. Укус клетки (r,c) съедает её и все клетки ниже И правее (в прямоугольнике от неё до края). Кто вынужден съесть X — проиграл. '-' — уже съедено.",
                       "Rules: a chocolate bar; the square X is poisoned. Biting square (r,c) eats it and every square below AND to the right of it (the rectangle to the edge). Whoever must eat X loses. '-' = already eaten.")
        state = self.L("Чомп.", "Chomp.") + f"\n{rules}\n{self.grid(grid)}\n" + self.L("Координаты (строка,столбец) с 1.", "Coordinates (row,column) from 1.")
        q = self.L("Какой укус выигрывает?", "Which bite wins?")
        return self._sample(rng, state, q, mvs, texts, good)

    def apply(self, option: int) -> None:
        mv = self._chosen(option)
        self.steps += 1
        if mv is None:
            return
        self.rows = self._bite(*mv)
        self.turn = 1
        self._opp()

    def result(self) -> Dict[str, float]:
        over = self.rows == (1,)
        return {"won": float(over and self.turn == 1), "lost": float(over and self.turn == 0), "drawn": 0.0}


@lru_cache(maxsize=None)
def _cram_wins(R: int, C: int, mask: int) -> bool:
    """Cram: players place dominoes on free cells; the player who cannot move loses. mask = occupied cells."""
    for i in range(R * C):
        if mask >> i & 1:
            continue
        r, c = divmod(i, C)
        for j in ((i + 1) if c + 1 < C else None, (i + C) if r + 1 < R else None):
            if j is not None and not (mask >> j & 1):
                if not _cram_wins(R, C, mask | 1 << i | 1 << j):
                    return True
    return False


class CramEnv(_Game):
    name = "cram"
    task = "game_cram"

    def __init__(self, rng: random.Random, spaced: bool = True):
        super().__init__(rng, spaced)
        self.R, self.C = rng.choice([(3, 3), (2, 4), (3, 4), (4, 4), (2, 5)])
        self.mask = 0
        self.turn = 0 if rng.random() < 0.5 else 1
        self._rng = random.Random(rng.random())
        self.steps = 0
        self.winner: Optional[int] = None
        self._opp()

    def _moves(self, mask: Optional[int] = None) -> List[Tuple[int, int]]:
        mask = self.mask if mask is None else mask
        out = []
        for i in range(self.R * self.C):
            if mask >> i & 1:
                continue
            r, c = divmod(i, self.C)
            if c + 1 < self.C and not (mask >> (i + 1) & 1):
                out.append((i, i + 1))
            if r + 1 < self.R and not (mask >> (i + self.C) & 1):
                out.append((i, i + self.C))
        return out

    @property
    def done(self) -> bool:  # type: ignore[override]
        return not self._moves()

    def _opp(self) -> None:
        if not self.done and self.turn == 1:
            a, b = self._rng.choice(self._moves())
            self.mask |= 1 << a | 1 << b
            self.turn = 0

    def decision(self, rng: random.Random) -> Sample:
        mvs = self._moves()
        good = {i for i, (a, b) in enumerate(mvs) if not _cram_wins(self.R, self.C, self.mask | 1 << a | 1 << b)}

        def cell(i: int) -> str:
            r, c = divmod(i, self.C)
            return f"({r + 1},{c + 1})"

        texts = [self.L(f"домино {cell(a)}–{cell(b)}", f"domino {cell(a)}–{cell(b)}") for a, b in mvs]
        rows = [["#" if self.mask >> (r * self.C + c) & 1 else "." for c in range(self.C)] for r in range(self.R)]
        rules = self.L("Правила: игроки по очереди кладут домино (две соседние свободные клетки). Кто не может сделать ход — проиграл. '#' — занято, '.' — свободно.",
                       "Rules: players take turns placing a domino on two adjacent free cells. The player who cannot move loses. '#' = taken, '.' = free.")
        state = self.L("Крэм.", "Cram.") + f"\n{rules}\n{self.grid(rows)}\n" + self.L("Координаты (строка,столбец) с 1.", "Coordinates (row,column) from 1.")
        q = self.L("Какое домино выигрывает?", "Which domino wins?")
        return self._sample(rng, state, q, mvs, texts, good)

    def apply(self, option: int) -> None:
        mv = self._chosen(option)
        self.steps += 1
        if mv is None:
            return
        a, b = mv
        if self.mask >> a & 1 or self.mask >> b & 1:
            return
        self.mask |= 1 << a | 1 << b
        self.turn = 1
        self._opp()

    def result(self) -> Dict[str, float]:
        # the player to move when no domino fits loses; after my move + the opponent's reply ``turn`` is 0 again (me)
        i_lost = self.done and self.turn == 0
        return {"won": float(self.done and not i_lost), "lost": float(i_lost), "drawn": 0.0}


# ------------------------------------------------------------------------------------------------ registries
def _opponents(cls, **kw) -> Callable[[random.Random], Env]:
    return lambda rng: cls(rng, **kw)


ENVS_MORE: Dict[str, Callable[[random.Random], Env]] = {
    "connect_vs_random": _opponents(ConnectEnv, opponent="random"),
    "connect_vs_search": _opponents(ConnectEnv, opponent="search"),
    "sokoban": SokobanEnv,
    "slide": SlideEnv,
    "nim_vs_random": _opponents(NimEnv, opponent="random"),
    "nim_vs_perfect": _opponents(NimEnv, opponent="perfect"),
    "keydoor": KeyDoorEnv,
    "lights": LightsEnv,
    "hanoi": HanoiEnv,
    "terrain": TerrainEnv,
}
ENVS_UNSEEN: Dict[str, Callable[[random.Random], Env]] = {"chomp": ChompEnv, "cram": CramEnv}
