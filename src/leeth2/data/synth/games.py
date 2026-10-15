"""Grid / board games with PERFECT experts (snake, maze, tic-tac-toe, treasure grid).

Several moves can be equally good, so labels are soft (uniform over the optimal set) instead of an arbitrary pick.
Everything the model needs is in the text: the legend, the symbols and the way moves are named change from sample
to sample, so it has to read the state instead of memorising a format.
"""

from __future__ import annotations

import random
from collections import deque
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ...schema import Sample

Cell = Tuple[int, int]
DIRS: Dict[str, Cell] = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DIR_ORDER = ["up", "down", "left", "right"]

MOVE_NAMES = {
    "ru": {"up": ["вверх", "наверх", "шаг вверх"], "down": ["вниз", "шаг вниз"], "left": ["влево", "шаг влево"], "right": ["вправо", "шаг вправо"]},
    "en": {"up": ["up", "move up", "go up"], "down": ["down", "move down", "go down"], "left": ["left", "move left", "go left"], "right": ["right", "move right", "go right"]},
    "arrow": {"up": ["↑"], "down": ["↓"], "left": ["←"], "right": ["→"]},
    "compass": {"up": ["north", "N"], "down": ["south", "S"], "left": ["west", "W"], "right": ["east", "E"]},
}


def _bfs_dist(start: Cell, goal: Cell, blocked: Set[Cell], H: int, W: int) -> Optional[int]:
    if start == goal:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        cur, d = q.popleft()
        for dr, dc in DIRS.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in seen and nxt not in blocked:
                if nxt == goal:
                    return d + 1
                seen.add(nxt)
                q.append((nxt, d + 1))
    return None


def _flood_count(start: Cell, blocked: Set[Cell], H: int, W: int) -> int:
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in DIRS.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in seen and nxt not in blocked:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) - 1


def _moves_text(rng: random.Random, style: str) -> Tuple[List[str], Dict[str, str]]:
    names = {d: rng.choice(MOVE_NAMES[style][d]) for d in DIR_ORDER}
    return [names[d] for d in DIR_ORDER], names


def _finish(task: str, lang: str, state: str, question: str, optimal: Dict[str, bool], names: Dict[str, str], rng: random.Random, family: str = "game") -> Optional[Sample]:
    if not any(optimal.values()):
        return None
    order = list(DIR_ORDER)
    rng.shuffle(order)
    options = [names[d] for d in order]
    if len(set(options)) != len(options):
        return None
    k = sum(1 for d in order if optimal[d])
    probs = [(1.0 / k) if optimal[d] else 0.0 for d in order]
    return Sample(task, family, lang, "choice", state, question, options, probs if k > 1 else probs.index(1.0))


# ------------------------------------------------------------------------------------------------ snake
def _random_snake_game(rng: random.Random, max_size: int = 11, min_size: int = 5):
    W, H = rng.randint(min_size, max_size), rng.randint(min_size, max_size)
    head = (rng.randrange(H), rng.randrange(W))
    target_len = rng.randint(3, max(3, (W * H) // 3))
    body = [head]
    seen = {head}
    cur = head
    for _ in range(4 * target_len):
        if len(body) >= target_len:
            break
        dr, dc = rng.choice(list(DIRS.values()))
        nxt = (cur[0] + dr, cur[1] + dc)
        if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in seen:
            body.append(nxt)
            seen.add(nxt)
            cur = nxt
    if len(body) < 3:
        return None
    free = [(r, c) for r in range(H) for c in range(W) if (r, c) not in seen]
    rng.shuffle(free)
    for food in free:
        if _bfs_dist(head, food, seen, H, W) is not None:
            return H, W, body, food
    return None


def snake_optimal(head: Cell, food: Cell, body: Sequence[Cell], H: int, W: int) -> Dict[str, bool]:
    body_set = set(body)
    out = {n: False for n in DIRS}
    legal = {}
    for n, (dr, dc) in DIRS.items():
        nxt = (head[0] + dr, head[1] + dc)
        if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in body_set:
            legal[n] = nxt
    if not legal:
        return out
    d_food = _bfs_dist(head, food, body_set, H, W)
    if d_food is not None:
        for n, nxt in legal.items():
            out[n] = _bfs_dist(nxt, food, body_set, H, W) == d_food - 1
        if any(out.values()):
            return out
    best = max(_flood_count(nxt, body_set | {head}, H, W) for nxt in legal.values())
    for n, nxt in legal.items():
        out[n] = _flood_count(nxt, body_set | {head}, H, W) == best
    return out


def gen_snake(rng: random.Random) -> Optional[Sample]:
    game = _random_snake_game(rng)
    if game is None:
        return None
    H, W, body, food = game
    head = body[0]
    opt = snake_optimal(head, food, body, H, W)
    sym = rng.choice([("H", "F", "B", "."), ("@", "*", "o", "."), ("S", "$", "s", "_"), ("^", "+", "#", " ")])
    lang = rng.choice(["ru", "en"])
    rows = []
    bs = set(body)
    for r in range(H):
        rows.append("".join(sym[0] if (r, c) == head else sym[1] if (r, c) == food else sym[2] if (r, c) in bs else sym[3] for c in range(W)))
    if lang == "ru":
        legend = f"Змейка {H}x{W}. Символы: {sym[0]} — голова, {sym[1]} — еда, {sym[2]} — тело, {sym[3]!r} — пусто."
        q = rng.choice(["Какой ход лучше всего?", "Куда двигаться, чтобы добраться до еды и не погибнуть?", "Выберите оптимальный ход."])
    else:
        legend = f"Snake game {H}x{W}. Symbols: {sym[0]} = head, {sym[1]} = food, {sym[2]} = body, {sym[3]!r} = empty."
        q = rng.choice(["Which move is best?", "Where should the snake go to reach the food without dying?", "Choose the optimal move."])
    style = rng.choice(["ru", "en", "arrow", "compass"]) if lang == "en" else rng.choice(["ru", "arrow", "en"])
    _, names = _moves_text(rng, style)
    return _finish("game_snake", lang, legend + "\n" + "\n".join(rows), q, opt, names, rng)


# ------------------------------------------------------------------------------------------------ maze
def _carve_maze(cw: int, ch: int, rng: random.Random) -> List[List[str]]:
    W, H = 2 * cw + 1, 2 * ch + 1
    grid = [["#"] * W for _ in range(H)]
    stack = [(1, 1)]
    grid[1][1] = "."
    while stack:
        r, c = stack[-1]
        nb = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < H - 1 and 1 <= nc < W - 1 and grid[nr][nc] == "#":
                nb.append((nr, nc, r + dr // 2, c + dc // 2))
        if not nb:
            stack.pop()
            continue
        nr, nc, mr, mc = rng.choice(nb)
        grid[mr][mc] = "."
        grid[nr][nc] = "."
        stack.append((nr, nc))
    return grid


def gen_maze(rng: random.Random) -> Optional[Sample]:
    cw, ch = rng.choice([(3, 3), (4, 4), (5, 5), (3, 4), (4, 3), (4, 5), (5, 4), (6, 5), (5, 6)])
    grid = _carve_maze(cw, ch, rng)
    H, W = len(grid), len(grid[0])
    # start anywhere on a free cell, exit on another free cell
    free = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == "."]
    start, goal = rng.sample(free, 2)
    walls = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == "#"}
    total = _bfs_dist(start, goal, walls, H, W)
    if total is None or total < 2:
        return None
    opt = {}
    for n, (dr, dc) in DIRS.items():
        nxt = (start[0] + dr, start[1] + dc)
        ok = 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in walls
        opt[n] = bool(ok and _bfs_dist(nxt, goal, walls, H, W) == total - 1)
    wall_c, free_c = rng.choice([("#", "."), ("█", " "), ("X", "_")])
    grid2 = [[(wall_c if ch_ == "#" else free_c) for ch_ in row] for row in grid]
    grid2[start[0]][start[1]] = "S"
    grid2[goal[0]][goal[1]] = "E"
    lang = rng.choice(["ru", "en"])
    if lang == "ru":
        legend = f"Лабиринт {H}x{W}. S — вы, E — выход, {wall_c} — стена, {free_c!r} — проход."
        q = rng.choice(["Какой шаг ведёт к выходу кратчайшим путём?", "Куда сделать шаг?"])
    else:
        legend = f"Maze {H}x{W}. S = you, E = exit, {wall_c} = wall, {free_c!r} = path."
        q = rng.choice(["Which step leads to the exit along a shortest path?", "Where should you step?"])
    _, names = _moves_text(rng, rng.choice(["ru", "arrow"]) if lang == "ru" else rng.choice(["en", "arrow", "compass"]))
    return _finish("game_maze", lang, legend + "\n" + "\n".join("".join(r) for r in grid2), q, opt, names, rng)


# ------------------------------------------------------------------------------------------------ tic-tac-toe
WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _winner(b: Tuple[str, ...]) -> Optional[str]:
    for a, c, d in WIN_LINES:
        if b[a] != "." and b[a] == b[c] == b[d]:
            return b[a]
    return None


@lru_cache(maxsize=None)
def _minimax(b: Tuple[str, ...], player: str) -> int:
    w = _winner(b)
    if w == "X":
        return 1
    if w == "O":
        return -1
    if all(x != "." for x in b):
        return 0
    vals = []
    for i in range(9):
        if b[i] == ".":
            vals.append(_minimax(b[:i] + (player,) + b[i + 1 :], "O" if player == "X" else "X"))
    return max(vals) if player == "X" else min(vals)


def _ttt_best(b: Tuple[str, ...], me: str) -> List[int]:
    other = "O" if me == "X" else "X"
    scores = {}
    for i in range(9):
        if b[i] == ".":
            v = _minimax(b[:i] + (me,) + b[i + 1 :], other)
            scores[i] = v if me == "X" else -v
    best = max(scores.values())
    return [i for i, v in scores.items() if v == best]


def gen_ttt(rng: random.Random) -> Optional[Sample]:
    for _ in range(60):
        b = tuple("." for _ in range(9))
        order = list(range(9))
        rng.shuffle(order)
        pl = "X"
        term = False
        for i in order[: rng.randint(0, 8)]:
            b = b[:i] + (pl,) + b[i + 1 :]
            if _winner(b) or all(x != "." for x in b):
                term = True
                break
            pl = "O" if pl == "X" else "X"
        if term or sum(x == "X" for x in b) != sum(x == "O" for x in b):
            continue
        break
    else:
        return None
    me = rng.choice(["X", "O"])
    # when playing O the position must have O to move: swap the marks so that "me" is to move
    if me == "O":
        b = tuple({"X": "O", "O": "X"}.get(x, x) for x in b)
    best = _ttt_best(b, "X" if me == "X" else "O")
    empty = [i for i in range(9) if b[i] == "."]
    lang = rng.choice(["ru", "en"])
    style = rng.choice(["rc", "num", "word"])

    def cell(i: int) -> str:
        r, c = divmod(i, 3)
        if style == "rc":
            return f"({r + 1},{c + 1})" if lang == "en" else f"({r + 1};{c + 1})"
        if style == "num":
            return f"cell {i + 1}" if lang == "en" else f"клетка {i + 1}"
        rows = ["top", "middle", "bottom"] if lang == "en" else ["верх", "середина", "низ"]
        cols = ["left", "center", "right"] if lang == "en" else ["лево", "центр", "право"]
        return f"{rows[r]}-{cols[c]}"

    board = "\n".join("".join(b[r * 3 : r * 3 + 3]) for r in range(3))
    if lang == "ru":
        state = f"Крестики-нолики 3x3. Вы играете за {me}. Клетки: X, O, '.' — пусто. Ваш ход.\n{board}\nНумерация клеток: слева направо, сверху вниз, от 1 до 9."
        q = rng.choice(["Какой ход оптимален?", "Куда поставить свой знак?"])
    else:
        state = f"Tic-tac-toe 3x3. You play {me}. Cells: X, O, '.' = empty. Your move.\n{board}\nCells are numbered left to right, top to bottom, 1 to 9."
        q = rng.choice(["Which move is optimal?", "Where should you place your mark?"])
    rng.shuffle(empty)
    options = [cell(i) for i in empty]
    if len(set(options)) != len(options):
        return None
    k = sum(1 for i in empty if i in best)
    probs = [(1.0 / k) if i in best else 0.0 for i in empty]
    if len(options) < 2:
        return None
    return Sample("game_ttt", "game", lang, "choice", state, q, options, probs if k > 1 else probs.index(1.0))


# ------------------------------------------------------------------------------------------------ treasure grid
def gen_treasure(rng: random.Random) -> Optional[Sample]:
    """Reach the nearest chest while avoiding traps; walls block. Options are the four moves."""
    H, W = rng.randint(5, 10), rng.randint(5, 10)
    cells = [(r, c) for r in range(H) for c in range(W)]
    rng.shuffle(cells)
    me = cells.pop()
    chests = [cells.pop() for _ in range(rng.randint(1, 3))]
    traps = {cells.pop() for _ in range(rng.randint(2, max(3, H * W // 8)))}
    walls = {cells.pop() for _ in range(rng.randint(0, H * W // 6))}
    blocked = traps | walls
    dists = [(d, ch) for ch in chests for d in [_bfs_dist(me, ch, blocked, H, W)] if d is not None]
    if not dists:
        return None
    dmin = min(d for d, _ in dists)
    opt = {}
    for n, (dr, dc) in DIRS.items():
        nxt = (me[0] + dr, me[1] + dc)
        ok = 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in blocked
        opt[n] = bool(ok and any(_bfs_dist(nxt, ch, blocked, H, W) == dmin - 1 for _, ch in dists))
    rows = []
    for r in range(H):
        rows.append("".join("P" if (r, c) == me else "T" if (r, c) in chests else "^" if (r, c) in traps else "#" if (r, c) in walls else "." for c in range(W)))
    lang = rng.choice(["ru", "en"])
    if lang == "ru":
        state = f"Карта {H}x{W}: P — вы, T — сундук, ^ — ловушка (смертельна), # — стена, '.' — пусто.\n" + "\n".join(rows)
        q = rng.choice(["Куда идти, чтобы быстрее добраться до ближайшего сундука?", "Выберите лучший шаг."])
    else:
        state = f"Map {H}x{W}: P = you, T = treasure chest, ^ = trap (deadly), # = wall, '.' = empty.\n" + "\n".join(rows)
        q = rng.choice(["Where should you go to reach the nearest chest fastest?", "Choose the best step."])
    _, names = _moves_text(rng, rng.choice(["ru", "arrow"]) if lang == "ru" else rng.choice(["en", "arrow", "compass"]))
    return _finish("game_treasure", lang, state, q, opt, names, rng)


GAMES = {"snake": (gen_snake, 3.0), "maze": (gen_maze, 3.0), "ttt": (gen_ttt, 2.0), "treasure": (gen_treasure, 3.0)}
