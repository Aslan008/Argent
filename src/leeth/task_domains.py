"""
src/leeth/task_domains.py — Multi-domain task generators for the Universal Leeth Task Solver.

Each domain exposes a PERFECT expert and produces TaskSample records:
state text + question + option list + the set of optimal option indices.

Domains:
- snake:        ASCII grid, 4 directional options, expert = BFS to food + flood-fill safety.
- maze:         ASCII maze, 4 directional options, expert = BFS shortest path to exit.
- tic_tac_toe:  ASCII 3x3 board, 9 cell options, expert = full minimax (unbeatable).

Pure Python (no torch) — cheap enough to run anywhere; the Kaggle kernel imports
this module to generate training data on the fly. Option order is shuffled per
sample so the model must compare option TEXT against the state, not list position.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]


@dataclass
class TaskSample:
    """One training/eval example for the DynamicOptionHead solver."""

    domain: str
    state_text: str
    question: str
    options: List[str]
    optimal_set: FrozenSet[int]  # indices of options the expert rates optimal

    @property
    def correct_idx(self) -> int:
        """Deterministic CE target: the first optimal option."""
        return min(self.optimal_set)


DIRS: Dict[str, Tuple[int, int]] = {
    "вверх": (-1, 0),
    "вниз": (1, 0),
    "влево": (0, -1),
    "вправо": (0, 1),
}
DIR_ORDER = ["вверх", "вниз", "влево", "вправо"]


def _dir_name(src: Cell, dst: Cell) -> str:
    dr, dc = dst[0] - src[0], dst[1] - src[1]
    for name, (d, dd) in DIRS.items():
        if (d, dd) == (dr, dc):
            return name
    raise ValueError(f"Not an adjacent step: {src} -> {dst}")


def _bfs_dist(start: Cell, goal: Cell, blocked, H: int, W: int) -> Optional[int]:
    """Shortest-path length start->goal avoiding blocked cells, or None."""
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


def _flood_count(start: Cell, blocked, H: int, W: int) -> int:
    """Number of cells reachable from start (start itself excluded)."""
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


def _shuffle_options(rng: random.Random, optimal: Sequence[bool]) -> Tuple[List[int], FrozenSet[int]]:
    """Return a permutation of option indices keeping optimality consistent."""
    perm = list(range(len(optimal)))
    rng.shuffle(perm)
    optimal_set = frozenset(perm.index(i) for i, o in enumerate(optimal) if o)
    return perm, optimal_set


# ---------------------------------------------------------------------------
# Domain 1: Snake
# ---------------------------------------------------------------------------

def _random_snake_game(rng: random.Random, max_size: int = 10, min_size: int = 5):
    """Random valid position: snake body (no self-overlap), reachable food."""
    W = rng.randint(min_size, max_size)
    H = rng.randint(min_size, max_size)
    head = (rng.randrange(H), rng.randrange(W))
    target_len = rng.randint(3, max(3, (W * H) // 3))
    body = [head]
    seen = {head}
    cur = head
    for _ in range(4 * target_len):  # bounded random walk
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
    if not free:
        return None
    rng.shuffle(free)
    for food in free:
        if _bfs_dist(head, food, seen, H, W) is not None:
            return H, W, body, food
    return None


def snake_expert_scores(head: Cell, food: Cell, body: Sequence[Cell], H: int, W: int) -> Dict[str, bool]:
    """For each direction: is it an optimal move?

    Optimal = the step lies on a shortest BFS path to food (body treated as
    static obstacles). If food is unreachable, optimal = moves maximizing the
    reachable free area (survival fallback). Lethal moves are never optimal.
    """
    body_set = set(body)
    scores: Dict[str, bool] = {name: False for name in DIRS}
    legal: Dict[str, Cell] = {}
    for name, (dr, dc) in DIRS.items():
        nxt = (head[0] + dr, head[1] + dc)
        if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in body_set:
            legal[name] = nxt
    if not legal:
        return {name: False for name in DIRS}  # doomed position, no optimal move

    d_food = _bfs_dist(head, food, body_set, H, W)
    if d_food is not None:
        for name, nxt in legal.items():
            d_next = _bfs_dist(nxt, food, body_set, H, W)
            scores[name] = d_next == d_food - 1
        if any(scores.values()):
            return scores

    # Survival fallback: maximize reachable area.
    best_n = max(_flood_count(nxt, body_set | {head}, H, W) for nxt in legal.values())
    for name, nxt in legal.items():
        scores[name] = _flood_count(nxt, body_set | {head}, H, W) == best_n
    return scores


def generate_snake_sample(rng: random.Random, min_size: int = 5, max_size: int = 10) -> Optional[TaskSample]:
    game = _random_snake_game(rng, max_size=max_size, min_size=min_size)
    if game is None:
        return None
    H, W, body, food = game
    head = body[0]
    scores = snake_expert_scores(head, food, body, H, W)
    if not any(scores.values()):
        return None

    rows = []
    body_set = set(body)
    for r in range(H):
        row = []
        for c in range(W):
            if (r, c) == head:
                row.append("H")
            elif (r, c) == food:
                row.append("F")
            elif (r, c) in body_set:
                row.append("B")
            else:
                row.append(".")
        rows.append("".join(row))
    state = f"змейка {H}x{W}, H=голова, F=еда, B=тело\n" + "\n".join(rows)

    optimal_flags = [scores[name] for name in DIR_ORDER]
    perm, optimal_set = _shuffle_options(rng, optimal_flags)
    options = [DIR_ORDER[i] for i in perm]
    return TaskSample("snake", state, "Какой ход оптимален?", options, optimal_set)


# ---------------------------------------------------------------------------
# Domain 2: Maze
# ---------------------------------------------------------------------------

def _carve_maze(cw: int, ch: int, rng: random.Random) -> List[List[str]]:
    """Iterative recursive-backtracker on a (2*ch+1) x (2*cw+1) grid."""
    W, H = 2 * cw + 1, 2 * ch + 1
    grid = [["#"] * W for _ in range(H)]
    stack = [(1, 1)]
    grid[1][1] = "."
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < H - 1 and 1 <= nc < W - 1 and grid[nr][nc] == "#":
                neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
        if not neighbors:
            stack.pop()
            continue
        nr, nc, mr, mc = rng.choice(neighbors)
        grid[mr][mc] = "."
        grid[nr][nc] = "."
        stack.append((nr, nc))
    return grid


MAZE_CELL_CHOICES = [(3, 3), (4, 4), (5, 5), (3, 4), (4, 3), (4, 5), (5, 4)]


def generate_maze_sample(rng: random.Random, cell_choices: Optional[Sequence[Tuple[int, int]]] = None) -> Optional[TaskSample]:
    cw, ch = rng.choice(cell_choices or MAZE_CELL_CHOICES)
    grid = _carve_maze(cw, ch, rng)
    H, W = len(grid), len(grid[0])
    start, goal = (1, 1), (H - 2, W - 2)
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"

    walls = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == "#"}
    d_total = _bfs_dist(start, goal, walls, H, W)
    if d_total is None:
        return None

    optimal_flags = []
    for name, (dr, dc) in DIRS.items():
        nxt = (start[0] + dr, start[1] + dc)
        ok = 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in walls
        d_next = _bfs_dist(nxt, goal, walls, H, W) if ok else None
        optimal_flags.append(d_next == d_total - 1)
    if not any(optimal_flags):
        return None

    rows = ["".join(row) for row in grid]
    state = f"лабиринт {H}x{W}, S=старт, E=выход, #=стена\n" + "\n".join(rows)
    perm, optimal_set = _shuffle_options(rng, optimal_flags)
    options = [DIR_ORDER[i] for i in perm]
    return TaskSample("maze", state, "Какой шаг к выходу оптимален?", options, optimal_set)


# ---------------------------------------------------------------------------
# Domain 3: Tic-Tac-Toe (minimax, X to move)
# ---------------------------------------------------------------------------

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def _ttt_winner(board: Tuple[str, ...]) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None


@lru_cache(maxsize=None)
def _ttt_minimax(board: Tuple[str, ...], player: str) -> int:
    """Game score from X's perspective: +1 X win, -1 O win, 0 draw."""
    w = _ttt_winner(board)
    if w == "X":
        return 1
    if w == "O":
        return -1
    if all(cell != "." for cell in board):
        return 0
    best = -2 if player == "X" else 2
    for i in range(9):
        if board[i] == ".":
            nxt = board[:i] + (player,) + board[i + 1:]
            val = _ttt_minimax(nxt, "O" if player == "X" else "X")
            best = max(best, val) if player == "X" else min(best, val)
    return best


def ttt_optimal_moves(board: Tuple[str, ...]) -> List[int]:
    """All moves achieving the best minimax score for X (X to move)."""
    best = max(
        _ttt_minimax(board[:i] + ("X",) + board[i + 1:], "O")
        for i in range(9) if board[i] == "."
    )
    return [i for i in range(9) if board[i] == "." and _ttt_minimax(board[:i] + ("X",) + board[i + 1:], "O") == best]


def _random_ttt_position(rng: random.Random) -> Optional[Tuple[str, ...]]:
    """Random non-terminal legal position with X to move (X count == O count)."""
    for _ in range(50):
        board = ("." * 9,)
        board = tuple("." for _ in range(9))
        moves = list(range(9))
        rng.shuffle(moves)
        player = "X"
        terminal = False
        for i in moves[: rng.randint(0, 8)]:
            board = board[:i] + (player,) + board[i + 1:]
            if _ttt_winner(board) is not None or all(c != "." for c in board):
                terminal = True
                break
            player = "O" if player == "X" else "X"
        if terminal:
            continue
        if sum(1 for c in board if c == "X") != sum(1 for c in board if c == "O"):
            continue  # ensure X to move
        return board
    return None


def generate_ttt_sample(rng: random.Random) -> Optional[TaskSample]:
    board = _random_ttt_position(rng)
    if board is None:
        return None
    optimal_cells = ttt_optimal_moves(board)

    rows = ["".join(board[r * 3 : r * 3 + 3]) for r in range(3)]
    state = "крестики-нолики 3x3, X=крестик, O=нолик, .=пусто, ход X\n" + "\n".join(rows)

    cell_names = [f"X в ({r},{c})" for r in range(3) for c in range(3)]
    optimal_flags = [i in optimal_cells for i in range(9)]
    perm, optimal_set = _shuffle_options(rng, optimal_flags)
    options = [cell_names[i] for i in perm]
    return TaskSample("tic_tac_toe", state, "Какой ход за X оптимален?", options, optimal_set)


# ---------------------------------------------------------------------------
# Prompt assembly (domain-agnostic)
# ---------------------------------------------------------------------------

def build_prompt(sample: TaskSample) -> str:
    """Assemble the full model input: [STATE] ... [QUESTION] ... [OPT] ... [OPT] ..."""
    parts = [f"[STATE] {sample.state_text}", f"[QUESTION] {sample.question}"]
    parts.extend(f"[OPT] {opt}" for opt in sample.options)
    return " ".join(parts)


GENERATORS = {
    "snake": generate_snake_sample,
    "maze": generate_maze_sample,
    "tic_tac_toe": generate_ttt_sample,
}


def generate_mixed_batch(rng: random.Random, n: int, domain_weights: Optional[Dict[str, float]] = None) -> List[TaskSample]:
    """Mixed-domain batch. Default weights equal; retries skip failed generations."""
    weights = domain_weights or {name: 1.0 for name in GENERATORS}
    domains = list(weights)
    probs = [weights[d] for d in domains]
    samples: List[TaskSample] = []
    attempts = 0
    while len(samples) < n and attempts < n * 10:
        attempts += 1
        domain = rng.choices(domains, weights=probs, k=1)[0]
        sample = GENERATORS[domain](rng)
        if sample is not None:
            samples.append(sample)
    return samples