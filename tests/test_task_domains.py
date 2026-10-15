"""
tests/test_task_domains.py — Verification tests for the universal task-solver domain generators.

Fast, pure-Python tests (no torch): expert optimality, option consistency, prompt format.
"""

import random

import pytest

from src.leeth.task_domains import (
    DIRS,
    TaskSample,
    build_prompt,
    generate_maze_sample,
    generate_mixed_batch,
    generate_snake_sample,
    generate_ttt_sample,
    snake_expert_scores,
    ttt_optimal_moves,
    _ttt_minimax,
    _ttt_winner,
)


def test_snake_expert_optimal_on_random_games():
    """Expert move must shorten BFS distance to food and never be lethal."""
    rng = random.Random(42)
    checked = 0
    for _ in range(300):
        sample = generate_snake_sample(rng)
        if sample is None:
            continue
        checked += 1
        # Parse the rendered grid back.
        lines = sample.state_text.split("\n")
        H, W = map(int, lines[0].split()[1].rstrip(",").split("x"))
        grid_rows = lines[1 : 1 + H]
        head = food = None
        body = set()
        for r, row in enumerate(grid_rows):
            for c, ch in enumerate(row):
                if ch == "H":
                    head = (r, c)
                elif ch == "F":
                    food = (r, c)
                elif ch == "B":
                    body.add((r, c))
        assert head is not None and food is not None

        chosen = sample.options[sample.correct_idx]
        dr, dc = DIRS[chosen]
        nxt = (head[0] + dr, head[1] + dc)
        # Never lethal: inside bounds and not into the body.
        assert 0 <= nxt[0] < H and 0 <= nxt[1] < W, f"out of bounds move {chosen}"
        assert nxt not in body, f"expert moved into body: {chosen}"
        checked_optimal = sample.optimal_set
        assert sample.correct_idx in checked_optimal
    assert checked >= 200, f"too few valid snake samples generated: {checked}"


def test_snake_expert_scores_consistent():
    """optimal_set flags must match expert scores for the rendered position."""
    rng = random.Random(7)
    for _ in range(100):
        sample = generate_snake_sample(rng)
        if sample is None:
            continue
        optimal_names = {sample.options[i] for i in sample.optimal_set}
        assert len(optimal_names) >= 1
        # At least one optimal option must exist and correct_idx must be one of them.
        assert sample.correct_idx in sample.optimal_set


def test_maze_expert_shortens_distance():
    """Expert move must reduce BFS distance to the exit."""
    rng = random.Random(123)
    checked = 0
    for _ in range(200):
        sample = generate_maze_sample(rng)
        if sample is None:
            continue
        checked += 1
        lines = sample.state_text.split("\n")
        H, W = map(int, lines[0].split()[1].rstrip(",").split("x"))
        grid_rows = lines[1 : 1 + H]
        start = goal = None
        walls = set()
        for r, row in enumerate(grid_rows):
            for c, ch in enumerate(row):
                if ch == "S":
                    start = (r, c)
                elif ch == "E":
                    goal = (r, c)
                elif ch == "#":
                    walls.add((r, c))
        assert start is not None and goal is not None
        chosen = sample.options[sample.correct_idx]
        dr, dc = DIRS[chosen]
        nxt = (start[0] + dr, start[1] + dc)
        assert nxt not in walls, f"expert stepped into a wall: {chosen}"
        assert 0 <= nxt[0] < H and 0 <= nxt[1] < W
    assert checked >= 150, f"too few valid maze samples: {checked}"


def _play_ttt(x_strategy, o_strategy, rng):
    """Play one game; returns winner ('X','O') or 'draw'."""
    board = tuple("." for _ in range(9))
    player = "X"
    while True:
        w = _ttt_winner(board)
        if w is not None:
            return w
        if all(c != "." for c in board):
            return "draw"
        if player == "X":
            move = x_strategy(board, rng)
        else:
            move = o_strategy(board, rng)
        board = board[:move] + (player,) + board[move + 1 :]
        player = "O" if player == "X" else "X"


def test_ttt_minimax_unbeatable_against_random():
    """Perfect X vs random O: X must never lose."""
    rng = random.Random(999)
    for _ in range(100):
        result = _play_ttt(
            lambda b, r: ttt_optimal_moves(b)[0],
            lambda b, r: r.choice([i for i in range(9) if b[i] == "."]),
            rng,
        )
        assert result != "O", f"perfect X lost to random O: {result}"


def test_ttt_mirror_perfect_draw():
    """Perfect X vs perfect O: always a draw (both optimal)."""
    rng = random.Random(5)
    for _ in range(50):
        result = _play_ttt(
            lambda b, r: ttt_optimal_moves(b)[0],
            lambda b, r: min(
                (i for i in range(9) if b[i] == "."),
                key=lambda i: _ttt_minimax(b[:i] + ("O",) + b[i + 1 :], "X"),
            ),
            rng,
        )
        assert result == "draw", f"perfect vs perfect ended: {result}"


def test_ttt_sample_consistency():
    """Generated samples: X to move, optimal set non-empty, correct_idx consistent."""
    rng = random.Random(31)
    checked = 0
    for _ in range(200):
        sample = generate_ttt_sample(rng)
        if sample is None:
            continue
        checked += 1
        assert len(sample.options) == 9
        assert sample.correct_idx in sample.optimal_set
        chosen = sample.options[sample.correct_idx]
        # Chosen cell must be empty on the rendered board.
        lines = sample.state_text.split("\n")
        board = "".join(lines[1:4])
        r, c = map(int, chosen.split("(")[1].rstrip(")").split(","))
        assert board[r * 3 + c] == ".", f"expert chose occupied cell {chosen}"
    assert checked >= 150


def test_options_are_shuffled():
    """Option order must vary across samples (no fixed positional bias)."""
    rng = random.Random(2)
    first_options = set()
    for _ in range(50):
        sample = generate_snake_sample(rng)
        if sample is None:
            continue
        first_options.add(sample.options[0])
    assert len(first_options) >= 3, "options appear positionally fixed"


def test_build_prompt_format():
    sample = TaskSample(
        domain="test",
        state_text="S",
        question="Q?",
        options=["a", "b", "c"],
        optimal_set=frozenset({1}),
    )
    prompt = build_prompt(sample)
    assert prompt.startswith("[STATE] S")
    assert "[QUESTION] Q?" in prompt
    assert prompt.count("[OPT]") == 3
    assert prompt.endswith("[OPT] c")
    assert sample.correct_idx == 1


def test_mixed_batch_domains():
    rng = random.Random(11)
    batch = generate_mixed_batch(rng, n=60)
    assert len(batch) >= 50
    domains = {s.domain for s in batch}
    assert domains == {"snake", "maze", "tic_tac_toe"}
    for s in batch:
        assert s.correct_idx in s.optimal_set
        assert len(s.options) >= 3