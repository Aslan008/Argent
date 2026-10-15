"""
snake_engine.py — Core game engine for Snake (matching Leeth task_domains formatting).

Features:
- Deterministic and random game initialization (6x6, 8x8, 12x12).
- Exact state text serialization conforming to src/leeth/task_domains.py.
- Built-in integration with snake_expert_scores BFS expert.
- Step execution with collision detection, growth, food placement, and win condition.
- Session statistics tracking (win-rate, average/median length, expert agreement rate).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.leeth.task_domains import DIRS, DIR_ORDER, snake_expert_scores

Cell = Tuple[int, int]


@dataclass
class StepRecord:
    """Detailed record of a single turn for playback, inspection, and stats."""
    step_number: int
    head: Cell
    food: Cell
    body_length: int
    chosen_move: str
    probabilities: Dict[str, float]
    expert_optimal_moves: List[str]
    is_optimal_by_expert: bool
    state_text: str
    result: str  # "continue", "food_eaten", "wall_collision", "self_collision", "win"


class SnakeGame:
    """Stateful Snake game environment."""

    def __init__(
        self,
        height: int = 6,
        width: int = 6,
        target_length: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.height = height
        self.width = width
        # Target length default: 12 on 6x6, 16 on 8x8, 24 on 12x12
        if target_length is None:
            if height == 6 and width == 6:
                self.target_length = 12
            elif height <= 8 and width <= 8:
                self.target_length = 16
            else:
                self.target_length = 24
        else:
            self.target_length = target_length

        self.rng = random.Random(seed)
        self.head: Cell = (0, 0)
        self.body: List[Cell] = []
        self.food: Cell = (0, 0)
        self.game_over: bool = False
        self.won: bool = False
        self.death_reason: Optional[str] = None
        self.steps_count: int = 0
        self.history: List[StepRecord] = []

        self.reset()

    def reset(self, body_length: int = 3) -> None:
        """Resets the game board with a healthy initial snake and food."""
        self.game_over = False
        self.won = False
        self.death_reason = None
        self.steps_count = 0
        self.history.clear()

        # Place initial snake near center
        mid_r = self.height // 2
        mid_c = self.width // 2

        # Horizontal snake heading right: (mid_r, mid_c) is head, (mid_r, mid_c-1)... are body
        self.body = [(mid_r, mid_c - i) for i in range(body_length)]
        self.head = self.body[0]

        self._spawn_food()

    def _spawn_food(self) -> bool:
        """Places food on a random unoccupied cell."""
        body_set = set(self.body)
        free_cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in body_set
        ]
        if not free_cells:
            self.food = (-1, -1)
            return False

        self.food = self.rng.choice(free_cells)
        return True

    def get_state_text(self) -> str:
        """Renders grid state string matching task_domains byte-for-byte:

        'змейка HxW, H=голова, F=еда, B=тело\\n' + lines of H, F, B, .
        """
        rows = []
        body_set = set(self.body)
        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                pos = (r, c)
                if pos == self.head:
                    row_chars.append("H")
                elif pos == self.food:
                    row_chars.append("F")
                elif pos in body_set:
                    row_chars.append("B")
                else:
                    row_chars.append(".")
            rows.append("".join(row_chars))

        return f"змейка {self.height}x{self.width}, H=голова, F=еда, B=тело\n" + "\n".join(rows)

    def get_expert_scores(self) -> Dict[str, bool]:
        """Calculates optimal directions according to perfect BFS / survival expert."""
        return snake_expert_scores(self.head, self.food, self.body, self.height, self.width)

    def get_optimal_moves(self) -> List[str]:
        """List of direction names rated optimal by the BFS expert."""
        scores = self.get_expert_scores()
        return [d for d in DIR_ORDER if scores.get(d, False)]

    def step(
        self,
        direction: str,
        probabilities: Optional[Dict[str, float]] = None,
    ) -> StepRecord:
        """Executes one step in the specified direction.

        Directions: 'вверх', 'вниз', 'влево', 'вправо'
        """
        if self.game_over:
            raise RuntimeError("Cannot step in a game that has already ended.")

        state_before = self.get_state_text()
        expert_optimal = self.get_optimal_moves()
        is_optimal = direction in expert_optimal

        if direction not in DIRS:
            raise ValueError(f"Unknown direction '{direction}'. Allowed: {list(DIRS.keys())}")

        dr, dc = DIRS[direction]
        new_head = (self.head[0] + dr, self.head[1] + dc)
        self.steps_count += 1

        result = "continue"

        # 1. Wall collision
        if not (0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width):
            self.game_over = True
            self.won = False
            self.death_reason = "wall_collision"
            result = "wall_collision"

        # 2. Self collision (tail moves away unless food is eaten, but standard check is body[:-1])
        elif new_head in self.body[:-1]:
            self.game_over = True
            self.won = False
            self.death_reason = "self_collision"
            result = "self_collision"

        else:
            # Valid step
            if new_head == self.food:
                # Food eaten: grow snake
                self.body.insert(0, new_head)
                self.head = new_head
                result = "food_eaten"

                if len(self.body) >= self.target_length:
                    self.won = True
                    self.game_over = True
                    result = "win"
                else:
                    if not self._spawn_food():
                        # Entire board filled
                        self.won = True
                        self.game_over = True
                        result = "win"
            else:
                # Normal move
                self.body.insert(0, new_head)
                self.body.pop()
                self.head = new_head

        record = StepRecord(
            step_number=self.steps_count,
            head=self.head,
            food=self.food,
            body_length=len(self.body),
            chosen_move=direction,
            probabilities=probabilities or {},
            expert_optimal_moves=expert_optimal,
            is_optimal_by_expert=is_optimal,
            state_text=state_before,
            result=result,
        )
        self.history.append(record)
        return record


class SessionTracker:
    """Maintains statistical metrics across multiple played games."""

    def __init__(self):
        self.games_played = 0
        self.wins = 0
        self.lengths: List[int] = []
        self.steps: List[int] = []
        self.total_moves = 0
        self.optimal_moves = 0
        self.death_reasons: Dict[str, int] = {"wall_collision": 0, "self_collision": 0}

    def record_game(self, game: SnakeGame) -> None:
        self.games_played += 1
        if game.won:
            self.wins += 1
        elif game.death_reason:
            self.death_reasons[game.death_reason] = (
                self.death_reasons.get(game.death_reason, 0) + 1
            )

        self.lengths.append(len(game.body))
        self.steps.append(game.steps_count)

        for step in game.history:
            self.total_moves += 1
            if step.is_optimal_by_expert:
                self.optimal_moves += 1

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games_played * 100.0) if self.games_played > 0 else 0.0

    @property
    def mean_length(self) -> float:
        return (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    @property
    def median_length(self) -> float:
        if not self.lengths:
            return 0.0
        sorted_lens = sorted(self.lengths)
        n = len(sorted_lens)
        mid = n // 2
        if n % 2 == 1:
            return float(sorted_lens[mid])
        return (sorted_lens[mid - 1] + sorted_lens[mid]) / 2.0

    @property
    def optimality_rate(self) -> float:
        return (
            (self.optimal_moves / self.total_moves * 100.0)
            if self.total_moves > 0
            else 0.0
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 2),
            "mean_length": round(self.mean_length, 2),
            "median_length": round(self.median_length, 2),
            "optimality_rate": round(self.optimality_rate, 2),
            "total_moves": self.total_moves,
            "death_reasons": dict(self.death_reasons),
        }
