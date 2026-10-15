"""
snake_benchmark.py — Automated verification script testing Leeth on Snake acceptance gates.

Tests:
1. 20 consecutive 6x6 games played by Leeth model (win-rate >= 80%, target len 12).
2. 5 games on 12x12 grid (zero-shot, move optimality >= 85%).
3. Expert BFS baseline comparison on identical seeds.
"""

import sys
import time
from leeth_player import LeethSnakePlayer
from snake_engine import SnakeGame, SessionTracker
from src.leeth.task_domains import DIR_ORDER


def run_benchmark(n_games_6x6: int = 20, n_games_12x12: int = 5):
    print("=" * 70)
    print("  LEETH SNAKE MODEL VERIFICATION BENCHMARK")
    print("=" * 70)
    
    print("\n[1/3] Loading Leeth Snake Player (mmBERT + DynamicOptionHead)...")
    t0 = time.time()
    player = LeethSnakePlayer(device="cpu")
    print(f"Model loaded in {time.time() - t0:.2f}s on device: {player.device}")

    # ---------------------------------------------------------
    # Gate 1: 20 games on 6x6 (target length 12)
    # ---------------------------------------------------------
    print(f"\n[2/3] Running {n_games_6x6} games on 6x6 grid (Target Length: 12)...")
    tracker_model = SessionTracker()
    tracker_expert = SessionTracker()

    for g_idx in range(1, n_games_6x6 + 1):
        seed = 1000 + g_idx
        # Game 1: Model
        game = SnakeGame(height=6, width=6, target_length=12, seed=seed)
        step_limit = 200
        while not game.game_over and game.steps_count < step_limit:
            state_text = game.get_state_text()
            pred = player.predict(state_text, options=DIR_ORDER, shuffle=True)
            game.step(pred["best_move"], probabilities=pred["probabilities"])
        tracker_model.record_game(game)

        # Game 2: BFS Expert on identical seed
        game_exp = SnakeGame(height=6, width=6, target_length=12, seed=seed)
        while not game_exp.game_over and game_exp.steps_count < step_limit:
            opt_moves = game_exp.get_optimal_moves()
            chosen = opt_moves[0] if opt_moves else "вверх"
            game_exp.step(chosen)
        tracker_expert.record_game(game_exp)

        status_str = "WIN" if game.won else f"FAIL ({game.death_reason})"
        print(f"  Game {g_idx:2d}/{n_games_6x6}: Final Length = {len(game.body):2d} | Steps = {game.steps_count:2d} | {status_str}")

    sum_model = tracker_model.summary()
    sum_exp = tracker_expert.summary()
    print("\n--- 6x6 Session Results ---")
    print(f"Model Win-Rate:       {sum_model['win_rate']:.1f}% ({sum_model['wins']}/{n_games_6x6}) [Gate target: >= 80%]")
    print(f"Expert Win-Rate:      {sum_exp['win_rate']:.1f}% ({sum_exp['wins']}/{n_games_6x6})")
    print(f"Model Mean Length:    {sum_model['mean_length']:.2f}")
    print(f"Expert Mean Length:   {sum_exp['mean_length']:.2f}")
    print(f"Move Optimality Rate: {sum_model['optimality_rate']:.1f}% [Target: ~98%]")

    # ---------------------------------------------------------
    # Gate 2: Zero-shot 12x12 grid
    # ---------------------------------------------------------
    print(f"\n[3/3] Running {n_games_12x12} games on 12x12 grid (Zero-Shot, Target Length: 24)...")
    tracker_12 = SessionTracker()
    for g_idx in range(1, n_games_12x12 + 1):
        seed = 5000 + g_idx
        game_12 = SnakeGame(height=12, width=12, target_length=24, seed=seed)
        while not game_12.game_over and game_12.steps_count < 300:
            state_text = game_12.get_state_text()
            pred = player.predict(state_text, options=DIR_ORDER, shuffle=True)
            game_12.step(pred["best_move"], probabilities=pred["probabilities"])
        tracker_12.record_game(game_12)
        print(f"  12x12 Game {g_idx}: Length = {len(game_12.body):2d} | Steps = {game_12.steps_count:2d} | Won: {game_12.won}")

    sum_12 = tracker_12.summary()
    print("\n--- 12x12 Zero-Shot Results ---")
    print(f"Zero-Shot Optimality: {sum_12['optimality_rate']:.1f}% [Gate target: >= 85%]")
    print(f"Zero-Shot Mean Len:   {sum_12['mean_length']:.2f}")

    print("\n" + "=" * 70)
    gate1_pass = sum_model["win_rate"] >= 80.0
    gate2_pass = sum_12["optimality_rate"] >= 85.0
    print(f"Gate 1 (6x6 Win Rate >= 80%):        {'PASSED' if gate1_pass else 'FAILED'} ({sum_model['win_rate']}%)")
    print(f"Gate 2 (12x12 Optimality >= 85%):    {'PASSED' if gate2_pass else 'FAILED'} ({sum_12['optimality_rate']}%)")
    print("=" * 70)
    return gate1_pass and gate2_pass


if __name__ == "__main__":
    success = run_benchmark(n_games_6x6=20, n_games_12x12=5)
    sys.exit(0 if success else 1)
