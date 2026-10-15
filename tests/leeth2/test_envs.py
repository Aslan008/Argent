"""The playable game environments: the perfect expert must play them (near) perfectly, a random player must not."""

from src.leeth2.envs import ENVS, expert_policy, random_policy, rollout, summarize


def play(name, policy, n=60, seed=1):
    return summarize(rollout(ENVS[name], n, policy, seed=seed))


def test_the_expert_solves_maze_and_treasure_and_never_loses_tic_tac_toe():
    assert play("maze", expert_policy)["solved"] == 1.0
    t = play("treasure", expert_policy)
    assert t["reached"] == 1.0 and t["died"] == 0.0
    assert play("ttt_vs_perfect", expert_policy, n=80)["lost"] == 0.0
    assert play("ttt_vs_random", expert_policy, n=80)["lost"] == 0.0


def test_the_expert_snake_eats_a_lot_and_every_move_is_optimal():
    s = play("snake", expert_policy, n=40)
    assert s["optimal_rate"] == 1.0
    assert s["eaten"] >= 5


def test_a_random_player_is_clearly_worse():
    assert play("maze", random_policy(0))["solved"] < 0.5
    assert play("treasure", random_policy(0))["reached"] < 0.5
    assert play("ttt_vs_perfect", random_policy(0), n=80)["lost"] > 0.4
    assert play("snake", random_policy(0), n=40)["eaten"] < play("snake", expert_policy, n=40)["eaten"]


def test_ttt_covers_both_players_and_second_player_positions():
    import random

    from src.leeth2.envs import TttEnv

    seen = set()
    for i in range(60):
        e = TttEnv(random.Random(i), opponent="perfect")
        s = e.decision(random.Random(i))
        seen.add((e.me, s.state.count("X") - s.state.count("O") if False else sum(c == "X" for c in e.b) - sum(c == "O" for c in e.b)))
    assert ("X", 0) in seen and ("O", 1) in seen            # equal marks as first player, one extra X as the second
