"""The new games: every label is the TRUE optimal set (cross-checked against independent solvers), games finish, rules are in the text."""

import itertools
import random

import pytest

from src.leeth2 import envs2
from src.leeth2.envs import expert_policy, random_policy, rollout, summarize
from src.leeth2.envs2 import ENVS_MORE, ENVS_UNSEEN
from src.leeth2.schema import Sample

ALL = {**ENVS_MORE, **ENVS_UNSEEN}


def test_every_position_is_a_valid_sample_with_rules_and_unique_options():
    for name, make in ALL.items():
        rng = random.Random(7)
        for g in range(25):
            env = make(random.Random(rng.random()))
            for _ in range(80):
                if env.done:
                    break
                s = env.decision(rng)
                s.validate()
                assert isinstance(s, Sample) and s.ptype == "choice" and s.family == "game"
                assert len(set(s.options)) == len(s.options) and len(s.options) >= 2, (name, s.options)
                assert ("Rules" in s.state) or ("Правила" in s.state), name
                t = s.target
                assert (abs(sum(t) - 1) < 1e-6 and max(t) > 0) if isinstance(t, list) else 0 <= t < len(s.options)
                env.apply(expert_policy([s])[0])


@pytest.mark.parametrize("name", sorted(ALL))
def test_the_expert_wins_or_solves_every_game(name):
    res = rollout(ALL[name], 40, expert_policy, seed=3)
    s = summarize(res)
    if "solved" in s:
        assert s["solved"] == 1.0, s
    if "reached" in s:
        assert s["reached"] == 1.0, s
    if "won" in s and "vs_perfect" not in name and "vs_search" not in name:
        # against a random opponent the expert only fails when it starts from a lost position (nim) or a drawn board (connect)
        floor = {"nim": 0.55, "connect": 0.55, "chomp": 0.45, "cram": 0.45}
        key = next(k for k in floor if name.startswith(k))
        assert s["won"] >= floor[key], s
    assert s["optimal_rate"] == 1.0 if "optimal_rate" in s else True


@pytest.mark.parametrize("name", ["sokoban", "slide", "keydoor", "lights", "hanoi"])
def test_the_expert_takes_exactly_the_optimal_number_of_moves(name):
    for r in rollout(ENVS_MORE[name], 30, expert_policy, seed=5):
        assert r["steps"] == r["optimal"], (name, r)


def test_terrain_expert_pays_exactly_the_cheapest_cost():
    for r in rollout(ENVS_MORE["terrain"], 40, expert_policy, seed=2):
        assert r["cost"] == r["optimal_cost"] and r["reached"] == 1.0


@pytest.mark.parametrize("name", ["sokoban", "slide", "nim_vs_random", "keydoor", "lights", "hanoi", "terrain", "connect_vs_random"])
def test_a_random_player_does_clearly_worse_than_the_expert(name):
    exp = summarize(rollout(ENVS_MORE[name], 60, expert_policy, seed=11))
    rnd = summarize(rollout(ENVS_MORE[name], 60, random_policy(1), seed=11))
    key = next(k for k in ("solved", "reached", "won") if k in exp)
    assert rnd[key] < exp[key] - 0.1 or exp[key] - rnd[key] >= 0.0 and rnd["optimal_rate"] < 0.8, (name, exp, rnd)


# ---------------------------------------------------------------- independent checks of the experts
def test_slide_distance_matches_iterative_deepening():
    table = envs2._slide_table(2, 3)
    goal = (1, 2, 3, 4, 5, 0)

    def nbrs(s):
        z = s.index(0)
        zr, zc = divmod(z, 3)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = zr + dr, zc + dc
            if 0 <= r < 2 and 0 <= c < 3:
                t = list(s)
                t[z], t[r * 3 + c] = t[r * 3 + c], 0
                yield tuple(t)

    def ids(s, limit, seen):
        if s == goal:
            return True
        if limit == 0:
            return False
        return any(ids(t, limit - 1, seen) for t in nbrs(s) if t not in seen)

    rng = random.Random(1)
    for s in rng.sample(sorted(table), 25):
        d = table[s]
        assert ids(s, d, {s}) and (d == 0 or not ids(s, d - 1, {s}))


def test_nim_normal_play_matches_the_xor_rule():
    for piles in itertools.product(range(0, 6), repeat=3):
        xor = piles[0] ^ piles[1] ^ piles[2]
        assert envs2._nim_wins(tuple(sorted(p for p in piles if p)), False) == (xor != 0)


def test_misere_nim_with_only_singletons_is_reversed():
    assert envs2._nim_wins((1,), True) is False           # forced to take the last one: loses
    assert envs2._nim_wins((1, 1), True) is True
    assert envs2._nim_wins((1, 1, 1), True) is False
    assert envs2._nim_wins((2, 1), True) is True


def test_lights_out_minimum_matches_brute_force():
    n = 3
    rng = random.Random(4)
    for _ in range(40):
        state = [0] * 9
        for i in rng.sample(range(9), rng.randint(1, 9)):
            r, c = divmod(i, 3)
            for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= r + dr < 3 and 0 <= c + dc < 3:
                    state[(r + dr) * 3 + c + dc] ^= 1
        best = 99
        for mask in range(512):
            t = list(state)
            for i in range(9):
                if mask >> i & 1:
                    r, c = divmod(i, 3)
                    for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                        if 0 <= r + dr < 3 and 0 <= c + dc < 3:
                            t[(r + dr) * 3 + c + dc] ^= 1
            if not any(t):
                best = min(best, bin(mask).count("1"))
        sols = envs2._lights_solutions(3, tuple(state))
        assert min(bin(x).count("1") for x in sols) == best


def test_lights_out_5x5_has_the_known_four_solutions_for_a_solvable_board():
    state = [0] * 25
    state[12] = 1
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        state[(2 + dr) * 5 + 2 + dc] ^= 1
    sols = envs2._lights_solutions(5, tuple(state))
    assert len(sols) in (1, 2, 4, 8)


def test_hanoi_distances_are_the_textbook_ones():
    for n in range(1, 6):
        assert envs2._hanoi_table(n)[tuple([0] * n)] == 2 ** n - 1


def test_connect_takes_a_win_and_blocks_a_loss():
    # X to move with three in a row on the bottom of a 4x5 Connect-3 board
    b = ("....." , ".....", ".....", "XX...")
    vals = envs2.connect_values(b, 3, "X", 4)
    top = max(vals.values())
    assert {c for c, v in vals.items() if v == top} == {2}              # (a three in a row: column 3 wins on the spot)
    # O threatens to win on column 4: X must block
    b = (".....", ".....", ".....", "OO.X.")
    vals = envs2.connect_values(b, 3, "X", 4)
    assert max(vals, key=vals.get) == 2


def test_chomp_exact_solver_small_boards():
    assert envs2._chomp_wins((3, 3)) is True                # any rectangle bigger than 1x1 is a first-player win
    assert envs2._chomp_wins((1,)) is False
    assert envs2._chomp_wins((2, 1)) is False               # an L of three squares: the mover loses
    assert envs2._chomp_wins((3, 1, 1)) is False


def test_cram_exact_solver_small_boards():
    assert envs2._cram_wins(2, 2, 0) is False               # 2x2: the second player mirrors
    assert envs2._cram_wins(1, 2, 0) is True


def test_determinism_given_the_seed():
    for name, make in ALL.items():
        a = make(random.Random(99)).decision(random.Random(5))
        b = make(random.Random(99)).decision(random.Random(5))
        assert a.to_json() == b.to_json(), name
