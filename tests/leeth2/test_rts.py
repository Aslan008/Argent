"""MiniRTS: rules, determinism, the option interface and the strength ordering of the scripted players."""

from src.leeth2 import rts_bots as B
from src.leeth2.rts import MAX_TICKS, RTS


def test_a_game_is_deterministic_and_always_ends():
    a = B.play(3, (B.teacher, B.light_rush))
    b = B.play(3, (B.teacher, B.light_rush))
    assert a[0] == b[0] and a[1].tick == b[1].tick and a[1].stats == b[1].stats
    assert a[0] in (-1, 0, 1) and a[1].tick <= MAX_TICKS


def test_map_is_connected_and_symmetric():
    for seed in range(30):
        g = RTS(seed)
        b0, b1 = g.base(0), g.base(1)
        assert g._path_len(b0.pos, b1.pos) is not None
        assert all(g._mirror(w) in g.walls for w in g.walls)
        assert {g._mirror(p.pos) for p in g.patches.values() if p.pos != g._mirror(p.pos)} <= {p.pos for p in g.patches.values()}


def test_every_decision_has_valid_options_and_describes_itself():
    g = RTS(5)
    for _ in range(120):
        for owner in (0, 1):
            for eid in g.pending(owner):
                e = g.entities[eid]
                state, question, texts, orders = g.describe(owner, eid)
                opts = g.options(eid)
                assert 1 <= len(texts) <= 40 and len(texts) == len(set(texts)) and len(texts) == len(orders)
                assert f"#{eid}" in state and "tick" in state and question
                g.set_order(eid, B.teacher(g, e, opts))
        g.step()


def test_harvest_train_and_build_work():
    g = RTS(2)
    start = g.resources[0]
    for _ in range(80):
        for eid in g.pending(0):
            e = g.entities[eid]
            g.set_order(eid, B.teacher(g, e, g.options(eid)))
        g.step()
    assert g.stats[0]["mined"] > 0
    assert g.mine(0, "barracks") and len(g.units(0)) > 1
    assert g.resources[0] != start


def test_teacher_beats_the_weak_players_and_random_loses():
    assert B.duel("teacher", "random", seeds=4)["win"] >= 0.9
    assert B.duel("teacher", "worker_rush", seeds=4)["win"] >= 0.5
    assert B.duel("random", "light_rush", seeds=4)["win"] <= 0.2
