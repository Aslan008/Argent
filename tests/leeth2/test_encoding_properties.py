"""Property tests: whatever the sample, the encoded sequence keeps its structure and its answer."""

import random

import pytest

from src.leeth2.data.synth import browser, games, npc, sorting, tools
from src.leeth2.encoding import MARKER_NAMES, Leeth2Encoder
from src.leeth2.schema import Sample

from .conftest import VOCAB

GENS = [games.gen_snake, games.gen_maze, games.gen_ttt, games.gen_treasure, npc.gen_policy, npc.gen_persona, sorting.gen_files, sorting.gen_tickets, sorting.gen_order,
        browser.gen_click, browser.gen_risk, tools.gen_toolpick]


def samples(n_per=60, seed=0):
    rng = random.Random(seed)
    out = []
    for g in GENS:
        k = 0
        while k < n_per:
            got = g(rng)
            if got:
                out.extend(got if isinstance(got, list) else [got])
                k += 1
    return out


@pytest.mark.parametrize("max_len", [48, 128, 512])
def test_structure_and_targets_survive_any_budget(tok, max_len):
    enc = Leeth2Encoder(tok, VOCAB, max_len=max_len, min_state=8)
    OPT, RD = enc.marker("OPT"), enc.marker("RD")
    rng = random.Random(1)
    for s in samples():
        for r in (None, rng):
            x = enc.encode(s, rng=r, want_offsets=(s.ptype == "extract"))
            ids = x.input_ids
            assert ids[0] == enc.bos and ids[-1] == enc.eos
            assert len(ids) == len(x.opt_index)
            assert ids[x.rd_pos] == RD and ids.count(RD) == 1
            assert 1 <= x.state_lo <= x.state_hi <= x.rd_pos
            K = len(s.options or [])
            assert x.n_opts == K and ids.count(OPT) == K
            if s.ptype == "choice":
                assert 0 <= x.choice_target < K and x.perm[x.choice_target] == s.hard_choice()
                # every option owns a contiguous block that starts with its marker
                for j in range(1, K + 1):
                    pos = [i for i, v in enumerate(x.opt_index) if v == j]
                    assert pos == list(range(pos[0], pos[-1] + 1)) and ids[pos[0]] == OPT
                if isinstance(s.target, list):
                    assert x.choice_probs is not None and abs(sum(x.choice_probs) - 1) < 1e-6
            if s.ptype == "extract":
                if s.target is None:
                    assert x.start_target == x.end_target == x.rd_pos
                else:
                    a, b = x.start_target, x.end_target
                    # either the answer survived truncation and is inside the state window, or it points at [RD]
                    assert (x.state_lo <= a <= b < x.state_hi) or (a == b == x.rd_pos)
                    if a != x.rd_pos and x.state_offsets:
                        lo, hi = x.state_offsets[a - x.state_lo][0], x.state_offsets[b - x.state_lo][1]
                        got, gold = s.state[lo:hi].strip(), s.state[s.target[0] : s.target[1]].strip()
                        # pointers are token-aligned: a word-level fake token may carry quotes/punctuation around the answer
                        assert gold in got and len(got) - len(gold) <= 6
            if s.ptype == "score":
                assert x.score_pos == pytest.approx(s.target - 1) and enc.marker(f"L{s.levels}") in ids


def test_all_markers_are_distinct_and_above_the_vocabulary(tok):
    enc = Leeth2Encoder(tok, VOCAB)
    ids = [enc.marker(n) for n in MARKER_NAMES]
    assert len(set(ids)) == len(ids) and min(ids) >= VOCAB


def test_long_context_samples_get_their_own_budget(tok):
    enc = Leeth2Encoder(tok, VOCAB, max_len=64)
    state = " ".join(f"w{i}" for i in range(500))
    plain = Sample("t", "f", "en", "noul", state, "q?", None, 1.0)
    long = Sample("t", "f", "en", "noul", state, "q?", None, 1.0, meta={"max_len": 300})
    assert len(enc.encode(plain, rng=random.Random(0))) <= 64
    assert 250 <= len(enc.encode(long, rng=random.Random(0))) <= 300
    # inference ignores the training-time hint: the runtime decides the budget
    assert len(enc.encode(long)) <= 64
