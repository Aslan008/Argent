import random

from src.leeth2.encoding import MARKER_NAMES, Leeth2Encoder, collate
from src.leeth2.schema import Sample

from .conftest import VOCAB


def enc(tok, **kw):
    return Leeth2Encoder(tok, VOCAB, **kw)


def mk(**kw):
    base = dict(task="t", family="f", lang="en", ptype="choice", state="the cat sat on the mat", question="what sat",
                options=["cat", "dog", "bird"], target=0)
    base.update(kw)
    return Sample(**base)


def test_markers_are_virtual_ids_above_the_vocab(tok):
    e = enc(tok)
    assert e.marker("OPT") == VOCAB + MARKER_NAMES.index("OPT")
    assert e.marker("RD") >= VOCAB


def test_choice_layout_and_option_index(tok):
    e = enc(tok)
    x = e.encode(mk())
    ids = x.input_ids
    assert ids[0] == e.bos and ids[-1] == e.eos
    assert ids.count(e.marker("OPT")) == 3
    assert ids[x.rd_pos] == e.marker("RD")
    # every option owns its marker + tokens, in order, and nothing else is tagged
    assert x.n_opts == 3
    assert sorted(set(x.opt_index)) == [0, 1, 2, 3]
    assert x.opt_index[: x.rd_pos + 1] == [0] * (x.rd_pos + 1)
    assert [ids[i] for i, j in enumerate(x.opt_index) if j and ids[i] >= VOCAB] == [e.marker("OPT")] * 3
    assert x.choice_target == 0 and x.perm == [0, 1, 2]


def test_shuffle_keeps_the_target_pointing_at_the_right_option(tok):
    e = enc(tok)
    s = mk(options=["cat", "dog", "bird", "fish", "cow"], target=3)  # "fish"
    fish_id = tok(["fish"])["input_ids"][0][0]
    seen = set()
    for seed in range(20):
        x = e.encode(s, rng=random.Random(seed))
        seen.add(tuple(x.perm))
        # the option shown at choice_target is the original target
        assert x.perm[x.choice_target] == 3
        pos = [i for i, j in enumerate(x.opt_index) if j == x.choice_target + 1 and x.input_ids[i] != e.marker("OPT")]
        assert [x.input_ids[i] for i in pos] == [fish_id]
    assert len(seen) > 5


def test_inference_is_deterministic(tok):
    e = enc(tok)
    s = mk()
    assert e.encode(s).input_ids == e.encode(s).input_ids


def test_noul_and_score_targets_and_level_marker(tok):
    e = enc(tok)
    n = e.encode(mk(ptype="noul", options=None, target=0.25))
    assert n.noul_target == 0.25 and n.choice_target == -100 and n.n_opts == 0
    assert e.marker("T_NOUL") in n.input_ids
    s = e.encode(mk(ptype="score", options=None, target=3.5, levels=5))
    assert s.score_pos == 2.5 and s.levels == 5
    assert e.marker("L5") in s.input_ids and e.marker("T_SCORE") in s.input_ids


def test_extract_span_maps_to_state_tokens(tok):
    e = enc(tok)
    state = "alpha beta gamma delta"
    s = mk(ptype="extract", options=None, state=state, question="which", target=[state.index("beta"), state.index("gamma") + 5])
    x = e.encode(s)
    toks = x.input_ids[x.start_target : x.end_target + 1]
    assert toks == tok(["beta gamma"])["input_ids"][0]
    assert x.state_lo <= x.start_target <= x.end_target < x.state_hi
    # char offsets let the runtime decode the span back to text
    a, b = x.state_offsets[x.start_target - x.state_lo][0], x.state_offsets[x.end_target - x.state_lo][1]
    assert state[a:b].strip() == "beta gamma"


def test_extract_without_answer_points_at_the_read_out_token(tok):
    e = enc(tok)
    x = e.encode(mk(ptype="extract", options=None, target=None))
    assert x.start_target == x.end_target == x.rd_pos


def test_long_state_is_windowed_around_the_answer(tok):
    e = enc(tok, max_len=64, min_state=8)
    words = [f"w{i}" for i in range(400)]
    state = " ".join(words)
    cs = state.index("w300")
    s = mk(ptype="extract", options=None, state=state, question="find", target=[cs, cs + 4])
    for seed in range(10):
        x = e.encode(s, rng=random.Random(seed))
        assert len(x) <= 64
        assert x.start_target != x.rd_pos, "the answer must stay inside the window"
        assert x.input_ids[x.start_target] == tok(["w300"])["input_ids"][0][0]


def test_long_state_without_span_keeps_head_and_tail(tok):
    e = enc(tok, max_len=60, min_state=8)
    state = " ".join(f"w{i}" for i in range(300))
    x = e.encode(mk(ptype="noul", options=None, state=state, target=1.0))
    assert len(x) <= 60
    kept = x.input_ids[x.state_lo : x.state_hi]
    ids = tok([state])["input_ids"][0]
    assert kept[0] == ids[0] and kept[-1] == ids[-1]


def test_many_options_extend_the_length_instead_of_dropping_options(tok):
    e = enc(tok, max_len=128)
    opts = [f"option number {i}" for i in range(255)]
    x = e.encode(mk(options=opts, target=200), rng=random.Random(1))
    assert x.n_opts == 255
    assert x.input_ids.count(e.marker("OPT")) == 255
    assert len(x) > 128, "length grows so that all 255 options fit"


def test_a_list_of_more_than_128_options_keeps_16_tokens_per_option_and_a_shorter_one_is_unchanged(tok):
    e = enc(tok, max_len=4096)
    long_option = " ".join(f"word{i}" for i in range(40))  # far more than 16 tokens
    for k, want in ((100, (4096 // 2) // 100 - 1), (128, 15), (255, 15)):
        x = e.encode(mk(options=[f"{i} {long_option}" for i in range(k)], target=0))
        sizes = [x.opt_index.count(j) for j in range(1, k + 1)]
        # an option owns its marker plus its (capped) tokens
        assert max(sizes) - 1 == min(want, e.opt_cap), (k, max(sizes))
        assert x.n_opts == k and len(x) <= e.hard_max_len


def test_collate_shapes_and_masks(tok):
    e = enc(tok)
    batch = [
        e.encode(mk()),
        e.encode(mk(ptype="noul", options=None, target=1.0)),
        e.encode(mk(ptype="extract", options=None, state="a b c", question="q", target=[2, 3])),
    ]
    b = collate(batch)
    B, L = b["input_ids"].shape
    assert B == 3 and b["attention_mask"].sum(1).tolist() == [len(x) for x in batch]
    assert b["k_max"] == 3 and b["n_opts"].tolist() == [3, 0, 0]
    assert b["choice_target"].tolist()[1:] == [-100, -100]
    assert b["noul_target"].tolist()[0] == -1.0
    # state mask covers the state tokens and the read-out token only
    for i, x in enumerate(batch):
        on = b["state_mask"][i].nonzero().flatten().tolist()
        assert on == list(range(x.state_lo, x.state_hi)) + [x.rd_pos]
