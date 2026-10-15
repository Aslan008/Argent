"""Leeth 2.1 data: multi-answer extraction, command-safety round 5 (with held-out programs), long states and big option lists."""

import random
import re

import pytest

from src.leeth2.data.synth import extract_multi as XM
from src.leeth2.data.synth import longctx as LC
from src.leeth2.data.synth import safety5 as S5
from src.leeth2.encoding import Leeth2Encoder
from src.leeth2.schema import Sample

from .conftest import VOCAB

HELD_WORDS = re.compile(r"diskpart|wevtutil|takeown|icacls|crontab|cryptsetup|filter-branch|update-ref|net stop|sc stop|sc config", re.I)


def draw(fn, n, seed=0, **kw):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out += fn(rng, **kw) or []
    return out


# ------------------------------------------------------------------------------------------------ extract_multi
def test_multi_documents_are_labelled_exactly():
    ss = draw(XM.gen_multi, 400)
    assert ss
    zero = one = many = 0
    for s in ss:
        s.validate()
        assert s.meta["multi"] and s.ptype == "extract"
        spans = s.spans()
        zero += not spans
        one += len(spans) == 1
        many += len(spans) > 1
        texts = [s.state[a:b] for a, b in spans]
        assert all(t == t.strip() and t for t in texts)
        if s.meta["etype"] == "email":
            assert all(re.fullmatch(r"[\w.]+@[\w.\-]+", t) for t in texts)
        if s.meta["etype"] == "hash":
            assert all(re.fullmatch(r"[0-9a-f]{7,40}", t) for t in texts)
    assert zero > 20 and one > 20 and many > 150  # "nothing there", one answer and several are all taught


def test_held_out_entity_types_stay_out_of_training_and_fill_the_exam():
    train = draw(XM.gen_multi, 600, seed=1)
    assert not {s.meta["etype"] for s in train} & set(XM.HOLDOUT_TYPES)
    held = draw(XM.gen_multi_holdout, 200, seed=2)
    assert {s.meta["etype"] for s in held} == set(XM.HOLDOUT_TYPES)
    assert all(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", s.state[a:b])
               for s in held if s.meta["etype"] == "uuid" for a, b in s.spans())


def test_distractors_of_other_types_are_present_without_being_labelled():
    ss = [s for s in draw(XM.gen_multi, 600, seed=3) if s.meta["etype"] == "email" and s.spans()]
    assert ss
    for s in ss:
        assert all("@" in s.state[a:b] for a, b in s.spans())


# ------------------------------------------------------------------------------------------------ safety5
def _cmd(s):
    return s.state.split("Command: ", 1)[1].split("\nCaller:", 1)[0]


def test_training_families_never_contain_the_held_out_programs():
    ss = draw(S5.gen_safety5, 1500, seed=4)
    assert ss
    bad = [_cmd(s) for s in ss if s.ptype == "noul" and HELD_WORDS.search(_cmd(s))]
    assert not bad, bad[:5]


def test_held_out_programs_cover_dangerous_and_harmless_forms():
    ss = [s for s in draw(S5.gen_safety5_held, 800, seed=5) if s.ptype == "noul"]
    safe = [s for s in ss if s.target >= 0.9]
    dang = [s for s in ss if s.target <= 0.05]
    assert len(safe) > 100 and len(dang) > 100


def test_a_command_is_never_both_harmless_and_harmful():
    ss = [s for s in draw(S5.gen_safety5, 4000, seed=6) if s.ptype == "noul"]
    label = {}
    quoting = ("grep", "echo", "git log", "Select-String", "rg ", "man ", "cat ", "head ")
    for s in ss:
        c = _cmd(s)
        hi = s.target >= 0.9
        lo = s.target <= 0.05
        if not (hi or lo):
            continue
        if c in label and label[c] != hi and not any(w in c for w in quoting):
            pytest.fail(f"contradictory labels for {c!r}")
        label[c] = hi


def test_samples_use_the_production_state_format():
    s = next(x for x in draw(S5.gen_safety5, 20, seed=7) if x.ptype == "noul")
    assert s.state.startswith("User shell execution request:\nCommand: ") and "\nCaller: " in s.state


# ------------------------------------------------------------------------------------------------ longctx
def _route(i):
    return Sample("argent_route_v2", "argent", "ru", "choice", f'Incoming user request in Argent session:\n"прочитай файл{i}.py"\nWorking directory: /opt', "Q",
                  ["read_file", "run_command", "NO_TOOL"], 0)


_FILL = ("Quarterly numbers were steady across the regions and nothing unusual was recorded in the logs of any of the build servers this week. "
         "The committee will meet on Tuesday to discuss the parking policy and the new coffee machine.")
POOL = LC.paragraphs([(_FILL + "\n\n") * 4] * 5 + ['Incoming user request in Argent session:\n"удали всё" and a long tail of other words to pass the length filter ' * 3])


def test_filler_never_contains_a_second_request():
    assert POOL and not any(LC.AMBIGUOUS.search(p) for p in POOL)


def test_inflated_state_keeps_the_original_block_whole_and_sets_a_window():
    s = _route(1)
    for seed in range(20):
        got = LC.inflate_state(s, POOL, random.Random(seed), 3000)
        assert got is not None
        assert s.state in got.state and got.state.count("Incoming user request") == 1
        assert 2500 < len(got.state) / 3.0 < 4000
        assert got.meta["long"] and 2048 <= got.meta["max_len"] <= 8192
        assert got.options == s.options and got.target == s.target


def test_inflated_extract_spans_move_with_the_text():
    state = "alpha beta see a.py and b.py end"
    sp = [[state.index("a.py"), state.index("a.py") + 4], [state.index("b.py"), state.index("b.py") + 4]]
    s = Sample("extract_multi", "x", "en", "extract", state, "files", None, sp, meta={"multi": True})
    for seed in range(10):
        g = LC.inflate_state(s, POOL, random.Random(seed), 2000)
        g.validate()
        assert [g.state[a:b] for a, b in g.spans()] == ["a.py", "b.py"]
        assert g.meta["multi"]
    one = Sample("x", "x", "en", "extract", state, "q", None, sp[0])
    g = LC.inflate_state(one, POOL, random.Random(0), 2000)
    assert g.state[g.target[0] : g.target[1]] == "a.py" and len(g.target) == 2 and isinstance(g.target[0], int)


def test_big_pools_add_real_alternatives_and_remap_the_target():
    labels = [f"label number {i}" for i in range(120)]
    rows = []
    rng = random.Random(0)
    for i in range(200):
        opts = rng.sample(labels, 8)
        rows.append(Sample("intent_x", "intent", "en", "choice", f"text {i}", "which intent", opts, rng.randrange(8)))
    vocab = LC.option_vocab(rows)
    assert len(vocab[("intent_x", "which intent")]) == 120
    s = rows[0]
    g = LC.inflate_options(s, vocab, random.Random(1), 100)
    assert len(g.options) == 100 and g.options[: len(s.options)] == s.options and g.options[g.target] == s.options[s.target]
    assert len(set(g.options)) == 100 and g.meta["big"] == 100 and g.meta["max_len"] >= 4096
    soft = Sample("intent_x", "intent", "en", "choice", "t", "which intent", s.options, [0.5, 0.5] + [0.0] * 6)
    gs = LC.inflate_options(soft, vocab, random.Random(1), 64)
    assert len(gs.target) == len(gs.options) and abs(sum(gs.target) - 1.0) < 1e-6
    gs.validate()


def test_a_task_with_one_fixed_option_list_is_left_alone():
    rows = [_route(i) for i in range(50)]
    assert LC.inflate_options(rows[0], LC.option_vocab(rows), random.Random(0), 200) is None


def test_inflate_rows_makes_both_kinds_and_counts_them():
    rows = []
    labels = [f"label {j}" for j in range(90)]
    for i in range(300):
        rows.append(_route(i).to_json())
        r = random.Random(i)
        rows.append(Sample("intent_y", "intent", "en", "choice", _FILL * 3, "which", r.sample(labels, 6), 0).to_json())
    out, counts = LC.inflate_rows(rows, n_long=40, n_big=30, seed=3)
    assert counts["long"] == 40 and counts["big"] == 30 and len(out) == 70
    for line in out:
        Sample.from_json(line).validate()


def test_options_follow_the_rows_own_window_in_training_and_the_runtime_window_at_inference(tok):
    e = Leeth2Encoder(tok, VOCAB, max_len=640)
    opts = [f"option number {i} with some extra words" for i in range(100)]
    base = Sample("t", "f", "en", "choice", "a state", "which", opts, 3)
    wide = Sample("t", "f", "en", "choice", "a state", "which", opts, 3, meta={"max_len": 4096})
    n_base = len(e.encode(base, rng=random.Random(0)))
    n_wide = len(e.encode(wide, rng=random.Random(0)))
    assert n_wide > n_base  # in the wide window every option keeps more of its tokens
    assert len(e.encode(wide)) == len(e.encode(base))  # at inference neither row's meta matters: the runtime's own window decides
