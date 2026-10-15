"""Leeth 2.1: several answers per extract question, the Brier term on option lists, windows over long states, gated choice."""

import random

import numpy as np
import pytest
import torch

from src.leeth.primitives import ChoiceResult, LeethDecisionBatch, NoulResult
from src.leeth2 import evaluate as E
from src.leeth2.encoding import IGNORE, Encoded, Leeth2Encoder, collate
from src.leeth2.losses import choice_brier, compute_losses, span_loss
from src.leeth2.runtime import Leeth2Runtime
from src.leeth2.schema import Sample, span_list

from .conftest import VOCAB

STATE = "see a.py then b.py and finally c.py ok"


def mk(**kw):
    base = dict(task="t", family="f", lang="en", ptype="extract", state=STATE, question="all python files", target=None)
    base.update(kw)
    return Sample(**base)


def span_of(text, sub):
    i = text.index(sub)
    return [i, i + len(sub)]


# ------------------------------------------------------------------------------------------ schema
def test_span_list_accepts_every_form():
    assert span_list(None) == []
    assert span_list([3, 5]) == [(3, 5)]
    assert span_list([[3, 5], [8, 9]]) == [(3, 5), (8, 9)]
    assert span_list([[3, 5]]) == [(3, 5)]
    with pytest.raises(ValueError):
        span_list([1, 2, 3])


def test_multi_span_target_is_validated():
    spans = [span_of(STATE, "a.py"), span_of(STATE, "b.py"), span_of(STATE, "c.py")]
    s = mk(target=spans, meta={"multi": True})
    s.validate()
    assert s.spans() == [tuple(x) for x in spans]
    for bad in ([spans[1], spans[0]], [[0, 5], [3, 9]], [[0, 5], [5, 999]], []):
        with pytest.raises(ValueError):
            mk(target=bad).validate()


# ------------------------------------------------------------------------------------------ encoding
def test_encoder_gives_every_answer_a_start_and_an_end(tok):
    e = Leeth2Encoder(tok, VOCAB, max_len=128)
    spans = [span_of(STATE, "a.py"), span_of(STATE, "b.py"), span_of(STATE, "c.py")]
    x = e.encode(mk(target=spans), want_offsets=True)
    assert len(x.start_set) == len(x.end_set) == 3
    assert x.start_target == x.start_set[0] and x.end_target == x.end_set[0]
    for a, b in zip(x.start_set, x.end_set):
        lo = a - x.state_lo
        hi = b - x.state_lo
        c0, c1 = x.state_offsets[lo][0], x.state_offsets[hi][1]
        assert STATE[c0:c1].strip() in ("a.py", "b.py", "c.py")


def test_single_answer_has_no_set_and_collate_pads_to_the_widest(tok):
    e = Leeth2Encoder(tok, VOCAB, max_len=128)
    one = e.encode(mk(target=span_of(STATE, "b.py")), want_offsets=True)
    none = e.encode(mk(target=None), want_offsets=True)
    three = e.encode(mk(target=[span_of(STATE, "a.py"), span_of(STATE, "b.py"), span_of(STATE, "c.py")]), want_offsets=True)
    assert one.start_set is None and none.start_set is None
    b = collate([one, none, three])
    assert b["start_set"].shape == (3, 3)
    assert b["start_set"][0].tolist() == [one.start_target, IGNORE, IGNORE]
    assert b["start_set"][1, 0].item() == none.rd_pos  # "no answer" points at the read-out token
    assert (b["start_set"][2] != IGNORE).all()
    assert b["start_target"].tolist()[0] == one.start_target  # the single-answer fields are unchanged


def test_answers_cut_off_by_a_window_are_dropped_not_misplaced(tok):
    words = [f"w{i}" for i in range(400)]
    words[10], words[390] = "alpha.py", "omega.py"
    state = " ".join(words)
    spans = [span_of(state, "alpha.py"), span_of(state, "omega.py")]
    e = Leeth2Encoder(tok, VOCAB, max_len=64)
    x = e.encode(mk(state=state, target=spans, question="files"), rng=random.Random(0), want_offsets=True)
    assert len(x) <= 64 + 8
    # one window cannot hold both: what is left is exactly one right answer (or the other), never a position outside the state
    kept = x.start_set if x.start_set else [x.start_target]
    assert all(x.state_lo <= p < x.state_hi for p in kept)


# ------------------------------------------------------------------------------------------ losses
def test_single_answer_span_loss_is_unchanged_by_the_set_arguments():
    torch.manual_seed(0)
    s, e = torch.randn(2, 9), torch.randn(2, 9)
    st, et = torch.tensor([2, 5]), torch.tensor([3, 7])
    plain = span_loss(s, e, st, et)
    sets = span_loss(s, e, st, et, torch.tensor([[2, IGNORE], [5, IGNORE]]), torch.tensor([[3, IGNORE], [7, IGNORE]]))
    assert torch.allclose(plain, sets)


def test_set_loss_is_minimised_by_spreading_mass_over_every_answer():
    st, et = torch.tensor([1]), torch.tensor([2])
    sset, eset = torch.tensor([[1, 5]]), torch.tensor([[2, 6]])

    def logits(peaks):
        z = torch.full((1, 8), -20.0)
        for p, w in peaks:
            z[0, p] = w
        return z

    two = span_loss(logits([(1, 0.0), (5, 0.0)]), logits([(2, 0.0), (6, 0.0)]), st, et, sset, eset)
    one = span_loss(logits([(1, 5.0), (5, 0.0)]), logits([(2, 5.0), (6, 0.0)]), st, et, sset, eset)
    assert two.item() < one.item()  # both answers at half the mass beats one answer at nearly all of it


def test_choice_brier_values_and_use_in_compute_losses():
    uniform = torch.zeros(1, 4)
    assert choice_brier(uniform, torch.tensor([2])).item() == pytest.approx(0.75)  # (0.25-1)^2 + 3 * 0.25^2
    sure = torch.tensor([[0.0, -30.0, -30.0]])
    assert choice_brier(sure, torch.tensor([0])).item() == pytest.approx(0.0, abs=1e-6)
    assert choice_brier(sure, torch.tensor([IGNORE])).item() == 0.0
    out = {"choice_logits": uniform, "noul_logit": torch.zeros(1), "score_logits": torch.zeros(1, 10),
           "start_logits": torch.zeros(1, 4), "end_logits": torch.zeros(1, 4)}
    batch = {"weight": torch.ones(1), "choice_target": torch.tensor([2]), "noul_target": torch.tensor([-1.0]), "score_pos": torch.tensor([-1.0]),
             "levels": torch.tensor([0]), "start_target": torch.tensor([IGNORE]), "end_target": torch.tensor([IGNORE])}
    _, off = compute_losses(out, batch, {"choice": 1.0, "noul": 1.0, "score": 0.7, "extract": 1.0, "choice_brier": 0.0})
    _, on = compute_losses(out, batch, {"choice": 1.0, "noul": 1.0, "score": 0.7, "extract": 1.0, "choice_brier": 0.5})
    assert "loss_choice_brier" not in off and on["loss_choice_brier"] == pytest.approx(0.75)


# ------------------------------------------------------------------------------------------ decoding
def _enc(n_state=12):
    offs = [(i * 6, i * 6 + 5) for i in range(n_state)]  # "tok00 tok01 ..."
    return Encoded(input_ids=[0] * (n_state + 8), opt_index=[0] * (n_state + 8), rd_pos=n_state + 3, state_lo=1, state_hi=1 + n_state, n_opts=0, ptype=3,
                   levels=0, choice_target=IGNORE, noul_target=-1.0, score_pos=-1.0, start_target=0, end_target=0, perm=[], state_offsets=offs)


def _logits(L, peaks, null=-5.0, rd=None):
    z = np.full(L, -30.0)
    for pos, v in peaks:
        z[pos] = v
    z[rd] = null
    return z


def test_decode_spans_reads_every_peak_in_text_order():
    enc = _enc()
    state = " ".join(f"tok{i:02d}" for i in range(12)) + " "
    L = enc.state_hi + 8
    starts = _logits(L, [(1 + 1, 3.0), (1 + 5, 3.0), (1 + 9, 2.8)], rd=enc.rd_pos)
    ends = _logits(L, [(1 + 2, 3.0), (1 + 6, 3.0), (1 + 9, 2.8)], rd=enc.rd_pos)
    got = E._decode_spans(starts, ends, enc, state)
    assert [o["text"] for o in got] == ["tok01 tok02", "tok05 tok06", "tok09"]
    assert all(0.0 < o["confidence"] <= 1.0 for o in got)
    assert all(a < b for (a, _), (b, _) in zip([o["span"] for o in got], [o["span"] for o in got][1:]))


def test_decode_spans_says_nothing_when_the_read_out_token_wins():
    enc = _enc()
    L = enc.state_hi + 8
    starts = _logits(L, [(3, 1.0)], null=4.0, rd=enc.rd_pos)
    ends = _logits(L, [(3, 1.0)], null=4.0, rd=enc.rd_pos)
    assert E._decode_spans(starts, ends, enc, "x " * 40) == []


def test_vectorised_best_span_matches_the_loop_it_replaced():
    rng = np.random.default_rng(3)
    for n in (1, 2, 7, 50, 300):
        s, e = rng.normal(size=n), rng.normal(size=n)
        best, bi, bj = -1e30, 0, 0
        for i in range(n):
            j = i + int(np.argmax(e[i : min(n, i + E.MAX_SPAN_TOKENS)]))
            v = float(s[i] + e[j])
            if v > best:
                best, bi, bj = v, i, j
        jb = E._best_ends(e, E.MAX_SPAN_TOKENS)
        v = s + e[jb]
        assert int(v.argmax()) == bi and int(jb[bi]) == bj


# ------------------------------------------------------------------------------------------ runtime
@pytest.fixture
def rt(tok, tiny_model):
    return Leeth2Runtime(tiny_model, Leeth2Encoder(tok, VOCAB, max_len=96), "cpu")


def test_long_state_is_cut_into_overlapping_windows_that_cover_it(rt):
    state = " ".join(f"w{i}" for i in range(600))
    wins = rt._state_windows(state)
    assert wins and wins[0][0] == 0 and wins[-1][1] == len(state)
    assert all(b > a for a, b in wins)
    assert all(wins[k + 1][0] < wins[k][1] for k in range(len(wins) - 1))  # overlap: an answer on a boundary is whole in one window
    assert rt._state_windows("a short state") is None


def test_extract_on_a_long_document_returns_offsets_into_the_whole_document(rt):
    from src.leeth.primitives import ExtractQuestion

    state = " ".join(f"w{i}" for i in range(600))
    r = rt.decide(state, [ExtractQuestion("q", "which word")])["q"]
    a, b = r.span
    assert r.text == "" or state[a:b] == r.text


def test_extract_all_is_sorted_and_free_of_overlaps(rt):
    state = " ".join(f"w{i}" for i in range(600))
    got = rt.extract_all(state, "all words")
    spans = [r.span for r in got]
    assert spans == sorted(spans)
    assert all(spans[k][1] <= spans[k + 1][0] for k in range(len(spans) - 1))
    assert all(state[a:b] == r.text for (a, b), r in zip(spans, got))


class _Stub(Leeth2Runtime):
    """A runtime whose model is replaced by fixed numbers, to test what choose_gated does with them."""

    def __init__(self, probs, gates):
        self.probs, self.gates, self.calls = probs, gates, []

    def decide(self, state, questions):
        self.calls.append([q.id for q in questions])
        res = {}
        for q in questions:
            if q.id == "c":
                res["c"] = ChoiceResult(choice=max(self.probs, key=self.probs.get), confidence=max(self.probs.values()), probabilities=dict(self.probs))
            else:
                opt = q.question.split("|")[1]
                p = self.gates[opt]
                res[q.id] = NoulResult(value=p >= 0.5, probability=p)
        return LeethDecisionBatch(state_preview="", results=res, elapsed_ms=0.0, device="cpu")


def test_choose_gated_skips_a_blocked_favourite_and_renormalises():
    st = _Stub({"delete": 0.6, "move": 0.3, "list": 0.1}, {"delete": 0.1, "move": 0.9, "list": 0.9})
    g = st.choose_gated("state", "which?", ["delete", "move", "list"], "ok|{option}|?")
    assert g.unconstrained == "delete" and g.choice == "move" and g.blocked == ["delete"]
    assert g.confidence == pytest.approx(0.3 / 0.4)
    assert len(st.calls) == 1  # few options: the choice and every gate go through one batch


def test_choose_gated_answers_none_when_every_checked_option_is_blocked():
    probs = {f"o{i}": 1 / 6 for i in range(6)}
    st = _Stub(probs, {o: 0.2 for o in probs})
    g = st.choose_gated("s", "which?", list(probs), lambda o: f"ok|{o}|?", top_k=3)
    assert g.choice is None and g.confidence == 0.0 and len(g.blocked) == 3
    assert len(st.calls) == 2  # many options: the choice first, then the gates of the top_k only


def test_punctuation_glued_to_the_last_token_is_not_part_of_a_listed_answer():
    from src.leeth2.encoding import Encoded

    state = "see foo(). then bar(), ok"
    offs = [(0, 3), (4, 10), (11, 15), (16, 22), (23, 25)]  # "see", "foo().", "then", "bar(),", "ok"
    enc = Encoded(input_ids=[0] * 12, opt_index=[0] * 12, rd_pos=8, state_lo=1, state_hi=6, n_opts=0, ptype=3, levels=0, choice_target=IGNORE,
                  noul_target=-1.0, score_pos=-1.0, start_target=0, end_target=0, perm=[], state_offsets=offs)
    z = np.full(12, -30.0)
    st, en = z.copy(), z.copy()
    st[8] = en[8] = -5.0
    st[2] = st[4] = en[2] = en[4] = 3.0
    got = E._decode_spans(st, en, enc, state)
    assert [o["text"] for o in got] == ["foo()", "bar()"]
    assert [state[a:b] for a, b in (o["span"] for o in got)] == ["foo()", "bar()"]


def test_long_option_lists_use_their_own_temperature_when_one_was_fitted(tok, tiny_model):
    from src.leeth2.evaluate import predict

    opts = [f"option number {i}" for i in range(70)]
    ss = [Sample("t", "f", "en", "choice", "a state", "which", opts, 3)]
    enc = Leeth2Encoder(tok, VOCAB, max_len=1024)
    tiny_model.cfg.temp_choice = 1.0
    tiny_model.cfg.temp_choice_big = 0.0
    base = predict(tiny_model, enc, ss)[0].probs
    tiny_model.cfg.temp_choice_big = 4.0
    flat = predict(tiny_model, enc, ss)[0].probs
    short = predict(tiny_model, enc, [Sample("t", "f", "en", "choice", "a state", "which", opts[:5], 3)])[0].probs
    tiny_model.cfg.temp_choice_big = 0.0
    assert flat.max() < base.max() and abs(flat.sum() - 1.0) < 1e-6
    assert abs(short.sum() - 1.0) < 1e-6  # a short list is untouched by the long-list temperature


def test_fit_temperatures_separates_long_lists_when_there_are_enough_of_them():
    import numpy as np

    from src.leeth2 import calibrate
    from src.leeth2.evaluate import Prediction

    rng = np.random.default_rng(0)
    samples, preds = [], []
    for i in range(160):
        k = 100 if i % 2 else 6
        lg = rng.normal(size=k) * (0.5 if k == 100 else 2.0)
        lg[0] += 2.0
        samples.append(Sample("t", "f", "en", "choice", "s", "q", [f"o{j}" for j in range(k)], 0))
        preds.append(Prediction("choice", probs=None, raw={"logits_shown": lg, "perm": list(range(k))}))
    t = calibrate.fit_temperatures(samples, preds)
    assert t["n_choice_big"] == 80 and t["n_choice"] == 80 and "temp_choice_big" in t and t["temp_choice_big"] != t["temp_choice"]
