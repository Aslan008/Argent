import pytest

from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion, ScoreQuestion
from src.leeth2.encoding import Leeth2Encoder
from src.leeth2.runtime import Leeth2Runtime

from .conftest import VOCAB


@pytest.fixture
def rt(tok, tiny_model):
    return Leeth2Runtime(tiny_model, Leeth2Encoder(tok, VOCAB, max_len=128), "cpu")


def test_decide_answers_all_four_primitives_in_one_call(rt):
    state = "Incoming request: please open the file report.pdf from the shared folder"
    qs = [
        ChoiceQuestion("tool", "which tool", ["read_file", "run_command", "web_search"]),
        NoulQuestion("safe", "is it safe"),
        ScoreQuestion("urg", "how urgent", levels=5),
        ExtractQuestion("what", "which file"),
    ]
    b = rt.decide(state, qs)
    assert len(b) == 4 and b.device == "cpu" and b.elapsed_ms > 0
    c = b["tool"]
    assert c.choice in ("read_file", "run_command", "web_search") and sum(c.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert c.confidence == pytest.approx(max(c.probabilities.values()), abs=1e-5)
    assert 0.0 <= b["safe"].probability <= 1.0 and b["safe"].value == (b["safe"].probability >= 0.5)
    s = b["urg"]
    assert 1 <= s.level <= 5 and set(s.probabilities) == {"1", "2", "3", "4", "5"} and 0 <= s.score <= 100
    e = b["what"]
    a, z = e.span
    assert state[a:z] == e.text or e.text == ""


def test_a_long_option_list_goes_through_the_retriever_first(rt):
    """Two stages, measured 2026-11-04: with the right option among 96 the same model answers 0.762 at 128-option lists
    against 0.569 in one pass, because every option keeps ~15 tokens of its description otherwise. Off unless installed."""
    options = [f"tool_{i}" for i in range(200)]
    seen = {}

    def retriever(state, question, opts, k):
        seen.update(k=k, n=len(opts))
        return opts[:k]

    rt.choose("a request", "which tool", options[:10])
    assert seen == {}                                        # a short list is asked as it is

    rt.set_shortlist(retriever, above=64, k=96)
    res = rt.choose("a request", "which tool", options)
    assert seen == {"k": 96, "n": 200}
    assert res.choice in options[:96] and set(res.probabilities) == set(options[:96])


def test_a_retriever_cannot_add_an_option_that_was_not_there(rt):
    options = [f"tool_{i}" for i in range(100)]
    rt.set_shortlist(lambda s, q, o, k: ["invented_tool"] + list(o)[:5], above=8, k=6)
    res = rt.choose("a request", "which tool", options)
    assert "invented_tool" not in res.probabilities and res.choice in options


def test_rubric_labels_are_returned(rt):
    s = rt.decide("some text", [ScoreQuestion("q", "rate it", levels=3, rubric=["bad", "ok", "good"])])["q"]
    assert s.label in ("bad", "ok", "good")


def test_convenience_helpers(rt):
    r = rt.choose("two cats and a dog", "which animal", ["cat", "dog"])
    assert r.choice in ("cat", "dog")
    assert 0.0 <= rt.judge("it rains", "is it wet").probability <= 1.0


def test_unknown_question_type_is_rejected(rt):
    class Weird:
        id = "w"
        question = "?"

    with pytest.raises(TypeError):
        rt.decide("x", [Weird()])
