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
