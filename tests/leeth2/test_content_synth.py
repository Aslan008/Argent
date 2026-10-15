"""The content-question bank (src/leeth2/data/synth/content.py): well-formed, balanced, and the held-out topics never leak into training."""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.leeth2.data.synth import content as C  # noqa: E402


def test_every_topic_has_every_tone_with_two_texts():
    for bank in (C.TEXTS_TRAIN, C.TEXTS_HELD):
        for topic, by_tone in bank.items():
            assert set(by_tone) == set(C.TONES), topic
            for tone, items in by_tone.items():
                assert len(items) == 2 and all(len(t) > 25 for t in items), (topic, tone)
    assert set(C.TEXTS_TRAIN) == set(C.TOPICS_TRAIN) and set(C.TEXTS_HELD) == set(C.TOPICS_HELD)


def test_held_out_topics_and_texts_are_disjoint_from_training():
    assert not set(C.TOPICS_TRAIN) & set(C.TOPICS_HELD)
    train_texts = {t for d in C.TEXTS_TRAIN.values() for v in d.values() for t in v}
    held_texts = {t for d in C.TEXTS_HELD.values() for v in d.values() for t in v}
    assert not train_texts & held_texts
    train_names = {n["acc"] for n in C.TOPICS_TRAIN.values()} | {n["en"] for n in C.TOPICS_TRAIN.values()}
    assert not train_names & ({n["acc"] for n in C.TOPICS_HELD.values()} | {n["en"] for n in C.TOPICS_HELD.values()})


def test_samples_are_valid_yes_no_questions_about_the_text():
    ss = C.build("train", seed=3)
    assert len(ss) > 3000
    for s in ss[:400]:
        s.validate()
        assert s.ptype == "noul" and s.target in (0.0, 1.0) and s.state and s.question.endswith("?")
    yes = sum(s.target > 0.5 for s in ss) / len(ss)
    assert 0.25 < yes < 0.45  # a feed holds more items that do not match than items that do
    kinds = Counter((s.meta["kind"], s.target) for s in ss)
    for kind in ("topic", "tone", "compound"):
        assert kinds[(kind, 1.0)] > 100 and kinds[(kind, 0.0)] > 100


def test_a_compound_question_is_true_only_when_topic_and_tone_both_match():
    names = C.TOPICS_TRAIN["weather"]
    crit = C.criteria("weather", "complaint", C.TOPICS_TRAIN)
    by_q = {q: truth for kind, q, truth, lang, qt, qu in crit if kind == "compound"}
    assert by_q["Это жалоба на погоду?"] is True
    assert by_q["Это положительный отзыв про погоду?"] is False  # same topic, other tone
    assert by_q["Это жалоба на музыку?"] is False  # same tone, other topic
    assert names["acc"] == "погоду"


def test_hard_negatives_of_the_compound_kind_are_drawn():
    ss = C.build("train", seed=5)
    hard = [s for s in ss if s.meta["kind"] == "compound" and s.target == 0.0]
    assert len(hard) > 800
    # at least some share the text's topic (another tone) -- the instructive ones
    same_topic = 0
    names = {t: n for t, n in C.TOPICS_TRAIN.items()}
    for s in hard[:600]:
        n = names[s.meta["topic"]]
        if n["acc"] in s.question or n["en"] in s.question:
            same_topic += 1
    assert same_topic > 60
