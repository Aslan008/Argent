import pytest

from src.leeth2.schema import Sample


def mk(**kw):
    base = dict(task="t", family="f", lang="en", ptype="noul", state="s", question="q?", target=1.0)
    base.update(kw)
    return Sample(**base)


def test_roundtrip_json_keeps_everything_needed():
    s = mk(ptype="choice", options=["a", "b"], target=1, meta={"k": 1})
    s.validate()
    back = Sample.from_json(s.to_json())
    assert back == s


def test_defaults_are_not_serialised():
    d = mk().to_dict()
    assert "weight" not in d and "split" not in d and "options" not in d


@pytest.mark.parametrize(
    "bad",
    [
        dict(ptype="nope"),
        dict(question="  "),
        dict(ptype="choice", options=["a"], target=0),
        dict(ptype="choice", options=["a", "a"], target=0),
        dict(ptype="choice", options=["a", "b"], target=2),
        dict(ptype="choice", options=["a", "b"], target=True),
        dict(ptype="noul", target=1.5),
        dict(ptype="score", levels=1, target=1),
        dict(ptype="score", levels=5, target=6),
        dict(ptype="extract", state="abc", target=[2, 9]),
        dict(ptype="extract", state="abc", target=[1]),
    ],
)
def test_validation_rejects(bad):
    with pytest.raises(ValueError):
        mk(**bad).validate()


def test_valid_extract_with_and_without_answer():
    mk(ptype="extract", state="hello world", target=[6, 11]).validate()
    mk(ptype="extract", state="hello world", target=None).validate()


def test_score_accepts_fractional_levels():
    mk(ptype="score", levels=5, target=3.5).validate()


def test_soft_noul_target():
    mk(target=0.3).validate()
