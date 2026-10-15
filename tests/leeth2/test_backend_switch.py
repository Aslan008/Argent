"""The legacy runtime can hand every question to Leeth 2 — only when asked to, and never at the price of crashing."""

import pytest

import config
from src.leeth.primitives import ChoiceQuestion, ChoiceResult, LeethDecisionBatch, NoulQuestion, NoulResult
from src.leeth.runtime import LeethRuntime


class StubV2:
    device = "stub"

    def decide(self, state, questions):
        res = {q.id: NoulResult(True, 0.9) if isinstance(q, NoulQuestion) else ChoiceResult(q.choices[0], 0.8, {q.choices[0]: 0.8}) for q in questions}
        return LeethDecisionBatch(state_preview=state[:20], results=res, device="stub")

    def choose(self, state, question, options):
        return ChoiceResult(options[-1], 0.7, {options[-1]: 0.7})


def test_default_backend_is_legacy(monkeypatch):
    monkeypatch.delenv("LEETH_BACKEND", raising=False)
    assert config.get_leeth_backend() == "legacy"
    assert LeethRuntime()._use_v2() is None


def test_backend_can_be_stored_and_validated():
    config.set_leeth_backend("leeth2")
    assert config.get_leeth_backend() == "leeth2"
    config.set_leeth_backend("legacy")
    with pytest.raises(ValueError):
        config.set_leeth_backend("gpt")


def test_environment_overrides_the_stored_setting(monkeypatch):
    config.set_leeth_backend("legacy")
    monkeypatch.setenv("LEETH_BACKEND", "leeth2")
    assert config.get_leeth_backend() == "leeth2"
    monkeypatch.setenv("LEETH_BACKEND", "nonsense")
    assert config.get_leeth_backend() == "legacy"


def test_selected_backend_answers_every_entry_point(monkeypatch):
    monkeypatch.setenv("LEETH_BACKEND", "leeth2")
    rt = LeethRuntime()
    rt._v2 = StubV2()
    out = rt.decide_batch(["state a", "state b"], [ChoiceQuestion("tool", "which", ["x", "y"]), NoulQuestion("safe", "safe?")])
    assert [o.device for o in out] == ["stub", "stub"]
    assert out[0]["tool"].choice == "x" and out[1]["safe"].probability == 0.9
    assert rt.decide("s", [NoulQuestion("q", "?")])["q"].value is True
    assert rt.decide_options("s", "which", ["a", "b", "c"]).choice == "c"
    assert rt.decide_dopt("s", "which", ["a", "b"]).choice == "b"
    assert rt.is_available


def test_missing_weights_fall_back_to_legacy_once(monkeypatch, tmp_path):
    monkeypatch.setenv("LEETH_BACKEND", "leeth2")
    monkeypatch.setenv("LEETH2_MODEL_DIR", str(tmp_path / "nothing_here"))
    rt = LeethRuntime()
    assert rt._use_v2() is None and rt._v2 is False
    assert rt._use_v2() is None  # no retry storm: the failure is remembered
