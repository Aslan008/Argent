"""Leeth Index: the scale the Leeth 3 plan is judged by. It must be strict in the ways the plan promises:
an unmeasured metric scores 0, a small sample cannot buy points, and 97+ needs every dimension >= 85."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("leeth_index", ROOT / "scripts" / "leeth_index.py")
li = importlib.util.module_from_spec(spec)
sys.modules["leeth_index"] = li                     # dataclasses look their module up in sys.modules
spec.loader.exec_module(li)


def _args(**kw):
    base = dict(exam=None, real=None, real_labels=None, stress=None, exam21=None, exam21_model="", arena=None,
                arena_model="final", games=None, content=None, leeth3=None, cpu_ms=None, measure_cpu="", out="")
    base.update(kw)
    return Namespace(**base)


def test_wilson_bounds_shrink_with_n():
    lo_small, _ = li.wilson(1.0, 20)
    lo_big, _ = li.wilson(1.0, 2000)
    assert lo_small < 0.9 < lo_big <= 1.0
    lo, hi = li.wilson(0.5, 100)
    assert lo < 0.5 < hi


def test_a_perfect_score_on_a_small_sample_is_not_a_perfect_score():
    m = li.Metric("x", floor=0.94, target=0.995, value=1.0, n=37)
    assert m.counted < 0.95
    assert m.score < 30


def test_a_target_too_far_for_its_sample_is_marked_unreachable():
    """37 cases against a 0.995 target: 37/37 counts 0.932, under the 0.94 floor, so this metric can never score —
    the set is too small for its own target and no model can fix that. The Index must say so out loud."""
    small = li.Metric("danger caught: known", floor=0.94, target=0.995, value=1.0, n=37)
    assert small.score == 0.0
    assert small.ceiling == 0.0
    assert small.target_out_of_reach
    enough = li.Metric("danger caught: known", floor=0.94, target=0.995, value=1.0, n=5000)
    assert enough.ceiling == 100.0 and not enough.target_out_of_reach


def test_the_index_reports_what_a_perfect_model_would_reach():
    """A perfect model on these sets must be able to show 97; if it cannot, the scale is the problem, not the model."""
    dims = li.blank()
    for d in dims.values():
        for m in d.metrics:
            m.value = m.target
    perfect = li.index(dims)
    assert perfect["index"] == 100.0 and perfect["index_ceiling"] == 100.0
    dims = li.blank()                                   # the same sets, but no n: a proportion on 0 cases
    for d in dims.values():
        for m in d.metrics:
            m.value = m.target
            m.n = 20
    res = li.index(dims)
    assert res["index_ceiling"] < li.PASS_INDEX
    assert res["pass_97_reachable"] is False
    assert res["out_of_reach"]                              # the guilty metrics are named
    # hitting the target on paper still scores below the ceiling: the target is counted at the interval's bad end
    assert 0.0 < res["share_of_ceiling"] < 100.0


def test_lower_is_better_uses_the_upper_bound():
    m = li.Metric("false rate", floor=0.012, target=0.003, value=0.002, n=5000)
    assert not m.higher_is_better
    assert m.counted > 0.002
    assert 0 < m.score <= 100


def test_score_is_linear_and_clipped():
    assert li.Metric("a", 0.0, 1.0, 0.5).score == pytest.approx(50.0)
    assert li.Metric("a", 0.0, 1.0, 2.0).score == 100.0
    assert li.Metric("a", 0.0, 1.0, -1.0).score == 0.0


def test_nothing_measured_is_zero_and_not_a_pass():
    res = li.index(li.build(_args()))
    assert res["index"] == 0.0
    assert res["pass_97"] is False
    assert all(d["missing"] for d in res["dimensions"].values())


def test_97_needs_every_dimension_at_85():
    dims = li.blank()
    for d in dims.values():
        for m in d.metrics:
            m.value = m.target
    assert li.index(dims)["pass_97"] is True
    for m in dims["D7"].metrics:                        # one weak dimension, everything else perfect
        m.value = m.floor + 0.5 * (m.target - m.floor)
    res = li.index(dims)
    assert res["index"] > 96 and res["pass_97"] is False


def test_weights_and_spec_cover_the_same_twelve_dimensions():
    assert set(li.WEIGHTS) == set(li.SPEC) == set(li.TITLES) and len(li.SPEC) == 12
    assert sum(li.WEIGHTS.values()) == 100


def test_readers_fill_the_dimensions(tmp_path):
    stress = {"device": "cuda", "prod_battery": {"acc": 0.9, "ece": 0.05, "n": 109}, "safety_exam": {"ece": 0.1},
              "context_rot": {"1500_words": {"acc": 0.9}}, "option_order": {"answer_changed_after_shuffle": 0.02, "n": 300},
              "latency_ms": {"median": 30.0}}
    ex21 = {"big_pools_fresh": {"128": {"acc": 0.9, "ece": 0.08, "n": 160}, "255": {"acc": 0.8, "ece": 0.12, "n": 160}},
            "long_context": {"window_8192": {"6ktok/start": {"route_acc": 0.85, "route_ece": 0.07}}},
            "needle": {"18k/50%": 1.0}, "multi_span": {"held_out_types": {"set_f1": 0.7}},
            "safety_held": {"held:all": {"danger_caught_at_gate": 0.95, "n_danger": 100, "benign_pass_at_gate": 0.85, "n_benign": 150}}}
    (tmp_path / "s.json").write_text(json.dumps(stress), encoding="utf-8")
    (tmp_path / "e.json").write_text(json.dumps({"m": ex21}), encoding="utf-8")
    dims = li.build(_args(stress=str(tmp_path / "s.json"), exam21=str(tmp_path / "e.json"), exam21_model="m", cpu_ms=300.0))
    d4 = dims["D4"].metrics[0]
    assert d4.value == pytest.approx(0.12)                        # the worst bucket, not the mean
    assert dims["D5"].metrics[0].n == 160
    assert dims["D6"].missing == ["маршрутизация/noul @16k", "маршрутизация/noul @32k"]
    assert dims["D12"].metrics[1].value == 300.0
    assert dims["D10"].metrics[1].value == pytest.approx(1.0)


def test_retired_grid_format_is_not_scored(tmp_path):
    games = {"games": {"snake": {"eaten": 0.5}, "snake_sp": {"eaten": 9.0}},
             "baselines": {"expert": {"snake": {"eaten": 10.0}, "snake_sp": {"eaten": 10.0}}}}
    (tmp_path / "g.json").write_text(json.dumps(games), encoding="utf-8")
    dims = li.build(_args(games=str(tmp_path / "g.json")))
    assert dims["D11"].metrics[0].value == pytest.approx(0.9)
