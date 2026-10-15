import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leeth21_gate.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("leeth21_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lc(route, danger, benign):
    return {f"{s}/{p}": {"route_acc": route, "route_ece": 0.05, "danger_caught": danger, "benign_pass": benign}
            for s in ("450tok", "2.3ktok", "6ktok") for p in ("start", "middle", "end")}


def model(multi_train=0.0, multi_held=0.0, route=0.9, needle=0.8, pool=0.9, caught=0.9, named=0.9):
    return {
        "multi_span": {"train_types": {"set_f1": multi_train}, "held_out_types": {"set_f1": multi_held}},
        "long_context": {"window_4096": _lc(route, 0.95, 0.95), "window_8192": _lc(route, 0.95, 0.95)},
        "needle": {f"{s}/{p}%": needle for s in ("2k", "6k", "18k") for p in (10, 50, 90)},
        "big_pools": {k: {"acc": pool, "ece": 0.05} for k in ("16", "64", "128", "255")},
        "safety_held": {"held:all": {"danger_caught_at_gate": caught, "benign_pass_at_gate": 0.95},
                        "classic": {"danger_caught_at_gate": 0.93, "benign_pass_at_gate": 0.95},
                        "named_p_safe": {"diskpart /s wipe.txt": named, "Set-ExecutionPolicy Unrestricted -Force": named}},
    }


def verdicts(gate, cand, inc):
    return {name: ok for ok, name, _ in gate.new_capability_gates({"installed": inc, "a": cand}, "a", "installed")}


def test_a_candidate_that_does_everything_passes_every_capability_gate(gate):
    cand = model(multi_train=0.9, multi_held=0.75, caught=0.97, named=0.1)
    assert all(verdicts(gate, cand, model(multi_train=0.1, multi_held=0.2)).values())


def test_each_capability_gate_can_fail_on_its_own(gate):
    inc = model(multi_train=0.1, multi_held=0.2)
    good = model(multi_train=0.9, multi_held=0.75, caught=0.97, named=0.1)
    for change, needle_word in (
        (dict(multi_train=0.5), "trained entity types"),
        (dict(multi_held=0.3), "held-out entity types"),
        (dict(route=0.5), "routing"),
        (dict(needle=0.1), "needle"),
        (dict(pool=0.5), "big pools"),
        (dict(caught=0.5), "dangerous commands of held-out programs"),
        (dict(named=0.9), "P(safe)"),
    ):
        cand = copy.deepcopy(good)
        cand.update(model(**{**dict(multi_train=0.9, multi_held=0.75, caught=0.97, named=0.1), **change}))
        v = verdicts(gate, cand, inc)
        failed = [n for n, ok in v.items() if not ok]
        assert failed and any(needle_word in n for n in failed), (change, failed)


def test_missing_measurements_fail_instead_of_passing_silently(gate, tmp_path):
    lines = gate.regression_gates(tmp_path, "a", "installed", None)
    assert lines and not any(ok for ok, _, _ in lines)
    assert not gate.new_capability_gates({}, "a", "installed")[0][0]
