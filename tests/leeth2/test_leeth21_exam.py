"""The Leeth 2.1 exam script runs end to end on a tiny model (numbers are meaningless there; the plumbing is what is tested)."""

import importlib.util
from pathlib import Path

import pytest

from src.leeth2.encoding import Leeth2Encoder
from src.leeth2.runtime import Leeth2Runtime

from .conftest import VOCAB

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "leeth21_exam.py"


@pytest.fixture(scope="module")
def exam():
    spec = importlib.util.spec_from_file_location("leeth21_exam", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rt(tok, tiny_model):
    return Leeth2Runtime(tiny_model, Leeth2Encoder(tok, VOCAB, max_len=256), "cpu")


def test_the_tool_library_is_large_enough_for_255_distinct_options(exam):
    lib = exam.tool_library()
    assert len(lib) >= 255 and len({n for n, _, _ in lib}) == len(lib)


def test_multi_span_section_reports_both_type_groups(exam, rt):
    r = exam.multi_span(rt, quick=True)
    assert set(r) == {"train_types", "held_out_types"}
    for v in r.values():
        assert 0.0 <= v["set_f1"] <= 1.0 and v["n"] > 0 and v["by_count"]


def test_big_pools_section_covers_every_pool_size(exam, rt):
    r = exam.big_pools(rt, quick=True)
    assert set(r) == {"16", "64", "128", "255"} and all(0.0 <= v["acc"] <= 1.0 for v in r.values())


def test_safety_section_names_every_held_out_program_and_the_slipping_commands(exam, rt):
    r = exam.safety_held(rt, quick=True)
    assert {f"held:{k}" for k in exam.S5.HELD} <= set(r)
    assert "diskpart /s wipe.txt" in r["named_p_safe"] and 0.0 <= r["named_p_safe"]["diskpart /s wipe.txt"] <= 1.0
    assert r["classic"]["n_danger"] == 27


def test_long_context_section_restores_the_window(exam, rt):
    before = rt.encoder.max_len
    r = exam.long_context(rt, quick=True)
    assert rt.encoder.max_len == 4096 and before == 256
    assert set(r) == {"window_4096", "window_8192"} and "6ktok/middle" in r["window_8192"]
