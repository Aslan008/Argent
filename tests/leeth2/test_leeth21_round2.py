"""Leeth 2.1, round 2: more entity types, ordinary commands, tool libraries of up to 255 entries, the temperature for long lists."""

import importlib.util
import random
import re
from pathlib import Path

import pytest

from src.leeth2.data.synth import bigtools as BT
from src.leeth2.data.synth import extract_multi as XM
from src.leeth2.data.synth import safety5 as S5
from src.leeth2.encoding import Leeth2Encoder
from src.leeth2.runtime import Leeth2Runtime

from .conftest import VOCAB

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
HELD_WORDS = re.compile(r"diskpart|wevtutil|takeown|icacls|crontab|cryptsetup|filter-branch|update-ref|net stop|sc stop|sc config", re.I)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw(fn, n, seed=0, **kw):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out += fn(rng, **kw) or []
    return out


def _cmd(s):
    return s.state.split("Command: ", 1)[1].split("\nCaller:", 1)[0]


# ------------------------------------------------------------------------------------------------ entity types
def test_there_are_many_more_training_types_and_the_three_held_out_ones_stay_out():
    assert len(XM.TRAIN_TYPES) >= 30
    assert set(XM.HOLDOUT_TYPES) == {"uuid", "mac", "color"} and not set(XM.HOLDOUT_TYPES) & set(XM.TRAIN_TYPES)


def test_every_entity_generator_makes_clean_distinct_values():
    rng = random.Random(0)
    for k, (gen, *_names) in XM.TYPES.items():
        vals = [gen(rng) for _ in range(60)]
        assert all(v and v == v.strip() and "\n" not in v for v in vals), k
        assert len(set(vals)) >= 8, k


def test_documents_of_every_training_type_are_labelled_exactly():
    seen = set()
    for s in draw(XM.gen_multi, 4000, seed=11):
        s.validate()
        seen.add(s.meta["etype"])
        for a, b in s.spans():
            t = s.state[a:b]
            assert t and t == t.strip()
    assert seen == set(XM.TRAIN_TYPES)


# ------------------------------------------------------------------------------------------------ ordinary commands
def test_ordinary_commands_are_mostly_harmless_and_never_use_the_held_out_programs():
    ss = [s for s in draw(S5.gen_ordinary, 3000, seed=12) if s.ptype == "noul"]
    assert len(ss) > 2500
    assert not [_cmd(s) for s in ss if HELD_WORDS.search(_cmd(s))]
    assert sum(s.target >= 0.9 for s in ss) / len(ss) > 0.85


def test_no_command_is_both_harmless_and_harmful_across_the_whole_round_two_mix():
    quoting = ("grep", "echo", "git log", "Select-String", "rg ", "man ", "cat ", "head ")
    label = {}
    rows = [s for s in draw(S5.gen_safety5, 4000, seed=13) + draw(S5.gen_ordinary, 4000, seed=14) if s.ptype == "noul"]
    for s in rows:
        c = _cmd(s)
        hi, lo = s.target >= 0.9, s.target <= 0.05
        if not (hi or lo):
            continue
        if c in label and label[c] != hi and not any(w in c for w in quoting):
            pytest.fail(f"contradictory labels for {c!r}")
        label[c] = hi


def test_build_output_cleanups_are_not_called_dangerous():
    ss = [s for s in draw(S5.gen_ordinary, 6000, seed=15) if s.ptype == "noul" and re.match(r"(rm |rd /s|Remove-Item)", _cmd(s))]
    assert ss
    assert all(s.target > 0.05 for s in ss)


# ------------------------------------------------------------------------------------------------ bigtools
def test_the_tool_library_is_large_and_disjoint_from_the_exams_vocabulary():
    exam = _load("leeth21_exam")
    assert len(BT._LIB) >= 600 and len({n for n, _ in BT._LIB}) == len(BT._LIB)
    assert not set(BT.VERBS) & set(exam.VERBS)
    assert not set(BT.OBJECTS) & set(exam.OBJECTS)
    cal = _load("leeth21_calibrate")
    assert not set(cal.VERBS) & (set(BT.VERBS) | set(exam.VERBS))
    assert not set(cal.OBJECTS) & (set(BT.OBJECTS) | set(exam.OBJECTS))


def test_bigtools_rows_are_valid_have_the_answer_and_use_big_windows_for_big_lists():
    ss = draw(BT.gen_bigtools, 400, seed=16)
    assert len(ss) > 350
    sizes = []
    for s in ss:
        s.validate()
        sizes.append(len(s.options))
        assert len(set(s.options)) == len(s.options)
        if len(s.options) > 40:
            assert s.meta["max_len"] >= 4096 and s.meta["big"] == len(s.options)
        if s.options[s.target] not in BT.NONE_OPTION:
            m = re.match(r"(\w+?)_", s.options[s.target]) or re.match(r"(\w+) (?:the|all) ", s.options[s.target].lower())
            assert m and m.group(1) in BT.VERBS, s.options[s.target]
    assert min(sizes) >= 6 and max(sizes) > 200 and 20 < sum(1 for n in sizes if n > 100)


def test_requests_are_paraphrases_not_the_tool_name_itself():
    for s in draw(BT.gen_bigtools, 300, seed=17):
        tgt = s.options[s.target]
        if tgt in BT.NONE_OPTION:
            continue
        name = tgt.split(" — ")[0]
        assert name not in s.state and "_" not in s.state


def test_small_bigtools_rows_fit_the_640_token_window_of_the_trainers_eval(tok):
    e = Leeth2Encoder(tok, VOCAB, max_len=640)
    for s in draw(BT.gen_bigtools_small, 60, seed=18):
        assert 6 <= len(s.options) <= 12 and "max_len" not in s.meta
        assert len(e.encode(s)) <= 640


# ------------------------------------------------------------------------------------------------ calibrate
def test_calibration_script_fits_a_temperature_and_writes_a_config_copy(tmp_path, tok, tiny_model):
    cal = _load("leeth21_calibrate")
    rt = Leeth2Runtime(tiny_model, Leeth2Encoder(tok, VOCAB, max_len=4096), "cpu")
    report = cal.fit_big_temperature(rt, n=4)
    assert 0.2 <= report["temp_choice_big"] <= 8.0 and set(report["by_k"]) == {"16", "64", "128", "255"}
    for v in report["by_k"].values():
        assert 0.0 <= v["acc"] <= 1.0 and 0.0 <= v["ece_after"] <= 1.0
    src = tmp_path / "src"
    src.mkdir()
    (src / "leeth2_config.json").write_text('{"temp_choice": 1.2, "big_k": 32}', encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"x" * 10)
    out = tmp_path / "out"
    cal.write_calibrated(src, out, report)
    import json

    cfg = json.loads((out / "leeth2_config.json").read_text(encoding="utf-8"))
    assert cfg["temp_choice"] == 1.2 and cfg["temp_choice_big"] == report["temp_choice_big"] and cfg["big_k"] == 64
    assert (out / "model.safetensors").read_bytes() == b"x" * 10 and (out / "calibration_big.json").exists()


# ------------------------------------------------------------------------------------------------ data kernel v2
def test_base_mix_is_thinned_by_task_and_the_protected_tasks_are_kept_whole():
    import json

    path = Path(__file__).resolve().parents[2] / ".argent" / "kaggle" / "leeth2" / "kernel_leeth21_data2" / "leeth21_data2.py"
    spec = importlib.util.spec_from_file_location("leeth21_data2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sizes = {"argent_route_v2": 8000, "argent_safety_v2": 7000, "games_a": 20000, "sites": 15000, "tiny": 30}
    lines = [json.dumps({"task": t, "i": i}, separators=(",", ":")) for t, n in sizes.items() for i in range(n)]
    out, cap, counts = mod.cap_by_task(lines, 45000, random.Random(0))
    assert len(out) <= 45000 and len(out) > 40000
    assert counts["argent_route_v2"] == (8000, 8000) and counts["argent_safety_v2"] == (7000, 7000) and counts["tiny"] == (30, 30)
    assert counts["games_a"][1] == counts["sites"][1] == cap and cap >= mod.MIN_CAP


# ------------------------------------------------------------------------------------------------ round 3
def test_heavy_bigtools_rows_have_100_to_255_options_and_the_window_policy_applies_to_them(tok):
    ss = draw(lambda rng: BT.gen_bigtools(rng, k_range=(100, 255)), 60, seed=19)
    assert len(ss) > 50 and all(99 < len(s.options) <= 256 for s in ss)  # +1: the "none" option may be inserted
    e = Leeth2Encoder(tok, VOCAB, max_len=640)
    big = [s for s in ss if len(s.options) > 128][0]
    import dataclasses

    wordy = [f"{o} " + " ".join(f"w{i}" for i in range(30)) for o in big.options]
    x = e.encode(dataclasses.replace(big, options=wordy))
    # more than 128 options: the window grows beyond the 640 of the evaluation so that an option keeps 15 tokens (+ its marker) instead of 1
    # (the cap of a 640 window is 640 // 2 // K - 1 = 1 for K = 130)
    sizes = [x.opt_index.count(j) for j in range(1, len(wordy) + 1)]
    assert x.n_opts == len(wordy) and max(sizes) - 1 == 15 and len(x) > 640


def test_the_trainer_can_fit_temperatures_with_the_run_time_window():
    from src.leeth2.train import TrainConfig

    cfg = TrainConfig()
    assert cfg.calib_max_len == 0
    cfg.calib_max_len = 4096
    assert cfg.calib_max_len == 4096


def test_a_list_longer_than_split_k_is_answered_in_two_rounds_and_a_short_one_is_untouched(tok, tiny_model):
    import numpy as np

    from src.leeth2.evaluate import _chunks, predict
    from src.leeth2.schema import Sample

    assert _chunks(255, 128) == [list(range(0, 128)), list(range(128, 255))]
    assert [len(c) for c in _chunks(161, 128)] == [81, 80] and _chunks(10, 128) == [list(range(10))]
    enc = Leeth2Encoder(tok, VOCAB, max_len=4096)
    mk = lambda k: Sample("t", "tools", "en", "choice", "User request: open the invoice", "Which tool?", [f"read_item_{i}" for i in range(k)], 3)
    short, long_ = mk(20), mk(150)
    cfg = tiny_model.cfg
    assert cfg.split_k == 0
    flat = predict(tiny_model, enc, [short, long_], device="cpu", calibrated=True)
    cfg.split_k = 64
    try:
        two = predict(tiny_model, enc, [short, long_], device="cpu", calibrated=True)
        raw = predict(tiny_model, enc, [long_], device="cpu", calibrated=False)
    finally:
        cfg.split_k = 0
    assert np.allclose(two[0].probs, flat[0].probs)  # 20 options: not split
    assert two[1].raw["split"] == 3 and two[1].probs.shape == (150,) and abs(float(two[1].probs.sum()) - 1.0) < 1e-6
    assert "split" not in raw[0].raw and abs(float(raw[0].probs.sum()) - 1.0) < 1e-6  # raw predictions are never split
