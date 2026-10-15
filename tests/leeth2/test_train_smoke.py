"""End-to-end smoke test of the trainer on CPU with a tiny random backbone and the real mmBERT tokenizer.

Skipped when the tokenizer is not on disk (CI). It exercises: corpus index, batch planning, the training
loop, EMA, evaluation, temperature calibration, final export, and resuming from state.pt.
"""

import json
import random
import shutil
from pathlib import Path

import pytest

TOK_DIR = Path(".argent/models/mmbert-argent")
pytestmark = pytest.mark.skipif(not (TOK_DIR / "tokenizer.json").exists(), reason="mmBERT tokenizer not on disk")


def make_corpus(path: Path, n: int, seed: int):
    from src.leeth2.schema import Sample

    rng = random.Random(seed)
    colors = ["red", "green", "blue"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            kind = i % 4
            w = [rng.choice(["apple", "river", "stone", "cloud", "wind", "lamp"]) for _ in range(rng.randint(3, 9))]
            if kind == 0:
                c = rng.randrange(3)
                s = Sample("toy_choice", "toy", "en", "choice", " ".join(w + [colors[c]]), "which colour is mentioned", colors, c)
            elif kind == 1:
                has = rng.random() < 0.5
                s = Sample("toy_noul", "toy", "en", "noul", " ".join(w + (["42"] if has else [])), "does it contain a number", None, float(has))
            elif kind == 2:
                lv = 1 + min(4, len(w) // 2)
                s = Sample("toy_score", "toy", "en", "score", " ".join(w), "how long is it", None, float(lv), levels=5)
            else:
                text = " ".join(w + ["code", "777"])
                a = text.index("777")
                s = Sample("toy_extract", "toy", "en", "extract", text, "extract the number", None, [a, a + 3])
            s.validate()
            f.write(s.to_json() + "\n")


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from transformers import ModernBertConfig, ModernBertModel

    d = tmp_path_factory.mktemp("tiny_base")
    cfg = ModernBertConfig(
        vocab_size=256000, hidden_size=32, num_hidden_layers=3, num_attention_heads=4, intermediate_size=48,
        max_position_embeddings=512, global_attn_every_n_layers=3, local_attention=16,
        pad_token_id=0, bos_token_id=2, eos_token_id=1, cls_token_id=2, sep_token_id=1,
    )
    import torch

    torch.manual_seed(0)
    ModernBertModel(cfg).save_pretrained(d)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOK_DIR / name, d / name)
    return d


def test_trainer_end_to_end_and_resume(tiny_base, tmp_path):
    from src.leeth2 import train as T
    from src.leeth2.datafile import JsonlReader, plan_batches
    from src.leeth2.model import Leeth2Model

    data = tmp_path / "data"
    data.mkdir()
    make_corpus(data / "train.jsonl", 1600, 1)
    make_corpus(data / "dev.jsonl", 80, 2)
    make_corpus(data / "calib.jsonl", 240, 3)
    make_corpus(data / "test.jsonl", 80, 4)

    # planning: every example exactly once, token budget respected, same count for every rank
    rd = JsonlReader(data / "train.jsonl")
    plan = plan_batches(rd.lens, tokens_per_batch=600, max_batch=32, seed=1, world=2, max_len=64)
    flat = [i for b in plan for i in b]
    assert len(plan) % 2 == 0 and len(flat) == len(set(flat)) and len(flat) > 1500
    assert all(len(b) <= 32 for b in plan)

    out = tmp_path / "run"
    cfg = T.TrainConfig(
        base=str(tiny_base), data_dir=str(data), out_dir=str(out), max_len=64, tokens_per_batch=600, max_batch=32, grad_accum=2,
        lr=3e-3, head_lr_mult=2.0, epochs=1.0, max_steps=120, eval_every=40, save_every=40, log_every=10, amp=False,
        workers=0, ema_decay=0.9, time_budget_s=600, dev_cap=80,
    )
    T.run(cfg)

    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    rows = [r for r in log if "gn" in r]

    def mean_loss(r):  # batches differ in which primitives they contain, so compare per-primitive means
        v = [x for k, x in r.items() if k.startswith("loss_")]
        return sum(v) / len(v)

    first = sum(mean_loss(r) for r in rows[:3]) / 3
    last = sum(mean_loss(r) for r in rows[-3:]) / 3
    assert last < first * 0.9, f"loss goes down: {first:.3f} -> {last:.3f}"
    evals = [r for r in log if "eval" in r]
    assert len(evals) >= 3
    assert (out / "state.pt").exists() and (out / "final" / "model.safetensors").exists()
    report = json.loads((out / "report.json").read_text())
    assert "temperatures" in report and "test" in report and report["steps"] == 120

    # the exported model loads and keeps the fitted temperatures
    m = Leeth2Model.from_pretrained(out / "final")
    assert m.cfg.temp_choice == pytest.approx(report["temperatures"]["temp_choice"])

    # resume continues from the saved step instead of starting over
    cfg2 = T.TrainConfig(**{**cfg.__dict__, "max_steps": 140})
    T.run(cfg2)
    report2 = json.loads((out / "report.json").read_text())
    assert report2["steps"] == 140
    steps = [r["step"] for r in map(json.loads, (out / "train_log.jsonl").read_text().splitlines()) if "gn" in r]
    assert steps == sorted(steps) and steps[-1] == 140 and steps.count(120) == 1

    # resuming WITHOUT an explicit length keeps the saved schedule
    cfg3 = T.TrainConfig(**{**cfg.__dict__, "max_steps": 0})
    T.run(cfg3)
    assert json.loads((out / "report.json").read_text())["steps"] == 140


def test_fine_tune_starts_from_the_ema_weights_of_a_finished_run(tiny_base, tmp_path):
    import torch

    from src.leeth2 import train as T
    from src.leeth2.model import Leeth2Model
    from transformers import AutoTokenizer

    data = tmp_path / "data"
    data.mkdir()
    for name, n, seed in (("train", 400, 1), ("dev", 40, 2), ("calib", 80, 3), ("test", 40, 4)):
        make_corpus(data / f"{name}.jsonl", n, seed)
    common = dict(base=str(tiny_base), data_dir=str(data), max_len=64, tokens_per_batch=600, max_batch=32, grad_accum=1, lr=3e-3,
                  eval_every=1000, save_every=1000, log_every=10, amp=False, workers=0, ema_decay=0.9, time_budget_s=600, dev_cap=40)
    first = tmp_path / "first"
    T.run(T.TrainConfig(out_dir=str(first), max_steps=10, **common))

    fresh = Leeth2Model.from_base(str(tiny_base), tokenizer=AutoTokenizer.from_pretrained(str(tiny_base)), attn_implementation="sdpa")
    what = T.load_init(fresh, first)
    assert "EMA" in what
    st = torch.load(first / "state.pt", map_location="cpu", weights_only=False)
    params = dict(fresh.named_parameters())
    assert st["ema"] and all(torch.allclose(params[k].detach(), v, atol=0) for k, v in st["ema"].items() if k in params)

    # a fine-tune is a NEW run: its own step counter, but the first evaluation already reflects the loaded weights
    second = tmp_path / "second"
    T.run(T.TrainConfig(out_dir=str(second), max_steps=4, init_from=str(first), **common))
    st2 = torch.load(second / "state.pt", map_location="cpu", weights_only=False)
    assert int(st2["step"]) == 4

    # without state.pt the half-precision final/ model is used
    (first / "state.pt").unlink()
    fresh2 = Leeth2Model.from_base(str(tiny_base), tokenizer=AutoTokenizer.from_pretrained(str(tiny_base)), attn_implementation="sdpa")
    assert "model.safetensors" in T.load_init(fresh2, first)


@pytest.fixture(scope="module")
def tiny_base6(tmp_path_factory):
    """Six layers: a block of three can be copied (layer 0 never is)."""
    import torch
    from transformers import ModernBertConfig, ModernBertModel

    d = tmp_path_factory.mktemp("tiny_base6")
    cfg = ModernBertConfig(
        vocab_size=256000, hidden_size=32, num_hidden_layers=6, num_attention_heads=4, intermediate_size=48,
        max_position_embeddings=512, global_attn_every_n_layers=3, local_attention=16,
        pad_token_id=0, bos_token_id=2, eos_token_id=1, cls_token_id=2, sep_token_id=1,
    )
    torch.manual_seed(0)
    ModernBertModel(cfg).save_pretrained(d)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOK_DIR / name, d / name)
    return d


def test_a_grown_model_trains_end_to_end(tiny_base6, tmp_path):
    """leeth2.grow -> train.run(arch_from=...): the grown network is built, loaded from the grown weights, trained, calibrated and exported."""
    import torch

    from src.leeth2 import grow
    from src.leeth2 import train as T
    from src.leeth2.model import Leeth2Model

    data = tmp_path / "data"
    data.mkdir()
    for name, n, seed in (("train", 800, 1), ("dev", 60, 2), ("calib", 160, 3), ("test", 60, 4)):
        make_corpus(data / f"{name}.jsonl", n, seed)
    common = dict(max_len=64, tokens_per_batch=600, max_batch=32, grad_accum=2, lr=3e-3, head_lr_mult=2.0, epochs=1.0, eval_every=20, save_every=20, log_every=5,
                  amp=False, workers=0, ema_decay=0.9, time_budget_s=600, dev_cap=60)
    T.run(T.TrainConfig(base=str(tiny_base6), data_dir=str(data), out_dir=str(tmp_path / "small"), max_steps=20, **common))
    small = tmp_path / "small" / "final"

    grown = tmp_path / "grown"
    info = grow.grow_checkpoint(small, grown, [5], mlp_mult=2.0)
    assert info["layers"] == 9 and info["intermediate"] == 96
    diff = grow.verify_equivalent(small, grown, seq=48)
    assert max(diff.values()) < 1e-3, diff                       # the exported model is fp16, hence the looser bound than in test_leeth2_grow

    out = tmp_path / "big"
    T.run(T.TrainConfig(base=str(tiny_base6), data_dir=str(data), out_dir=str(out), max_steps=30, arch_from=str(grown), init_from=str(grown), llrd=0.95, new_lr_mult=3.0, **common))
    report = json.loads((out / "report.json").read_text())
    assert report["steps"] == 30 and "test" in report
    m = Leeth2Model.from_pretrained(out / "final")
    assert len(m.backbone.layers) == 9 and m.cfg.grown["new_layers"] == [6, 7, 8]
    assert sum(p.numel() for p in m.parameters()) > sum(p.numel() for p in Leeth2Model.from_pretrained(small).parameters())
    log = [json.loads(l) for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert all(torch.isfinite(torch.tensor(r["gn"])) for r in log if "gn" in r)

    # the second stage the final kernel runs: the start model is a TRAINED grown model, built from its own config (grow.is_grown says so)
    assert grow.is_grown(out / "final") and not grow.is_grown(small)
    again = tmp_path / "again"
    T.run(T.TrainConfig(base=str(tiny_base6), data_dir=str(data), out_dir=str(again), max_steps=10, arch_from=str(out / "final"), init_from=str(out / "final"),
                        llrd=0.95, new_lr_mult=1.5, **common))
    m2 = Leeth2Model.from_pretrained(again / "final")
    assert len(m2.backbone.layers) == 9 and m2.cfg.grown["new_layers"] == [6, 7, 8]
