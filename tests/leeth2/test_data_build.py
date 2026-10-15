import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.leeth2.data import build
from src.leeth2.data.kinds import Spec
from src.leeth2.datafile import JsonlReader


def fake_hub(monkeypatch, data):
    """data[(repo, cfg, split)] -> rows. No network."""

    def open_split(spec, split):
        return SimpleNamespace(features=None)

    def iter_rows(spec, split, seed, limit=None):
        rows = data[(spec.repo, spec.cfg, split)]
        for i, r in enumerate(rows):
            if limit is not None and i >= limit:
                return
            yield i, r

    monkeypatch.setattr(build, "open_split", open_split)
    monkeypatch.setattr(build, "iter_rows", iter_rows)


def intents(n, prefix):
    labs = ["alarm_set", "music_play", "weather_query"]
    return [{"text": f"{prefix} request number {i} about {labs[i % 3]}", "label": labs[i % 3]} for i in range(n)]


def test_run_spec_official_splits_and_string_label_prepass(tmp_path, monkeypatch):
    data = {
        ("fake/intents", None, "train"): intents(300, "train"),
        ("fake/intents", None, "validation"): intents(120, "val"),
        ("fake/intents", None, "test"): intents(80, "test"),
    }
    fake_hub(monkeypatch, data)
    sp = Spec(id="fake_intents", repo="fake/intents", kind="cls", family="intent", lang="en", dev="validation", test="test", caps=(200, 20, 30, 40),
              opts=dict(text="text", string_labels=True, what_en=["intent"], what_ru=["намерение"]))
    info = build.run_spec(sp, tmp_path, seed=1, scale=1.0)
    assert info["error"] is None
    c = info["counts"]
    assert 200 <= c["train"] <= 260 or c["train"] == 200  # cap is on samples; one row may yield a choice and a noul sample
    assert c["dev"] == 20 and c["calib"] == 30 and c["test"] == 40
    ch = [json.loads(l) for l in open(tmp_path / "shards" / "fake_intents.dev.jsonl", encoding="utf-8")]
    assert all(d["family"] == "intent" and d["task"] == "fake_intents" for d in ch)
    # the label set came from a pre-pass over the data and became the options
    opts = {o for d in ch if d["ptype"] == "choice" for o in d["options"]}
    assert opts <= {"alarm set", "music play", "weather query"}
    # resumable: a second call must not touch the network again
    monkeypatch.setattr(build, "iter_rows", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-read")))
    assert build.run_spec(sp, tmp_path, seed=1, scale=1.0)["counts"] == c


def test_carved_eval_pools_never_overlap_train(tmp_path, monkeypatch):
    rows = [{"input": f"Person number {i} lives in Town{i}.", "output": f"['Person number {i} <> Person', 'Town{i} <> City']"} for i in range(6000)]
    fake_hub(monkeypatch, {("fake/ner", None, "entity"): rows})
    from src.leeth2.data.specs import ner_pairs_from_numind

    sp = Spec(id="fake_ner", repo="fake/ner", kind="entlist", family="ner", lang="en", train="entity", dev=None, caps=(3000, 0, 0, 0), opts=dict(get=ner_pairs_from_numind))
    info = build.run_spec(sp, tmp_path, seed=3, scale=1.0)
    c = info["counts"]
    assert c["train"] >= 3000 and c["dev"] > 0 and c["calib"] > 0 and c["test"] > 0
    summary = build.assemble(tmp_path, seed=3, infos=[info])
    assert summary["counts"]["train"] == c["train"] - summary["train_removed_leak_or_dup"] + 0 or summary["counts"]["train"] <= c["train"]
    ev = set()
    for role in ("dev", "calib", "test"):
        for l in open(tmp_path / f"{role}.jsonl", encoding="utf-8"):
            d = json.loads(l)
            ev.add((d["state"], d["question"]))
    for l in open(tmp_path / "train.jsonl", encoding="utf-8"):
        d = json.loads(l)
        assert (d["state"], d["question"]) not in ev, "train/eval leakage"


def test_assemble_scrubs_duplicates_and_leaks_and_indexes(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()

    def row(state, q="what?"):
        return json.dumps({"task": "t", "family": "f", "lang": "en", "ptype": "noul", "state": state, "question": q, "target": 1.0})

    (shards / "a.train.jsonl").write_text("\n".join([row("x1"), row("x1"), row("leaky"), row("x2")]) + "\n", encoding="utf-8")
    (shards / "a.test.jsonl").write_text(row("leaky") + "\n", encoding="utf-8")
    out = build.assemble(tmp_path, seed=1, infos=[])
    assert out["counts"] == {"train": 2, "dev": 0, "calib": 0, "test": 1}
    assert out["train_removed_leak_or_dup"] == 2
    rd = JsonlReader(tmp_path / "train.jsonl")
    assert len(rd) == 2 and {rd.get(i).state for i in range(2)} == {"x1", "x2"}


def test_a_broken_dataset_is_reported_not_fatal(tmp_path, monkeypatch):
    def boom(spec, split):
        raise RuntimeError("Dataset scripts are no longer supported")

    monkeypatch.setattr(build, "open_split", boom)
    monkeypatch.setattr(build, "iter_rows", lambda *a, **k: iter(()))
    sp = Spec(id="broken", repo="fake/broken", kind="cls", family="x", lang="en", dev="validation", caps=(10, 5, 5, 5), opts=dict(text="text", string_labels=True))
    info = build.run_spec(sp, tmp_path, seed=1, scale=1.0)
    assert info["error"] and info["counts"]["train"] == 0
