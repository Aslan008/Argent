"""The v2 fine-tune data: valid, balanced, and unable to leak the hand-written exams."""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from src.leeth2.data import finetune as F
from src.leeth2.data.argent import SAFETY_QUESTION, TOOL_OPTIONS
from src.leeth2.data.synth import argent_router as AR

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def samples(fn, n, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(fn(rng) or [])
    return out


def test_every_generated_sample_is_valid_and_every_tool_appears():
    got = samples(AR.gen_route, 3000)
    for s in got:
        s.validate()
    routed = Counter(s.meta["tool"] for s in got if s.task == "argent_route_v2")
    assert set(routed) == set(TOOL_OPTIONS)
    assert min(routed.values()) > 40
    for s in samples(AR.gen_safety, 2000):
        s.validate()


def test_the_label_follows_the_tool_not_the_wording():
    got = samples(AR.gen_route, 4000, seed=3)
    need = [s for s in got if s.task == "argent_need_tool_v2"]
    routes = {s.state: s for s in got if s.task == "argent_route_v2"}
    assert need
    for s in need:
        tool = routes[s.state].meta["tool"]
        assert s.target == (0.0 if tool == "NO_TOOL" else 1.0)
    for s in got:
        if s.task == "argent_arg_v2":
            assert s.state[s.target[0]:s.target[1]]            # a real, non-empty span of the request


def test_hard_contrasts_are_separated_by_the_object_named():
    """The same lead verb routes differently depending on what follows — that is what the data must teach."""
    rng = random.Random(0)
    lead_find, lead_grep, lead_open_url, lead_open_file = set(), set(), set(), set()
    for _ in range(3000):
        for tool, bucket in (("find_by_name", lead_find), ("grep_search", lead_grep), ("browser_open", lead_open_url), ("read_file", lead_open_file)):
            text, _k = AR.MAKERS[tool][0](rng)
            bucket.add(text.split()[0].lower())
    assert "найди" in lead_find and "найди" in lead_grep           # same verb, two tools
    assert "открой" in lead_open_url and "открой" in lead_open_file


def test_safety_labels_have_three_classes():
    got = [s for s in samples(AR.gen_safety, 3000, seed=5) if s.ptype == "noul"]
    assert all(s.question == SAFETY_QUESTION for s in got)
    targets = [s.target for s in got]
    assert sum(t >= 0.9 for t in targets) > 800 and sum(t <= 0.05 for t in targets) > 800 and sum(0.3 <= t <= 0.5 for t in targets) > 100
    by_cmd = {}
    for s in got:
        by_cmd.setdefault(s.state.split("\n")[1], set()).add(round(s.target, 1))
    assert by_cmd["Command: rm -rf /"] == {0.0}
    assert min(by_cmd["Command: git status"]) >= 0.9


def test_exam_texts_cover_all_exams_and_catch_leaks():
    blocked = F.exam_texts(SCRIPTS)
    assert F.norm("открой main.py") in blocked and F.norm("Remove-Item -Recurse -Force C:\\") in blocked and F.norm("найди все TODO в проекте") in blocked
    assert F.leaks('Incoming user request in Argent session:\n"Открой  main.py"\nWorking directory: /x', blocked)
    assert F.leaks("User shell execution request:\nCommand: rm -rf /\nCaller: Argent Agent", blocked)
    assert not F.leaks('Incoming user request in Argent session:\n"открой main.py и перепиши"\nWorking directory: /x', blocked)


def test_build_never_emits_an_exam_item_and_keeps_roles_apart(tmp_path):
    F.build(tmp_path, seed=7, scale=0.02, exam_dir=SCRIPTS)
    blocked = F.exam_texts(SCRIPTS)
    keys = {}
    for role in ("train", "dev", "calib", "test"):
        rows = [json.loads(l) for l in (tmp_path / f"{role}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows
        for d in rows:
            assert not F.leaks(d["state"], blocked)
            keys.setdefault((d["ptype"], d["state"], d["question"]), set()).add(role)
    assert all(not ({"train"} & r and r - {"train"}) for r in keys.values()), "a training sample also sits in an evaluation split"


def test_mix_replays_v1_without_the_superseded_rows(tmp_path):
    from src.leeth2.schema import Sample

    base, targeted, out = tmp_path / "base", tmp_path / "targeted", tmp_path / "mix"
    base.mkdir()
    targeted.mkdir()

    def write(path, rows):
        path.write_text("\n".join(s.to_json() for s in rows) + "\n", encoding="utf-8")

    def toy(i, q="which?"):
        return Sample("toy", "toy", "en", "choice", f"state {i}", q, ["a", "b"], i % 2)

    def safety(i):
        return Sample("argent_old", "argent", "ru", "noul", f"User shell execution request:\nCommand: c{i}\nCaller: x", SAFETY_QUESTION, None, 1.0)

    write(base / "train.jsonl", [toy(i) for i in range(300)] + [safety(i) for i in range(50)])
    for role in ("dev", "calib", "test"):
        write(base / f"{role}.jsonl", [toy(1000 + i) for i in range(20)] + [safety(500 + i) for i in range(5)])
        write(targeted / f"{role}.jsonl", [toy(2000 + i, "new?") for i in range(10)])
    write(targeted / "train.jsonl", [toy(5000 + i, "new?") for i in range(40)])

    counts = F.mix(targeted, base, out, replay=100)
    assert counts["train"] == 40 + 100
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(d["question"] == "new?" for d in rows) == 40
    assert not any(d["question"] == SAFETY_QUESTION for d in rows), "superseded v1 rows must not be replayed"
    for role in ("dev", "calib", "test"):
        part = [json.loads(l) for l in (out / f"{role}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {d["question"] for d in part} == {"which?", "new?"}
    assert (out / "train.jsonl.idx.npy").exists()
