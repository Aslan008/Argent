"""The v3 data: states byte-identical to the ones Argent sends, the 16-option list Argent asks, and labels decided by rule."""

import json
import random
from collections import Counter
from pathlib import Path

from src.leeth.dispatcher import _CANONICAL_WORKDIR, LeethDispatcher
from src.leeth2.data import finetune as F
from src.leeth2.data.synth import argent_context as CX
from src.leeth2.data.synth import argent_router as AR

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def samples(fn, n, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.extend(fn(rng) or [])
    return out


def test_state_is_what_the_dispatcher_builds():
    rng = random.Random(0)
    wd = _CANONICAL_WORKDIR                      # a constant of the dispatcher; the generators vary it, the layout is what matters
    for i in range(300):
        ctx = CX.make_context(rng) if i % 3 else None
        query = rng.choice(["Прочитай main.py", "останови сервер", "ok, continue", "найди файл app.py"])
        want = LeethDispatcher._build_state(query, context=ctx)
        assert CX.build_state(query, ctx, wd) == want
    # an empty context is the bare state, as in production
    empty = {"active_tasks": [], "last_target": None, "last_tool": None, "prev_turn": None}
    assert CX.build_state("привет", empty, wd) == LeethDispatcher._build_state("привет", context=empty)


def test_options_are_the_ones_the_dispatcher_asks():
    _state, questions = LeethDispatcher._build_state_and_questions("x")
    asked = next(q for q in questions if q.id == "tool")
    assert CX.PROD_OPTIONS == list(asked.choices)


def test_every_sample_is_valid_and_labels_follow_the_rules():
    got = samples(CX.gen_background, 3000) + samples(CX.gen_delegate, 3000) + samples(CX.gen_route16, 3000)
    for s in got:
        s.validate()
    routes = [s for s in got if s.ptype == "choice"]
    assert all(s.options == CX.PROD_OPTIONS for s in routes)
    tools = Counter(s.meta["tool"] for s in routes)
    assert set(tools) == set(CX.PROD_OPTIONS), "every one of the 16 options must be taught"
    assert tools["read_background_command"] > 200
    for s in routes:
        if s.meta["tool"] == "read_background_command":
            request = s.state.split("\n")[1]
            assert "Active background processes: PID" in s.state or any(ch.isdigit() for ch in request), "reading a background output needs a running process or a named pid"
    for s in got:
        if s.task == "argent_need_tool_v3":
            assert s.target in (0.0, 1.0)
        if s.task == "argent_arg_v3":
            assert s.state[s.target[0]:s.target[1]]


def test_context_changes_the_answer_only_where_the_request_depends_on_it():
    got = samples(CX.gen_background, 4000, seed=9)
    with_proc = [s for s in got if s.ptype == "choice" and "Active background processes: PID" in s.state]
    without = [s for s in got if s.ptype == "choice" and "Active background processes: None" in s.state]
    assert with_proc and without
    assert any(s.meta["tool"] == "read_background_command" for s in with_proc)
    assert any(s.meta["tool"] == "run_command" for s in with_proc)                 # "stop the server" while one runs
    # the same words with nothing running are NOT answered with a tool that needs a process
    assert not any(s.meta["tool"] == "read_background_command" and "Active background processes: None" in s.state and "Incoming user request" in s.state
                   and not any(ch.isdigit() for ch in s.state.split("\n")[1]) for s in without)


def test_delegation_data_never_sends_a_pasted_error_or_a_skill_request_to_a_tool():
    got = [s for s in samples(CX.gen_delegate, 4000, seed=4) if s.ptype == "choice"]
    labels = Counter(s.meta["tool"] for s in got)
    assert set(labels) == {"UNFAMILIAR", "NO_TOOL"}
    for s in got:
        q = s.state.split("\n")[1].lower()
        if "скилл" in q or "the " in q and " skill" in q or "error:" in q or "ошибка:" in q or "погод" in q:
            assert s.meta["tool"] == "UNFAMILIAR"


def test_harmless_commands_are_harmless_and_review_commands_are_in_between():
    got = [s for s in samples(CX.gen_safety_more, 4000, seed=2) if s.ptype == "noul"]
    by = {}
    for s in got:
        by.setdefault(s.state.split("\n")[1], set()).add(round(s.target, 2))
    assert min(by["Command: npm ci"]) >= 0.9
    assert min(by["Command: python -m http.server 8000"]) >= 0.9
    assert max(by["Command: docker compose down -v"]) <= 0.5


def test_v3_build_keeps_exams_out_including_context_states(tmp_path):
    F.build(tmp_path, seed=11, scale=0.01, exam_dir=SCRIPTS, v3=True)
    blocked = F.exam_texts(SCRIPTS)
    seen_context = 0
    for role in ("train", "dev", "calib", "test"):
        for line in (tmp_path / f"{role}.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            assert not F.leaks(d["state"], blocked)
            seen_context += "Active background processes:" in d["state"]
    assert seen_context > 100
    # a context state must be recognised by the leak filter (the request is followed by 'Active background processes', not by the working directory)
    assert F.leaks('Incoming user request in Argent session:\n"Открой  main.py"\nActive background processes: None\nRecent session context:', blocked)


def test_v3_mix_drops_the_v1_template_routing_rows(tmp_path):
    from src.leeth2.data.argent import TOOL_QUESTION
    from src.leeth2.schema import Sample

    base, targeted, out = tmp_path / "base", tmp_path / "targeted", tmp_path / "mix"
    base.mkdir()
    targeted.mkdir()

    def write(path, rows):
        path.write_text("\n".join(s.to_json() for s in rows) + "\n", encoding="utf-8")

    old_route = [Sample("argent_route", "argent", "ru", "choice", f"s{i}", TOOL_QUESTION, ["a", "b"], 0) for i in range(30)]
    toy = [Sample("toy", "toy", "en", "choice", f"t{i}", "which?", ["a", "b"], i % 2) for i in range(60)]
    new = [Sample("new", "toy", "en", "choice", f"n{i}", "new?", ["a", "b"], 0) for i in range(10)]
    write(base / "train.jsonl", old_route + toy)
    for role in ("dev", "calib", "test"):
        write(base / f"{role}.jsonl", old_route[:3] + toy[:5])
        write(targeted / f"{role}.jsonl", new)
    write(targeted / "train.jsonl", new)
    F.mix(targeted, base, out, replay=1000, v3=True)
    assert not any(json.loads(l)["question"] == TOOL_QUESTION for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines())
    F.mix(targeted, base, tmp_path / "mix2", replay=1000, v3=False)
    assert any(json.loads(l)["question"] == TOOL_QUESTION for l in (tmp_path / "mix2" / "train.jsonl").read_text(encoding="utf-8").splitlines())
