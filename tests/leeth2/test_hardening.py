"""The false-dispatch clusters: delegate-side requests carry no concrete target, tool-side ones do, and no exam sentence leaks in."""

import json
import re
from collections import Counter
from pathlib import Path

from src.leeth2.data import finetune
from src.leeth2.data.synth import argent_hardening as H
from src.leeth2.data.synth import generate

REQUEST = re.compile(r'Incoming user request in Argent session:\n"(.*?)"\n', re.S)
URL = re.compile(r"https?://|\b[\w-]+\.(?:com|org|kz|dev|io|net|ru)\b|localhost|127\.0\.0\.1")


def route_rows(n: int = 3000, seed: int = 11):
    out = []
    for s in generate(H.gen_hardening, n, seed):
        if s.ptype == "choice":
            out.append((REQUEST.search(s.state).group(1), s.options[s.hard_choice()], s))
    return out


def test_labels_are_only_the_five_the_clusters_teach():
    got = Counter(tool for _, tool, _ in route_rows())
    assert set(got) == {"UNFAMILIAR", "NO_TOOL", "browser_open", "run_command", "calculate"}, got
    assert got["UNFAMILIAR"] > got["browser_open"]            # most of the cluster material is "hand it to the LLM"


def test_browser_open_only_when_there_is_an_address():
    rows = route_rows()
    assert all(URL.search(q) for q, tool, _ in rows if tool == "browser_open")
    # a site NAME is not an address
    named = [q for q, tool, _ in rows if tool == "UNFAMILIAR" and re.search(r"epic games|steam|ozon|avito|wildberries|яндекс карты|мой банк", q)]
    assert named and not any(URL.search(q) for q in named)


def test_run_command_only_with_an_explicit_command_and_calculate_only_with_clean_arithmetic():
    rows = route_rows()
    for q, tool, _ in rows:
        if tool == "run_command":
            assert re.search(r"[a-z]{3}|гит", q), q                      # a command (typos from the noise layer allowed)
        if tool == "calculate":
            assert re.search(r"\d", q) and "\\" not in q, q


def test_latex_and_chatty_math_go_to_the_llm():
    qs = [q for q, tool, _ in route_rows(6000) if tool == "UNFAMILIAR" and ("\\" in q or "сколько будет" in q.lower() or "рассуди" in q)]
    assert any("\\" in q for q in qs) and any("сколько будет" in q.lower() for q in qs)


def test_no_context_anaphora_is_delegated():
    rows = [(q, s) for q, tool, s in route_rows(6000) if tool == "UNFAMILIAR" and q.split()[0] in ("открой", "покажи", "прочитай") and "его" in q and len(q.split()) <= 4]
    assert rows
    for q, s in rows:
        assert "Last target: None" in s.state or "Recent session context" not in s.state


def test_every_sample_validates_and_options_are_the_production_sixteen():
    from src.leeth2.data.synth.argent_context import PROD_OPTIONS

    for _, _, s in route_rows(800):
        s.validate()
        assert s.options == PROD_OPTIONS


def test_the_build_drops_anything_equal_to_an_exam_request(tmp_path):
    exams = tmp_path / "scripts"
    exams.mkdir()
    (exams / "real_queries.json").write_text(json.dumps(["загугли", "проверь сеть"]), encoding="utf-8")
    finetune.build(tmp_path / "out", seed=3, scale=0.02, exam_dir=exams, v3=True, v4=True)
    for role in ("train", "dev", "test"):
        for line in (tmp_path / "out" / f"{role}.jsonl").read_text(encoding="utf-8").splitlines():
            m = REQUEST.search(json.loads(line)["state"]) if '"state"' in line else None
            if m:
                assert m.group(1).lower() not in ("загугли", "проверь сеть")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["v4"] is True


def test_v4_adds_the_hardening_generator_and_v3_does_not():
    names3 = [n for n, _, _ in finetune.generators(True, False)]
    names4 = [n for n, _, _ in finetune.generators(True, True)]
    assert "hardening" in names4 and "hardening" not in names3


# ─── two-step requests: the owner's convention ────────────────────────────────

JOB_WORDS = re.compile(r"войди|заполни|нажми|выполни \w[\w-]* через|сделай \w[\w-]* с помощью|потом \w[\w-]* через|log in|fill in|screenshot|скриншот|выполни там поиск|быстрее двух секунд", re.I)


def test_compound_requests_follow_the_owner_batteries():
    import random

    from src.leeth2.data.synth import argent_router as AR

    rng = random.Random(5)
    report, job = set(), set()
    for _ in range(4000):
        text, label = AR.compound(rng, "read_file", "прочитай config.py", owner=True)
        if text == "прочитай config.py":
            continue
        (report if label == "read_file" else job).add((text, label))
    assert report and job
    assert all(label == "UNFAMILIAR" for _, label in job)
    assert any("исправь ошибки" in t for t, _ in job) and any("кратко" in t or "скажи" in t or "перескажи" in t for t, _ in report)
    assert not any("исправь" in t for t, _ in report)                 # a second JOB step is never labelled with the first tool


def test_before_v4_every_compound_keeps_the_first_tool_so_old_builds_repeat():
    import random

    from src.leeth2.data.synth import argent_router as AR

    rng = random.Random(5)
    seen = [AR.compound(rng, "read_file", "прочитай config.py", owner=False) for _ in range(2000)]
    assert any(t != "прочитай config.py" for t, _ in seen)
    assert {label for _, label in seen} == {"read_file"}


def test_v4_uses_the_owner_convention_in_both_route_grammars_and_v3_does_not():
    import functools

    from src.leeth2.data.synth import argent_context as CX
    from src.leeth2.data.synth import argent_router as AR

    v4 = {n: f for n, f, _ in finetune.generators(True, True)}
    v3 = {n: f for n, f, _ in finetune.generators(True, False)}
    for name in ("route", "route16"):
        assert isinstance(v4[name], functools.partial) and v4[name].keywords == {"owner": True}
    assert v3["route"] is AR.gen_route and v3["route16"] is CX.gen_route16


def test_open_a_page_and_do_more_goes_to_the_llm_but_open_and_report_stays_one_call():
    rows = route_rows(8000, seed=21)
    browser = [q for q, tool, _ in rows if tool == "browser_open"]
    assert browser and not any(JOB_WORDS.search(q) for q in browser), [q for q in browser if JOB_WORDS.search(q)][:3]
    jobs = [q for q, tool, _ in rows if tool == "UNFAMILIAR" and URL.search(q) and JOB_WORDS.search(q)]
    assert len(jobs) > 40, len(jobs)
    assert any(re.search(r"скажи, что там написано|назови заголовок|расскажи, что на странице", q) for q in browser)


def test_questions_about_running_things_are_answers_and_reproaches_need_the_llm():
    rows = route_rows(8000, seed=22)
    feasibility = [(q, tool) for q, tool, _ in rows if re.search(r"смогу ли я запустить|потянет ли|какая ссылка на|where can I find|can I run", q)]
    assert feasibility and {tool for _, tool in feasibility} == {"NO_TOOL"}
    reproach = [(q, tool) for q, tool, _ in rows if re.search(r"я же просил|ты разве не|ты так и не|you never|didn't I ask", q)]
    assert reproach and {tool for _, tool in reproach} == {"UNFAMILIAR"}


def test_two_requests_in_one_message_go_to_the_llm():
    rows = route_rows(12000, seed=31)
    arith = re.compile(r"\d+\s*[-+*/]\s*\d+|сколько будет|calculate|what is")
    files = re.compile(r"\.(py|md|json|yaml|toml|rs|ts|js|txt)\b|в папке|папк|git |pytest|npm ")
    both = [(q, tool) for q, tool, _ in rows if arith.search(q) and files.search(q) and re.search(r"\.\s+\S|; | и | а ещё|затем|Then|Also|also", q)]
    unfamiliar = [q for q, tool in both if tool == "UNFAMILIAR"]
    assert len(unfamiliar) > 25, len(unfamiliar)
    # a lone arithmetic request is still calculate (that cluster is untouched)
    assert any(tool == "calculate" for q, tool, _ in rows if arith.search(q) and not files.search(q))
