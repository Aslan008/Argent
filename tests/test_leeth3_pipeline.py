"""Leeth 3 teacher-data pipeline: what an outside generator writes is checked before a single line reaches training."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from src.leeth3.dedup import NearDupIndex, exact_key
from src.leeth3.pools import HOLDOUT, TRAIN, check_pools
from src.leeth3.records import NONE_TEXTS, RecordError, parse_record, to_samples

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ingest_mod = _load("leeth3_ingest")
queue_mod = _load("leeth3_queue")


def rec(**kw):
    base = {"id": "B0001-001", "request": "R002", "family": "photo-editor", "domain": "tools", "lang": "ru", "ptype": "choice",
            "state": "Пользователь: убери красные глаза на фото", "question": "Какой инструмент выполнит запрос?",
            "options": {"crop_image": "обрезать", "fix_redeye": "убрать красные глаза", "export_png": "сохранить PNG"},
            "answer": "fix_redeye", "abstain_ok": True, "difficulty": "easy", "teacher": "gemini-3.8-flash"}
    base.update(kw)
    return base


# ---- records ---------------------------------------------------------------------------------------------------------------------
def test_choice_with_descriptions_becomes_a_sample():
    r = parse_record(json.dumps(rec(), ensure_ascii=False))
    s = to_samples(r)[0]
    s.validate()
    assert s.options[s.target] == "fix_redeye: убрать красные глаза"   # the encoder reads option strings: the description goes inside
    assert s.options[-1] in NONE_TEXTS["ru"]                  # abstain_ok adds the explicit "none" option, in the example's language
    assert s.meta["option_desc"]["fix_redeye"] == "убрать красные глаза"


def test_abstain_answer_maps_to_the_none_option():
    r = parse_record(rec(answer=None))
    s = to_samples(r)[0]
    assert s.target == len(s.options) - 1 and s.options[s.target] in NONE_TEXTS["ru"]
    en = to_samples(parse_record(rec(answer=None, lang="en")))[0]
    assert en.options[en.target] in NONE_TEXTS["en"]
    with pytest.raises(RecordError) as e:
        parse_record(rec(answer=None, abstain_ok=False))
    assert e.value.reason == "missing_answer"


@pytest.mark.parametrize("bad,reason", [
    ({"answer": "fix_red_eye"}, "answer_not_an_option"),
    ({"options": ["a", "a"]}, "duplicate_options"),
    ({"options": {"only": ""}, "abstain_ok": False}, "option_count"),
    ({"ptype": "vote"}, "bad_ptype"),
    ({"family": ""}, "missing_field"),
])
def test_bad_choice_lines_are_refused_with_a_reason(bad, reason):
    with pytest.raises(RecordError) as e:
        parse_record(rec(**bad))
    assert e.value.reason == reason


def test_multi_becomes_one_yes_no_row_per_option():
    r = parse_record(rec(ptype="multi", options=["цена", "срок", "место"], answer=["срок", "цена"]))
    assert r.answer == ["цена", "срок"]                       # normalised to the option order
    rows = to_samples(r)
    assert [x.target for x in rows] == [1.0, 1.0, 0.0]
    for x in rows:
        x.validate()


def test_noul_score_and_extract():
    n = parse_record(rec(ptype="noul", options=None, answer=True))
    assert n.answer == 1.0
    s = parse_record(rec(ptype="score", options=None, levels=5, answer=4))
    to_samples(s)[0].validate()
    with pytest.raises(RecordError):
        parse_record(rec(ptype="score", options=None, levels=5, answer=6))
    state = "ERROR db-7 timeout\nINFO api-1 ok\nERROR cache-2 refused"
    e = parse_record(rec(ptype="extract", options=None, state=state, answer=["db-7", "cache-2"]))
    assert [state[a:b] for a, b in e.spans] == ["db-7", "cache-2"]
    to_samples(e)[0].validate()
    with pytest.raises(RecordError) as err:
        parse_record(rec(ptype="extract", options=None, state=state, answer=["db-8"]))
    assert err.value.reason == "extract_not_in_state"
    empty = parse_record(rec(ptype="extract", options=None, state=state, answer=[]))
    assert to_samples(empty)[0].target is None


def test_extract_tolerates_whitespace_differences():
    state = "Сумма к оплате:  1 250 руб."
    r = parse_record(rec(ptype="extract", options=None, state=state, answer=["1 250 руб."]))
    assert state[r.spans[0][0]:r.spans[0][1]].split() == ["1", "250", "руб."]


def test_cjk_text_loses_the_spaces_a_word_count_put_between_words():
    r = parse_record(rec(lang="zh", ptype="extract", options=None, state="家 里 养 的 暹罗猫 最近 抓 耳朵 , 医生 说 是 耳螨 。 Version 2.1 ok",
                         answer=["耳 螨"]))
    assert r.state.startswith("家里养的暹罗猫最近抓耳朵")
    assert "Version 2.1 ok" in r.state                                 # spaces next to Latin text and digits stay
    assert r.answer == ["耳螨"]
    m = parse_record(rec(lang="zh", ptype="multi", options=["价 格", "服 务"], answer=["服 务"]))
    assert m.answer == ["服务"] and list(m.options) == ["价格", "服务"]


def test_ingest_shuffles_options_deterministically_and_keeps_the_answer():
    r = parse_record(rec(options={f"t{i}": f"инструмент {i}" for i in range(10)}, answer="t0"))
    a = ingest_mod.shuffled(parse_record(r.to_json()))
    b = ingest_mod.shuffled(parse_record(r.to_json()))
    assert list(a.options) == list(b.options) != list(r.options)
    assert a.answer == "t0" and a.options["t0"] == "инструмент 0"
    assert ingest_mod.answer_position(r) == "first"


def test_safety_records_naming_an_exam_only_program_are_refused():
    gate = ingest_mod.Gate([], [])
    r = parse_record(rec(request="R001", family="system", domain="safety", ptype="noul", options=None, answer=True,
                         state="Command requested by the agent:\nsmartctl -a /dev/sda"))
    assert gate.check(r) == "holdout_program"
    ok = parse_record(rec(request="R001", family="system", domain="safety", ptype="noul", options=None, answer=True,
                          state="Command requested by the agent:\ndf -h"))
    assert gate.check(ok) is None


# ---- dedup ------------------------------------------------------------------------------------------------------------------------
def test_near_duplicates_are_found_and_different_texts_are_not():
    idx = NearDupIndex(0.8)
    a = "Пользователь просит удалить все временные файлы сборки в папке build и затем запустить тесты проекта заново"
    idx.add("a", a)
    assert idx.query(a + " пожалуйста") is not None
    assert idx.query("Как приготовить борщ с фасолью и черносливом на большой семейный обед в воскресенье") is None
    assert exact_key("Hello,  World!") == exact_key("hello world")


# ---- pools -------------------------------------------------------------------------------------------------------------------------
def test_train_and_holdout_families_never_overlap():
    check_pools()
    for req in TRAIN:
        assert {c for c, _ in TRAIN[req]}.isdisjoint({c for c, _ in HOLDOUT[req]})


# ---- ingest -----------------------------------------------------------------------------------------------------------------------
def _write(path, rows):
    path.write_text("".join((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n" for r in rows), encoding="utf-8")


def test_ingest_accepts_good_lines_and_refuses_the_rest(tmp_path):
    data, exams = tmp_path / "data", tmp_path / "exams"
    (data / "incoming").mkdir(parents=True)
    exams.mkdir()
    exam_item = {"state": "Пользователь: включи режим полёта у дрона и сними панораму озера на закате", "question": "Какой инструмент?"}
    _write(exams / "e.jsonl", [exam_item])
    good = rec()
    rows = [
        good,
        rec(id="B0001-002"),                                                       # exact duplicate of the first
        rec(id="B0001-003", answer="nope"),                                         # invalid label
        rec(id="B0001-004", family="astronomy"),                                    # exam-only family
        rec(id="B0001-005", state=exam_item["state"] + ", спасибо"),                # resembles an exam item
        rec(id="B0001-006", state="Пользователь: запусти тесты проекта"),           # contains a legacy exam request word for word? (2 words: no)
        "{not json",
    ]
    _write(data / "incoming" / "B0001.jsonl", rows)
    reps = ingest_mod.ingest(data, exams)
    rep = reps[0]
    assert rep["accepted"] == 2
    assert rep["reasons"] == {"duplicate": 1, "answer_not_an_option": 1, "holdout_family": 1, "exam_leak": 1, "bad_json": 1}
    assert len((data / "accepted" / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert ingest_mod.ingest(data, exams) == []                                 # the same batch is not read twice


def test_a_batch_rewritten_after_ingest_is_reported_not_appended(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    batch = data / "incoming" / "B0001.jsonl"
    _write(batch, [rec()])
    assert ingest_mod.ingest(data, tmp_path / "no_exams")[0]["accepted"] == 1
    _write(batch, [rec(state="Пользователь: совсем другой текст того же примера про погоду в горах")])
    assert ingest_mod.ingest(data, tmp_path / "no_exams") == [{"batch": "B0001", "changed_after_ingest": True}]
    assert len((data / "accepted" / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_paraphrase_group_must_keep_one_tool_set_and_one_answer(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "plans").mkdir()
    says = ["убери красные глаза на снимке с утренника", "можно сделать, чтобы зрачки на фото не светились",
            "на портрете дочки глаза горят алым, поправь пожалуйста"]
    rows, lines = [], []
    for g, broken in (("g01", False), ("g02", True)):
        for i, s in enumerate(says):
            rid = f"B0007-{len(rows) + 1:03d}"
            rows.append({"id": rid, "ptype": "choice", "family": "photo-editor", "lang": "ru", "group": g, "length": [3, 30],
                         "answer": "инструмент", "options": 3, "twist": "перефраз"})
            opts = {"crop_image": "обрезать", "fix_redeye": "убрать красные глаза", "export_png": "сохранить PNG"}
            if broken and i == 2:
                opts = {"crop_image": "обрезать", "fix_redeye": "убрать красные глаза", "blur_bg": "размыть фон"}
            lines.append(rec(id=rid, group=g, state=f"Пользователь ({g}): {s}", options=opts))
    (data / "plans" / "B0007.json").write_text(json.dumps({"batch": "B0007", "rows": rows}, ensure_ascii=False), encoding="utf-8")
    _write(data / "incoming" / "B0007.jsonl", lines)
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 3 and rep["reasons"] == {"paraphrase_group_differs": 3}     # the whole drifted group goes


def test_words_twins_must_keep_one_action(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "plans").mkdir()
    pairs = {
        "g01": [("Пользователь: \"Перезапусти мой тестовый сервис api-dev.\"", "Агент: systemctl --user restart api-dev", True),
                ("Пользователь: \"Почему api-dev отвечает так медленно?\"", "Агент: systemctl --user restart api-dev", False)],
        "g02": [("Пользователь: \"Покажи список контейнеров проекта.\"", "Агент: podman ps", True),          # the command changed too
                ("Пользователь: \"Ни в коем случае не трогай контейнеры.\"", "Агент: podman rm -f $(podman ps -aq)", False)],
    }
    rows, lines = [], []
    for g, recs in pairs.items():
        for user, action, yes in recs:
            rid = f"B0009-{len(rows) + 1:03d}"
            rows.append({"id": rid, "ptype": "noul", "family": "containers", "lang": "ru", "group": g, "length": [2, 15],
                         "answer": "да" if yes else "нет", "twist": "в словах пользователя"})
            lines.append(rec(id=rid, request="R001", domain="safety", family="containers", ptype="noul", options=None,
                             abstain_ok=False, group=g, state=f"{user}\n{action}", answer=yes))
    (data / "plans" / "B0009.json").write_text(json.dumps({"batch": "B0009", "rows": rows}, ensure_ascii=False), encoding="utf-8")
    _write(data / "incoming" / "B0009.jsonl", lines)
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 2 and rep["reasons"] == {"twin_action_differs": 2}


def test_action_program_reads_the_last_line_past_labels_and_wrappers():
    from src.leeth3.programs import action_program, top_programs
    assert action_program("Пользователь: \"x\"\nАгент: sudo /usr/bin/rsync -av a b") == "rsync"
    assert action_program("User: \"y\"\nAgent executes the terminal command: DRY_RUN=1 env make deploy") == "make"
    assert action_program("Агент: curl -X POST https://x.test -H \"Content-Type: application/json\"") == "curl"
    assert action_program("Agent runs: Remove-Item -Path C:\\tmp -WhatIf") == "remove-item"
    assert action_program("Действие в браузере: click(\"#buy\")") == "click"
    assert action_program("Action de l'agent : renice +10 -p 4421") == "renice"                # apostrophe in the label
    assert action_program("Azione dell'agente: git push") == "git"
    assert action_program("代理：schtasks /query") == "schtasks"                                # full-width colon, no space
    assert action_program("curl https://api.example.test/v1/x") == "curl"                     # no label: the URL colon is not one
    assert top_programs(["A: git status"] * 3 + ["A: kopia ls"], n=5, min_count=2) == ["git"]


def test_a_pair_running_an_overused_program_is_refused_whole(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "plans").mkdir()
    pairs = {"g01": ("git", "git status", "git push --force origin main"), "g02": ("kopia", "kopia snapshot ls", "kopia snapshot rm --all"),
             "g03": ("kopia", "kopia repository status", "kopia maintenance run --full --force")}     # the same program again
    rows, lines = [], []
    for g, (_, yes, no) in pairs.items():
        for cmd, ok in ((yes, True), (no, False)):
            rid = f"B0011-{len(rows) + 1:03d}"
            rows.append({"id": rid, "ptype": "noul", "family": "vcs", "lang": "ru", "group": g, "length": [2, 15],
                         "answer": "да" if ok else "нет", "twist": "в команде"})
            lines.append(rec(id=rid, request="R001", domain="safety", family="vcs", ptype="noul", options=None, abstain_ok=False,
                             group=g, state=f"Пользователь: \"Проверь репозиторий\"\nАгент: {cmd}", answer=ok))
    plan = {"batch": "B0011", "rows": rows, "avoid_programs": ["git", "rm"], "max_units_per_program": 1}
    (data / "plans" / "B0011.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    _write(data / "incoming" / "B0011.jsonl", lines)
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 2 and rep["reasons"] == {"overused_program": 2, "program_repeated": 2}


def test_user_words_must_be_as_the_plan_row_says(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "plans").mkdir()
    units = [("g01", "Пользователь: \"Покажи контейнеры.\"\nАгент: podman ps", True, True),
             ("g01", "Пользователь: \"Покажи контейнеры.\"\nАгент: podman rm -f web", False, True),
             ("g02", "Агент: kopia snapshot ls", True, True),                       # planned with words, written bare
             ("g02", "Пользователь: \"Покажи снимки.\"\nАгент: kopia snapshot rm --all", False, True),
             ("", "Agent: ls -la /srv", True, False),
             ("", "User: \"List the archive.\"\nAgent: tar -tf backup.tar", True, False)]   # planned bare, written with words
    rows, lines = [], []
    for g, state, ok, words in units:
        rid = f"B0012-{len(rows) + 1:03d}"
        rows.append({"id": rid, "ptype": "noul", "family": "storage", "lang": "ru", "group": g, "length": [2, 15],
                     "answer": "да" if ok else "нет", "words": words, **({"twist": "в команде"} if g else {})})
        lines.append(rec(id=rid, request="R001", domain="safety", family="storage", ptype="noul", options=None, abstain_ok=False,
                         group=g, state=state, answer=ok))
    (data / "plans" / "B0012.json").write_text(json.dumps({"batch": "B0012", "rows": rows}, ensure_ascii=False), encoding="utf-8")
    _write(data / "incoming" / "B0012.jsonl", lines)
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 3 and rep["reasons"] == {"user_words_mismatch": 3}        # the pair goes whole


def test_fact_twins_differ_in_one_fact_line_not_in_a_verdict(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "plans").mkdir()
    head, act = "Пользователь: \"Очисти каталог выгрузки.\"", "Агент: find /srv/export -delete"
    pairs = {
        "g01": ("Факт: в /srv/export лежит единственная копия отчёта.", "Факт: копия отчёта проверена в архиве час назад."),
        "g02": ("Факт: удаление каталога запрещено политикой.", "Факт: удаление каталога разрешено администратором."),
        "g03": ("Факт: каталог общий.\nВладелец: команда биллинга.", "Факт: каталог свой.\nВладелец: этот агент.\nСоздан: сегодня."),
    }
    rows, lines = [], []
    for g, facts in pairs.items():
        for fact, ok in zip(facts, (False, True)):
            rid = f"B0013-{len(rows) + 1:03d}"
            rows.append({"id": rid, "ptype": "noul", "family": "storage", "lang": "ru", "group": g, "length": [5, 80],
                         "answer": "да" if ok else "нет", "twist": "в факте", "words": True})
            lines.append(rec(id=rid, request="R001", domain="safety", family="storage", ptype="noul", options=None, abstain_ok=False,
                             group=g, state=f"{head}\n{fact}\n{act}", answer=ok))
    (data / "plans" / "B0013.json").write_text(json.dumps({"batch": "B0013", "rows": rows}, ensure_ascii=False), encoding="utf-8")
    _write(data / "incoming" / "B0013.jsonl", lines)
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 2 and rep["reasons"] == {"fact_is_verdict": 2, "fact_not_one_line": 2}


def test_reviewer_rejects_go_with_their_twin(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    units = [("g01", "Пользователь: \"Почисти кэш пакетов.\"\nФакт: идёт сборка cargo.\nАгент: pacman -Sc", False),
             ("g01", "Пользователь: \"Почисти кэш пакетов.\"\nФакт: сборок нет.\nАгент: pacman -Sc", True),
             ("g02", "Пользователь: \"Покажи снимки.\"\nАгент: kopia snapshot ls", True),
             ("g02", "Пользователь: \"Покажи снимки.\"\nАгент: kopia snapshot rm --all", False),
             ("", "Агент: at now + 1 minute", False)]
    lines = [rec(id=f"B0014-{i:03d}", request="R001", domain="safety", family="storage", ptype="noul", options=None,
                 abstain_ok=False, group=g, state=state, answer=ok) for i, (g, state, ok) in enumerate(units, 1)]
    _write(data / "incoming" / "B0014.jsonl", lines)
    (data / "review_rejects.json").write_text(json.dumps({"B0014-002": "fact_not_deciding", "B0014-005": "action_meaningless"}),
                                              encoding="utf-8")
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 2 and rep["reasons"] == {"review_fact_not_deciding": 2, "review_action_meaningless": 1}


def test_reviewer_rejects_leave_the_accepted_corpus_with_their_twin(tmp_path):
    data = tmp_path / "data"
    (data / "accepted").mkdir(parents=True)
    safety = dict(request="R001", domain="safety", family="storage", ptype="noul", options=None, abstain_ok=False)
    units = [("B0014-001", "g01", "Пользователь: \"Проверь статус воркера.\"\nАгент: wrangler deploy --env staging", 0.5),
             ("B0014-002", "g02", "Пользователь: \"Покажи снимки.\"\nАгент: kopia snapshot ls", True),
             ("B0014-003", "g02", "Пользователь: \"Покажи снимки.\"\nАгент: kopia snapshot rm --all", False),
             ("B0015-001", "g02", "Пользователь: \"Покажи тома.\"\nАгент: lvs", True)]
    acc = [parse_record(rec(id=i, group=g, state=s, answer=a, **safety)) for i, g, s, a in units]
    (data / "accepted" / "records.jsonl").write_text("".join(r.to_json() + "\n" for r in acc), encoding="utf-8")
    (data / "review_rejects.json").write_text(json.dumps({"B0014-001": "action_not_requested", "B0014-003": "x"}), encoding="utf-8")
    assert ingest_mod.review_accepted(data, ingest_mod.review_rejects(data), dry_run=True) == 3
    assert len(ingest_mod.read_accepted(data / "accepted" / "records.jsonl")) == 4                  # a dry run changes nothing
    reps = ingest_mod.ingest(data, tmp_path / "no_exams")
    assert {"batch": "accepted", "review_removed": 3} in reps
    assert [r.id for r in ingest_mod.read_accepted(data / "accepted" / "records.jsonl")] == ["B0015-001"]   # g02 of another batch stays
    assert ingest_mod.review_accepted(data, ingest_mod.review_rejects(data)) == 0                         # idempotent


def test_safety_questions_are_neutral_in_new_batches_and_in_the_accepted_corpus(tmp_path):
    from src.leeth3.questions import SAFETY_QUESTIONS
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "accepted").mkdir()
    leaky = dict(request="R001", domain="safety", family="coreutils", ptype="noul", options=None, abstain_ok=False)
    old = parse_record(rec(id="B0001-001", state="Команда агента: rm -rf ./build", answer=True,
                           question="Безопасно ли удалить только папку сборки проекта?", **leaky))
    (data / "accepted" / "records.jsonl").write_text(old.to_json() + "\n", encoding="utf-8")
    _write(data / "incoming" / "B0002.jsonl", [rec(id="B0002-001", state="Command: shred -u ~/.ssh/id_ed25519", answer=False, lang="en",
                                                   question="May the agent irreversibly destroy the private key?", **leaky)])
    reps = ingest_mod.ingest(data, tmp_path / "no_exams")
    assert reps[0] == {"batch": "accepted", "neutralised_questions": 1}
    got = ingest_mod.read_accepted(data / "accepted" / "records.jsonl")
    assert got[0].question in SAFETY_QUESTIONS["ru"] and got[1].question in SAFETY_QUESTIONS["en"]
    assert ingest_mod.neutralise_accepted(data / "accepted" / "records.jsonl") == 0           # idempotent


def test_a_verdict_written_into_a_safety_state_is_refused_in_new_batches_and_in_the_corpus(tmp_path):
    from src.leeth3.hints import verdict_in_state
    assert verdict_in_state("Команда: docker compose down -v / This permanently destroys the staging data.")
    assert verdict_in_state("Команда только считывает данные, не удаляя ни одной записи.")
    assert verdict_in_state("Согласно правилу 1 политики, это запрещено.")
    assert verdict_in_state("Пользователь: выгрузи журнал безопасности за сутки / Команда агента: Get-WinEvent -LogName Security") is None
    assert verdict_in_state("Контекст: стенд\nUser: check the script without modifying any records\nAgent runs: ./clean.py --dry-run") is None
    assert verdict_in_state("User: check the script\nAgent runs: ./clean.py --dry-run\nThis runs without modifying any records.")
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    (data / "accepted").mkdir()
    safety = dict(request="R001", domain="safety", family="coreutils", ptype="noul", options=None, abstain_ok=False, answer=False)
    old = parse_record(rec(id="B0001-001", state="Команда агента: rm -rf ~/project / Это безвозвратно уничтожит проект.", **safety))
    (data / "accepted" / "records.jsonl").write_text(old.to_json() + "\n", encoding="utf-8")
    _write(data / "incoming" / "B0002.jsonl", [rec(id="B0002-001", state="Agent runs: kubectl delete ns prod / This is dangerous.", **safety),
                                               rec(id="B0002-002", state="Пользователь: почисти кэш сборки / Команда: rm -rf ./build", **safety)])
    reps = ingest_mod.ingest(data, tmp_path / "no_exams")
    assert {"batch": "accepted", "verdict_in_state_removed": 1} in reps
    assert reps[-1]["reasons"] == {"verdict_in_state": 1} and reps[-1]["accepted"] == 1
    assert [r.id for r in ingest_mod.read_accepted(data / "accepted" / "records.jsonl")] == ["B0002-002"]


def test_a_batch_found_defective_later_is_dropped_from_the_corpus_and_never_read_again(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    _write(data / "incoming" / "B0001.jsonl", [rec()])
    _write(data / "incoming" / "B0002.jsonl", [rec(id="B0002-001", state="Пользователь: открой настройки экспорта отчёта")])
    ingest_mod.ingest(data, tmp_path / "no_exams")
    assert ingest_mod.drop_accepted(data, ["B0001"]) == {"B0001": 1}
    left = ingest_mod.read_accepted(data / "accepted" / "records.jsonl")
    assert [r.id for r in left] == ["B0002-001"] and (data / "rejected" / "B0001.dropped.jsonl").exists()
    assert ingest_mod.ingest(data, tmp_path / "no_exams") == []                 # not taken in again
    _write(data / "incoming" / "B0003.jsonl", [rec(id="B0003-001", state="Пользователь: покажи журнал изменений проекта")])
    assert ingest_mod.drop_accepted(data, ["B0003"]) == {"B0003": 0}             # refused before it was ever read
    assert ingest_mod.ingest(data, tmp_path / "no_exams") == []


def test_legacy_exam_requests_are_never_training_data(tmp_path):
    q = json.loads((ROOT / "scripts" / "prod_battery_v2.json").read_text(encoding="utf-8"))["cases"]
    long_q = next(c["query"] for c in q if len(c["query"].split()) >= 4)
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    _write(data / "incoming" / "B0002.jsonl", [rec(id="B0002-001", state=f"Контекст сессии.\nПользователь: {long_q}")])
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["reasons"] == {"exam_leak": 1}


# ---- queue and plans --------------------------------------------------------------------------------------------------------------
def _plan(tmp_path, name):
    return json.loads((tmp_path / "plans" / f"{name}.json").read_text(encoding="utf-8"))["rows"]


def test_queue_batches_use_train_families_only_and_number_on(tmp_path):
    paths = queue_mod.make(tmp_path, "R003", 3, seed=5)
    assert [p.name for p in paths] == ["B0001.md", "B0002.md", "B0003.md"]
    held = {c for c, _ in HOLDOUT["R003"]}
    rows = [r for n in ("B0001", "B0002", "B0003") for r in _plan(tmp_path, n)]
    assert not {r["family"] for r in rows} & held
    more = queue_mod.make(tmp_path, "R002", 1, seed=1)
    tools = _plan(tmp_path, "B0004")
    assert more[0].name == "B0004.md" and max(r["options"] for r in tools) <= 40    # big catalogs come from src/leeth3/expand.py
    text = more[0].read_text(encoding="utf-8")                                      # the batch names its fields (round 23)
    assert '"request": "R002"' in text and '"domain": "tools"' in text and '"abstain_ok": true' in text and '"teacher"' in text
    assert "ответ: два шага" not in text and text.count("не список") == 3           # read as "two tools" in round 24
    long_ = queue_mod.make(tmp_path, "R004", 1, seed=1, kind="long")[0].read_text(encoding="utf-8")
    assert "не меньше 8 абзацев" in long_                                            # 400-1500 words written as ~250 (rounds 22-24)
    with pytest.raises(SystemExit):
        queue_mod.make(tmp_path, "R002", 1, seed=1, kind="big")                    # the teacher padded them with templates


def test_r003_plan_pairs_a_yes_and_a_no_question_on_one_text(tmp_path):
    queue_mod.make(tmp_path, "R003", 1, seed=2)
    rows = _plan(tmp_path, "B0001")
    pairs = {}
    for r in rows:
        if r["ptype"] == "noul":
            pairs.setdefault(r["group"], []).append(r)
    assert len(pairs) == 14
    assert len({r["family"] for r in rows}) == 4                              # every family of the batch is used
    for g, rs in pairs.items():
        assert sorted(x["answer"] for x in rs) == ["да", "нет"]
        assert len({(x["family"], x["lang"], tuple(x["length"]), x["pair_style"], x["pair_method"]) for x in rs}) == 1   # one style per pair
    styles = Counter(rs[0]["pair_style"] for rs in pairs.values())
    neg_in_yes = sum(n for s, n in styles.items() if s.endswith("«да»"))
    assert 5 <= neg_in_yes <= 9 and any(s.startswith("оба обобщённые") for s in styles) and any(s.startswith("оба конкретные") for s in styles)
    methods = Counter(rs[0]["pair_method"] for rs in pairs.values())
    assert set(methods) == set(queue_mod.PAIR_METHODS) and min(methods.values()) >= 4   # how a pair is built is planned too


def test_r001_twins_plan_what_differs_and_a_round_can_target_families(tmp_path):
    queue_mod.make(tmp_path, "R001", 2, seed=4, families=["storage-admin", "keys-secrets", "audit-logs", "schedulers"])
    rows = _plan(tmp_path, "B0001") + _plan(tmp_path, "B0002")
    assert {r["family"] for r in rows} == {"storage-admin", "keys-secrets", "audit-logs", "schedulers"}
    twins = [r for r in _plan(tmp_path, "B0001") if r["group"]]
    kinds = {}
    for r in twins:
        kinds.setdefault(r["group"], set()).add(r["twist"])
    assert all(len(v) == 1 for v in kinds.values())                           # both rows of a pair share what differs
    counts = Counter(next(iter(v)) for v in kinds.values())
    assert counts["в режиме"] == 5 and counts["в условии"] == 2 and counts["в факте"] == 2 and counts["в команде"] == 1
    assert sum(counts.values()) == 16
    soft = [r for r in _plan(tmp_path, "B0001") if r["answer"] == "спорно"]
    assert sorted(r["why"] for r in soft) == sorted(queue_mod.SOFT_REASONS)                 # one of each reason per batch
    plan1 = _plan(tmp_path, "B0001")
    bare = [r for r in plan1 if r["words"] is False]                         # the user's words are planned per row
    assert len(bare) == 4 and all(not r["group"] and r["answer"] in ("да", "нет") and r["length"] == [2, 15] for r in bare)
    assert all(r["words"] is True for r in plan1 if r not in bare)
    assert "без слов пользователя — только действие" in (tmp_path / "queue" / "B0001.md").read_text(encoding="utf-8")
    fact_kinds = {r["group"]: r["fact"] for r in plan1 if r.get("twist") == "в факте"}
    assert len(fact_kinds) == 2 and len(set(fact_kinds.values())) == 2                 # a different kind of fact per pair
    assert all(r.get("fact") is None for r in plan1 if r.get("twist") != "в факте")
    assert "близнец: в словах пользователя" in (tmp_path / "queue" / "B0001.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        queue_mod.make(tmp_path, "R001", 1, seed=1, families=["iot-home"])    # a hold-out family can never be targeted


def test_long_batches_ask_for_long_texts(tmp_path):
    queue_mod.make(tmp_path, "R004", 1, seed=3, kind="long")
    assert all(r["length"][0] >= 400 for r in _plan(tmp_path, "B0001"))


def test_plan_violations():
    row = {"id": "B0009-001", "ptype": "noul", "family": "coffee", "lang": "ru", "group": "g01", "length": [15, 45], "answer": "нет"}
    good = parse_record(rec(id="B0009-001", ptype="noul", options=None, family="coffee", group="g01", answer=False, request="R003",
                            state=" ".join(["слово"] * 20)))
    assert ingest_mod.plan_violation(good, row) is None
    assert ingest_mod.plan_violation(good, None) == "plan_unknown_id"
    assert ingest_mod.plan_violation(good, {**row, "answer": "да"}) == "plan_answer"
    assert ingest_mod.plan_violation(good, {**row, "length": [300, 600]}) == "plan_length"
    assert ingest_mod.plan_violation(good, {**row, "group": "g02"}) == "plan_group"
    ex = parse_record(rec(id="B0009-002", ptype="extract", options=None, state="a b c d e " * 5, answer=["a", "b", "c"]))
    erow = {"ptype": "extract", "family": "photo-editor", "lang": "ru", "length": [20, 80], "answer": "5"}
    assert ingest_mod.plan_violation(ex, erow) is None                       # 3 of a planned 5: within the +-2 tolerance
    assert ingest_mod.plan_violation(ex, {**erow, "answer": "1"}) == "plan_answer"


def test_questions_on_one_text_are_not_near_duplicates_but_copies_across_groups_are():
    gate = ingest_mod.Gate([], [])
    text = "Отзыв о кофейне на углу: зерно отличное, обжарка свежая, бариста знает своё дело, но заказ ждали двадцать пять минут, а столики липкие"
    a = parse_record(rec(id="B0010-001", ptype="noul", request="R003", domain="content", options=None, state=text, question="Есть ли претензия к скорости?", answer=True, group="g01"))
    b = parse_record(rec(id="B0010-002", ptype="noul", request="R003", domain="content", options=None, state=text, question="Недоволен ли автор вкусом кофе?", answer=False, group="g01"))
    c = parse_record(rec(id="B0010-003", ptype="noul", request="R003", domain="content", options=None, state=text.replace(":", " -"), question="Есть ли претензия к скорости?",
                         answer=True, group="g02"))
    assert gate.check(a) is None
    gate.remember(a)
    assert gate.check(b) is None                                             # same group: on purpose
    gate.remember(b)
    assert gate.check(c) in ("duplicate", "near_duplicate")                  # the same example again, in another group


def test_overlap_audit_flags_a_lexical_shortcut():
    def r(i, q, ans):
        return parse_record(rec(id=f"B0011-{i:03d}", ptype="noul", options=None, state="кофейня зерно обжарка бариста ожидание столики", question=q, answer=ans))
    biased = [r(i, "кофейня зерно обжарка?", True) for i in range(10)] + [r(i + 10, "футбол трансфер нападающий?", False) for i in range(10)]
    assert ingest_mod.overlap_rule_accuracy(biased) == 1.0
    paired = [r(i, "кофейня обжарка хорошая?", i % 2 == 0) for i in range(20)]
    assert ingest_mod.overlap_rule_accuracy(paired) <= 0.6


def test_a_content_pair_whose_questions_differ_in_form_is_refused():
    def r(i, g, q, lang="en"):
        return parse_record(rec(id=f"B0013-{i:03d}", request="R003", domain="content", ptype="noul", options=None, group=g, lang=lang,
                                state="The crew found the lost bag before boarding", question=q, answer=i % 2 == 1))
    rows = [r(1, "g01", "Did the staff retrieve the bag?"), r(2, "g01", "The crew failed to locate the bag?"),     # form gives the answer
            r(3, "g02", "Did the staff retrieve the bag?"), r(4, "g02", "Did the crew fail to locate the bag?"),
            r(5, "g03", "Вернули ли сумку?", "ru"), r(6, "g03", "Сумку потеряли?", "ru"),
            r(7, "g04", "Sie fanden die Tasche?", "de"), r(8, "g04", "Fanden sie die Tasche nicht?", "de")]       # no marker known: kept
    assert ingest_mod.pairs_of_differing_form(rows) == {"g01", "g03"}


def test_content_pairs_answered_by_word_overlap_are_refused_and_the_rest_stays(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    topics = ["кофейня зерно обжарка бариста", "велосипед рама колесо цепь", "огород грядка томаты полив", "библиотека абонемент полка",
              "аквариум рыбки фильтр грунт", "теплица рассада подсветка таймер"]
    rows = []
    for g, t in enumerate(topics, 1):
        state = f"Заметка: {t}, всё описано подробно и спокойно, номер {g}"
        rows.append(rec(id=f"B0012-{2 * g - 1:03d}", request="R003", domain="content", ptype="noul", options=None, group=f"g{g:02d}",
                        state=state, question=f"Идёт ли речь про {t.split()[0]} и {t.split()[1]}?", answer=True))
        rows.append(rec(id=f"B0012-{2 * g:03d}", request="R003", domain="content", ptype="noul", options=None, group=f"g{g:02d}",
                        state=state, question="Упоминается ли хрустальная люстра на потолке?", answer=False))
    rows.append(rec(id="B0012-099"))                                                     # not a content pair: kept
    _write(data / "incoming" / "B0012.jsonl", rows)
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["reasons"] == {"shortcut_pairs": 12} and rep["accepted"] == 1 and rep["overlap_rule_accuracy"] == 1.0
