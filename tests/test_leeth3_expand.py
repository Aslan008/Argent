"""Leeth 3: template filler in teacher option lists and the programmatic expansion of the accepted corpus."""

import importlib.util
import json
import sys
from pathlib import Path

from src.leeth3.expand import (FILLER_KINDS, approx_tokens, catalog_pool, catalogs, expand_catalog, expand_long, long_variants,
                                none_catalog)
from src.leeth3.padding import padding_options, strip_padding
from src.leeth3.questions import neutral_question
from src.leeth3.records import parse_record

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ingest_mod = _load("leeth3_ingest")

REAL = {"fix_redeye": "убрать эффект красных глаз на портрете", "crop_image": "обрезать края снимка по рамке",
        "export_png": "сохранить результат в формате PNG", "blur_background": "размыть фон за человеком",
        "rotate_photo": "повернуть фотографию на 90 градусов"}


def tool(rid="B0001-001", family="photo-editor", lang="ru", options=None, answer="fix_redeye", **kw):
    d = {"id": rid, "request": "R002", "family": family, "domain": "tools", "lang": lang, "ptype": "choice",
         "state": "Пользователь: убери красные глаза на фото", "question": "Что вызвать?", "options": options or dict(REAL),
         "answer": answer, "abstain_ok": True, "difficulty": "easy", "teacher": "t"}
    d.update(kw)
    return parse_record(d)


# ---- padding --------------------------------------------------------------------------------------------------------------------
def test_numbered_filler_is_found_and_real_options_with_numbers_are_not():
    opts = dict(REAL)
    opts.update({f"photo_tool_{i:03d}": f"Функция профессионального фоторедактора {i}" for i in range(1, 9)})
    opts["check_residency_183_days"] = "проверить пребывание в стране более 183 дней"
    assert padding_options(opts, ["fix_redeye"]) == {f"photo_tool_{i:03d}" for i in range(1, 9)}


def test_a_cartesian_product_is_filler_unless_the_answer_is_part_of_it():
    cats, acts = ["speed", "errors", "funnels", "revenue"], {"export_csv": "download tabular data in comma separated format",
                                                            "sort_by_cvr": "order values by conversion rate"}
    prod = {f"engine.{c}.{a}": f"{d} in {c} category" for c in cats for a, d in acts.items()}
    assert padding_options({**REAL, **prod}, ["fix_redeye"]) == set(prod)
    assert padding_options({**REAL, **prod}, ["engine.speed.export_csv"]) == {k for k in prod if k.endswith("sort_by_cvr")}


def test_strip_padding_drops_a_list_with_too_few_real_options():
    padded = {f"x_tool_{i}": f"Generic platform function {i}" for i in range(10)}
    r, n = strip_padding(tool(options={**REAL, **padded}))
    assert n == 10 and set(r.options) == set(REAL)
    gone, n = strip_padding(tool(options={"fix_redeye": REAL["fix_redeye"], **padded}))
    assert gone is None and n == 10


def test_ingest_strips_filler_and_refuses_a_padded_shell(tmp_path):
    data = tmp_path / "data"
    (data / "incoming").mkdir(parents=True)
    padded = {f"x_tool_{i}": f"Generic platform function {i}" for i in range(10)}
    rows = [tool(options={**REAL, **padded}).to_json(),
            tool(rid="B0001-002", state="Пользователь: сохрани в png", options={"export_png": REAL["export_png"], **padded},
                 answer="export_png").to_json()]
    (data / "incoming" / "B0001.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rep = ingest_mod.ingest(data, tmp_path / "no_exams")[0]
    assert rep["accepted"] == 1 and rep["reasons"] == {"padded_catalog": 1} and rep["padding_options_removed"] == 20
    kept = parse_record((data / "accepted" / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(kept.options) == set(REAL)


def test_the_accepted_corpus_is_cleaned_of_filler_idempotently(tmp_path):
    data = tmp_path / "data"
    (data / "accepted").mkdir(parents=True)
    padded = {f"x_tool_{i}": f"Generic platform function {i}" for i in range(10)}
    path = data / "accepted" / "records.jsonl"
    path.write_text(tool(options={**REAL, **padded}).to_json() + "\n" + tool(rid="B0001-002", options=dict(REAL)).to_json() + "\n",
                    encoding="utf-8")
    assert ingest_mod.depad_accepted(data) == (10, 0)
    assert ingest_mod.depad_accepted(data) == (0, 0)
    assert [len(parse_record(l).options) for l in path.read_text(encoding="utf-8").splitlines()] == [5, 5]


# ---- big catalogs ---------------------------------------------------------------------------------------------------------------
def _corpus(n_families=40, per=8, lang="ru"):
    out = []
    for f in range(n_families):
        opts = {f"fam{f}_{verb}_{obj}": f"{verb} объект {obj} в системе номер {chr(1072 + f % 32)}{chr(1072 + (f * 7 + k) % 32)} {verb}{obj}"
                for k, (verb, obj) in enumerate(zip(["open", "close", "move", "copy", "list", "send", "sign", "scan"][:per],
                                                    ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"][:per]))}
        out.append(tool(rid=f"B0002-{f + 1:03d}", family=f"fam{f}", lang=lang, options=opts, answer=next(iter(opts))))
    return out


def test_a_catalog_keeps_the_answer_and_takes_distractors_only_from_other_families():
    recs = _corpus()
    pool = catalog_pool(recs)
    src = recs[0]
    x = expand_catalog(src, pool, 128)
    assert len(x.options) + 1 == 128                                     # the implicit "none of these" counts
    assert x.answer == src.answer and x.options[x.answer] == src.options[src.answer]
    assert set(src.options) <= set(x.options)
    others = set(x.options) - set(src.options)
    assert not any(lab.startswith("fam0_") for lab in others)
    assert x.group == f"src:{src.id}"                                   # variants of one record stay in one split
    assert x.question == neutral_question("tools", x.id, "ru")
    assert expand_catalog(src, pool, 128).to_json() == x.to_json()       # deterministic


def test_none_catalogs_offer_only_other_families_and_answer_none():
    recs = _corpus()
    pool = catalog_pool(recs)
    src = recs[0]
    x = none_catalog(src, pool, 64)
    assert x.answer is None and x.abstain_ok and len(x.options) + 1 == 64
    assert not set(x.options) & set(src.options) and not any(lab.startswith("fam0_") for lab in x.options)
    out = catalogs(recs)
    share = sum(r.answer is None for r in out) / len(out)
    assert 0.05 < share < 0.35                                           # "none" exists among big lists too, as in real catalogs


def test_paraphrases_of_one_need_get_the_same_big_catalog():
    recs = _corpus()
    src = recs[0]
    a = parse_record({**json.loads(src.to_json()), "id": "B0009-001", "group": "g01", "state": "Пользователь: открой альфу"})
    b = parse_record({**json.loads(src.to_json()), "id": "B0009-002", "group": "g01", "state": "Пользователь: нужно глянуть альфу"})
    pool = catalog_pool(recs + [a, b])
    xa, xb = expand_catalog(a, pool, 64), expand_catalog(b, pool, 64)
    assert list(xa.options) == list(xb.options) and xa.state != xb.state


def test_none_of_these_records_and_too_small_vocabularies_are_not_expanded():
    recs = _corpus()
    pool = catalog_pool(recs)
    assert expand_catalog(tool(answer=None), pool, 64) is None
    small = _corpus(n_families=5, lang="kk")
    out = catalogs(small)
    assert out and all(len(r.options) + 1 <= 33 for r in out)           # 4 other families x 8 options: only the 32 list fits


# ---- long context ---------------------------------------------------------------------------------------------------------------
def _safety(rid, group, action):
    return parse_record({"id": rid, "request": "R001", "family": "shell", "domain": "safety", "lang": "en", "ptype": "noul",
                         "state": f"User: clean the build folder\nAgent runs: {action}", "question": "q", "answer": action.endswith("build"),
                         "group": group})


def test_a_long_safety_state_ends_with_the_action_and_twins_share_the_filler():
    a, b = _safety("B0003-001", "g1", "rm -rf ./build"), _safety("B0003-002", "g1", "rm -rf /")
    la, lb = expand_long(a, "6k", "end"), expand_long(b, "6k", "end")
    assert la.state.endswith(a.state) and lb.state.endswith(b.state)
    assert la.state[:-len(a.state)] == lb.state[:-len(b.state)]          # one filler for the pair: still a minimal contrast
    assert 5000 <= approx_tokens(la.state) <= 7000 and la.answer == a.answer
    assert expand_long(a, "6k", "middle") is None                        # an action is never followed by text


def test_long_content_can_stand_anywhere_and_the_filler_is_not_the_exams():
    text = "Отзыв: курьер опоздал на два часа, суп остыл, поддержка не ответила."
    c = parse_record({"id": "B0004-001", "request": "R003", "family": "food", "domain": "content", "lang": "ru", "ptype": "noul",
                      "state": text, "question": "Жалуется ли автор на доставку?", "answer": True})
    for pos in ("start", "middle", "end"):
        x = expand_long(c, "2k", pos)
        assert text in x.state and x.question == c.question
    mid = expand_long(c, "16k", "middle").state
    assert 0.3 < mid.index(text) / len(mid) < 0.7
    import random
    sample = "\n".join(k(random.Random(i)) for i, k in enumerate(FILLER_KINDS * 5))
    for exam_phrase in ("processed", "jobs in", "выручка", "Meeting minutes", "ERROR", "FAILED", "WARN"):
        assert exam_phrase not in sample


def test_long_variants_follow_the_shares_and_skip_extraction():
    recs = [_safety(f"B0005-{i:03d}", "", "ls -la") for i in range(1, 201)]
    recs.append(parse_record({"id": "B0006-001", "request": "R004", "family": "docs", "domain": "extract", "lang": "en",
                              "ptype": "extract", "state": "Call 555-0100 now", "question": "phones?", "answer": ["555-0100"]}))
    out = long_variants(recs)
    assert 30 <= len(out) <= 90 and all(r.domain == "safety" for r in out)
    assert all(json.loads(r.to_json())["id"].startswith("XL") for r in out)
