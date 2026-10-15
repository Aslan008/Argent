"""Form planning (src/leeth/webforms.py), bulk classification (src/leeth/bulk.py) and the tools built on them -- with a scripted stand-in for
the model, so what is tested is the logic around it (claims, units, options, guards), not the weights."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.leeth import bulk, webforms  # noqa: E402


class FakeRT:
    """decide() answers extraction / yes-no questions from tables keyed by a substring of the question."""

    def __init__(self, extract=None, yesno=None, choices=None, many_yes=None):
        self.extract = extract or {}
        self.yesno = yesno or {}
        self.choices = choices or {}
        self.many_yes = many_yes or {}
        self.asked = []

    def decide(self, state, questions):
        self.asked.append([q.question for q in questions])
        out = {}
        for q in questions:
            if hasattr(q, "choices"):
                out[q.id] = self._choice(state, q.question, q.choices)
            elif any(k in q.question for k in self.yesno):
                p = next(v for k, v in self.yesno.items() if k in q.question)
                out[q.id] = SimpleNamespace(value=p >= 0.5, probability=p)
            else:
                key = next((k for k in self.extract if k in q.question), None)
                text, conf = self.extract.get(key, ("", 0.4))
                a = state.find(text) if text else -1
                out[q.id] = SimpleNamespace(text=text, span=(a, a + len(text)) if a >= 0 else (0, 0), confidence=conf)
        return out

    def _choice(self, state, question, options):
        probs = {o: 0.01 for o in options}
        pick = self.choices.get("pick")
        if pick in options:
            probs[pick] = 0.9
        else:
            probs[options[0]] = 0.9
        best = max(probs, key=probs.get)
        return SimpleNamespace(choice=best, confidence=probs[best], probabilities=probs)

    def choose(self, state, question, options):
        self.asked.append([question])
        return self._choice(state, question, list(options))

    def judge_many(self, states, question):
        return [SimpleNamespace(probability=self.many_yes.get(s, 0.5)) for s in states]

    def choose_many(self, states, question, options):
        out = []
        for s in states:
            pick = self.many_yes.get(s, options[0])
            probs = {o: (0.8 if o == pick else 0.2 / max(1, len(options) - 1)) for o in options}
            out.append(SimpleNamespace(choice=pick, confidence=0.8, probabilities=probs))
        return out


def el(idx, tag, label, value="", checked=None, disabled=False, href=""):
    return {"idx": idx, "tag": tag, "label": label, "value": value, "checked": checked, "disabled": disabled, "href": href}


# ------------------------------------------------------------------------------------------------------------------ fields
def test_parse_fields_keeps_the_fillable_controls_only():
    els = [el(1, "input[type=text]", "Логин"), el(2, "input[type=number]", "Рост"), el(3, "select", "Город", "Москва"), el(4, "input[type=checkbox]", "Рассылка", checked=False),
           el(5, "input[type=hidden]", "csrf"), el(6, "button", "Отправить"), el(7, "input[type=password]", "Пароль"), el(8, "input[type=text]", "Код", disabled=True),
           el(9, "textarea", "Комментарий"), el(10, "a", "Главная", href="/"), el(11, "input[type=submit]", "Go")]
    fs = {f.index: f for f in webforms.parse_fields(els)}
    assert set(fs) == {1, 2, 3, 4, 7, 9}
    assert fs[2].kind == "number" and fs[3].kind == "select" and fs[4].kind == "checkbox" and fs[7].kind == "password" and fs[9].kind == "textarea"
    assert fs[4].checked is False and fs[3].value == "Москва"


def test_buttons_lists_buttons_and_submits():
    els = [el(1, "button", "Сохранить"), el(2, "input[type=submit]", "Отправить"), el(3, "a", "Ссылка"), el(4, "button", "")]
    assert webforms.buttons(els) == [(1, "Сохранить"), (2, "Отправить")]


@pytest.mark.parametrize("raw,kind,want", [("175 см", "text", "175"), ("29 лет", "number", "29"), ("Qwerty123!,", "password", "Qwerty123!"), ("+7 999 123-45-67", "text", "+7 999 123-45-67"),
                                            ("ivan@mail.ru", "text", "ivan@mail.ru"), ("12,5 кг", "number", "12.5"), ("42", "number", "42"), ("Валера", "text", "Валера"),
                                            ("5 Baker Street", "text", "5 Baker Street")])
def test_clean_value(raw, kind, want):
    assert webforms.clean_value(raw, kind) == want


def test_match_option_exact_contains_and_placeholders():
    opts = ["— выберите —", "Москва", "Hong Kong", "Hong Kong Island", "Токио"]
    assert webforms.match_option("москва", opts) == ("exact", "Москва")
    assert webforms.match_option("токио", opts) == ("exact", "Токио")
    assert webforms.match_option("Hong Kong", opts) == ("exact", "Hong Kong")
    assert webforms.match_option("Island", opts) == ("contains", "Hong Kong Island")
    assert webforms.match_option("Гонконг", opts) is None  # a translation is Leeth's job, not a string match
    assert webforms.match_option("выберите", ["— выберите —"]) is None


# ------------------------------------------------------------------------------------------------------------------ plan
INSTR = "Введи имя в поле логина Валера, рост 175 см, возраст 29 лет, выбери в выпадающем списке город Гонконг"


def _form():
    return webforms.parse_fields([el(1, "input[type=text]", "Логин"), el(2, "input[type=number]", "Рост"), el(3, "input[type=text]", "Возраст"), el(4, "select", "Город")])


def test_the_example_instruction_fills_four_fields_in_one_plan():
    rt = FakeRT(extract={"«Логин»": ("Валера", 0.88), "«Рост»": ("175 см", 0.93), "«Возраст»": ("29 лет", 0.94), "«Город»": ("Гонконг", 0.99)},
                choices={"pick": "Hong Kong"})
    plan = webforms.plan_fill(rt, INSTR, _form(), {4: ["Москва", "Hong Kong", "Токио"]})
    got = {s.label: (s.kind, s.value) for s in plan.steps}
    assert got == {"Логин": ("type", "Валера"), "Рост": ("type", "175"), "Возраст": ("type", "29"), "Город": ("select", "Hong Kong")}
    assert len(rt.asked[0]) == 4, "one batched pass: a question per field"
    assert plan.language == "ru" and not plan.skipped


def test_a_value_claimed_by_two_fields_goes_to_the_more_confident_one():
    form = webforms.parse_fields([el(1, "input[type=text]", "Name"), el(2, "input[type=text]", "Company")])
    rt = FakeRT(extract={"'Name'": ("Anna", 0.99), "'Company'": ("Anna", 0.7)})
    plan = webforms.plan_fill(rt, "Please enter the name Anna, leave the rest empty", form)
    assert [(s.label, s.value) for s in plan.steps] == [("Name", "Anna")]
    assert plan.skipped[0]["label"] == "Company" and "more confident" in plan.skipped[0]["reason"]
    assert plan.language == "en"


def test_weak_and_empty_answers_are_reported_not_entered():
    form = webforms.parse_fields([el(1, "input[type=text]", "Email"), el(2, "input[type=text]", "Телефон"), el(3, "input[type=text]", "Имя")])
    rt = FakeRT(extract={"«Email»": ("", 0.47), "«Телефон»": ("999", 0.3), "«Имя»": ("Иван", 0.99)})
    plan = webforms.plan_fill(rt, "Введи имя Иван, 999", form)
    assert [s.label for s in plan.steps] == ["Имя"]
    reasons = {s["label"]: s["reason"] for s in plan.skipped}
    assert "no value" in reasons["Email"] and "not sure" in reasons["Телефон"]


def test_a_dropdown_takes_the_option_that_contains_the_value_without_asking_the_model():
    form = webforms.parse_fields([el(1, "select", "Country")])
    rt = FakeRT(extract={"'Country'": ("Germany", 1.0)})
    plan = webforms.plan_fill(rt, "country Germany", form, {1: ["Select...", "France", "Germany", "Spain"]})
    assert [(s.kind, s.value, s.note) for s in plan.steps] == [("select", "Germany", "exact match")]
    assert len(rt.asked) == 1  # only the extraction pass; no choice question


def test_a_dropdown_without_a_matching_option_is_left_alone_when_the_model_is_not_sure():
    form = webforms.parse_fields([el(1, "select", "Город")])
    rt = FakeRT(extract={"«Город»": ("Гонконг", 0.99)})
    rt._choice = lambda state, q, options: SimpleNamespace(choice=options[0], confidence=0.2, probabilities={o: 0.2 for o in options})
    plan = webforms.plan_fill(rt, "город Гонконг", form, {1: ["Москва", "Токио", "Париж"]})
    assert not plan.steps and plan.skipped[0]["reason"] == "no option of the list matches"


def test_checkboxes_are_ticked_on_request_and_consent_boxes_need_a_mention_of_consent():
    form = webforms.parse_fields([el(1, "input[type=checkbox]", "Subscribe to newsletter", checked=False), el(2, "input[type=checkbox]", "I accept the terms", checked=False),
                                  el(3, "input[type=checkbox]", "Remember me", checked=False)])
    rt = FakeRT(yesno={"Subscribe": 0.9, "terms": 0.9, "Remember": 0.2})
    plan = webforms.plan_fill(rt, "register and tick the newsletter box", form)
    assert [(s.kind, s.label) for s in plan.steps] == [("check", "Subscribe to newsletter")]
    assert any(s["label"] == "I accept the terms" and "consent" in s["reason"] for s in plan.skipped)
    plan2 = webforms.plan_fill(rt, "tick the newsletter box and accept the terms", form)
    assert {s.label for s in plan2.steps} == {"Subscribe to newsletter", "I accept the terms"}


def test_an_already_ticked_box_is_not_clicked_again():
    form = webforms.parse_fields([el(1, "input[type=checkbox]", "Newsletter", checked=True)])
    plan = webforms.plan_fill(FakeRT(yesno={"Newsletter": 0.95}), "tick newsletter", form)
    assert not plan.steps


def test_a_password_is_masked_in_the_report():
    form = webforms.parse_fields([el(1, "input[type=password]", "Пароль")])
    plan = webforms.plan_fill(FakeRT(extract={"«Пароль»": ("Qwerty123!", 0.96)}), "пароль Qwerty123!", form)
    assert plan.steps[0].value == "Qwerty123!"
    dumped = json.dumps(plan.to_dict(), ensure_ascii=False)
    assert "Qwerty123" not in dumped and "••••" in dumped


def test_pick_submit_never_marks_a_payment_button_as_safe():
    els = [el(1, "button", "Назад"), el(2, "button", "Оплатить заказ")]
    rt = FakeRT(choices={"pick": "Оплатить заказ"})
    r = webforms.pick_submit(rt, "оформи", els)
    assert r["label"] != "Оплатить заказ" and not r["dangerous"]  # the payment button is not even offered to the choice
    assert webforms.pick_submit(rt, "оформи", [els[1]])["dangerous"] is True
    assert webforms.is_dangerous_label("Delete account") and webforms.is_dangerous_label("Confirm purchase") and not webforms.is_dangerous_label("Save")


# ------------------------------------------------------------------------------------------------------------------ bulk
def test_classify_yesno_has_an_unsure_band():
    rt = FakeRT(many_yes={"a": 0.95, "b": 0.5, "c": 0.05})
    res = bulk.classify_yesno(rt, ["a", "b", "c"], "Is it about cooking?")
    assert [r["decision"] for r in res] == ["yes", "unsure", "no"]
    rep = bulk.summarize(res, ["a", "b", "c"], "yesno")
    assert rep["counts"] == {"yes": 1, "no": 1, "unsure": 1} and rep["yes"] == [0] and rep["no"] == [2] and rep["unsure"][0]["text"] == "b"


def test_classify_categories_and_validation():
    rt = FakeRT(many_yes={"x": "news", "y": "recipes"})
    res = bulk.classify_categories(rt, ["x", "y"], ["news", "recipes"])
    assert [r["category"] for r in res] == ["news", "recipes"]
    with pytest.raises(ValueError):
        bulk.classify_categories(rt, ["x"], ["only one"])
    with pytest.raises(ValueError):
        bulk.classify_categories(rt, ["x"], ["a", "a"])
    assert bulk.summarize(res, ["x", "y"], "categories")["categories"] == {"news": [0], "recipes": [1]}


# ------------------------------------------------------------------------------------------------------------------ tools
def test_leeth_map_tool(monkeypatch):
    import tools.leeth_bulk_tools as T

    monkeypatch.setattr(T, "_leeth2", lambda: FakeRT(many_yes={"борщ": 0.97, "новости": 0.02}))
    out = json.loads(T.leeth_map(json.dumps(["борщ", "новости"]), "Это рецепт?"))
    assert out["yes"] == [0] and out["no"] == [1]
    assert T.leeth_map("", "q").startswith("Error")
    assert T.leeth_map("a\nb", "").startswith("Error")
    assert "limit" in T.leeth_map("\n".join("x" * 3 for _ in range(10)), "q?", max_items=5)
    cats = json.loads(T.leeth_map("новости", "О чём это?", mode="categories", categories="новости\nрецепты"))
    assert cats["mode"] == "categories"


def test_click_many_refuses_without_a_scan_and_guards_destructive_elements(monkeypatch):
    import tools.leeth_bulk_tools as T

    T._LAST_SCAN.clear()
    assert T.leeth_click_many("[1]").startswith("Error: nothing to click")
    T._LAST_SCAN["default"] = {1: {"label": "Удалить (Рецепт борща)", "href": ""}, 2: {"label": "Рецепт борща", "href": "https://x/1"}}
    msg = T.leeth_click_many("1")
    assert msg.startswith("Not clicked") and "confirm=True" in msg
    assert T.leeth_click_many("abc").startswith("Error: indices")


def test_a_button_whose_label_is_the_items_title_is_still_destructive_by_its_text(monkeypatch):
    import tools.leeth_bulk_tools as T

    T._LAST_SCAN["default"] = {7: {"label": "Рецепт борща с говядиной", "href": "", "text": "Удалить"}}
    msg = T.leeth_click_many("[7]")
    assert msg.startswith("Not clicked") and "confirm=True" in msg


def test_unlabeled_boxes_are_reported_and_a_payment_button_is_never_the_submit_candidate():
    form = webforms.parse_fields([el(1, "input[type=checkbox]", "", checked=False), el(2, "input[type=text]", "Имя")])
    plan = webforms.plan_fill(FakeRT(extract={"«Имя»": ("Иван", 0.99)}), "Имя Иван", form)
    assert [s.label for s in plan.steps] == ["Имя"] and any("no label" in s["reason"] for s in plan.skipped)
    els = [el(1, "button", "Сохранить профиль"), el(2, "button", "Оплатить подписку")]
    r = webforms.pick_submit(FakeRT(choices={"pick": "Оплатить подписку"}), "подпишись", els)
    assert r["label"] == "Сохранить профиль" and not r["dangerous"]
    only_pay = webforms.pick_submit(FakeRT(), "оплати", [el(2, "button", "Оплатить подписку")])
    assert only_pay["dangerous"] is True


def test_the_new_tools_are_registered_with_schemas_and_plan_mode_knows_which_change_the_page():
    from src.agent.plan_mode import MUTATING_TOOLS
    from tools.lazy_registry import DEFAULT_TOOL_LOADERS
    from tools.schemas import TOOL_SCHEMAS

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    for t in ("leeth_map", "leeth_scan_page", "leeth_click_many", "leeth_fill_form"):
        assert t in DEFAULT_TOOL_LOADERS and t in names
    assert {"leeth_fill_form", "leeth_click_many"} <= MUTATING_TOOLS and not {"leeth_map", "leeth_scan_page"} & MUTATING_TOOLS


# ------------------------------------------------------------------------------------------------------------------ examples / posts / scrolling
def _logit(p):
    import math

    return math.log(p / (1 - p))


def test_cut_from_examples_separable_overlapping_and_one_sided():
    c = bulk.cut_from_examples([0.30, 0.45], [0.05, 0.10])
    assert c["yes"] == pytest.approx(_logit(0.30)) and c["no"] == pytest.approx(_logit(0.10))
    ov = bulk.cut_from_examples([0.20, 0.50], [0.15, 0.40])  # the scores of the two kinds interleave: a wide unsure band
    assert ov["no"] < ov["yes"]
    only_pos = bulk.cut_from_examples([0.3], [])
    assert only_pos["yes"] == pytest.approx(_logit(0.3)) and only_pos["no"] < only_pos["yes"]
    only_neg = bulk.cut_from_examples([], [0.1])
    assert only_neg["no"] == pytest.approx(_logit(0.1)) and only_neg["yes"] > only_neg["no"]
    with pytest.raises(ValueError):
        bulk.cut_from_examples([], [])


def test_examples_move_the_cut_to_where_the_scores_of_this_question_separate():
    # a content question: clear matches score 0.3-0.4, not 0.9 -- the fixed 0.7 / 0.3 bounds call them unsure or no
    scores = {"pos1": 0.30, "neg1": 0.05, "a": 0.38, "b": 0.12, "c": 0.04}
    rt = FakeRT(many_yes=scores)
    plain = bulk.classify_yesno(rt, ["a", "b", "c"], "Это жалоба?")
    assert [r["decision"] for r in plain] == ["unsure", "no", "no"]
    meta = {}
    res = bulk.classify_yesno(rt, ["a", "b", "c"], "Это жалоба?", positives=["pos1"], negatives=["neg1"], meta=meta)
    assert [r["decision"] for r in res] == ["yes", "unsure", "no"]  # b sits between the examples: it goes back to the caller
    assert meta["calibrated_by"] == "examples" and meta["examples"] == {"positive": 1, "negative": 1}
    assert meta["yes_from_p"] == pytest.approx(0.3, abs=1e-3) and meta["no_up_to_p"] == pytest.approx(0.05, abs=1e-3)
    assert [r["i"] for r in res] == [0, 1, 2]  # the examples are not part of the answer


def test_summary_lists_unsure_items_strongest_first():
    res = [{"i": 0, "decision": "unsure", "p": 0.4}, {"i": 1, "decision": "unsure", "p": 0.6}, {"i": 2, "decision": "no", "p": 0.1}]
    rep = bulk.summarize(res, ["a", "b", "c"], "yesno")
    assert [u["i"] for u in rep["unsure"]] == [1, 0]


def test_leeth_map_hints_at_examples_when_most_items_are_unsure(monkeypatch):
    import tools.leeth_bulk_tools as T

    items = [f"текст {i}" for i in range(8)]
    monkeypatch.setattr(T, "_leeth2", lambda: FakeRT(many_yes={t: 0.5 for t in items}))
    rep = json.loads(T.leeth_map(json.dumps(items), "Это рецепт?"))
    assert "positive_examples" in rep["hint"]
    rt = FakeRT(many_yes={**{t: 0.5 for t in items}, "эталон": 0.45, "антипример": 0.05})
    monkeypatch.setattr(T, "_leeth2", lambda: rt)
    rep = json.loads(T.leeth_map(json.dumps(items), "Это рецепт?", positive_examples=["эталон"], negative_examples=["антипример"]))
    assert "hint" not in rep and rep["calibrated_by"] == "examples" and rep["yes"] == list(range(8))


def test_post_text_loses_its_buttons_captions_but_keeps_the_context_key():
    from tools.leeth_bulk_tools import _post_items, _strip_ui_tail

    assert _strip_ui_tail("Курьер опоздал Нравится В закладки", ["Нравится", "В закладки"]) == "Курьер опоздал"
    assert _strip_ui_tail("Нравится", ["Нравится"]) == "Нравится"  # never to an empty text
    els = [{"idx": 2, "tag": "button", "label": "Нравится", "href": ""}, {"idx": 3, "tag": "button", "label": "В закладки", "href": ""}]
    blocks = [{"text": "Курьер опоздал на три часа Нравится В закладки", "href": "https://x/1", "idxs": [2, 3]}]
    it = _post_items(blocks, els, {})[0]
    assert it["label"] == "Курьер опоздал на три часа"
    assert it["ctx"] == blocks[0]["text"][:80] and [b["label"] for b in it["buttons"]] == ["Нравится", "В закладки"]


class FakeEngine:
    """A page as a list of posts that grows as it is scrolled; click() records what was pressed."""

    def __init__(self, posts_by_scroll, like_label="Нравится"):
        self.posts_by_scroll = posts_by_scroll  # [[post texts visible at scroll 0], [at scroll 1], ...] (cumulative)
        self.k = 0
        self.like = like_label
        self.clicked = []
        self.scrolls = 0

    def run(self, coro):
        import asyncio

        return asyncio.run(coro)

    def _posts(self):
        return self.posts_by_scroll[min(self.k, len(self.posts_by_scroll) - 1)]

    async def get_interactive_elements_raw(self, session="default"):
        # every post has the same "like" button: idx 10 * (n + 1)
        return [{"idx": 10 * (n + 1), "tag": "button", "label": self.like, "href": "", "text": self.like} for n, _ in enumerate(self._posts())]

    async def get_block_contexts(self, session="default"):
        return {10 * (n + 1): t[:80] for n, t in enumerate(self._posts())}

    async def get_text_blocks(self, session="default", **kw):
        return [{"text": f"{t} {self.like}", "href": f"https://x/{n}", "idxs": [10 * (n + 1)], "y": n} for n, t in enumerate(self._posts())]

    async def scroll(self, direction="down", amount=500, **kw):
        self.k += 1
        self.scrolls += 1
        return "ok"

    async def click(self, index=None, **kw):
        self.clicked.append(index)
        return "clicked"


def _engine(monkeypatch, fake):
    import browser_engine as BE

    monkeypatch.setattr(BE, "browser_engine", fake)
    return fake


def test_scan_posts_scrolls_collects_new_posts_and_stops_when_nothing_new_appears(monkeypatch):
    import tools.leeth_bulk_tools as T

    a, b, c = "Курьер опоздал, жалоба на доставку", "Рецепт борща", "Заказ не пришёл, жалоба"
    eng = _engine(monkeypatch, FakeEngine([[a], [a, b], [a, b, c], [a, b, c], [a, b, c], [a, b, c]]))
    monkeypatch.setattr(T, "_leeth2", lambda: FakeRT(many_yes={a: 0.9, b: 0.02, c: 0.85}))
    rep = json.loads(T.leeth_scan_page("Жалоба на доставку?", include="posts", scroll_pages=10))
    assert rep["judged"] == 3 and [p["post"] for p in rep["yes"]] == [1, 3] and rep["no_count"] == 1
    assert eng.scrolls < 10  # two scrolls in a row brought nothing: the feed is over
    assert T.leeth_scan_page("q?", include="nonsense").startswith("Error: include")


def test_identical_buttons_of_different_posts_are_told_apart_by_the_posts_text(monkeypatch):
    import tools.leeth_bulk_tools as T

    a, b = "Курьер опоздал, жалоба на доставку", "Рецепт борща"
    eng = _engine(monkeypatch, FakeEngine([[a, b]]))
    monkeypatch.setattr(T, "_leeth2", lambda: FakeRT(many_yes={a: 0.9, b: 0.02}))
    rep = json.loads(T.leeth_scan_page("Жалоба?", include="posts"))
    assert [p["post"] for p in rep["yes"]] == [1]
    like = rep["yes"][0]["buttons"][0]
    assert like["label"] == "Нравится"
    out = json.loads(T.leeth_click_many(str(like["index"])))
    assert out["clicked"] == [like["index"]] and eng.clicked == [10]  # the first post's button, not the second one's (idx 20)


def test_a_post_that_left_the_page_is_skipped_not_clicked_by_mistake(monkeypatch):
    import tools.leeth_bulk_tools as T

    a = "Курьер опоздал, жалоба на доставку"
    eng = _engine(monkeypatch, FakeEngine([[a, "Другой пост про погоду и ветер"], ["Совсем другой пост про море и пляж", "И ещё один про лес"]]))
    monkeypatch.setattr(T, "_leeth2", lambda: FakeRT(many_yes={a: 0.9}))
    rep = json.loads(T.leeth_scan_page("Жалоба?", include="posts"))
    like = rep["yes"][0]["buttons"][0]["index"]
    eng.k = 1  # the feed re-rendered
    out = json.loads(T.leeth_click_many(str(like)))
    assert out["clicked"] == [] and out["skipped"] and eng.clicked == []


def test_one_option_of_a_radio_group_is_ticked():
    r = {"tag": "input[type=radio]", "checked": False, "disabled": False, "href": ""}
    els = [{**r, "idx": 1, "label": "Мужской", "name": "sex"}, {**r, "idx": 2, "label": "Женский", "name": "sex"},
           {**r, "idx": 3, "label": "Курьер", "name": "ship"}]
    fields = webforms.parse_fields(els)
    assert {f.index: f.group for f in fields} == {1: "sex", 2: "sex", 3: "ship"}
    plan = webforms.plan_fill(FakeRT(yesno={"«Мужской»": 0.9, "«Женский»": 0.8, "«Курьер»": 0.95}), "Пол мужской, доставка курьером", fields)
    assert sorted(s.label for s in plan.steps) == ["Курьер", "Мужской"]
    assert any(s["label"] == "Женский" and "radio group" in s["reason"] for s in plan.skipped)


def test_the_page_scripts_of_block_reading_are_valid_javascript():
    """The scripts live in Python strings (escapes!) and only run in a real page: parse them with node when it is installed."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    from browser_engine import BrowserEngine

    lib = BrowserEngine._BLOCKS_LIB_JS
    for name in ("_TEXT_BLOCKS_JS", "_BLOCK_CONTEXT_JS"):
        src = getattr(BrowserEngine, name).replace("__LIB__", lib)
        assert "__LIB__" not in src and "\x08" not in src
        # the arrow function must at least parse: wrap it as an expression
        cp = subprocess.run([node, "-e", "new Function('return (' + require('fs').readFileSync(0, 'utf8') + ')')"],
                            input=src, capture_output=True, text=True, timeout=30)
        assert cp.returncode == 0, f"{name}: {cp.stderr[:300]}"
