"""Browser decisions over an accessibility-tree style page: which element serves the goal, what text to type, is a click risky.

Pages are assembled from templates with near-duplicate distractors (``Search`` vs ``Search settings``), disabled
twins, row-level actions (many ``Delete`` buttons) and icon-only controls that only carry an aria-label.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...schema import Sample

# (en, ru)
BUTTONS = [("Search", "Поиск"), ("Sign in", "Войти"), ("Sign up", "Регистрация"), ("Submit", "Отправить"), ("Cancel", "Отмена"), ("Save", "Сохранить"), ("Delete", "Удалить"),
           ("Add to cart", "В корзину"), ("Buy now", "Купить сейчас"), ("Next", "Далее"), ("Back", "Назад"), ("Close", "Закрыть"), ("Accept all cookies", "Принять все cookies"),
           ("Reject cookies", "Отклонить cookies"), ("Download", "Скачать"), ("Share", "Поделиться"), ("Edit", "Изменить"), ("Subscribe", "Подписаться"), ("Log out", "Выйти"),
           ("Settings", "Настройки"), ("Help", "Помощь"), ("Upload", "Загрузить"), ("Pay now", "Оплатить"), ("Apply", "Применить"), ("Reset", "Сбросить"), ("Continue", "Продолжить"),
           ("Confirm", "Подтвердить"), ("Play", "Воспроизвести"), ("Like", "Нравится"), ("Compose", "Написать"), ("Refresh", "Обновить"), ("Print", "Печать")]
LINKS = [("Home", "Главная"), ("About us", "О нас"), ("Contact", "Контакты"), ("Pricing", "Цены"), ("Blog", "Блог"), ("Careers", "Вакансии"), ("Privacy policy", "Политика конфиденциальности"),
         ("Terms of service", "Условия использования"), ("FAQ", "Частые вопросы"), ("Documentation", "Документация"), ("Support", "Поддержка"), ("My orders", "Мои заказы"), ("Wishlist", "Избранное")]
FIELDS = [("Email", "Электронная почта"), ("Password", "Пароль"), ("Username", "Имя пользователя"), ("First name", "Имя"), ("Last name", "Фамилия"), ("Phone", "Телефон"),
          ("Address", "Адрес"), ("Comment", "Комментарий"), ("Coupon code", "Промокод"), ("Quantity", "Количество"), ("Search query", "Поисковый запрос"), ("City", "Город")]
CHECKS = [("Remember me", "Запомнить меня"), ("I agree to the terms", "Я согласен с условиями"), ("Subscribe to the newsletter", "Подписаться на рассылку"), ("Ship to a different address", "Доставить по другому адресу")]
COMBOS = [("Country", "Страна"), ("Language", "Язык"), ("Sort by", "Сортировка"), ("Category", "Категория")]
ICONS = [("magnifier", "Search", "Поиск"), ("cart", "Cart", "Корзина"), ("gear", "Settings", "Настройки"), ("bell", "Notifications", "Уведомления"), ("user", "Profile", "Профиль"),
         ("menu", "Menu", "Меню"), ("heart", "Favourites", "Избранное"), ("trash", "Delete", "Удалить"), ("pencil", "Edit", "Изменить"), ("x", "Close", "Закрыть")]
DANGEROUS_EN = ["Delete", "Delete account", "Remove", "Pay now", "Buy now", "Place order", "Send", "Unsubscribe", "Format disk", "Reset all data", "Log out", "Confirm payment", "Publish", "Transfer funds"]
DANGEROUS_RU = ["Удалить", "Удалить аккаунт", "Убрать", "Оплатить", "Купить сейчас", "Оформить заказ", "Отправить", "Отписаться", "Форматировать диск", "Сбросить все данные", "Выйти", "Подтвердить платёж", "Опубликовать", "Перевести деньги"]
SAFE_EN = ["Search", "Next", "Back", "Cancel", "Close", "Help", "Settings", "Download", "Share", "Edit", "Refresh", "Print", "Like", "Show more", "Sort by", "Expand", "Preview", "Filter"]
SAFE_RU = ["Поиск", "Далее", "Назад", "Отмена", "Закрыть", "Помощь", "Настройки", "Скачать", "Поделиться", "Изменить", "Обновить", "Печать", "Нравится", "Показать ещё", "Сортировка", "Развернуть", "Предпросмотр", "Фильтр"]
VALUES = ["john@example.com", "anna.k@mail.ru", "blue running shoes", "красные кроссовки", "Moscow", "Berlin", "42", "SPRING15", "+1 555 0100", "hunter2", "Ivan Petrov", "Смирнова", "iphone 15 case", "ресторан рядом"]
ITEM_NAMES = ["Invoice #1043", "Invoice #1044", "Order 7781", "Order 7782", "Report Q3", "Report Q4", "Счёт №201", "Счёт №202", "Заказ 5512", "Заказ 5513", "Photo album", "Backup 2023"]


def _label(rng: random.Random, pair: Tuple[str, str], lang: str) -> str:
    return pair[1 if lang == "ru" else 0]


class El:
    def __init__(self, role: str, label: str, **kw: Any):
        self.role, self.label, self.kw = role, label, kw
        self.disabled = kw.get("disabled", False)
        self.context = kw.get("context")
        self.icon = kw.get("icon")
        self.id = 0

    def text(self, style: str, lang: str) -> str:
        extra = []
        if self.disabled:
            extra.append("disabled" if lang == "en" else "недоступно")
        if self.context:
            extra.append(("in row " if lang == "en" else "в строке ") + f"'{self.context}'")
        if self.kw.get("placeholder"):
            extra.append(("placeholder " if lang == "en" else "подсказка ") + f"'{self.kw['placeholder']}'")
        tail = (" (" + ", ".join(extra) + ")") if extra else ""
        name = f"[icon: {self.icon}] aria-label='{self.label}'" if self.icon else f"'{self.label}'"
        if style == "bracket":
            return f"[{self.id}] {self.role} {name}{tail}"
        if style == "tag":
            return f"<{self.role} id={self.id}> {self.label if not self.icon else name}{tail}"
        return f"{self.role} #{self.id}: {self.label if not self.icon else name}{tail}"


def _decorate(rng: random.Random, target_pair: Tuple[str, str], lang: str, pool: List[Tuple[str, str]], n_extra: int) -> List[El]:
    """Near duplicates of the target label: contain it / are contained in it / differ by a word."""
    base = _label(rng, target_pair, lang)
    out = []
    for suf_en, suf_ru in rng.sample([("settings", "настройки"), ("history", "история"), ("options", "параметры"), ("all", "все"), ("help", "справка")], min(n_extra, 5)):
        out.append(El("button", base + " " + (suf_ru if lang == "ru" else suf_en)))
    return out


def gen_click(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    style = rng.choice(["bracket", "tag", "hash"])
    kind = rng.choice(["button", "link", "icon", "row", "field", "check", "combo"])
    n_other = rng.randint(4, 30)
    els: List[El] = []
    target_el: Optional[El] = None
    goal = ""
    if kind == "button":
        pair = rng.choice(BUTTONS)
        target_el = El("button", _label(rng, pair, lang))
        goal = rng.choice([f"Click the “{target_el.label}” button.", f"Press {target_el.label}."] if lang == "en" else [f"Нажмите кнопку «{target_el.label}».", f"Нажми «{target_el.label}»."])
        els += _decorate(rng, pair, lang, BUTTONS, rng.randint(0, 3))
    elif kind == "link":
        pair = rng.choice(LINKS)
        target_el = El("link", _label(rng, pair, lang))
        goal = rng.choice([f"Open the “{target_el.label}” page.", f"Go to {target_el.label}."] if lang == "en" else [f"Откройте страницу «{target_el.label}».", f"Перейди в раздел «{target_el.label}»."])
    elif kind == "icon":
        ic = rng.choice(ICONS)
        target_el = El("button", ic[1 if lang == "en" else 2], icon=ic[0])
        goal = {"magnifier": ("Open the search.", "Открой поиск."), "cart": ("Open the shopping cart.", "Открой корзину."), "gear": ("Open the settings.", "Открой настройки."),
                "bell": ("Show my notifications.", "Покажи уведомления."), "user": ("Go to my profile.", "Перейди в профиль."), "menu": ("Open the main menu.", "Открой главное меню."),
                "heart": ("Open my favourites.", "Открой избранное."), "trash": ("Delete the item.", "Удали элемент."), "pencil": ("Edit the item.", "Редактируй элемент."), "x": ("Close the dialog.", "Закрой окно.")}[ic[0]][lang == "ru"]
        for other in rng.sample(ICONS, 4):
            if other[0] != ic[0]:
                els.append(El("button", other[1 if lang == "en" else 2], icon=other[0]))
    elif kind == "row":
        names = rng.sample(ITEM_NAMES, rng.randint(3, 5))
        act = rng.choice([("Delete", "Удалить"), ("Edit", "Изменить"), ("Download", "Скачать")])
        tgt_name = rng.choice(names)
        for nm in names:
            e = El("button", _label(rng, act, lang), context=nm)
            els.append(e)
            if nm == tgt_name:
                target_el = e
        goal = (f"Click “{_label(rng, act, lang)}” for {tgt_name}." if lang == "en" else f"Нажмите «{_label(rng, act, lang)}» для {tgt_name}.")
    elif kind == "field":
        pair = rng.choice(FIELDS)
        val = rng.choice(VALUES)
        target_el = El("textbox", _label(rng, pair, lang), placeholder=rng.choice(["", "...", ""]) or None)
        goal = (f"Type “{val}” into the {target_el.label} field." if lang == "en" else f"Введите «{val}» в поле «{target_el.label}».")
        for f in rng.sample(FIELDS, 3):
            if f[0] != pair[0]:
                els.append(El("textbox", _label(rng, f, lang)))
    elif kind == "check":
        pair = rng.choice(CHECKS)
        target_el = El("checkbox", _label(rng, pair, lang))
        goal = (f"Tick the box “{target_el.label}”." if lang == "en" else f"Отметьте флажок «{target_el.label}».")
        for c in CHECKS:
            if c[0] != pair[0]:
                els.append(El("checkbox", _label(rng, c, lang)))
    else:
        pair = rng.choice(COMBOS)
        target_el = El("combobox", _label(rng, pair, lang))
        goal = (f"Open the “{target_el.label}” dropdown." if lang == "en" else f"Откройте выпадающий список «{target_el.label}».")
        for c in COMBOS:
            if c[0] != pair[0]:
                els.append(El("combobox", _label(rng, c, lang)))
    # filler elements
    fillers: List[El] = []
    for _ in range(n_other):
        r = rng.random()
        if r < 0.45:
            fillers.append(El("button", _label(rng, rng.choice(BUTTONS), lang), disabled=rng.random() < 0.1))
        elif r < 0.8:
            fillers.append(El("link", _label(rng, rng.choice(LINKS), lang)))
        elif r < 0.9:
            fillers.append(El("textbox", _label(rng, rng.choice(FIELDS), lang)))
        else:
            ic = rng.choice(ICONS)
            fillers.append(El("button", ic[1 if lang == "en" else 2], icon=ic[0]))
    allv = [target_el] + els + fillers
    # drop fillers that duplicate the target exactly (same role+label+context): the label would be ambiguous
    uniq: List[El] = []
    seen = set()
    for e in allv:
        key = (e.role, e.label, e.context, e.icon, e.disabled)
        if key in seen and e is not target_el:
            continue
        if e is not target_el and (e.role, e.label, e.context, e.icon) == (target_el.role, target_el.label, target_el.context, target_el.icon):
            continue
        seen.add(key)
        uniq.append(e)
    rng.shuffle(uniq)
    for i, e in enumerate(uniq, start=1):
        e.id = i
    texts = [e.text(style, lang) for e in uniq]
    if len(set(texts)) != len(texts) or len(texts) < 3:
        return None
    page_titles = [("Checkout", "Оформление заказа"), ("Account settings", "Настройки аккаунта"), ("Search results", "Результаты поиска"), ("Sign in", "Вход"), ("Dashboard", "Панель управления")]
    pt = rng.choice(page_titles)
    url = rng.choice(["shop.example.com/checkout", "app.example.org/settings", "example.com/search?q=shoes", "mail.example.com/inbox", "docs.example.com"])
    state = (f"Page: {pt[0]} — {url}\nGoal: {goal}" if lang == "en" else f"Страница: {pt[1]} — {url}\nЦель: {goal}")
    q = rng.choice(["Which element should be used next?", "Which element do you interact with to achieve the goal?"] if lang == "en" else ["С каким элементом нужно взаимодействовать дальше?", "Какой элемент нужен для достижения цели?"])
    out = [Sample("browser_element", "browser", lang, "choice", state, q, texts, uniq.index(target_el))]
    # typing goals also ask what to type
    if kind == "field":
        val_i = state.find(val)
        if val_i >= 0:
            qe = "What exact text must be typed?" if lang == "en" else "Какой именно текст нужно ввести?"
            out.append(Sample("browser_type_value", "browser", lang, "extract", state, qe, None, [val_i, val_i + len(val)]))
    return out


def gen_risk(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    risky = rng.random() < 0.5
    if lang == "en":
        label = rng.choice(DANGEROUS_EN if risky else SAFE_EN)
    else:
        label = rng.choice(DANGEROUS_RU if risky else SAFE_RU)
    ctx = rng.choice(["Page: Account — example.com/account", "Page: Cart — shop.example.com/cart", "Page: Files — drive.example.com", "Page: Inbox — mail.example.com"])
    ctx_ru = rng.choice(["Страница: Аккаунт — example.com/account", "Страница: Корзина — shop.example.com/cart", "Страница: Файлы — drive.example.com", "Страница: Почта — mail.example.com"])
    state = (ctx if lang == "en" else ctx_ru) + "\n" + (f"Element: button '{label}'" if lang == "en" else f"Элемент: кнопка «{label}»")
    q = "Could clicking this element cause an irreversible or costly action?" if lang == "en" else "Может ли нажатие на этот элемент привести к необратимому или платному действию?"
    return [Sample("browser_risk", "browser", lang, "noul", state, q, None, 1.0 if risky else 0.0)]


GENERATORS: Dict[str, Tuple[Callable[[random.Random], Optional[List[Sample]]], float]] = {"click": (gen_click, 6.0), "risk": (gen_risk, 1.0)}
