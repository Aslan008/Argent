"""Sorting / triage with rules written in the prompt: files into folders, tickets by keywords, ordering and comparison.

The categories, their definitions and the criterion all come with the example, so "sorting" is a skill of reading the
rule, not a fixed label set.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from ...schema import Sample

# ---------------------------------------------------------------------------------------------- files into folders
EXT_GROUPS: Dict[str, Tuple[List[str], str, str]] = {
    "documents": (["pdf", "docx", "doc", "txt", "odt", "md", "rtf"], "Documents", "Документы"),
    "images": (["jpg", "jpeg", "png", "gif", "webp", "svg", "bmp"], "Images", "Изображения"),
    "video": (["mp4", "mkv", "avi", "mov", "webm"], "Video", "Видео"),
    "audio": (["mp3", "flac", "wav", "ogg", "m4a"], "Audio", "Аудио"),
    "archives": (["zip", "rar", "7z", "tar.gz", "tgz"], "Archives", "Архивы"),
    "code": (["py", "js", "ts", "cpp", "rs", "go", "java", "cs", "rb"], "Code", "Код"),
    "tables": (["xlsx", "csv", "ods", "xls"], "Spreadsheets", "Таблицы"),
    "installers": (["exe", "msi", "dmg", "deb", "apk"], "Installers", "Установщики"),
    "fonts": (["ttf", "otf", "woff"], "Fonts", "Шрифты"),
}
STEMS_EN = ["report", "invoice", "holiday", "backup", "notes", "draft", "meeting", "photo", "song", "setup", "project", "final", "scan", "budget", "lecture", "game", "movie", "résumé", "plan", "data"]
STEMS_RU = ["отчёт", "счёт", "отпуск", "резерв", "заметки", "черновик", "встреча", "фото", "песня", "установка", "проект", "итог", "скан", "бюджет", "лекция", "игра", "фильм", "резюме", "план", "данные"]
STEMS = STEMS_EN + STEMS_RU


def _file(rng: random.Random, group: str) -> str:
    exts = EXT_GROUPS[group][0]
    stem = "_".join(rng.sample(STEMS, rng.randint(1, 2))) + (f"_{rng.choice(['v2', 'final', '2021', '03', 'copy', 'new'])}" if rng.random() < 0.6 else "")
    return f"{stem}.{rng.choice(exts)}"


def gen_files(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    li = 1 if lang == "ru" else 0
    groups = list(EXT_GROUPS)
    k = rng.randint(3, 7)
    chosen = rng.sample(groups, k)
    target_group = rng.choice(chosen)
    name = _file(rng, target_group)
    with_ext = rng.random() < 0.5
    fallback = ("Other", "Разное")[li]
    opt_names = {g: EXT_GROUPS[g][1 + li] for g in chosen}
    options = []
    for g in chosen:
        o = opt_names[g]
        if with_ext:
            o += " (" + ", ".join(EXT_GROUPS[g][0][:4]) + ")"
        options.append((g, o))
    options.append(("other", fallback + (" (" + ("всё остальное" if lang == "ru" else "everything else") + ")" if with_ext else "")))
    rng.shuffle(options)
    texts = [o for _, o in options]
    target = [g for g, _ in options].index(target_group)
    size = rng.choice(["12 KB", "340 KB", "2.3 MB", "48 MB", "1.2 GB"])
    d = date(2019, 1, 1) + timedelta(days=rng.randint(0, 2400))
    if lang == "ru":
        state = f"Файл: {name}\nРазмер: {size}\nИзменён: {d.isoformat()}\nПравило: раскладывайте файлы по папкам согласно расширению."
        q = rng.choice(["В какую папку положить файл?", "Выберите папку для этого файла."])
    else:
        state = f"File: {name}\nSize: {size}\nModified: {d.isoformat()}\nRule: sort files into folders by extension."
        q = rng.choice(["Which folder should the file go to?", "Pick the destination folder for this file."])
    out = [Sample("sort_files", "sorting", lang, "choice", state, q, texts, target)]
    # a file whose extension matches no listed folder goes to "other"
    unlisted = [g for g in groups if g not in chosen]
    if unlisted and rng.random() < 0.5:
        g2 = rng.choice(unlisted)
        name2 = _file(rng, g2)
        state2 = state.replace(name, name2)
        out.append(Sample("sort_files", "sorting", lang, "choice", state2, q, texts, [g for g, _ in options].index("other")))
    return out


# ---------------------------------------------------------------------------------------------- tickets by keyword rules
CATS = {
    "billing": (["invoice", "payment", "refund", "charge", "receipt"], ["счёт", "оплата", "возврат", "списание", "чек"], "Billing", "Оплата"),
    "shipping": (["delivery", "package", "tracking", "courier", "shipment"], ["доставка", "посылка", "трекинг", "курьер", "отправление"], "Shipping", "Доставка"),
    "tech": (["error", "crash", "bug", "login", "password"], ["ошибка", "сбой", "баг", "вход", "пароль"], "Technical", "Техподдержка"),
    "sales": (["price", "discount", "quote", "offer", "catalog"], ["цена", "скидка", "расчёт", "предложение", "каталог"], "Sales", "Продажи"),
    "hr": (["vacation", "salary", "contract", "interview", "leave"], ["отпуск", "зарплата", "договор", "собеседование", "больничный"], "HR", "Кадры"),
    "legal": (["lawsuit", "terms", "gdpr", "license", "complaint"], ["иск", "условия", "персональные данные", "лицензия", "претензия"], "Legal", "Юристы"),
}
FILLER_EN = ["Hello,", "Hi team,", "Quick question:", "Please help:", "Dear support,", "Good afternoon,"]
FILLER_RU = ["Здравствуйте,", "Добрый день,", "Подскажите, пожалуйста:", "Коллеги,", "Прошу помочь:", "Привет,"]


def gen_tickets(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    li = 1 if lang == "ru" else 0
    k = rng.randint(3, 6)
    cats = rng.sample(list(CATS), k)
    target = rng.choice(cats)
    others = [c for c in cats if c != target]
    n_t = rng.randint(1, 3)
    msg_kws = rng.sample(CATS[target][li], n_t)
    if n_t >= 2 and others and rng.random() < 0.5:  # a weaker hint from another department: the majority rule decides
        msg_kws.append(rng.choice(CATS[rng.choice(others)][li]))
    rng.shuffle(msg_kws)
    filler = rng.choice(FILLER_RU if lang == "ru" else FILLER_EN)
    joiner = " и " if lang == "ru" else " and "
    body = " ".join(msg_kws) if rng.random() < 0.3 else f"{filler} {joiner.join(msg_kws)}."
    fallback = ("Other", "Прочее")[li]
    defs = [(c, f"{CATS[c][2 + li]}: " + ", ".join(CATS[c][li])) for c in cats]
    rng.shuffle(defs)
    options = [d for _, d in defs] + [fallback]
    counts = {c: sum(1 for kw in msg_kws if kw in CATS[c][li]) for c in cats}
    best = max(counts.values())
    winners = [c for c, n in counts.items() if n == best]
    if len(winners) != 1 or winners[0] != target:
        return None
    if lang == "ru":
        state = f"Входящее письмо: «{body}»\nПравило: направьте письмо в отдел, ключевых слов которого в нём больше всего. Если ключевых слов нет — «{fallback}»."
        q = "В какой отдел направить письмо?"
    else:
        state = f"Incoming message: “{body}”\nRule: route the message to the department with the most of its keywords in it. If there are no keywords, use “{fallback}”."
        q = "Which department should receive the message?"
    out = [Sample("sort_tickets", "sorting", lang, "choice", state, q, options, [c for c, _ in defs].index(target))]
    if rng.random() < 0.3:  # no keyword at all -> the fallback bin
        plain = rng.choice(["Thanks, talk later!", "See you on Monday.", "Ok."] if lang == "en" else ["Спасибо, до связи!", "Увидимся в понедельник.", "Хорошо."])
        out.append(Sample("sort_tickets", "sorting", lang, "choice", state.replace(body, plain), q, options, len(options) - 1))
    return out


# ---------------------------------------------------------------------------------------------- ordering / comparison
THINGS_EN = ["apple", "laptop", "chair", "lamp", "book", "phone", "bike", "watch", "jacket", "mug", "camera", "desk", "guitar", "backpack"]
THINGS_RU = ["яблоко", "ноутбук", "стул", "лампа", "книга", "телефон", "велосипед", "часы", "куртка", "кружка", "камера", "стол", "гитара", "рюкзак"]
ATTRS = {
    # key: (word_en, word_ru, unit_en, unit_ru, min_en, max_en, min_ru, max_ru)
    "price": ("price", "цена", "$", " ₽", "cheapest", "most expensive", "самый дешёвый", "самый дорогой"),
    "weight": ("weight", "вес", " kg", " кг", "lightest", "heaviest", "самый лёгкий", "самый тяжёлый"),
    "rating": ("rating", "рейтинг", "", "", "lowest rated", "highest rated", "с самым низким рейтингом", "с самым высоким рейтингом"),
}


def _fmt_val(attr: str, v: float, lang: str) -> str:
    a = ATTRS[attr]
    unit = a[3] if lang == "ru" else a[2]
    return f"{v}{unit}" if attr != "price" or lang == "ru" else f"${v}"


def gen_order(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    n = rng.randint(3, 8)
    names = rng.sample(THINGS_RU if lang == "ru" else THINGS_EN, n)
    attr = rng.choice(list(ATTRS))
    a = ATTRS[attr]
    vals = rng.sample(range(2, 500), n) if attr != "rating" else rng.sample([x / 10 for x in range(10, 50)], n)
    pairs = list(zip(names, vals))
    rng.shuffle(pairs)
    word = a[1] if lang == "ru" else a[0]
    lines = [f"- {nm}: {word} {_fmt_val(attr, v, lang)}" for nm, v in pairs]
    ordered = sorted(pairs, key=lambda p: p[1])
    mode = rng.choice(["min", "max", "second", "first_desc"])
    if mode == "min":
        tgt, qq = ordered[0][0], (f"Какой товар {a[6]}?" if lang == "ru" else f"Which item is the {a[4]}?")
    elif mode == "max":
        tgt, qq = ordered[-1][0], (f"Какой товар {a[7]}?" if lang == "ru" else f"Which item is the {a[5]}?")
    elif mode == "second":
        tgt, qq = ordered[1][0], ("Какой товар вторым при сортировке по возрастанию?" if lang == "ru" else "Which item comes second when sorted in ascending order?")
    else:
        tgt, qq = ordered[-1][0], ("Какой товар первым при сортировке по убыванию?" if lang == "ru" else "Which item comes first when sorted in descending order?")
    state = ("Список:\n" if lang == "ru" else "List:\n") + "\n".join(lines)
    opts = list(names)
    rng.shuffle(opts)
    out = [Sample("sort_order", "sorting", lang, "choice", state, qq, opts, opts.index(tgt))]
    seq = [v for _, v in pairs]
    asc = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    qn = "Отсортирован ли этот список по возрастанию?" if lang == "ru" else "Is this list already sorted in ascending order?"
    out.append(Sample("sort_is_sorted", "sorting", lang, "noul", state, qn, None, float(asc)))
    if rng.random() < 0.5:  # a correctly sorted copy, so that "yes" is as common as "no"
        sorted_lines = [f"- {nm}: {word} {_fmt_val(attr, v, lang)}" for nm, v in ordered]
        out.append(Sample("sort_is_sorted", "sorting", lang, "noul", ("Список:\n" if lang == "ru" else "List:\n") + "\n".join(sorted_lines), qn, None, 1.0))
    return out


def gen_compare_numbers(rng: random.Random) -> Optional[List[Sample]]:
    lang = rng.choice(["en", "ru"])
    kind = rng.choice(["int", "dec", "date"])
    if kind == "date":
        d1 = date(2020, 1, 1) + timedelta(days=rng.randint(0, 1500))
        d2 = d1 + timedelta(days=rng.choice([-1, 1]) * rng.randint(1, 400))
        a, b = d1.isoformat(), d2.isoformat()
        which_later = a if d1 > d2 else b
        q = "Which date is later?" if lang == "en" else "Какая дата позже?"
        opts = [a, b]
        tgt = opts.index(which_later)
    else:
        x = rng.randint(1, 10000) if kind == "int" else round(rng.uniform(0.01, 99.99), 2)
        y = x + (rng.choice([-1, 1]) * (rng.randint(1, 500) if kind == "int" else round(rng.uniform(0.01, 5), 2)))
        if y == x or y <= 0:
            return None
        opts = [str(x), str(y)]
        q = "Which number is larger?" if lang == "en" else "Какое число больше?"
        tgt = opts.index(str(max(x, y)))
    state = ("Сравните значения." if lang == "ru" else "Compare the values.")
    return [Sample("compare_numbers", "sorting", lang, "choice", state, q, opts, tgt)]


GENERATORS: Dict[str, Tuple[Callable[[random.Random], Optional[List[Sample]]], float]] = {
    "files": (gen_files, 3.0), "tickets": (gen_tickets, 3.0), "order": (gen_order, 3.0), "compare": (gen_compare_numbers, 1.0),
}
