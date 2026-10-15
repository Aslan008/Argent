"""Bilingual (EN/RU) wording banks.

Diversity of *phrasing* is what lets one model answer tasks it has never seen: every task is asked in
several ways, in two languages, with several verbalisations of its labels. Nothing here is task-specific
magic — a new task only has to provide its own ``what`` words.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

LANG_MIX_SAME = 0.62  # probability that the question language follows the state language (en/ru)

CLS_Q = {
    "en": [
        "What is the {what} of this text?",
        "Determine the {what}.",
        "Which {what} fits best?",
        "Select the correct {what} for the text above.",
        "Classify the text by {what}.",
        "Read the text and choose its {what}.",
        "What {what} does this text have?",
        "Pick the option that gives the {what} of the text.",
        "Identify the {what}.",
    ],
    "ru": [
        "Определите: {what}.",
        "Выберите подходящий вариант ({what}).",
        "Классифицируйте текст ({what}).",
        "Прочитайте текст и укажите: {what}.",
        "Какой вариант верен для текста? Критерий — {what}.",
        "Укажите {what} текста.",
        "Выберите один вариант: {what}.",
        "Что подходит к тексту ({what})?",
    ],
}

CLS_NOUL_Q = {
    "en": [
        "Is the {what} of the text “{label}”?",
        "Does the text match “{label}” ({what})?",
        "True or false: the {what} is “{label}”.",
        "Would you label this text as “{label}” ({what})?",
        "Is “{label}” the right {what} for this text?",
    ],
    "ru": [
        "Верно ли, что ({what}) текста — «{label}»?",
        "Соответствует ли текст варианту «{label}» ({what})?",
        "Правда ли, что текст можно отнести к «{label}»? Критерий: {what}.",
        "Подходит ли «{label}» для текста? Критерий: {what}.",
    ],
}

NLI_LABELS = {
    "en": [("entailment", "neutral", "contradiction"), ("yes", "maybe", "no"), ("follows", "unclear", "contradicts"), ("true", "neither", "false"), ("supported", "not enough information", "refuted")],
    "ru": [("следует", "неопределённо", "противоречит"), ("да", "возможно", "нет"), ("верно", "не определено", "неверно"), ("подтверждается", "недостаточно данных", "опровергается")],
}
NLI_Q = {
    "en": [
        "Does the text above entail the following statement, contradict it, or neither?\nStatement: {h}",
        "Given the text, is this statement true, false, or undetermined?\nStatement: {h}",
        "What is the relation between the text and the claim?\nClaim: {h}",
        "Based only on the text, judge the hypothesis.\nHypothesis: {h}",
        "Text ⇒ hypothesis. Does the hypothesis follow?\n{h}",
    ],
    "ru": [
        "Следует ли из текста выше такое утверждение, противоречит ли оно тексту или неопределённо?\nУтверждение: {h}",
        "Исходя из текста, это утверждение верно, неверно или неопределённо?\nУтверждение: {h}",
        "Каково отношение между текстом и утверждением?\nУтверждение: {h}",
        "Опираясь только на текст, оцените гипотезу.\nГипотеза: {h}",
    ],
}
NLI_NOUL_Q = {
    "en": [
        "Is the following statement definitely supported by the text?\nStatement: {h}",
        "Does the text entail this claim?\nClaim: {h}",
        "Based on the text, is this true?\nStatement: {h}",
        "Can you conclude the following from the text?\nConclusion: {h}",
    ],
    "ru": [
        "Подтверждается ли это утверждение текстом?\nУтверждение: {h}",
        "Следует ли из текста этот вывод?\nВывод: {h}",
        "Верно ли это утверждение, исходя из текста?\nУтверждение: {h}",
    ],
}

PARA_NOUL_Q = {
    "en": ["Do the two sentences mean the same thing?\nSentence B: {b}", "Is sentence B a paraphrase of sentence A?\nSentence B: {b}", "Are these two texts equivalent in meaning?\nText 2: {b}"],
    "ru": ["Означают ли два предложения одно и то же?\nПредложение Б: {b}", "Является ли предложение Б перефразом предложения А?\nПредложение Б: {b}", "Эквивалентны ли тексты по смыслу?\nТекст 2: {b}"],
}
PARA_STATE = {"en": "Sentence A: {a}", "ru": "Предложение А: {a}"}

MCQ_Q = {
    "en": ["{q}", "Answer the question.\n{q}", "Choose the correct answer.\n{q}", "{q}\nSelect the best option."],
    "ru": ["{q}", "Ответьте на вопрос.\n{q}", "Выберите правильный ответ.\n{q}", "{q}\nВыберите лучший вариант."],
}
ENDING_Q = {
    "en": ["Which ending is the most plausible continuation?", "How does the text most likely continue?", "Choose the most sensible continuation."],
    "ru": ["Какое продолжение наиболее правдоподобно?", "Как, скорее всего, продолжается текст?", "Выберите наиболее разумное продолжение."],
}
BLANK_Q = {
    "en": ["Which option correctly fills the blank (_)?", "Replace _ with the right option.", "Choose what the blank _ refers to."],
    "ru": ["Какой вариант правильно заполняет пропуск (_)?", "Подставьте вместо _ верный вариант.", "Выберите, к чему относится пропуск _."],
}

EXTRACT_Q = {
    "en": ["{q}", "Answer with a span from the text: {q}", "Find the answer in the text. {q}", "Extract the answer. {q}"],
    "ru": ["{q}", "Ответьте фрагментом текста: {q}", "Найдите ответ в тексте. {q}", "Извлеките ответ. {q}"],
}
ENT_Q = {
    "en": ["Extract the {ent} mentioned in the text.", "Find the {ent} in the text.", "Which {ent} is mentioned? Copy it from the text.", "Return the first {ent} from the text."],
    "ru": ["Извлеките из текста: {ent}.", "Найдите в тексте: {ent}.", "Какой объект типа «{ent}» упомянут? Скопируйте его из текста.", "Верните первое упоминание: {ent}."],
}
ENT_NOUL_Q = {
    "en": ["Does the text mention any {ent}?", "Is there a {ent} in this text?"],
    "ru": ["Упоминается ли в тексте: {ent}?", "Есть ли в тексте объект типа «{ent}»?"],
}
ENT_NAMES = {
    "PER": {"en": ["person name", "person", "name of a person"], "ru": ["имя человека", "человек", "персона"]},
    "ORG": {"en": ["organization", "company or organization", "organisation name"], "ru": ["организация", "компания или организация", "название организации"]},
    "LOC": {"en": ["location", "place name", "geographic location"], "ru": ["место", "географическое название", "локация"]},
}

SCORE_Q = {
    "en": ["Rate the {what} from 1 to {n}.", "How would you rate the {what} on a scale of 1 to {n}?", "Score the {what} (1 = lowest, {n} = highest).", "Give a {n}-level rating for the {what}."],
    "ru": ["Оцените: {what} — по шкале от 1 до {n}.", "Как бы вы оценили ({what}) по шкале от 1 до {n}?", "Поставьте балл ({what}); 1 — минимум, {n} — максимум.", "Дайте оценку по {n}-балльной шкале: {what}."],
}
SIM_STATE = {"en": "Text A: {a}\nText B: {b}", "ru": "Текст А: {a}\nТекст Б: {b}"}
SIM_Q = {"en": ["How similar in meaning are Text A and Text B?", "Rate the semantic similarity of the two texts."], "ru": ["Насколько схожи по смыслу текст А и текст Б?", "Оцените семантическую близость двух текстов."]}

PAIR_Q = {
    "en": ["Which response is better?", "Which answer would a careful reader prefer?", "Pick the better of the two responses."],
    "ru": ["Какой ответ лучше?", "Какой ответ предпочёл бы внимательный читатель?", "Выберите лучший из двух ответов."],
}
PAIR_OPTS = {"en": [("A", "B"), ("first", "second"), ("Response A", "Response B")], "ru": [("А", "Б"), ("первый", "второй"), ("Ответ А", "Ответ Б")]}


def qlang(rng: random.Random, state_lang: str) -> str:
    """Language of the question/labels for a given state language."""
    if state_lang in ("en", "ru"):
        return state_lang if rng.random() < LANG_MIX_SAME else ("ru" if state_lang == "en" else "en")
    return "en" if rng.random() < 0.7 else "ru"


def fmt(template: str, **kw: str) -> str:
    return template.format(**kw)


def pick(rng: random.Random, seq: Sequence):
    return seq[rng.randrange(len(seq))]
