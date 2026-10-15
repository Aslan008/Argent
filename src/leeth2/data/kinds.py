"""Row -> Sample converters, one per *kind* of task. A dataset is described by a ``Spec`` and a kind.

Every converter is a pure function ``(spec, row, rng, ctx) -> List[Sample]`` so it can be unit-tested with
hand-written rows and no network.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..schema import Sample
from . import banks
from .common import clean, humanize, make_choice, make_extract, make_noul, make_score, pick, subset_options


@dataclass
class Spec:
    id: str
    repo: str
    kind: str
    family: str
    lang: str
    cfg: Optional[str] = None
    train: Optional[str] = "train"
    dev: Optional[str] = None  # official split feeding dev (+ calibration); None = carve the eval pools out of train by hash
    test: Optional[str] = None  # official split for the test pool; None = take it from the ``dev`` split after dev+calibration
    caps: Tuple[int, int, int, int] = (50000, 150, 300, 1000)  # max samples: train / dev / calibration / test
    lic: str = "unknown"  # permissive | sa | unknown | nc | other
    heldout: bool = False  # never trained on: zero-shot evaluation of unseen tasks
    eval_only: bool = False  # excluded from train (licence or by design)
    opts: Dict[str, Any] = field(default_factory=dict)
    revision: Optional[str] = None
    note: str = ""


# ----------------------------------------------------------------------------------------------
def _col(row: Dict[str, Any], key: Any) -> str:
    if callable(key):
        return key(row)
    if isinstance(key, (list, tuple)):
        return "\n".join(clean(row.get(k)) for k in key if row.get(k) not in (None, ""))
    return clean(row.get(key))


def _word(rng: random.Random, v: Any) -> str:
    return pick(rng, v) if isinstance(v, (list, tuple)) else str(v)


# common label words in both directions, so a Russian question does not carry English label names (and vice versa)
LABEL_TR: Dict[str, Tuple[str, ...]] = {
    "negative": ("негативный", "отрицательный"), "positive": ("позитивный", "положительный"), "neutral": ("нейтральный",), "good": ("хороший", "позитивный"), "bad": ("плохой", "негативный"),
    "spam": ("спам",), "ham": ("не спам",), "world": ("мировые новости",), "sports": ("спорт",), "business": ("бизнес",), "politics": ("политика",), "culture": ("культура",),
    "science": ("наука",), "economics": ("экономика",), "incidents": ("происшествия",), "technology": ("технологии",), "hateful": ("враждебный",), "not hateful": ("не враждебный",),
    "offensive": ("оскорбительный",), "not offensive": ("не оскорбительный",), "ironic": ("ироничный",), "not ironic": ("без иронии",),
}
_REV: Dict[str, Tuple[str, ...]] = {}
for _en, _ru in LABEL_TR.items():
    for _w in _ru:
        _REV.setdefault(_w, (_en,))
    _REV.setdefault(_en, (_en,))


def _translate_label(name: str, lang: str, rng: random.Random) -> str:
    key = name.strip().lower()
    cyr = bool(re.search("[а-яё]", key))
    if lang == "ru" and not cyr and key in LABEL_TR:
        return rng.choice(LABEL_TR[key])
    if lang == "en" and cyr and key in _REV:
        return rng.choice(_REV[key])
    return name


def _class_name(spec: Spec, ctx: Dict[str, Any], lang: str, i: int, rng: random.Random) -> str:
    bank = ctx.get(f"names_{lang}") or spec.opts.get(f"names_{lang}")
    if bank:
        entry = bank[i]
        return _word(rng, entry) if isinstance(entry, (list, tuple)) else str(entry)
    entry = ctx["names"][i]
    name = _word(rng, entry) if isinstance(entry, (list, tuple)) else str(entry)
    return _translate_label(name, lang, rng)


def _label_id(spec: Spec, row: Dict[str, Any], ctx: Dict[str, Any]) -> Optional[int]:
    v = row.get(spec.opts.get("label", "label"))
    fn = spec.opts.get("label_fn")
    if fn:
        v = fn(v, row)
    if v is None or v == "":
        return None
    if isinstance(v, str) and ctx.get("label_to_id") is not None:
        return ctx["label_to_id"].get(v)
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v < ctx["n_classes"] else None


def _nonempty(*samples: Optional[Sample]) -> List[Sample]:
    return [s for s in samples if s is not None]


# ---------------------------------------------------------------------------------------------- cls
def conv_cls(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """Single-label classification -> Choice over the label names (+ Noul 'is it X?')."""
    text = _col(row, spec.opts["text"])
    y = _label_id(spec, row, ctx)
    if not text or y is None:
        return []
    ql = banks.qlang(rng, spec.lang)
    what = _word(rng, spec.opts.get(f"what_{ql}") or spec.opts.get("what_en", "category"))
    n = ctx["n_classes"]
    out: List[Sample] = []
    show = subset_options(rng, n, y, k_max=spec.opts.get("k_max", 40), full_prob=spec.opts.get("full_prob", 0.45), force_full=(spec.eval_only or spec.heldout) and n <= 255)
    options = [_class_name(spec, ctx, ql, i, rng) for i in show]
    q = pick(rng, banks.CLS_Q[ql]).format(what=what)
    extra = spec.opts.get(f"q_{ql}")
    if extra and rng.random() < spec.opts.get("q_prob", 0.5):
        q = _word(rng, extra)
    s = make_choice(spec.id, spec.family, spec.lang, text, q, options, show.index(y))
    if s is None:  # two classes verbalised identically: fall back to the canonical names
        options = [ctx["names"][i] if isinstance(ctx["names"][i], str) else ctx["names"][i][0] for i in show]
        s = make_choice(spec.id, spec.family, spec.lang, text, q, options, show.index(y))
    if s is not None:
        out.append(s)
    if rng.random() < spec.opts.get("noul_frac", 0.25) and n >= 2:
        truth = rng.random() < 0.5
        j = y if truth else rng.choice([i for i in range(n) if i != y])
        lab = _class_name(spec, ctx, ql, j, rng)
        nq = pick(rng, banks.CLS_NOUL_Q[ql]).format(what=what, label=lab)
        out.append(make_noul(spec.id, spec.family, spec.lang, text, nq, 1.0 if truth else 0.0))
    return out


# ---------------------------------------------------------------------------------------------- nli
def conv_nli(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """Premise/hypothesis with entailment/neutral/contradiction -> Choice (3 verbalisations) + strict Noul."""
    p = _col(row, spec.opts.get("premise", "premise"))
    h = _col(row, spec.opts.get("hypothesis", "hypothesis"))
    y = _label_id(spec, row, ctx)
    if not p or not h or y is None or y not in (0, 1, 2):
        return []
    ql = banks.qlang(rng, spec.lang)
    labels = pick(rng, banks.NLI_LABELS[ql])
    q = pick(rng, banks.NLI_Q[ql]).format(h=h)
    out = _nonempty(make_choice(spec.id, spec.family, spec.lang, p, q, list(labels), y)) if spec.opts.get("choice", True) else []
    if rng.random() < spec.opts.get("noul_frac", 0.5) or not out:
        nq = pick(rng, banks.NLI_NOUL_Q[ql]).format(h=h)
        out.append(make_noul(spec.id, spec.family, spec.lang, p, nq, 1.0 if y == 0 else 0.0))
    return out


def conv_para(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    a = _col(row, spec.opts.get("a", "sentence1"))
    b = _col(row, spec.opts.get("b", "sentence2"))
    y = row.get(spec.opts.get("label", "label"))
    if not a or not b or y is None:
        return []
    ql = banks.qlang(rng, spec.lang)
    state = banks.PARA_STATE[ql].format(a=a)
    q = pick(rng, banks.PARA_NOUL_Q[ql]).format(b=b)
    out = [make_noul(spec.id, spec.family, spec.lang, state, q, float(int(y) == 1))]
    if rng.random() < 0.3:
        yes, no = pick(rng, [("yes", "no"), ("да", "нет")] if ql == "ru" else [("yes", "no"), ("same", "different")])
        out.extend(_nonempty(make_choice(spec.id, spec.family, spec.lang, state, q, [yes, no], 0 if int(y) == 1 else 1)))
    return out


# ---------------------------------------------------------------------------------------------- mcq
def conv_mcq(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """``opts['get'](row)`` -> (context, question, options, answer_index, style) with style in {'qa','ending','blank'}."""
    got = spec.opts["get"](row)
    if not got:
        return []
    context, question, options, ans, *rest = got
    style = rest[0] if rest else "qa"
    if ans is None or not options or not (0 <= ans < len(options)):
        return []
    ql = banks.qlang(rng, spec.lang) if spec.opts.get("mix_lang", False) else spec.lang if spec.lang in ("en", "ru") else "en"
    if style == "ending":
        q = pick(rng, banks.ENDING_Q[ql])
    elif style == "blank":
        q = pick(rng, banks.BLANK_Q[ql])
    else:
        q = pick(rng, banks.MCQ_Q[ql]).format(q=question)
    return _nonempty(make_choice(spec.id, spec.family, spec.lang, context or "", q, list(options), ans))


# ---------------------------------------------------------------------------------------------- span
def conv_span(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """``opts['get'](row)`` -> (context, question, [(answer_text, start_char)])  (empty list = unanswerable)."""
    got = spec.opts["get"](row)
    if not got:
        return []
    context, question, answers = got
    if not context or not question:
        return []
    ql = spec.lang if spec.lang in ("en", "ru") else "en"
    q = pick(rng, banks.EXTRACT_Q[ql]).format(q=question)
    if answers:
        text, start = answers[0]
        s = make_extract(spec.id, spec.family, spec.lang, context, q, text, start)
    else:
        s = make_extract(spec.id, spec.family, spec.lang, context, q, None)
    return _nonempty(s)


# ---------------------------------------------------------------------------------------------- ner
_ENT_MAP = {"PER": "PER", "ORG": "ORG", "LOC": "LOC", "person": "PER", "organization": "ORG", "location": "LOC"}


def _spans_from_bio(tokens: Sequence[str], tags: Sequence[str]) -> Tuple[str, List[Tuple[int, int, str]]]:
    """Rebuild text from tokens (single spaces) and return entity char spans [(s, e, TYPE)]."""
    text_parts: List[str] = []
    offs: List[Tuple[int, int]] = []
    pos = 0
    for t in tokens:
        if text_parts:
            pos += 1
        offs.append((pos, pos + len(t)))
        text_parts.append(t)
        pos += len(t)
    text = " ".join(text_parts)
    spans: List[Tuple[int, int, str]] = []
    cur: Optional[List[Any]] = None
    for i, tag in enumerate(tags):
        if tag == "O" or tag is None:
            if cur:
                spans.append((cur[0], cur[1], cur[2]))
                cur = None
            continue
        pre, _, typ = tag.partition("-")
        if pre == "B" or cur is None or cur[2] != typ:
            if cur:
                spans.append((cur[0], cur[1], cur[2]))
            cur = [offs[i][0], offs[i][1], typ]
        else:
            cur[1] = offs[i][1]
    if cur:
        spans.append((cur[0], cur[1], cur[2]))
    return text, spans


def conv_ner(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """BIO-tagged tokens -> Extract 'first <type>' (+ unanswerable variants + Noul 'is there a <type>?')."""
    tokens = row.get(spec.opts.get("tokens", "tokens"))
    tag_ids = row.get(spec.opts.get("tags", "ner_tags"))
    names = ctx.get("tag_names")
    if not tokens or tag_ids is None or not names:
        return []
    tags = [names[t] if isinstance(t, int) else t for t in tag_ids]
    if not spec.opts.get("bio", True):  # plain type tags (few-nerd): a run of the same tag is one entity
        tags = ["O" if t == "O" else ("I-" if i > 0 and tags[i - 1] == t else "B-") + t for i, t in enumerate(tags)]
    text, spans = _spans_from_bio(tokens, tags)
    if len(text) < 8:
        return []
    by_type: Dict[str, List[Tuple[int, int]]] = {}
    for s, e, typ in spans:
        by_type.setdefault(typ, []).append((s, e))
    all_types = list(spec.opts.get("types") or sorted({n.partition("-")[2] or n for n in names if n != "O"}))
    ql = banks.qlang(rng, spec.lang)
    out: List[Sample] = []

    def ent_word(typ: str) -> str:
        n = banks.ENT_NAMES.get(_ENT_MAP.get(typ, typ))
        return pick(rng, n[ql]) if n else humanize(typ).lower()

    if by_type and rng.random() < 0.8:
        typ = rng.choice(sorted(by_type))
        s, e = by_type[typ][0]
        q = pick(rng, banks.ENT_Q[ql]).format(ent=ent_word(typ))
        out.extend(_nonempty(make_extract(spec.id, spec.family, spec.lang, text, q, text[s:e], s)))
    absent = [t for t in all_types if t not in by_type]
    if absent and rng.random() < 0.25:
        typ = rng.choice(absent)
        if rng.random() < 0.5:
            q = pick(rng, banks.ENT_Q[ql]).format(ent=ent_word(typ))
            out.extend(_nonempty(make_extract(spec.id, spec.family, spec.lang, text, q, None)))
        else:
            out.append(make_noul(spec.id, spec.family, spec.lang, text, pick(rng, banks.ENT_NOUL_Q[ql]).format(ent=ent_word(typ)), 0.0))
    if by_type and rng.random() < 0.15:
        typ = rng.choice(sorted(by_type))
        out.append(make_noul(spec.id, spec.family, spec.lang, text, pick(rng, banks.ENT_NOUL_Q[ql]).format(ent=ent_word(typ)), 1.0))
    return out


def conv_entlist(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """``opts['get'](row)`` -> (text, [(start_char, end_char, label_text)]) with free-form labels (GLiNER-style data)."""
    got = spec.opts["get"](row)
    if not got:
        return []
    text, ents = got
    if len(text) < 8 or not ents:
        return []
    by: Dict[str, List[Tuple[int, int]]] = {}
    for s, e, lab in ents:
        if 0 <= s < e <= len(text) and lab:
            by.setdefault(lab.strip(), []).append((s, e))
    if not by:
        return []
    pool: List[str] = ctx.setdefault("_pool", [])
    for lab in by:  # free-text entity types seen so far: the vocabulary for the many-option task below
        if len(pool) < 20000:
            pool.append(lab)
    lab = rng.choice(sorted(by))
    s, e = by[lab][0]
    ql = "en" if rng.random() < 0.75 else "ru"
    q = pick(rng, banks.ENT_Q[ql]).format(ent=lab)  # labels stay in their own language; the instruction language varies
    out = _nonempty(make_extract(spec.id, spec.family, spec.lang, text, q, text[s:e], s))
    # Many-option Choice (8..255 options): what type is this mention? Teaches the model to rank very long option lists.
    if len(set(pool)) > 24 and rng.random() < spec.opts.get("bigk_prob", 0.3):
        k = int(min(255, math.exp(rng.uniform(math.log(8), math.log(255)))))
        wrong = [l for l in dict.fromkeys(rng.sample(pool, min(len(pool), k * 3))) if l.strip().lower() != lab.strip().lower()][: k - 1]
        options = wrong + [lab]
        rng.shuffle(options)
        mention = text[s:e]
        qq = (f"What type is the entity “{mention}” in this text?" if ql == "en" else f"Какого типа сущность «{mention}» в этом тексте?")
        out += _nonempty(make_choice(spec.id + "_type", spec.family, spec.lang, text, qq, options, options.index(lab)))
    return out


# ---------------------------------------------------------------------------------------------- score / binary
def conv_score(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """``opts['get'](row)`` -> list of (state, what_en, what_ru, level, levels)."""
    items = spec.opts["get"](row) or []
    out: List[Sample] = []
    for state, what_en, what_ru, level, levels in items:
        if not state:
            continue
        ql = banks.qlang(rng, spec.lang)
        what = what_ru if ql == "ru" else what_en
        q = pick(rng, banks.SCORE_Q[ql]).format(what=what, n=levels)
        out.append(make_score(spec.id, spec.family, spec.lang, state, q, level, levels))
    return out


def conv_binary(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """``opts['get'](row)`` -> list of (state, question_en, question_ru, p_true)  (soft targets allowed)."""
    items = spec.opts["get"](row) or []
    out: List[Sample] = []
    for state, q_en, q_ru, p in items:
        if not state:
            continue
        ql = banks.qlang(rng, spec.lang)
        out.append(make_noul(spec.id, spec.family, spec.lang, state, _word(rng, q_ru if ql == "ru" else q_en), p))
    return out


def conv_sim(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """Sentence-pair similarity (STS) -> Score with 6 levels (0..5 mapped to 1..6) — fractional labels kept."""
    a = _col(row, spec.opts.get("a", "sentence1"))
    b = _col(row, spec.opts.get("b", "sentence2"))
    v = row.get(spec.opts.get("score", "score"))
    if not a or not b or v is None:
        return []
    ql = banks.qlang(rng, spec.lang)
    state = banks.SIM_STATE[ql].format(a=a, b=b)
    q = pick(rng, banks.SIM_Q[ql])
    lv = 6
    out = [make_score(spec.id, spec.family, spec.lang, state, q + " (1–6)", 1.0 + float(v) / spec.opts.get("max", 5.0) * (lv - 1), lv)]
    if rng.random() < 0.25:
        out.append(make_noul(spec.id, spec.family, spec.lang, state, "Do the two texts mean essentially the same thing?" if ql == "en" else "Означают ли два текста по сути одно и то же?", float(float(v) >= 4.0)))
    return out


def conv_pairwise(spec: Spec, row: Dict[str, Any], rng: random.Random, ctx: Dict[str, Any]) -> List[Sample]:
    """``opts['get'](row)`` -> (prompt, chosen, rejected): which of two responses is better (position randomised)."""
    got = spec.opts["get"](row)
    if not got:
        return []
    prompt, chosen, rejected = got
    if not chosen or not rejected or chosen.strip() == rejected.strip():
        return []
    ql = "en" if rng.random() < 0.8 else "ru"
    la, lb = pick(rng, banks.PAIR_OPTS[ql])
    flip = rng.random() < 0.5
    a, b = (rejected, chosen) if flip else (chosen, rejected)
    cap = spec.opts.get("resp_chars", 900)
    state = f"{clean(prompt, 1800)}\n\n[{la}] {clean(a, cap)}\n\n[{lb}] {clean(b, cap)}"
    q = pick(rng, banks.PAIR_Q[ql])
    return _nonempty(make_choice(spec.id, spec.family, spec.lang, state, q, [la, lb], 1 if flip else 0))


def register_extra() -> None:
    """Kinds that live in other modules (they import ``Spec`` from here, so they are attached lazily)."""
    from . import webtools

    KINDS["mind2web"] = webtools.conv_mind2web
    KINDS["glaive"] = webtools.conv_glaive


KINDS: Dict[str, Callable[..., List[Sample]]] = {
    "cls": conv_cls,
    "nli": conv_nli,
    "para": conv_para,
    "mcq": conv_mcq,
    "span": conv_span,
    "ner": conv_ner,
    "entlist": conv_entlist,
    "score": conv_score,
    "binary": conv_binary,
    "sim": conv_sim,
    "pairwise": conv_pairwise,
}
