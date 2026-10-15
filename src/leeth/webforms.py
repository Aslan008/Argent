"""src/leeth/webforms.py -- a form and an instruction -> a plan: which field gets which value (pure logic, no browser).

"Введи имя в поле логина Валера, рост 175 см, возраст 29 лет, выбери в выпадающем списке город Гонконг" is one instruction for four
fields.  The planner asks Leeth 2 one question per field over the WHOLE instruction ("what value goes into the field «Рост»?") in ONE
batched pass; the answers are spans of the instruction, so nothing is invented.  Then, deterministically:

  * a value claimed by two fields goes to the more confident one (a field with no value in the instruction gets "no answer", but a weak
    model can echo a neighbour's value -- that is what this rule removes);
  * "175 см" into a field -> "175" (a number followed by a unit word);
  * a dropdown is answered by the option that matches the extracted value (exact / contains, otherwise Leeth's choice among the options:
    "Гонконг" -> "Hong Kong");
  * a checkbox is ticked when the instruction asks for it (calibrated yes/no); consent boxes (terms, privacy) only when the instruction
    mentions consent at all.

Nothing is executed here: ``plan_fill`` returns a ``FillPlan`` of steps and of fields it did NOT fill together with the reason, so the
caller (tools/leeth_tools.py) can run the steps in one batch, verify them and hand the rest back to the user or to the main model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKIP_INPUT_TYPES = frozenset({"hidden", "submit", "button", "reset", "image", "file", "color", "range"})
TEXT_LIKE = frozenset({"text", "number", "password", "textarea"})
PLACEHOLDER_OPTION = re.compile(r"^[\W_]*$|^[\W_]*(?:выберите|выбрать|не выбрано|select|choose|please select|pick|none selected)(?=[ :.…–—-]|$).*$", re.I)
CONSENT_LABEL = re.compile(r"соглас|принима|услови|политик|конфиденц|оферт|agree|accept|terms|privacy|policy|consent|gdpr", re.I)
CONSENT_IN_INSTRUCTION = re.compile(r"соглас|принима|принят|услови|политик|оферт|agree|accept|terms|privacy|policy|consent|gdpr", re.I)
UNCHECK_WORDS = re.compile(r"сними|снять|убери|убрать|отключи|выключи|без\s|uncheck|untick|deselect|turn off|disable", re.I)
NUMBER_WITH_UNIT = re.compile(r"^([+-]?\d+(?:[.,]\d+)?)\s*[^\W\d_]{1,9}\.?$", re.U)
CYRILLIC = re.compile(r"[а-яё]", re.I)

Q_EXTRACT = {"ru": "Какое значение нужно указать в поле «{}»?", "en": "What value should be entered in the field '{}'?"}
Q_TICK = {"ru": "Нужно ли в итоге отметить флажок «{}»?", "en": "Should the checkbox '{}' end up ticked?"}
Q_OPTION = {"ru": "Какой вариант списка «{}» соответствует этому значению?", "en": "Which option of the list '{}' matches this value?"}
STATE_VALUE = {"ru": "Значение: {}", "en": "Value: {}"}
Q_SUBMIT = {"ru": "Какую кнопку нужно нажать, чтобы отправить заполненную форму?", "en": "Which button submits the filled form?"}
NO_BUTTON = {"ru": "ни одну", "en": "none of them"}

DANGEROUS_BUTTON = [
    re.compile(r"\b(?:delete|remove|erase|destroy)\b", re.I),
    re.compile(r"\b(?:pay|payment|checkout|purchase|place order|buy now|buy|order)\b", re.I),
    re.compile(r"\bconfirm\b|\bauthori[sz]e\b|\bsubmit payment\b", re.I),
    re.compile(r"\b(?:удалить|удали|оплатить|оплата|подтвердить|купить|заказать|оформить заказ|отправить деньги|перевести)\b", re.I),
]


def is_dangerous_label(label: str) -> bool:
    return any(p.search(label or "") for p in DANGEROUS_BUTTON)


def lang_of(*texts: str) -> str:
    return "ru" if any(CYRILLIC.search(t or "") for t in texts) else "en"


@dataclass
class FormField:
    index: int
    kind: str  # text | number | password | textarea | select | checkbox | radio
    label: str
    value: str = ""
    checked: Optional[bool] = None
    options: Optional[List[str]] = None
    input_type: str = ""
    group: str = ""  # radio buttons: the shared name -- one of a group can be selected


@dataclass
class FillStep:
    index: int
    kind: str  # type | select | check | uncheck
    label: str
    value: str  # what is typed / the option chosen / "" for a checkbox
    confidence: float
    raw: str = ""  # the span of the instruction the value came from
    note: str = ""
    secret: bool = False  # never echo the value (password fields)


@dataclass
class FillPlan:
    steps: List[FillStep] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)  # {label, reason, candidate?, confidence?}
    language: str = "ru"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [{"index": s.index, "kind": s.kind, "label": s.label, "value": "••••" if s.secret else s.value, "confidence": round(s.confidence, 3),
                       **({"note": s.note} if s.note else {})} for s in self.steps],
            "skipped": self.skipped,
        }


# ------------------------------------------------------------------------------------------------------------------ elements -> fields
def parse_fields(elements: Sequence[Dict[str, Any]]) -> List[FormField]:
    """The fillable controls among ``browser_engine.get_interactive_elements_raw()`` elements (tag descriptors like ``input[type=email]``)."""
    out: List[FormField] = []
    for el in elements:
        tag = str(el.get("tag") or "")
        base = tag.split("[", 1)[0].lower()
        m = re.search(r"type=([\w-]+)", tag)
        typ = m.group(1).lower() if m else ""
        if base not in ("input", "textarea", "select") or el.get("disabled"):
            continue
        label = str(el.get("label") or "").strip()
        idx = int(el.get("idx"))
        if base == "select":
            out.append(FormField(idx, "select", label, str(el.get("value") or ""), input_type="select"))
        elif base == "textarea":
            out.append(FormField(idx, "textarea", label, str(el.get("value") or ""), input_type="textarea"))
        elif typ in ("checkbox", "radio"):
            out.append(FormField(idx, typ, label, checked=bool(el.get("checked")), input_type=typ,
                                 group=str(el.get("name") or "") if typ == "radio" else ""))
        elif typ in SKIP_INPUT_TYPES:
            continue
        else:
            kind = "number" if typ == "number" else "password" if typ == "password" else "text"
            out.append(FormField(idx, kind, label, str(el.get("value") or ""), input_type=typ or "text"))
    return out


def buttons(elements: Sequence[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """(index, label) of the buttons and submit inputs."""
    out = []
    for el in elements:
        tag = str(el.get("tag") or "").lower()
        if el.get("disabled"):
            continue
        if tag.startswith("button") or "type=submit" in tag or "type=button" in tag or tag.startswith("input[type=image"):
            label = str(el.get("label") or "").strip()
            if label:
                out.append((int(el.get("idx")), label))
    return out


# ------------------------------------------------------------------------------------------------------------------ values
def clean_value(raw: str, kind: str) -> str:
    v = (raw or "").strip().strip(",;").strip()
    if kind == "number":
        m = NUMBER_WITH_UNIT.match(v) or re.match(r"^([+-]?\d+(?:[.,]\d+)?)$", v)
        if m:
            return m.group(1).replace(",", ".")
        m = re.search(r"[+-]?\d+(?:[.,]\d+)?", v)
        return m.group(0).replace(",", ".") if m else v
    m = NUMBER_WITH_UNIT.match(v)
    if m:  # "175 см", "29 лет": forms ask for the number
        return m.group(1)
    return v


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").casefold().replace("ё", "е")).strip()


def match_option(value: str, options: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Deterministic option match: ('exact' | 'contains', option) or None."""
    v = _norm(value)
    real = [o for o in options if not PLACEHOLDER_OPTION.match(o or "")]
    for o in real:
        if _norm(o) == v:
            return "exact", o
    hits = [o for o in real if len(v) >= 3 and (v in _norm(o) or (len(_norm(o)) >= 3 and _norm(o) in v))]
    if len(hits) == 1:
        return "contains", hits[0]
    return None


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ------------------------------------------------------------------------------------------------------------------ the plan
def plan_fill(rt: Any, instruction: str, fields: Sequence[FormField], options: Optional[Dict[int, List[str]]] = None, accept: float = 0.55,
              option_accept: float = 0.4, tick_accept: float = 0.75) -> FillPlan:
    """``rt`` is a Leeth2Runtime (decide / choose / judge).  ``options`` maps a select's index to its option texts."""
    from .primitives import ExtractQuestion, NoulQuestion

    options = options or {}
    lang = lang_of(instruction, " ".join(f.label for f in fields))
    plan = FillPlan(language=lang)
    valued = [f for f in fields if f.kind in TEXT_LIKE or f.kind == "select"]
    labelled = [f for f in valued if f.label]
    for f in valued:
        if not f.label:
            plan.skipped.append({"label": f"#{f.index}", "reason": "no label: cannot tell what it is for"})

    # one batched pass: a question per field over the whole instruction
    cands: Dict[int, Tuple[str, Tuple[int, int], float]] = {}
    if labelled:
        batch = rt.decide(instruction, [ExtractQuestion(f"f{f.index}", Q_EXTRACT[lang].format(f.label)) for f in labelled])
        for f in labelled:
            r = batch[f"f{f.index}"]
            text = (getattr(r, "text", "") or "").strip()
            span = tuple(getattr(r, "span", (0, 0)) or (0, 0))
            cands[f.index] = (text, span, float(getattr(r, "confidence", 0.0)))

    # a span goes to one field only: the most confident claim wins
    taken: List[Tuple[int, int]] = []
    winners: Dict[int, Tuple[str, Tuple[int, int], float]] = {}
    for f in sorted(labelled, key=lambda f: -cands[f.index][2]):
        text, span, conf = cands[f.index]
        if not text or span[1] <= span[0]:
            plan.skipped.append({"label": f.label, "reason": "the instruction gives no value for it", "confidence": round(conf, 3)})
            continue
        if conf < accept:
            plan.skipped.append({"label": f.label, "reason": "not sure enough", "candidate": text, "confidence": round(conf, 3)})
            continue
        if any(_overlap(span, t) for t in taken):
            plan.skipped.append({"label": f.label, "reason": "the same words are claimed by a more confident field", "candidate": text, "confidence": round(conf, 3)})
            continue
        taken.append(span)
        winners[f.index] = (text, span, conf)

    for f in labelled:
        if f.index not in winners:
            continue
        text, span, conf = winners[f.index]
        if f.kind == "select":
            opts = options.get(f.index) or []
            real = [o for o in opts if not PLACEHOLDER_OPTION.match(o or "")]
            if not real:
                plan.skipped.append({"label": f.label, "reason": "the options of the list could not be read", "candidate": text, "confidence": round(conf, 3)})
                continue
            m = match_option(text, real)
            if m:
                plan.steps.append(FillStep(f.index, "select", f.label, m[1], conf, raw=text, note=f"{m[0]} match"))
                continue
            res = rt.choose(STATE_VALUE[lang].format(text), Q_OPTION[lang].format(f.label), real)
            if float(res.confidence) >= option_accept:
                plan.steps.append(FillStep(f.index, "select", f.label, res.choice, min(conf, float(res.confidence)), raw=text,
                                           note="chosen by meaning" + ("" if res.confidence >= 0.6 else "; check it")))
            else:
                plan.skipped.append({"label": f.label, "reason": "no option of the list matches", "candidate": text, "confidence": round(float(res.confidence), 3)})
            continue
        value = clean_value(text, "number" if f.kind == "number" else f.kind)
        plan.steps.append(FillStep(f.index, "type", f.label, value, conf, raw=text, secret=(f.kind == "password")))

    # checkboxes / radios: only what the instruction asks for
    boxes = [f for f in fields if f.kind in ("checkbox", "radio") and f.label]
    for f in fields:
        if f.kind in ("checkbox", "radio") and not f.label:
            plan.skipped.append({"label": f"#{f.index}", "reason": "no label: cannot tell what the box is for"})
    if boxes:
        batch = rt.decide(instruction, [NoulQuestion(f"c{f.index}", Q_TICK[lang].format(f.label)) for f in boxes])
        wants_uncheck = bool(UNCHECK_WORDS.search(instruction))
        consent_ok = bool(CONSENT_IN_INSTRUCTION.search(instruction))
        for f in boxes:
            p = float(batch[f"c{f.index}"].probability)
            if p >= tick_accept and not f.checked:
                if CONSENT_LABEL.search(f.label) and not consent_ok:
                    plan.skipped.append({"label": f.label, "reason": "consent boxes are ticked only when the instruction mentions consent", "confidence": round(p, 3)})
                else:
                    plan.steps.append(FillStep(f.index, "check", f.label, "", p))
            elif f.checked and f.kind == "checkbox" and p <= 0.1 and wants_uncheck:
                plan.steps.append(FillStep(f.index, "uncheck", f.label, "", 1.0 - p))
        _one_per_radio_group(plan, boxes)
    return plan


def _one_per_radio_group(plan: "FillPlan", boxes: Sequence[FormField]) -> None:
    """Radio buttons with one name are alternatives: when the plan ticks several of them, only the most confident is kept and the
    others are reported (the browser would silently let the last click win)."""
    group_of = {f.index: f.group for f in boxes if f.kind == "radio" and f.group}
    best: Dict[str, FillStep] = {}
    for st in plan.steps:
        g = group_of.get(st.index)
        if g and st.kind == "check" and (g not in best or st.confidence > best[g].confidence):
            best[g] = st
    for st in list(plan.steps):
        g = group_of.get(st.index)
        if g and st.kind == "check" and best[g] is not st:
            plan.steps.remove(st)
            plan.skipped.append({"label": st.label, "reason": "another option of the same radio group is more likely",
                                 "confidence": round(st.confidence, 3)})


def pick_submit(rt: Any, instruction: str, elements: Sequence[Dict[str, Any]], lang: str = "ru", accept: float = 0.5) -> Dict[str, Any]:
    """Which button submits the form: {'index', 'label', 'confidence', 'dangerous'} or {'index': None, 'reason'}."""
    bs_all = buttons(elements)
    if not bs_all:
        return {"index": None, "reason": "no buttons on the page"}
    # a payment / delete / confirm button is never offered as "the submit button": the model weighs words, and "оплата подписки" matches "подпишись"
    bs = [b for b in bs_all if not is_dangerous_label(b[1])]
    if not bs:
        d = bs_all[0]
        return {"index": d[0], "label": d[1], "confidence": 0.0, "dangerous": True}
    labels = [lab for _, lab in bs]
    seen: Dict[str, int] = {}
    unique = []
    for lab in labels:  # equal labels must stay distinguishable for the choice head
        seen[lab] = seen.get(lab, 0) + 1
        unique.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
    none = NO_BUTTON[lang]
    res = rt.choose(instruction, Q_SUBMIT[lang], unique + [none])
    if res.choice == none or float(res.confidence) < accept:
        return {"index": None, "reason": "no button clearly submits the form", "confidence": round(float(res.confidence), 3)}
    i = unique.index(res.choice)
    return {"index": bs[i][0], "label": bs[i][1], "confidence": round(float(res.confidence), 3), "dangerous": is_dangerous_label(bs[i][1])}
