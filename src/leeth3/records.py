"""Leeth 3 teacher records: what an outside generator (Gemini in Antigravity, a local model) writes, and how it becomes training data.

A teacher writes one JSON object per line. Labels are written as TEXT (the option itself, the substring itself), never as indices:
language models miscount positions, they do not misspell a label they copy.

    {"id": "B0007-031", "request": "R002", "family": "fintech-cli", "domain": "tools", "lang": "ru", "ptype": "choice",
     "state": "...", "question": "...",
     "options": {"label": "description", ...}  or  ["label", ...],
     "answer": "label" | ["label", ...] (multi) | true/false/0.8 (noul) | 3 (score) | ["exact substring", ...] (extract) | null,
     "levels": 5, "abstain_ok": false, "difficulty": "easy|medium|hard", "teacher": "gemini-3.8-flash"}

ptype: choice (one option), multi (every option that applies; may be empty), noul (yes/no probability), score (1..levels),
extract (every substring of the state that answers; empty list = no answer). ``abstain_ok`` adds an implicit "none of these"
answer to choice: ``answer: null``.

``parse_record`` validates a line and normalises it (options as an ordered dict, extract answers as character spans);
``to_samples`` turns a record into Leeth 2 ``Sample`` rows so the current trainer can use them today (multi -> one yes/no row per
option; option descriptions travel in ``meta``) - the Leeth 3 encoder will read the record itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.leeth2.schema import MAX_LEVELS, MAX_OPTIONS, MIN_LEVELS, Sample

PTYPES = ("choice", "multi", "noul", "score", "extract")
NONE_OPTION = "ни один из вариантов"
# The "none of these" option as a client writes it: in the example's language, in more than one wording (the exam runner and Argent
# each use their own; training on a single fixed string would tie the skill to that string).
NONE_TEXTS: Dict[str, List[str]] = {
    "ru": ["ни один из вариантов не подходит", NONE_OPTION, "подходящего инструмента нет"],
    "en": ["none of these fits", "none of the options", "no suitable tool"],
    "uk": ["жоден з варіантів не підходить", "немає відповідного інструмента"],
    "de": ["keine der Optionen passt", "kein passendes Werkzeug"],
    "es": ["ninguna de estas opciones encaja", "no hay herramienta adecuada"],
    "fr": ["aucune de ces options ne convient", "aucun outil adapté"],
    "it": ["nessuna di queste opzioni è adatta", "nessuno strumento adatto"],
    "pt": ["nenhuma destas opções serve", "nenhuma ferramenta adequada"],
    "pl": ["żadna z tych opcji nie pasuje", "brak odpowiedniego narzędzia"],
    "tr": ["bu seçeneklerin hiçbiri uygun değil", "uygun araç yok"],
    "kk": ["бұл нұсқалардың ешқайсысы сәйкес келмейді", "сәйкес құрал жоқ"],
    "zh": ["以上选项都不合适", "没有合适的工具"],
    "ja": ["どの選択肢も当てはまらない", "適切なツールがない"],
}
QUESTION_FREE_DOMAINS = ("safety", "tools")     # their question is replaced by a neutral one at ingest (src/leeth3/questions.py)
REQUIRED = ("id","request", "family", "domain", "lang", "ptype", "state", "question")
MAX_STATE_CHARS = 120_000


class RecordError(ValueError):
    """A teacher line that cannot be used; ``reason`` is a short machine-readable code for the ingest report."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass
class Record:
    id: str
    request: str
    family: str
    domain: str
    lang: str
    ptype: str
    state: str
    question: str
    options: Dict[str, str] = field(default_factory=dict)     # label -> description ("" when none was given), in the teacher's order
    answer: Any = None
    levels: Optional[int] = None
    abstain_ok: bool = False
    difficulty: str = ""
    teacher: str = ""
    group: str = ""                                            # records of one group are deliberately alike (questions on one text, twins)
    spans: List[Tuple[int, int]] = field(default_factory=list)  # extract: character spans of the answer substrings

    @property
    def batch(self) -> str:
        return self.id.rsplit("-", 1)[0]

    @property
    def group_key(self) -> str:
        return f"{self.batch}:{self.group}" if self.group else ""

    def text_for_dedup(self) -> str:
        # In these domains the question is a neutral formula picked by the record id (src/leeth3/questions.py), so it says nothing
        # about the example: two records with the same state and options are the same example whatever their questions.
        q = "" if self.domain in QUESTION_FREE_DOMAINS else self.question
        return f"{self.state}\n{q}\n" + "\n".join(self.options)

    def to_json(self) -> str:
        d = {k: getattr(self, k) for k in ("id", "request", "family", "domain", "lang", "ptype", "state", "question", "options",
                                           "answer", "levels", "abstain_ok", "difficulty", "teacher", "group")}
        if self.spans:
            d["spans"] = [list(s) for s in self.spans]
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def _norm_label(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _options(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        out = {_norm_label(k): ("" if v is None else str(v).strip()) for k, v in raw.items()}
        if len(out) != len(raw):
            raise RecordError("duplicate_options")
    elif isinstance(raw, list):
        labels = [_norm_label(x) for x in raw]
        if len(set(labels)) != len(labels):
            raise RecordError("duplicate_options")
        out = {lab: "" for lab in labels}
    else:
        raise RecordError("bad_options", type(raw).__name__)
    if any(not k for k in out):
        raise RecordError("empty_option")
    return out


def _find_spans(state: str, parts: List[Any]) -> List[Tuple[int, int]]:
    """Each answer substring -> its first occurrence after the previous one (answers are listed in reading order); exact match
    first, then whitespace-insensitive. A substring that is not in the state makes the record unusable."""
    spans: List[Tuple[int, int]] = []
    pos = 0
    for p in parts:
        s = str(p)
        if not s.strip():
            raise RecordError("empty_extract_answer")
        i = state.find(s, pos)
        if i < 0:
            i = state.find(s)                            # listed out of order: accept the first occurrence anywhere
        if i < 0:
            pat = r"\s+".join(re.escape(t) for t in s.split())
            m = re.search(pat, state)
            if not m:
                raise RecordError("extract_not_in_state", s[:60])
            i, j = m.start(), m.end()
        else:
            j = i + len(s)
        spans.append((i, j))
        pos = j
    spans = sorted(set(spans))
    for (a, b), (c, d) in zip(spans, spans[1:]):
        if c < b:
            raise RecordError("extract_overlap")
    return spans


_CJK = r"[぀-ヿ㐀-䶿一-鿿豈-﫿　-〿＀-￯]"
_CJK_GAP = re.compile(rf"(?<={_CJK})[ \t]+(?={_CJK})")


def unspace_cjk(text: str) -> str:
    """Chinese and Japanese are written without spaces between words; a generator asked for a word count inserted them
    ("家 里 养 的 猫"). Spaces between two CJK characters are removed; spaces next to Latin text and digits stay."""
    return _CJK_GAP.sub("", text)


def parse_record(obj: Any) -> Record:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as e:
            raise RecordError("bad_json", str(e)) from None
    if not isinstance(obj, dict):
        raise RecordError("not_an_object")
    missing = [k for k in REQUIRED if not (isinstance(obj.get(k), str) and obj[k].strip())]
    if missing:
        raise RecordError("missing_field", ",".join(missing))
    ptype = obj["ptype"].strip().lower()
    if ptype not in PTYPES:
        raise RecordError("bad_ptype", ptype)
    state, question = obj["state"], obj["question"].strip()
    if obj["lang"].strip().lower() in ("zh", "ja"):
        state, question = unspace_cjk(state), unspace_cjk(question)
        obj = dict(obj)
        if isinstance(obj.get("answer"), list):
            obj["answer"] = [unspace_cjk(str(a)) for a in obj["answer"]]
        elif isinstance(obj.get("answer"), str):
            obj["answer"] = unspace_cjk(obj["answer"])
        opts = obj.get("options")
        if isinstance(opts, dict):
            obj["options"] = {unspace_cjk(str(k)): (unspace_cjk(str(v)) if v is not None else v) for k, v in opts.items()}
        elif isinstance(opts, list):
            obj["options"] = [unspace_cjk(str(o)) for o in opts]
    if len(state) > MAX_STATE_CHARS:
        raise RecordError("state_too_long")
    rec = Record(id=obj["id"].strip(), request=obj["request"].strip(), family=obj["family"].strip(), domain=obj["domain"].strip(),
                 lang=obj["lang"].strip().lower(), ptype=ptype, state=state, question=question,
                 abstain_ok=bool(obj.get("abstain_ok", False)), difficulty=str(obj.get("difficulty", "")).strip().lower(),
                 teacher=str(obj.get("teacher", "")).strip(), group=str(obj.get("group") or "").strip())
    ans = obj.get("answer")

    if ptype in ("choice", "multi"):
        rec.options = _options(obj.get("options"))
        k = len(rec.options) + (1 if ptype == "choice" and rec.abstain_ok else 0)
        if not 2 <= k <= MAX_OPTIONS:
            raise RecordError("option_count", str(len(rec.options)))
        if ptype == "choice":
            if ans is None:
                if not rec.abstain_ok:
                    raise RecordError("missing_answer")
                rec.answer = None
            else:
                a = _norm_label(ans)
                if a not in rec.options:
                    raise RecordError("answer_not_an_option", a[:60])
                rec.answer = a
        else:
            if not isinstance(ans, list):
                raise RecordError("multi_answer_not_list")
            a = [_norm_label(x) for x in ans]
            if any(x not in rec.options for x in a):
                raise RecordError("answer_not_an_option")
            if len(set(a)) != len(a):
                raise RecordError("duplicate_answer")
            rec.answer = sorted(a, key=list(rec.options).index)
    elif ptype == "noul":
        if isinstance(ans, bool):
            rec.answer = 1.0 if ans else 0.0
        elif isinstance(ans, (int, float)) and 0.0 <= float(ans) <= 1.0:
            rec.answer = float(ans)
        else:
            raise RecordError("bad_noul_answer", repr(ans)[:40])
    elif ptype == "score":
        lv = obj.get("levels")
        if not isinstance(lv, int) or isinstance(lv, bool) or not MIN_LEVELS <= lv <= MAX_LEVELS:
            raise RecordError("bad_levels", repr(lv))
        if isinstance(ans, bool) or not isinstance(ans, (int, float)) or not 1 <= float(ans) <= lv:
            raise RecordError("bad_score_answer", repr(ans)[:40])
        rec.levels, rec.answer = lv, float(ans)
    else:  # extract
        if ans is None:
            ans = []
        if not isinstance(ans, list):
            raise RecordError("extract_answer_not_list")
        rec.spans = _find_spans(state, ans)
        rec.answer = [state[a:b] for a, b in rec.spans]
    return rec


def none_text(rec: Record) -> str:
    texts = NONE_TEXTS.get(rec.lang, NONE_TEXTS["en"])
    return texts[sum(map(ord, rec.id)) % len(texts)]


def option_text(label: str, desc: str) -> str:
    """How an option reaches the model: the Leeth 2 encoder reads option strings only, so the description travels inside it - the
    form the exam runner and a client with a described tool catalog use ("name: what it does")."""
    return f"{label}: {desc}" if desc else label


def to_samples(rec: Record, split: str = "train") -> List[Sample]:
    """The record as Leeth 2 training rows (what the current trainer reads)."""
    meta: Dict[str, Any] = {"record": rec.id, "request": rec.request, "teacher": rec.teacher, "difficulty": rec.difficulty}
    if any(rec.options.values()):
        meta["option_desc"] = dict(rec.options)
    task, fam = f"l3_{rec.request}", rec.domain
    if rec.ptype == "choice":
        labels = list(rec.options) + ([None] if rec.abstain_ok else [])
        opts = [option_text(lab, rec.options[lab]) if lab is not None else none_text(rec) for lab in labels]
        target = labels.index(rec.answer)
        return [Sample(task, fam, rec.lang, "choice", rec.state, rec.question, opts, target, split=split, meta=meta)]
    if rec.ptype == "multi":
        chosen = set(rec.answer)
        out = []
        for lab, desc in rec.options.items():
            q = f"{rec.question}\nПодходит ли вариант: {lab}" + (f" ({desc})" if desc else "") + "?"
            out.append(Sample(task, fam, rec.lang, "noul", rec.state, q, None, 1.0 if lab in chosen else 0.0, split=split,
                              meta={**meta, "multi_option": lab}))
        return out
    if rec.ptype == "noul":
        return [Sample(task, fam, rec.lang, "noul", rec.state, rec.question, None, rec.answer, split=split, meta=meta)]
    if rec.ptype == "score":
        return [Sample(task, fam, rec.lang, "score", rec.state, rec.question, None, rec.answer, levels=rec.levels, split=split, meta=meta)]
    target = [list(s) for s in rec.spans] if rec.spans else None
    return [Sample(task, fam, rec.lang, "extract", rec.state, rec.question, None, target, split=split, meta=meta)]
