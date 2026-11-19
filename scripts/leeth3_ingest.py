"""Take teacher batches (Gemini in Antigravity, a local model) into the Leeth 3 corpus - or refuse them, with the reason.

    python scripts/leeth3_ingest.py [--data leeth3_data] [--exams ~/.argent/leeth3_exams] [--dry-run]

Every new ``<data>/incoming/*.jsonl`` is read once (tracked by content hash). A line is accepted only when it
  * parses and validates (src/leeth3/records.py: labels are option texts / exact substrings, counts and ranges are right);
  * does not belong to a family reserved for the exams (src/leeth3/pools.py);
  * is not an exact or near duplicate of anything already accepted (a generator repeating itself);
  * does not resemble any exam - the old batteries in scripts/ and every Leeth 3 exam (kept OUTSIDE the folder the generator
    can read). A leak is a near duplicate of an exam item, or a short exam request contained word for word;
  * is not named in ``<data>/review_rejects.json`` - the reviewer's {id: reason} for defects of meaning no gate can see.
Accepted lines go to ``<data>/accepted/records.jsonl``, refused ones to ``<data>/rejected/<batch>.jsonl`` with the reason, and a
report per batch to ``<data>/reports/<batch>.json`` - including the answer-position histogram (a generator that always puts the
right option first teaches a shortcut; the trainer shuffles options, but the report shows the habit).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leeth3.dedup import NearDupIndex, exact_key, normalise  
from src.leeth3.pools import HOLDOUT, SAFETY_HOLDOUT_PROGRAMS  
from src.leeth3.hints import norm_in_fact, verdict_in_state  
from src.leeth3.padding import strip_padding  
from src.leeth3.probes import is_user_line  
from src.leeth3.programs import action_program  
from src.leeth3.questions import NEUTRAL, neutral_question  
from src.leeth3.records import Record, RecordError, parse_record  

HELD = {req: {code for code, _ in pool} for req, pool in HOLDOUT.items()}   # exam-only families, per request
_HELD_PROGRAM = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(p) for p in SAFETY_HOLDOUT_PROGRAMS) + r")(?![\w-])", re.I)

DEFAULT_DATA = ROOT / "leeth3_data"
DEFAULT_EXAMS = Path.home() / ".argent" / "leeth3_exams"
TRAIN_DUP = 0.80          # near-duplicate inside the training data (one inserted word in a 25-word text is ~0.82)
EXAM_LEAK = 0.60          # resemblance to an exam item that counts as a leak (stricter: an exam must stay unseen)
MIN_CONTAIN_WORDS = 3     # a short exam request (>= 3 words) found word for word inside a training line is a leak
SHORTCUT_MAX = 0.65       # the word-overlap rule should not predict a batch's yes/no labels much better than chance


def legacy_exam_texts(scripts: Path = ROOT / "scripts") -> List[str]:
    """The requests and commands of the Leeth 2 batteries: they stay exams for the whole Leeth 3 plan too."""
    out: List[str] = []

    def load(name: str) -> Any:
        p = scripts / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    for name in ("prod_battery.json", "prod_battery_v2.json"):
        d = load(name)
        out += [c["query"] for c in (d or {}).get("cases", [])]
    s = load("safety_exam_v2.json") or {}
    out += [c for k in ("dangerous", "benign", "lookalike", "review") for c in s.get(k, [])]
    o = load("ood_exam_v3.json") or {}
    out += [c["query"] for c in o.get("routing", [])]
    out += [c for k in ("dangerous", "benign", "lookalike") for c in o.get("safety", {}).get(k, [])]
    return [t for t in out if isinstance(t, str) and t.strip()]


def leeth3_exam_texts(exams: Path) -> List[str]:
    out: List[str] = []
    if not exams.exists():
        return out
    for p in sorted(exams.rglob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(d.get("state") or d.get("question") or "")      # the content an exam item is about is its state
    return [t for t in out if t.strip()]


class Gate:
    """Everything an incoming line is checked against; built once per run."""

    def __init__(self, accepted: Iterable[Record], exam_texts: List[str]):
        self.exact = set()
        self.group_of: Dict[str, str] = {}
        self.train = NearDupIndex(TRAIN_DUP)
        for r in accepted:
            self.remember(r)
        self.exam = NearDupIndex(EXAM_LEAK)
        self.short: List[str] = []
        for i, t in enumerate(exam_texts):
            self.exam.add(f"exam{i}", t)
            n = normalise(t)
            if len(n.split()) >= MIN_CONTAIN_WORDS:
                self.short.append(n)

    def remember(self, r: Record) -> None:
        t = r.text_for_dedup()
        self.exact.add(exact_key(t))
        self.train.add(r.id, t)
        self.group_of[r.id] = r.group_key

    def check(self, r: Record) -> Optional[str]:
        if r.family in HELD.get(r.request, set()):
            return "holdout_family"
        if r.domain == "safety" and _HELD_PROGRAM.search(r.state):
            return "holdout_program"
        if r.domain == "safety" and verdict_in_state(r.state):
            return "verdict_in_state"
        t = r.text_for_dedup()
        if exact_key(t) in self.exact:
            return "duplicate"
        if self.exam.query(r.state) is not None:                # a leak is about content: the record's state against the exam's
            return "exam_leak"
        nt = " " + normalise(f"{r.state} {r.question}") + " "
        if any(f" {s} " in nt for s in self.short if len(s) <= len(nt)):
            return "exam_leak"
        same_group = (lambda key: self.group_of.get(key) == r.group_key) if r.group_key else None   # questions on one text, twins
        if self.train.query(t, ignore=same_group) is not None:
            return "near_duplicate"
        return None


# ---- the batch plan (scripts/leeth3_queue.py) ---------------------------------------------------------------------------------------
def word_count(text: str, lang: str) -> float:
    if lang in ("zh", "ja"):                                    # no spaces between words: ~2 characters per word
        return len("".join(text.split())) / 2.0
    return float(len(text.split()))


def plan_violation(r: Record, row: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a record does not follow its row of the batch plan, or None."""
    if row is None:
        return "plan_unknown_id"
    if r.ptype != row["ptype"]:
        return "plan_ptype"
    if r.family != row["family"] or r.lang != row["lang"]:
        return "plan_family_or_lang"
    if row.get("group") and r.group != row["group"]:
        return "plan_group"
    lo, hi = row["length"]
    if not 0.7 * lo <= word_count(r.state, r.lang) <= 1.5 * hi:
        return "plan_length"
    want = str(row["answer"])
    if r.ptype == "noul":
        p = float(r.answer)
        ok = {"да": p >= 0.7, "нет": p <= 0.3, "спорно": 0.2 <= p <= 0.8}.get(want, True)
        return None if ok else "plan_answer"
    if r.ptype == "choice":
        if len(r.options) != int(row["options"]):
            return "plan_options"
        if want == "null" and r.answer is not None or want == "инструмент" and r.answer is None:
            return "plan_answer"
        return None
    if r.ptype == "multi":
        n, k = (int(x) for x in want.split(" из "))
        if len(r.options) != k:
            return "plan_options"
        return None if len(r.answer) == n else "plan_answer"
    if r.ptype == "score":
        lv_want, n = int(row["levels"]), int(want.split(" из ")[0])
        if r.levels != lv_want:
            return "plan_options"
        return None if round(r.answer) == n else "plan_answer"
    if r.ptype == "extract":
        n, got = int(want), len(r.spans)
        ok = got == n if n <= 1 else (got >= 2 and abs(got - n) <= 2)
        return None if ok else "plan_answer"
    return None


def broken_paraphrase_groups(records: List[Record], plan: Optional[Dict[str, Dict[str, Any]]]) -> set:
    """Paraphrase groups (plan twist "перефраз") whose records do not share ONE tool set and ONE answer. Such a group teaches nothing
    about wording - the need itself changed - so it is refused whole."""
    if not plan:
        return set()
    seen: Dict[str, set] = {}
    for r in records:
        row = plan.get(r.id) or {}
        if row.get("twist") == "перефраз" and r.group:
            seen.setdefault(r.group, set()).add((frozenset((r.options or {}).items()), r.answer))
    return {g for g, variants in seen.items() if len(variants) > 1}


# Twins whose two records must end with the same action: what differs is the user's words, the condition's state in the context,
# or one fact in the middle of a long context - never the command.
SAME_ACTION_TWISTS = ("в словах пользователя", "в условии", "в факте", "факт в середине контекста")


def plan_meta(data: Path, batch: str) -> Dict[str, Any]:
    """The batch-level fields of a plan (``avoid_programs``, ``max_units_per_program``), {} without a plan."""
    p = data / "plans" / f"{batch}.json"
    if not p.exists():
        return {}
    return {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k != "rows"}


def plan_avoid(data: Path, batch: str) -> set:
    """Programs the batch plan says not to run (the corpus's most frequent ones, scripts/leeth3_queue.py)."""
    return set(plan_meta(data, batch).get("avoid_programs", []))


def review_rejects(data: Path) -> Dict[str, str]:
    """{record id: reason} refused by the round's reviewer for what no gate can see. Round 18: 4 of 36 fact pairs had a fact the
    command does not depend on (a running `cargo build` deciding `pacman -Sc`, 140 000 archives deciding `find -name "*.tmp"
    -delete`); an object-name overlap rule missed two of them and failed 12 good pairs, so the reviewer names them. A twin pair
    goes whole: refusing one record refuses its group."""
    p = data / "review_rejects.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def review_accepted(data: Path, reviewed: Dict[str, str], dry_run: bool = False) -> int:
    """Take the reviewer's refusals out of the accepted corpus too, twin with twin (a group is per batch). Round 19 traced a plan
    reason that asked for "a minor setting without an explicit request" to 6 records already taken in where the user asked to look
    and the agent changed something (`wrangler deploy` for "check the status") - the label 0.5 is wrong there, not the gates.
    The records go to rejected/accepted.review.jsonl. Idempotent: returns how many records were taken out."""
    path = data / "accepted" / "records.jsonl"
    recs = read_accepted(path)
    by_group = {(r.id.split("-")[0], r.group) for r in recs if r.id in reviewed and r.group}
    gone = lambda r: r.id in reviewed or (r.id.split("-")[0], r.group) in by_group   # noqa: E731
    out = [r for r in recs if gone(r)]
    if out and not dry_run:
        (data / "rejected").mkdir(parents=True, exist_ok=True)
        with (data / "rejected" / "accepted.review.jsonl").open("a", encoding="utf-8") as f:
            f.write("".join(r.to_json() + "\n" for r in out))
        path.write_text("".join(r.to_json() + "\n" for r in recs if not gone(r)), encoding="utf-8")
    return len(out)


def repeated_programs(records: List[Record], limit: int) -> Tuple[set, set]:
    """(groups, ungrouped ids) of safety units - a twin pair or a single record - beyond the ``limit``-th unit of the same program
    in the batch, in record order. Round 15: with the 25 most frequent programs excluded, the teacher moved to new favourites
    (chown 15 times, unlink 9, gh 9); one program per unit would have cut 68 of 288 records."""
    if not limit:
        return set(), set()
    units: Dict[str, List[Record]] = {}
    for r in records:
        if r.domain == "safety":
            units.setdefault(r.group or r.id, []).append(r)
    seen: Counter = Counter()
    groups, ids = set(), set()
    for key, rs in units.items():
        progs = {action_program(r.state) for r in rs} - {""}
        seen.update(progs)
        if any(seen[p] > limit for p in progs):
            (groups.add(key) if rs[0].group else ids.add(key))
    return groups, ids


def overused(records: List[Record], avoid: set) -> Tuple[set, set]:
    """(groups, ungrouped ids) of safety records whose action runs a program the plan excluded. A twin pair goes whole: its other
    record alone would be a lone example of a pair's contrast."""
    groups, ids = set(), set()
    for r in records:
        if r.domain == "safety" and avoid and action_program(r.state) in avoid:
            (groups.add(r.group) if r.group else ids.add(r.id))
    return groups, ids


def has_user_words(state: str) -> bool:
    """A part before the action holds what a person said (a speaker label, src/leeth3/probes.py). Rounds 11-15: found in all
    1216 safety records that had them.

    Chinese and Japanese records are written on ONE line with 。 between the parts (round 26, B0761: `…用户：“清理邮件队列。”代理执行：
    postsuper -d ALL`), so the label sits mid-line and a line-based check reported "the user never spoke" for 10 good records
    and refused them. One line: split on sentence ends and judge every part but the last (the action)."""
    lines = [l for l in state.strip().splitlines() if l.strip()]
    if len(lines) > 1:
        return any(is_user_line(l) for l in lines[:-1])
    parts = [p.strip() for p in re.split(r"[。｡！？!?]", state) if p.strip()]
    return any(is_user_line(p) for p in parts[:-1])


def words_mismatch(records: List[Record], plan: Optional[Dict[str, Dict[str, Any]]]) -> Tuple[set, set]:
    """(groups, ungrouped ids) of safety records whose user's words are not as the plan row says (``words``: true / false).
    Asked in words for "at least two thirds with the user's words, about 10 % the action alone", the teacher wrote 0 bare actions of
    1216 in rounds 11-15 and 99 of 288 in round 16 (every short row, twins included). A pair goes whole."""
    groups, ids = set(), set()
    for r in records:
        row = (plan or {}).get(r.id) or {}
        if r.domain == "safety" and "words" in row and has_user_words(r.state) != bool(row["words"]):
            (groups.add(r.group) if r.group else ids.add(r.id))
    return groups, ids


FACT_TWISTS = ("в факте", "факт в середине контекста")


def broken_fact_twins(records: List[Record], plan: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, str]:
    """{group: reason} of fact twins that are not "the same text, ONE fact line differs". ``fact_not_one_line``: more than one block
    or more than two lines differ, or (long twins) the difference sits in the last 15 % of the context; ``fact_is_verdict``: the
    differing line states a norm or a forecast (src/leeth3/hints.py). Round 17: long twins changed a block of 3-5 lines in all 24
    pairs (rounds 14-16: 72 of 72 within one or two lines), and 18 of 36 fact pairs wrote the verdict as the fact."""
    pairs: Dict[str, List[Record]] = {}
    long_: set = set()
    for r in records:
        twist = ((plan or {}).get(r.id) or {}).get("twist")
        if r.domain == "safety" and twist in FACT_TWISTS and r.group:
            pairs.setdefault(r.group, []).append(r)
            if twist == FACT_TWISTS[1]:
                long_.add(r.group)
    out: Dict[str, str] = {}
    for g, rs in pairs.items():
        if len(rs) != 2:
            continue
        la = [l for l in rs[0].state.strip().splitlines() if l.strip()]
        lb = [l for l in rs[1].state.strip().splitlines() if l.strip()]
        ops = [o for o in difflib.SequenceMatcher(None, la, lb, autojunk=False).get_opcodes() if o[0] != "equal"]
        if not ops:
            continue
        n = max(sum(i2 - i1 for _, i1, i2, _, _ in ops), sum(j2 - j1 for _, _, _, j1, j2 in ops))
        end = (max(i2 for _, _, i2, _, _ in ops) - 1) / max(1, len(la) - 1)
        if len(ops) > 1 or n > 2 or (g in long_ and end > 0.85):
            out[g] = "fact_not_one_line"
            continue
        diff = la[ops[0][1]:ops[0][2]] + lb[ops[0][3]:ops[0][4]]
        if any(norm_in_fact(l) for l in diff):
            out[g] = "fact_is_verdict"
    return out


def twins_of_differing_action(records: List[Record], plan: Optional[Dict[str, Dict[str, Any]]]) -> set:
    """Safety twins of SAME_ACTION_TWISTS whose last line (the agent's action) is not the same in both records. When the command
    changes too, the pair is a read command against a write command and the label is read off the command alone (round 12: 28 of
    30 "в словах пользователя" pairs, 22 of them with a prohibition in the "no" twin's words). Refused whole."""
    if not plan:
        return set()
    seen: Dict[str, set] = {}
    for r in records:
        row = plan.get(r.id) or {}
        if r.domain == "safety" and row.get("twist") in SAME_ACTION_TWISTS and r.group:
            lines = r.state.strip().splitlines()
            seen.setdefault(r.group, set()).add(lines[-1].strip() if lines else "")
    return {g for g, actions in seen.items() if len(actions) > 1}


# A yes/no question is either marked as a question (an auxiliary first, a particle) or a statement with "?". B0083-B0086 wrote the
# "no" question as a statement and the "yes" question with the marker (English statements: "no" 11 of 11; Russian "ли": "yes" 11 of
# 12), so the form alone gave the answer. Both questions of a pair must share one form; only unambiguous markers are recognised.
_Q_MARK = {
    "en": re.compile(r"^\s*(did|does|do|is|are|was|were|has|have|had|can|could|will|would|should|may|might|must)\b", re.I),
    "ru": re.compile(r"\bли\b", re.I), "uk": re.compile(r"\bчи\b", re.I), "pl": re.compile(r"^\s*czy\b", re.I),
    "tr": re.compile(r"\bm[ıiuü](\w*)?\s*\?", re.I), "kk": re.compile(r"\b(ма|ме|ба|бе|па|пе)\s*\?", re.I),
    "zh": re.compile(r"(吗|是否|是不是|有没有)"), "ja": re.compile(r"か\s*[?？]?\s*$"),
    "es": re.compile(r"^\s*¿"),
}


def question_form(question: str, lang: str) -> Optional[str]:
    pat = _Q_MARK.get(lang)
    if pat is None:
        return None
    return "marked" if pat.search(question) else "statement"


def pairs_of_differing_form(records: List[Record]) -> set:
    """Content yes/no pairs whose two questions differ in grammatical form (only languages with an unambiguous marker)."""
    by_group: Dict[str, List[Record]] = {}
    for r in records:
        if r.domain == "content" and r.ptype == "noul" and r.group:
            by_group.setdefault(r.group, []).append(r)
    out = set()
    for g, rs in by_group.items():
        forms = {question_form(r.question, r.lang) for r in rs}
        if len(rs) == 2 and None not in forms and len(forms) > 1:
            out.add(g)
    return out


def answer_position(r: Record) -> Optional[str]:
    if r.ptype != "choice" or r.answer is None:
        return None
    i, k = list(r.options).index(r.answer), len(r.options)
    return "first" if i == 0 else "last" if i == k - 1 else "middle"


def shuffled(r: Record) -> Record:
    """The options in an order that does not depend on the generator: round 2 put the right tool first 45 times in 46. The order
    is a deterministic function of the record id, so a re-ingest reproduces the corpus byte for byte."""
    if r.ptype in ("choice", "multi") and len(r.options) > 1:
        items = list(r.options.items())
        random.Random(r.id).shuffle(items)
        r.options = dict(items)
        if r.ptype == "multi":
            r.answer = sorted(r.answer, key=list(r.options).index)
    return r


def load_plan(data: Path, batch: str) -> Optional[Dict[str, Dict[str, Any]]]:
    p = data / "plans" / f"{batch}.json"
    if not p.exists():
        return None
    return {row["id"]: row for row in json.loads(p.read_text(encoding="utf-8"))["rows"]}


def overlap_rule_accuracy(records: List[Record]) -> Optional[float]:
    """How well "answer yes when the question shares words with the text" alone predicts the yes/no labels of a batch. A high value
    means the batch teaches that shortcut (pilot R003: 0.80); paired questions on one text bring it to chance."""
    rows = [r for r in records if r.ptype == "noul" and (r.answer >= 0.7 or r.answer <= 0.3)]
    if len(rows) < 10:
        return None
    stem = lambda w: w.lower()[:5]                               # noqa: E731  (crude, language-agnostic)
    words = lambda s: {stem(w) for w in re.findall(r"\w{4,}", s)}   # noqa: E731
    ov = [(len(words(r.question) & words(r.state)) / max(1, len(words(r.question))), r.answer >= 0.7) for r in rows]
    return max(sum((o >= t) == y for o, y in ov) / len(ov) for t in [i / 20 for i in range(21)])


def neutralise(r: Record) -> Record:
    """Safety and tool records ask a neutral question (src/leeth3/questions.py): the teacher's own question leaked the label."""
    if r.domain in NEUTRAL:
        r.question = neutral_question(r.domain, r.id, r.lang)
    return r


def read_accepted(path: Path) -> List[Record]:
    if not path.exists():
        return []
    return [parse_record(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def drop_accepted(data: Path, batches: Iterable[str]) -> Dict[str, int]:
    """Take whole batches back out of the accepted corpus when a defect is found after they were taken in (2026-10-19: every long
    R001 record ended with a narrator's sentence on what the action would or would not destroy - the label, said in the text).
    The records go to rejected/<batch>.dropped.jsonl; the batch stays in ingested.json, so it is never read again."""
    path = data / "accepted" / "records.jsonl"
    want = set(batches)
    keep, gone = [], {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if not line.strip():
            continue
        b = json.loads(line)["id"].split("-")[0]
        if b in want:
            gone.setdefault(b, []).append(line)
        else:
            keep.append(line)
    if gone:
        (data / "rejected").mkdir(parents=True, exist_ok=True)
        for b, lines in gone.items():
            (data / "rejected" / f"{b}.dropped.jsonl").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    # A refused batch that was never taken in is marked as read, so no later run picks it up (no --skip list to remember).
    seen_path = data / "accepted" / "ingested.json"
    seen: Dict[str, str] = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.exists() else {}
    for b in sorted(want):
        f = data / "incoming" / f"{b}.jsonl"
        if f.exists() and f.name not in seen:
            seen[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
            gone.setdefault(b, [])
    if seen:
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps(seen, indent=1), encoding="utf-8")
    return {b: len(v) for b, v in gone.items()}


def refilter_accepted(data: Path, dry_run: bool = False) -> int:
    """Apply the state checks added after a batch was taken in (verdict_in_state) to the accepted corpus too; the refused records
    go to rejected/accepted.verdict_in_state.jsonl. Idempotent: returns how many records were taken out."""
    path = data / "accepted" / "records.jsonl"
    if not path.exists():
        return 0
    keep, out = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = parse_record(line)
        (out if r.domain == "safety" and verdict_in_state(r.state) else keep).append(line)
    if out and not dry_run:
        (data / "rejected").mkdir(parents=True, exist_ok=True)
        with (data / "rejected" / "accepted.verdict_in_state.jsonl").open("a", encoding="utf-8") as f:
            f.write("".join(l + "\n" for l in out))
        path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
    return len(out)


def depad_accepted(data: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Remove template filler options from the accepted corpus (src/leeth3/padding.py); records with too few real options left go
    to rejected/accepted.padded_catalog.jsonl. Idempotent: returns (options removed, records taken out)."""
    path = data / "accepted" / "records.jsonl"
    recs = read_accepted(path)
    keep, out, removed = [], [], 0
    for r in recs:
        line = r.to_json()
        fixed, n = strip_padding(r)
        removed += n
        (keep.append(fixed) if fixed is not None else out.append(line))
    if removed and not dry_run:
        if out:
            (data / "rejected").mkdir(parents=True, exist_ok=True)
            with (data / "rejected" / "accepted.padded_catalog.jsonl").open("a", encoding="utf-8") as f:
                f.write("".join(l + "\n" for l in out))
        path.write_text("".join(r.to_json() + "\n" for r in keep), encoding="utf-8")
    return removed, len(out)


def neutralise_accepted(path: Path, dry_run: bool = False) -> int:
    """Bring the accepted corpus to the same rule (idempotent): returns how many questions were replaced."""
    recs = read_accepted(path)
    changed = 0
    for r in recs:
        before = r.question
        if neutralise(r).question != before:
            changed += 1
    if changed and not dry_run:
        path.write_text("".join(r.to_json() + "\n" for r in recs), encoding="utf-8")
    return changed


def batch_report(name: str, accepted: List[Record], rejected: List[Tuple[str, str]], plan_missing: int = 0) -> Dict[str, Any]:
    pos = Counter()
    for r in accepted:
        if r.ptype == "choice" and r.answer is not None:
            i, k = list(r.options).index(r.answer), len(r.options)
            pos["first" if i == 0 else "last" if i == k - 1 else "middle"] += 1
    noul = [r.answer for r in accepted if r.ptype == "noul"]
    return {
        "batch": name, "accepted": len(accepted), "rejected": len(rejected),
        "reasons": dict(Counter(reason for reason, _ in rejected).most_common()),
        "by_ptype": dict(Counter(r.ptype for r in accepted)), "by_lang": dict(Counter(r.lang for r in accepted)),
        "by_request": dict(Counter(r.request for r in accepted)), "families": len({(r.request, r.family) for r in accepted}),
        "answer_position": dict(pos), "noul_yes_share": round(sum(1 for a in noul if a >= 0.5) / len(noul), 3) if noul else None,
        "option_counts": dict(Counter(len(r.options) for r in accepted if r.options)),
        "overlap_rule_accuracy": overlap_rule_accuracy([r for r in accepted if r.domain == "content"]), "plan_rows_missing": plan_missing,
    }


def ingest(data: Path, exams: Path, dry_run: bool = False, skip: Iterable[str] = ()) -> List[Dict[str, Any]]:
    inc, acc_dir, rej_dir, rep_dir = data / "incoming", data / "accepted", data / "rejected", data / "reports"
    acc_path, seen_path = acc_dir / "records.jsonl", acc_dir / "ingested.json"
    seen: Dict[str, str] = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.exists() else {}
    reports: List[Dict[str, Any]] = []
    fixed = neutralise_accepted(acc_path, dry_run)
    if fixed:
        reports.append({"batch": "accepted", "neutralised_questions": fixed})
    taken = refilter_accepted(data, dry_run)
    if taken:
        reports.append({"batch": "accepted", "verdict_in_state_removed": taken})
    pad_opts, pad_recs = depad_accepted(data, dry_run)
    if pad_opts:
        reports.append({"batch": "accepted", "padding_options_removed": pad_opts, "padded_records_removed": pad_recs})
    reviewed = review_rejects(data)
    unreviewed = review_accepted(data, reviewed, dry_run)
    if unreviewed:
        reports.append({"batch": "accepted", "review_removed": unreviewed})
    gate = Gate([neutralise(r) for r in read_accepted(acc_path)], legacy_exam_texts() + leeth3_exam_texts(exams))
    skip = set(skip)
    for f in sorted(inc.glob("*.jsonl")) if inc.exists() else []:
        if f.stem in skip:
            continue
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if seen.get(f.name) == digest:
            continue
        if f.name in seen:
            # Rewritten after it was taken in: its records are already in the corpus, so reading it again would add near-copies
            # of the same items next to the old ones. Left untouched and reported until someone decides what to keep.
            reports.append({"batch": f.stem, "changed_after_ingest": True})
            continue
        plan = load_plan(data, f.stem)
        ok: List[Record] = []
        bad: List[Tuple[str, str]] = []
        got_ids = set()
        gen_pos: Counter = Counter()
        padding = 0
        for n, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = neutralise(parse_record(line))
            except RecordError as e:
                bad.append((e.reason, line))
                continue
            got_ids.add(r.id)
            why = plan_violation(r, plan.get(r.id)) if plan is not None else None
            if why is None:                                     # the plan counts the options as written, filler included
                fixed, n = strip_padding(r)
                padding += n
                why = "padded_catalog" if fixed is None else None
            why = why or gate.check(r)
            if why:
                bad.append((why, line))
                continue
            gate.remember(r)
            gen_pos.update([answer_position(r)] if answer_position(r) else [])     # the generator's habit, before the shuffle
            ok.append(r)
        by_group = {r.group: reviewed[r.id] for r in ok if r.id in reviewed and r.group}
        rv_why = lambda r: reviewed.get(r.id) or (by_group.get(r.group) if r.group else None)   # noqa: E731
        bad += [(f"review_{rv_why(r)}", r.to_json()) for r in ok if rv_why(r)]
        ok = [r for r in ok if not rv_why(r)]
        broken = broken_paraphrase_groups(ok, plan)
        bad += [("paraphrase_group_differs", r.to_json()) for r in ok if r.group in broken]
        ok = [r for r in ok if r.group not in broken]
        wg, wi = words_mismatch(ok, plan)
        w_hit = lambda r: r.domain == "safety" and (r.group in wg if r.group else r.id in wi)   # noqa: E731
        bad += [("user_words_mismatch", r.to_json()) for r in ok if w_hit(r)]
        ok = [r for r in ok if not w_hit(r)]
        facts = broken_fact_twins(ok, plan)
        bad += [(facts[r.group], r.to_json()) for r in ok if r.domain == "safety" and r.group in facts]
        ok = [r for r in ok if not (r.domain == "safety" and r.group in facts)]
        moved = twins_of_differing_action(ok, plan)
        bad += [("twin_action_differs", r.to_json()) for r in ok if r.domain == "safety" and r.group in moved]
        ok = [r for r in ok if not (r.domain == "safety" and r.group in moved)]
        og, oi = overused(ok, plan_avoid(data, f.stem))
        hit = lambda r: r.domain == "safety" and (r.group in og if r.group else r.id in oi)   # noqa: E731
        bad += [("overused_program", r.to_json()) for r in ok if hit(r)]
        ok = [r for r in ok if not hit(r)]
        rg, ri = repeated_programs(ok, int(plan_meta(data, f.stem).get("max_units_per_program", 0)))
        rep_hit = lambda r: r.domain == "safety" and (r.group in rg if r.group else r.id in ri)   # noqa: E731
        bad += [("program_repeated", r.to_json()) for r in ok if rep_hit(r)]
        ok = [r for r in ok if not rep_hit(r)]
        odd = pairs_of_differing_form(ok)
        bad += [("pair_form_differs", r.to_json()) for r in ok if r.domain == "content" and r.group in odd]
        ok = [r for r in ok if not (r.domain == "content" and r.group in odd)]
        # Content yes/no pairs that the word-overlap rule answers teach exactly that rule (the pilot's failed content head, and
        # round 4 again: "no" questions made of details the text never mentions). A warning did not stop it, so they are refused;
        # the batch's multi and score rows stay.
        pairs = [r for r in ok if r.domain == "content" and r.ptype == "noul"]
        acc = overlap_rule_accuracy(pairs)
        if acc is not None and acc > SHORTCUT_MAX:
            bad += [("shortcut_pairs", r.to_json()) for r in pairs]
            ok = [r for r in ok if not (r.domain == "content" and r.ptype == "noul")]
        ok = [shuffled(r) for r in ok]
        rep = batch_report(f.stem, ok, bad, plan_missing=len(set(plan) - got_ids) if plan is not None else 0)
        rep["answer_position"] = dict(gen_pos)
        if padding:
            rep["padding_options_removed"] = padding
        if acc is not None:
            rep["overlap_rule_accuracy"] = acc                  # measured on the pairs as written, refused or not
        reports.append(rep)
        if dry_run:
            continue
        for d in (acc_dir, rej_dir, rep_dir):
            d.mkdir(parents=True, exist_ok=True)
        with acc_path.open("a", encoding="utf-8") as out:
            for r in ok:
                out.write(r.to_json() + "\n")
        if bad:
            (rej_dir / f"{f.stem}.jsonl").write_text(
                "".join(json.dumps({"reason": why, "line": line}, ensure_ascii=False) + "\n" for why, line in bad), encoding="utf-8")
        (rep_dir / f"{f.stem}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        seen[f.name] = digest
        seen_path.write_text(json.dumps(seen, indent=1), encoding="utf-8")
        _file_batch_away(data, f.stem)
    return reports


def _file_batch_away(data: Path, batch: str) -> None:
    """Move the batch's task file to queue/done/ once it has been read. AGENT.md asks the teacher to do it, and round 26 shows it
    does not always: 24 already-read task files were left in queue/, so the next round would have written them again."""
    src = data / "queue" / f"{batch}.md"
    if not src.exists():
        return
    dst = data / "queue" / "done" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        src.unlink()
    else:
        src.replace(dst)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--exams", default=str(DEFAULT_EXAMS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip", default="", help="comma-separated batches to leave out (e.g. a pilot batch found unfit)")
    ap.add_argument("--drop-accepted", default="", help="comma-separated batches to take back out of the accepted corpus, then exit")
    a = ap.parse_args(argv)
    if a.drop_accepted:
        print("dropped from the accepted corpus:", drop_accepted(Path(a.data), [s.strip() for s in a.drop_accepted.split(",") if s.strip()]))
        return 0
    reps = ingest(Path(a.data), Path(a.exams), a.dry_run, [s.strip() for s in a.skip.split(",") if s.strip()])
    if not reps:
        print("nothing new in incoming/")
    for r in reps:
        if r.get("changed_after_ingest"):
            print(f"{r['batch']}: CHANGED AFTER INGEST - not read again; its accepted records stay as they were")
            continue
        if "verdict_in_state_removed" in r:
            print(f"accepted corpus: {r['verdict_in_state_removed']} safety records removed (verdict written in the state)")
            continue
        if "review_removed" in r:
            print(f"accepted corpus: {r['review_removed']} records removed by the reviewer (review_rejects.json)")
            continue
        if "neutralised_questions" in r:
            print(f"accepted corpus: {r['neutralised_questions']} safety/tool questions replaced by the neutral one")
            continue
        flag ="  SHORTCUT!" if (r["overlap_rule_accuracy"] or 0) > SHORTCUT_MAX else ""
        print(f"{r['batch']}: accepted {r['accepted']}, rejected {r['rejected']} {r['reasons'] or ''}  "
              f"ptype {r['by_ptype']}  lang {r['by_lang']}  answer position {r['answer_position'] or '-'}  "
              f"yes-share {r['noul_yes_share']}  overlap-rule {r['overlap_rule_accuracy']}{flag}"
              + (f"  plan rows missing {r['plan_rows_missing']}" if r["plan_rows_missing"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
