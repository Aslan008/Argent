"""Programmatic expansion of the accepted teacher corpus: new training records whose answer is known by construction.

A teacher round gives ~800 records; the plan needs hundreds of thousands. Two of the weakest Leeth Index dimensions can be
fed from records the ingest already accepted, without new labels:

* **Big catalogs (D5, 255 options on an unfamiliar vocabulary).** A tool-choice record keeps its request, its answer and its own
  options; the catalog is filled up to 32/64/128/255 with options written by the teacher for OTHER families in the same language
  (a model must read descriptions, not recognise a familiar list). Records answered "none of these" are not expanded: a tool from
  another family could fit their request. Options too close to the right one (shared name words, similar description) are kept out.
* **Long context (D6, the decisive text buried in 2k-32k tokens).** The state is surrounded by machine output (telemetry, checksums,
  test runs, access logs...) that is topic-free and has no errors or warnings, so it cannot change the answer of a content, safety or
  tool question. A safety action and a tool request stay LAST (that is how an agent sees them); a content text may stand at the
  start, in the middle or at the end. The filler is not the exam's (scripts/leeth3_exam.py ``neutral_filler``: job logs, regional
  sales, meeting minutes) - the long-context exam must not test a filler the model has learned to skip.

Extraction is not expanded: filler numbers, dates and codes could be answers of an extraction question.
Records of one teacher group (twins, the "yes" and "no" questions on one text) get the same filler at the same place, so the pair
stays a minimal contrast when long. Everything is deterministic from (record id or group, seed): rebuilding after an ingest gives the same records for unchanged sources.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.leeth2.schema import MAX_OPTIONS
from src.leeth3.questions import neutral_question
from src.leeth3.records import QUESTION_FREE_DOMAINS, Record

CATALOG_SIZES = (32, 64, 128, 255)
CATALOG_WEIGHTS = (1, 1, 2, 2)              # the big lists are the ones s70 fails (0.57 @128, 0.62 @255 on new vocabulary)
CATALOGS_PER_RECORD = 2
# Share of catalogs whose right answer is "none of these": the request's own family is taken out and only other families' tools
# stay. Pilot 2026-10-20: with answered catalogs only, recall of "none" on the exam fell 0.72 -> 0.61 (a big list "always has it").
NONE_SHARE = 0.18

LONG_TIERS = {"2k": 2000, "6k": 6000, "16k": 16000, "32k": 30000}   # approximate tokens of the whole state
LONG_WEIGHTS = (3, 3, 2, 1)
LONG_SHARE = {"content": 1.0, "safety": 0.3, "tools": 0.3}          # share of source records that get one long variant


def _rng(*parts: object) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16], 16))


def _source_group(rec: Record) -> str:
    """Variants of one record (and of one teacher group: twins, paraphrases, questions on one text) must land in one split."""
    return f"src:{rec.group_key or rec.id}"


def _derived(rec: Record, new_id: str, **changes) -> Record:
    out = replace(rec, id=new_id, group=_source_group(rec), spans=list(rec.spans), **changes)
    if rec.domain in QUESTION_FREE_DOMAINS:
        out.question = neutral_question(rec.domain, new_id, rec.lang)
    return out


# ---- big catalogs ------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _name_words(label: str) -> frozenset:
    return frozenset(w.lower() for w in re.split(r"[_\-\s.]+", label) if len(w) >= 3)


@lru_cache(maxsize=None)
def _trigrams(text: str) -> frozenset:
    t = re.sub(r"\s+", " ", text.lower())
    return frozenset(t[i:i + 3] for i in range(max(0, len(t) - 2)))


def _too_close(label: str, desc: str, gold_label: str, gold_desc: str) -> bool:
    """A distractor that may be a second right answer: two shared name words (send_invoice_email ~ email_invoice_send) or a
    description much like the right one."""
    if len(_name_words(label) & _name_words(gold_label)) >= 2:
        return True
    a, b = _trigrams(desc), _trigrams(gold_desc)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.45


def catalog_pool(records: Iterable[Record]) -> Dict[str, List[Tuple[str, str, str]]]:
    """lang -> [(label, description, family)], one entry per label (the first the teacher wrote)."""
    pool: Dict[str, Dict[str, Tuple[str, str, str]]] = defaultdict(dict)
    for r in records:
        if r.domain == "tools" and r.ptype == "choice":
            for lab, desc in r.options.items():
                pool[r.lang].setdefault(lab, (lab, desc, r.family))
    return {lang: list(d.values()) for lang, d in pool.items()}


_CAND: Dict[Tuple[int, str], List[Tuple[str, str]]] = {}


def _candidates(rec: Record, pool: Dict[str, List[Tuple[str, str, str]]]) -> List[Tuple[str, str]]:
    key = (id(pool), rec.id)
    if key not in _CAND:
        gold_desc = rec.options[rec.answer]
        _CAND[key] = [(lab, desc) for lab, desc, fam in pool.get(rec.lang, ())
                      if fam != rec.family and lab not in rec.options and not _too_close(lab, desc, rec.answer, gold_desc)]
    return _CAND[key]


def expand_catalog(rec: Record, pool: Dict[str, List[Tuple[str, str, str]]], size: int, seed: int = 0) -> Optional[Record]:
    """The record with its options plus distractors from other families up to ``size`` choices (the implicit "none of these" of an
    abstain record counts). None when the language has too few options or the record cannot be expanded safely."""
    if rec.domain != "tools" or rec.ptype != "choice" or rec.answer is None:
        return None
    room = min(size, MAX_OPTIONS) - len(rec.options) - (1 if rec.abstain_ok else 0)
    if room <= 0:
        return None
    cand = _candidates(rec, pool)
    if len(cand) < room:
        return None
    rng = _rng("catalog", rec.group_key or rec.id, size, seed)     # a paraphrase group shares one tool set - and one big catalog
    items = list(rec.options.items()) + rng.sample(cand, room)
    rng.shuffle(items)
    return _derived(rec, f"XC{size:03d}-{rec.id}", options=dict(items), difficulty="hard")


def none_catalog(rec: Record, pool: Dict[str, List[Tuple[str, str, str]]], size: int, seed: int = 0) -> Optional[Record]:
    """The record's request over a catalog of OTHER families only: none of its own tools is offered, so the answer is "none of
    these" - the same assumption the distractors of ``expand_catalog`` rest on (another family's tool does not do this request)."""
    if rec.domain != "tools" or rec.ptype != "choice" or rec.answer is None or not rec.abstain_ok:
        return None
    room = min(size, MAX_OPTIONS) - 1
    cand = _candidates(rec, pool)
    if len(cand) < room:
        return None
    rng = _rng("none-catalog", rec.group_key or rec.id, size, seed)
    return _derived(rec, f"XN{size:03d}-{rec.id}", options=dict(rng.sample(cand, room)), answer=None, difficulty="hard")


def catalogs(records: Sequence[Record], seed: int = 0, per_record: int = CATALOGS_PER_RECORD) -> List[Record]:
    pool = catalog_pool(records)
    _CAND.clear()
    out: List[Record] = []
    for rec in records:
        if rec.domain != "tools" or rec.ptype != "choice" or rec.answer is None:
            continue
        rng = _rng("catalog-sizes", rec.group_key or rec.id, seed)
        sizes: List[int] = []
        while len(sizes) < min(per_record, len(CATALOG_SIZES)):
            s = rng.choices(CATALOG_SIZES, CATALOG_WEIGHTS)[0]
            if s not in sizes:
                sizes.append(s)
        for s in sorted(sizes):
            x = expand_catalog(rec, pool, s, seed)
            if x is None and s > 32:          # a small language: the biggest catalog its own vocabulary can fill
                for smaller in [c for c in CATALOG_SIZES if c < s and c not in sizes][::-1]:
                    x = expand_catalog(rec, pool, smaller, seed)
                    if x is not None:
                        sizes.append(smaller)
                        break
            if x is not None and _rng("none-pick", rec.group_key or rec.id, len(x.options), seed).random() < NONE_SHARE:
                x = none_catalog(rec, pool, len(x.options) + 1, seed) or x
            if x is not None:
                out.append(x)
    return out


# ---- long context ------------------------------------------------------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    """Gemma splits every digit into its own token; other text is ~4 characters per token."""
    digits = sum(c.isdigit() for c in text)
    return digits + (len(text) - digits) // 4


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _telemetry(rng: random.Random) -> str:
    st = f"ST-{rng.randint(10, 99)}"
    rows = ["ts,station,temp_c,humidity_pct,pressure_hpa,wind_ms"]
    h, m = rng.randint(0, 20), 0
    for _ in range(rng.randint(14, 30)):
        m += rng.choice((5, 10, 15))
        rows.append(f"2026-08-{rng.randint(1, 28):02d}T{(h + m // 60) % 24:02d}:{m % 60:02d},{st},{rng.uniform(-5, 31):.1f},"
                    f"{rng.randint(20, 95)},{rng.randint(985, 1035)},{rng.uniform(0, 14):.1f}")
    return "\n".join(rows)


def _checksums(rng: random.Random) -> str:
    stem = rng.choice(("chunk", "part", "seg", "blk", "tile", "frame"))
    return "\n".join(f"{_hex(rng, 40)}  {stem}_{i:05d}.bin" for i in range(rng.randint(10, 24)))


def _metrics(rng: random.Random) -> str:
    keys = ("p50_ms", "p95_ms", "p99_ms", "rps", "cpu_pct", "rss_mb", "gc_pauses", "open_fds", "threads", "queue_depth")
    blocks = []
    for i in range(rng.randint(3, 7)):
        inner = ", ".join(f'"{k}": {rng.randint(1, 999) if rng.random() < .6 else round(rng.uniform(0, 100), 2)}'
                          for k in rng.sample(keys, rng.randint(4, len(keys))))
        blocks.append(f'  "node-{rng.randint(1, 64):02d}": {{{inner}}}')
    return "{\n" + ",\n".join(blocks) + "\n}"


def _matrix(rng: random.Random) -> str:
    n = rng.randint(6, 12)
    return "\n".join(" ".join(f"{rng.uniform(-1, 1):+.4f}" for _ in range(n)) for _ in range(rng.randint(6, 14)))


def _access_log(rng: random.Random) -> str:
    kinds = (("css", "text/css"), ("js", "application/javascript"), ("png", "image/png"), ("woff2", "font/woff2"), ("svg", "image/svg+xml"))
    rows = []
    for _ in range(rng.randint(10, 22)):
        ext, mime = rng.choice(kinds)
        rows.append(f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)} - - [12/Aug/2026:{rng.randint(0, 23):02d}:"
                    f"{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d} +0000] \"GET /static/{_hex(rng, 8)}.{ext} HTTP/1.1\" 200 "
                    f"{rng.randint(300, 90000)} \"{mime}\"")
    return "\n".join(rows)


def _test_run(rng: random.Random) -> str:
    mod = f"tests/unit/test_{rng.choice(('parse', 'codec', 'grid', 'units', 'format', 'layout', 'merge'))}_{rng.randint(1, 40)}.py"
    n = rng.randint(8, 20)
    rows = [f"{mod}::test_case_{i:03d} PASSED [{round(100 * (i + 1) / n):3d}%]" for i in range(n)]
    rows.append(f"==== {n} passed in {rng.uniform(0.2, 9.9):.2f}s ====")
    return "\n".join(rows)


def _hexdump(rng: random.Random) -> str:
    base = rng.randrange(0, 1 << 20, 16)
    return "\n".join(f"{base + 16 * i:08x}  " + " ".join(_hex(rng, 2) for _ in range(16)) for i in range(rng.randint(8, 20)))


def _track(rng: random.Random) -> str:
    lat, lon = rng.uniform(-60, 70), rng.uniform(-170, 170)
    rows = ["seq;lat;lon;alt_m;sats"]
    for i in range(rng.randint(12, 26)):
        lat += rng.uniform(-.002, .002)
        lon += rng.uniform(-.002, .002)
        rows.append(f"{i};{lat:.6f};{lon:.6f};{rng.randint(0, 2400)};{rng.randint(4, 14)}")
    return "\n".join(rows)


def _usage_table(rng: random.Random) -> str:
    rows = ["volume        size_gb  used_gb  inodes_used"]
    for i in range(rng.randint(5, 12)):
        size = rng.randint(20, 4000)
        rows.append(f"vol{rng.randint(0, 99):02d}-{i:<6d} {size:>8d} {rng.randint(1, size):>8d} {rng.randint(1000, 9_000_000):>12d}")
    return "\n".join(rows)


FILLER_KINDS = (_telemetry, _checksums, _metrics, _matrix, _access_log, _test_run, _hexdump, _track, _usage_table)


def filler_docs(rng: random.Random, tokens: int) -> List[str]:
    docs: List[str] = []
    have = 0
    while have < tokens:
        d = rng.choice(FILLER_KINDS)(rng)
        docs.append(d)
        have += approx_tokens(d) + 1
    return docs


def bury(state: str, docs: List[str], position: str) -> str:
    if position == "start":
        parts = [state] + docs
    elif position == "end":
        parts = docs + [state]
    else:
        k = len(docs) // 2
        parts = docs[:k] + [state] + docs[k:]
    return "\n\n".join(parts)


def expand_long(rec: Record, tier: str, position: str, seed: int = 0) -> Optional[Record]:
    if rec.domain not in LONG_SHARE or (rec.domain != "content" and position != "end"):
        return None
    rng = _rng("long", rec.group_key or rec.id, tier, position, seed)     # one group (twins, a text's two questions) - one filler
    need = LONG_TIERS[tier] - approx_tokens(rec.state)
    if need <= 0:
        return None
    state = bury(rec.state, filler_docs(rng, need), position)
    return _derived(rec, f"XL{tier}{position[0]}-{rec.id}", state=state)


def long_variants(records: Sequence[Record], seed: int = 0) -> List[Record]:
    out: List[Record] = []
    for rec in records:
        share = LONG_SHARE.get(rec.domain, 0.0)
        rng = _rng("long-pick", rec.group_key or rec.id, seed)
        if rng.random() >= share:
            continue
        tier = rng.choices(list(LONG_TIERS), LONG_WEIGHTS)[0]
        position = rng.choice(("start", "middle", "end")) if rec.domain == "content" else "end"
        x = expand_long(rec, tier, position, seed)
        if x is not None:
            out.append(x)
    return out
