"""Template padding in a teacher's option list: the label is the only real option, the rest is filler.

Found 2026-10-19 in every big tool-choice batch (B0022 ... B0170, 128-254 options): the teacher wrote ~10-60 real tools and padded
the list with generated ones - ``herramienta_paquetes_042: "Función del registro de paquetes ... 42"`` (one description, a new
number), or a cartesian product ``engine.<category>.<action>: "<action> in <category> category"``. In B0046 631 of 634 options were
such filler. A model trained on it learns "the answer is the option that is not a template" - the 255-option skill (D5) reduced to
spotting the odd one out. Big catalogs are now built from real options of other records (src/leeth3/expand.py); here the filler
is recognised and removed.

An option is filler when its description SKELETON - lower case, without digits, identifiers (words with ``_`` or ``.``) and the
words of its own label - is shared by at least ``MIN_TEMPLATE`` options of the list, and none of them is a right answer (when the
answer itself is one of a product, the product is the contrast set of the task, not padding).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, Optional, Set, Tuple

MIN_TEMPLATE = 4

_SPLIT = re.compile(r"[._\-\s/:]+")
_TOKEN = re.compile(r"[^\W\d_]+(?:[._][^\W\d_]+)*|\S", re.UNICODE)


def skeleton(label: str, desc: str) -> str:
    own = {w.lower() for w in _SPLIT.split(label) if w}
    words = []
    for tok in _TOKEN.findall(desc.lower()):
        if "_" in tok or "." in tok or tok in own or not any(ch.isalpha() for ch in tok):
            continue
        words.append(tok)
    return " ".join(words)


def padding_options(options: Dict[str, str], answers: Iterable[str] = ()) -> Set[str]:
    """Labels of template filler in ``options`` (label -> description)."""
    keep = set(answers)
    groups: Dict[str, list] = defaultdict(list)
    for lab, desc in options.items():
        if desc:
            groups[skeleton(lab, desc)].append(lab)
    out: Set[str] = set()
    for sk, labs in groups.items():
        if len(labs) >= MIN_TEMPLATE and not keep.intersection(labs):
            out.update(labs)
    return out


MIN_REAL = 4        # fewer real options left: the record was a padded list around its answer, nothing to choose between


def strip_padding(rec) -> Tuple[Optional[object], int]:
    """(the record without filler options or None when too few real ones remain, how many options were removed)."""
    if rec.ptype not in ("choice", "multi") or not rec.options:
        return rec, 0
    answers = [rec.answer] if isinstance(rec.answer, str) else list(rec.answer or [])
    pad = padding_options(rec.options, answers)
    if not pad:
        return rec, 0
    if len(rec.options) - len(pad) < MIN_REAL:
        return None, len(pad)
    rec.options = {lab: desc for lab, desc in rec.options.items() if lab not in pad}
    return rec, len(pad)
