"""Small helpers shared by every adapter: text cleaning, option building, sample constructors."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..schema import MAX_OPTIONS, Sample

_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")


def clean(text: Any, max_chars: int = 6000) -> str:
    """Normalise whitespace and cut very long texts at a word boundary."""
    if text is None:
        return ""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n")
    t = _WS.sub(" ", t)
    t = _NL.sub("\n\n", t).strip()
    if len(t) > max_chars:
        cut = t.rfind(" ", 0, max_chars)
        t = t[: cut if cut > max_chars * 0.6 else max_chars].rstrip()
    return t


def humanize(label: Any) -> str:
    """'alarm_set' -> 'alarm set', 'B-PER' stays, numbers become strings."""
    s = str(label).strip()
    if re.fullmatch(r"[A-Za-z0-9]+([_\-.][A-Za-z0-9]+)+", s) and not re.match(r"^[A-Z]-", s):
        s = re.sub(r"[_]+", " ", s)
    return s


def stable_hash(*parts: Any) -> int:
    h = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")


def pick(rng: random.Random, seq: Sequence[Any]) -> Any:
    return seq[rng.randrange(len(seq))]


def subset_options(rng: random.Random, n_classes: int, target: int, k_min: int = 4, k_max: int = 40, full_prob: float = 0.45, force_full: bool = False) -> List[int]:
    """Class indices to show: the full label set when small, by chance or when forced (benchmarks), otherwise a random subset containing the target."""
    if force_full or n_classes <= k_min or (n_classes <= 80 and rng.random() < full_prob):
        return list(range(n_classes))
    k = min(n_classes, rng.randint(k_min, k_max))
    others = [i for i in range(n_classes) if i != target]
    rng.shuffle(others)
    chosen = [target] + others[: k - 1]
    chosen.sort()
    return chosen


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def make_choice(task: str, family: str, lang: str, state: str, question: str, options: List[str], target: int, **kw: Any) -> Optional[Sample]:
    options = [clean(o, 400) or "-" for o in options]
    if len(set(o.lower() for o in options)) != len(options):
        return None  # ambiguous options would make the label meaningless
    if not (2 <= len(options) <= MAX_OPTIONS):
        return None
    return Sample(task=task, family=family, lang=lang, ptype="choice", state=clean(state), question=clean(question, 1200), options=options, target=int(target), **kw)


def make_noul(task: str, family: str, lang: str, state: str, question: str, p_true: float, **kw: Any) -> Sample:
    return Sample(task=task, family=family, lang=lang, ptype="noul", state=clean(state), question=clean(question, 1200), target=float(min(1.0, max(0.0, p_true))), **kw)


def make_score(task: str, family: str, lang: str, state: str, question: str, level: float, levels: int, **kw: Any) -> Sample:
    return Sample(task=task, family=family, lang=lang, ptype="score", state=clean(state), question=clean(question, 1200), target=float(level), levels=int(levels), **kw)


def make_extract(task: str, family: str, lang: str, state: str, question: str, answer: Optional[str], start: Optional[int] = None, **kw: Any) -> Optional[Sample]:
    """Span from ``answer`` text; ``start`` is its character offset in the ORIGINAL state (cleaning may shift it, so
    the span is re-located in the cleaned text)."""
    st = clean(state, 20000)
    if answer is None or answer == "":
        return Sample(task=task, family=family, lang=lang, ptype="extract", state=st, question=clean(question, 1200), target=None, **kw)
    ans = answer.strip()
    if not ans:
        return None
    idx = -1
    if start is not None:
        raw_prefix = str(state)[: max(0, int(start))]
        approx = len(clean(raw_prefix, 10**9))
        if st[approx : approx + len(ans)] == ans:
            idx = approx
    if idx < 0:
        idx = st.find(ans)
    if idx < 0:
        return None
    return Sample(task=task, family=family, lang=lang, ptype="extract", state=st, question=clean(question, 1200), target=[idx, idx + len(ans)], **kw)
