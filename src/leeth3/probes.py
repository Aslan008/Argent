"""What a Leeth model actually reads: input interventions on records no model was trained on, and averaging over re-asks.

The ingest and the round audit look for shortcuts in the DATA (word overlap, question-only classifiers, twins that change the command).
These probes ask the MODEL instead, the way the HybridAgent authors swapped latent thoughts between problems to show they carried the
answer (2026-10-20 review of arXiv:2610.18861):

* twins - two records that differ in one thing and have opposite labels (the user's words, the command, its scope, its mode, a fact in
  the middle of a long context): the model reads that thing only if it puts the "yes" twin above the "no" twin. Chance is 0.5.
* without the user's words - the user's lines taken out of a safety state: how far the probability moves.
* question only - a content pair with its text taken away: if the "yes" question still wins, the model answers from the question.
* wording - the neutral safety question asked in each of its phrasings: the answer should not depend on which one.
* option order - a choice asked again with the options shuffled: the answer should not depend on the order.

Averaging the re-asks (all phrasings, several orders) is the cheap test-time ensemble; ``gated`` applies it only when the first answer
is unsure, which is what a dispatcher could afford. Pure functions here; scripts/leeth3_probe.py runs a model.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.leeth3.records import NONE_TEXTS, Record

# A line that holds what a person said: a speaker label (user, developer, operator ... in the corpus languages) and a colon or a quote
# within the first words. The same labels as the round audit (scripts/leeth3_audit.py), anchored at the line start.
_SPEAKER = re.compile(
    r"^\s*[-*>]?\s*(пользовател\w*|user|користувач\w*|utilisateur|usuario|usuário|benutzer|użytkownik\w*|kullanıcı\w*|қолданушы|"
    r"пайдаланушы|utente|ユーザー|用户|разработчик\w*|инженер\w*|developer|engineer|admin\w*|оператор\w*|operator)"
    r"[^\n:：«\"]{0,60}[:：«\"]", re.I)


def is_user_line(line: str) -> bool:
    return bool(_SPEAKER.search(line))


def without_user_words(state: str) -> Optional[str]:
    """The state with the user's lines removed; None when there is nothing to remove or nothing would be left but the action. The
    last line (the agent's action) always stays, even if it happens to start with a label."""
    lines = state.strip().splitlines()
    if len(lines) < 2:
        return None
    head, last = lines[:-1], lines[-1]
    kept = [l for l in head if not is_user_line(l)]
    if len(kept) == len(head):
        return None
    return "\n".join(kept + [last])


def option_texts(r: Record) -> Tuple[List[str], List[Optional[str]]]:
    """Option strings as the client and the exam write them ("name: description"), plus "none of these" when the record allows it."""
    texts, labels = [], []
    for name, desc in r.options.items():
        texts.append(f"{name}: {desc}" if desc else name)
        labels.append(name)
    if r.abstain_ok:
        texts.append(NONE_TEXTS.get(r.lang, NONE_TEXTS["en"])[0])
        labels.append(None)
    return texts, labels


def widen(r: Record, pool: Sequence[Record], k: int, rng: random.Random) -> Optional[Record]:
    """The record with its options grown to ``k`` by real options of OTHER families (so none of them fits the request better). None
    when the pool cannot supply enough distinct names."""
    have = dict(r.options)
    if len(have) >= k:
        return r
    donors = [x for x in pool if x.family != r.family and x.ptype == "choice"]
    rng.shuffle(donors)
    for d in donors:
        for name, desc in d.options.items():
            if name not in have:
                have[name] = desc
                if len(have) >= k:
                    items = list(have.items())
                    rng.shuffle(items)
                    return Record(**{**r.__dict__, "options": dict(items), "id": f"{r.id}@{k}"})
    return None


def permuted(texts: Sequence[str], seed: str, n: int) -> List[List[int]]:
    """``n`` orders of the options: the original first, then ``n - 1`` shuffles fixed by ``seed`` (reproducible)."""
    rng = random.Random(seed)
    orders = [list(range(len(texts)))]
    for _ in range(n - 1):
        o = list(range(len(texts)))
        rng.shuffle(o)
        orders.append(o)
    return orders


# ---- metrics --------------------------------------------------------------------------------------------------------------------------
def hard(answer: float) -> Optional[bool]:
    return True if answer >= 0.7 else False if answer <= 0.3 else None


def auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(probs: Sequence[float], labels: Sequence[bool], bins: int = 10) -> Optional[float]:
    """Of a yes/no head: confidence max(p, 1-p) against "the 0.5 decision was right"."""
    if not probs:
        return None
    conf = [max(p, 1 - p) for p in probs]
    ok = [(p >= 0.5) == y for p, y in zip(probs, labels)]
    err = 0.0
    for b in range(bins):
        lo, hi = 0.5 + b * 0.5 / bins, 0.5 + (b + 1) * 0.5 / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == hi)]
        if idx:
            err += len(idx) / len(conf) * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return err


def binary_summary(probs: Sequence[float], labels: Sequence[bool]) -> Dict[str, Optional[float]]:
    pos = [p for p, y in zip(probs, labels) if y]
    neg = [p for p, y in zip(probs, labels) if not y]
    return {"n": len(probs), "auc": auc(pos, neg), "acc": sum((p >= 0.5) == y for p, y in zip(probs, labels)) / len(probs) if probs else None,
            "ece": ece(probs, labels)}


def twin_pairs(records: Iterable[Record]) -> List[Tuple[Record, Record]]:
    """(yes, no) pairs: two records of one group with opposite hard labels. Groups of another size or with soft labels are skipped."""
    by: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        if r.group:
            by[r.group_key].append(r)
    out = []
    for rs in by.values():
        if len(rs) != 2:
            continue
        a, b = rs
        ya, yb = hard(float(a.answer)), hard(float(b.answer))
        if ya is None or yb is None or ya == yb:
            continue
        out.append((a, b) if ya else (b, a))
    return out


def ordering(pairs: Sequence[Tuple[str, str]], p: Dict[str, float]) -> Optional[float]:
    """Share of (yes id, no id) pairs where the yes record gets the higher probability; ties count half. Chance 0.5."""
    got = [(p[y] > p[n]) + 0.5 * (p[y] == p[n]) for y, n in pairs if y in p and n in p]
    return sum(got) / len(got) if got else None


def wilson(k: float, n: int, z: float = 1.645) -> Tuple[Optional[float], Optional[float]]:
    """90 % interval of a share (as the Leeth Index counts)."""
    if n == 0:
        return None, None
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mean_probs(rows: Sequence[Sequence[float]]) -> List[float]:
    return [sum(col) / len(col) for col in zip(*rows)]


def gated(base: Sequence[float], ens: Sequence[float], conf: Sequence[float], tau: float) -> Tuple[List[float], float]:
    """The ensemble answer only where the first answer's confidence is below ``tau``; returns the answers and the share re-asked."""
    out = [e if c < tau else b for b, e, c in zip(base, ens, conf)]
    return out, sum(c < tau for c in conf) / len(conf) if conf else 0.0


def _bit(key: str, salt: str = "") -> int:
    return int(hashlib.sha256((salt + key).encode("utf-8")).hexdigest()[:8], 16) % 2


def record_key(r: Record) -> str:
    """By group: twins and the two questions on one text share it (an expanded record's ``src:`` group names its source)."""
    if r.group.startswith("src:"):
        return r.group[4:]
    return r.group_key or r.id.split("@")[0]


def half(r: Record) -> int:
    """A fixed split of the probe pool, so a threshold chosen on one half is reported on the other."""
    return _bit(record_key(r), "gate:")


PROBE_FIRST = 215          # the pilots of 2026-10-20 were trained on B0001-B0214


def in_probe_holdout(key: str) -> bool:
    """Half of the groups of B0215 and later, never trained on by any later pilot, so the probes keep measuring unseen records.
    ``key`` as ``record_key`` gives it: "B0300:g01" or "B0300-017"."""
    try:
        batch = int(key[1:5])
    except ValueError:
        return False
    return batch >= PROBE_FIRST and _bit(key, "probe:") == 1
