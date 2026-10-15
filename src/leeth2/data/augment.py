"""Context-rot augmentation: bury the real state in unrelated text.

Hard lesson from the previous Leeth: a router that is perfect on clean states dropped to 0/20 once 700 tokens of
logs surrounded the request. A small share of examples now carry distractor text before and/or after the state
(taken from OTHER examples, so it looks like real prose, code, reviews, logs). Families whose label depends on the
whole text (sentiment, topic, NLI...) are never touched — extra text would change the answer.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

RATES = {"argent": 0.25, "qa_extract": 0.10, "browser": 0.08, "tools": 0.10, "npc": 0.04, "sorting": 0.04}
LONG_MAX_LEN = {1: 1024, 2: 2048, 3: 3072}


def make_pool(states: List[str], rng: random.Random, n: int = 4000) -> List[str]:
    cand = [s for s in states if 200 <= len(s) <= 3000]
    rng.shuffle(cand)
    return cand[:n]


def _noise(pool: List[str], rng: random.Random, n_chars: int) -> str:
    parts: List[str] = []
    total = 0
    while total < n_chars and pool:
        p = rng.choice(pool)
        parts.append(p)
        total += len(p) + 2
    text = "\n\n".join(parts)
    if len(text) > n_chars:
        cut = text.rfind(" ", 0, n_chars)
        text = text[: cut if cut > n_chars * 0.6 else n_chars]
    return text.strip()


def augment(d: Dict[str, Any], pool: List[str], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Return an augmented copy of the sample dict, or None when the sample must stay clean."""
    fam = d.get("family", "")
    rate = RATES.get(fam, 0.0)
    if not pool or rate <= 0 or rng.random() >= rate:
        return None
    r = rng.random()
    n_chars = rng.randint(300, 1200) if r < 0.60 else rng.randint(1200, 4000) if r < 0.92 else rng.randint(4000, 9000)
    where = rng.choice(["prefix", "suffix", "both"])
    state = d.get("state", "")
    before = after = ""
    if where in ("prefix", "both"):
        before = _noise(pool, rng, n_chars if where == "prefix" else n_chars // 2)
    if where in ("suffix", "both"):
        after = _noise(pool, rng, n_chars if where == "suffix" else n_chars // 2)
    if not before and not after:
        return None
    sep = "\n\n"
    new = dict(d)
    new["state"] = (before + sep if before else "") + state + (sep + after if after else "")
    if d.get("ptype") == "extract" and d.get("target"):
        shift = len(before) + (len(sep) if before else 0)
        new["target"] = [d["target"][0] + shift, d["target"][1] + shift]
    meta = dict(d.get("meta") or {})
    meta["rot"] = where
    tier = 1 if n_chars <= 1200 else 2 if n_chars <= 4000 else 3
    meta["max_len"] = LONG_MAX_LEN[tier]
    new["meta"] = meta
    return new
