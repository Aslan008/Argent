"""Long states and long option lists, made from rows the corpus already has (labels carried over, not guessed).

Two post-processing steps that run on JSONL rows of the training mix:

``inflate_state``   the original state is put at a random place (start, middle, end) among paragraphs of OTHER samples, so a
                    1.5k - 6k token state keeps its answer.  The run so far trained on windows of <= 640 tokens; the backbone
                    reads 8192, and every untrained length is a guess.  Paragraphs that look like another request or another
                    command are never used as filler: they would make the label ambiguous.
``inflate_options`` a choice with few options gets REAL alternatives of the same task added, up to 255: the labels of a classification
                    task, the other pages' buttons, the other tools.  This is the "large pool" regime of Jev (255 options), with
                    hard negatives instead of made-up names.  A task whose options are the same short fixed list in every row
                    (the 15 Argent tools) is left alone: there is nothing real to add.

Both set ``meta.max_len`` so the encoder neither cuts the state nor squeezes the options (see Leeth2Encoder.encode).
"""

from __future__ import annotations

import json
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...schema import MAX_OPTIONS, Sample

SKIP_TASK_PREFIXES = ("game_", "web_", "rts", "play_", "snake", "maze", "ttt", "treasure", "sokoban", "connect", "nim", "hanoi", "lights", "slide", "chomp", "cram", "keys", "terrain")
AMBIGUOUS = re.compile(r"Incoming user request|User shell execution request|Command:|Caller:|Working directory:|Active background processes|Recent session context", re.I)
CHARS_PER_TOKEN_TRUE = 3.0  # to size a state
CHARS_PER_TOKEN_SAFE = 2.4  # to size max_len: lower = bigger budget, so nothing is ever cut
MAX_LONG_TOKENS = 6000


def _skip(task: str) -> bool:
    t = task.lower()
    return any(t.startswith(p) for p in SKIP_TASK_PREFIXES) or "rts" in t.split("_")[:1]


def paragraphs(states: Iterable[str], min_chars: int = 160, max_chars: int = 900) -> List[str]:
    """Filler paragraphs from sample states; anything that reads like a request or a command is excluded."""
    out: List[str] = []
    for st in states:
        for p in re.split(r"\n{2,}", st):
            p = p.strip()
            if min_chars <= len(p) <= max_chars and not AMBIGUOUS.search(p):
                out.append(p)
    return out


def inflate_state(s: Sample, pool: Sequence[str], rng: random.Random, target_tokens: int) -> Optional[Sample]:
    if _skip(s.task) or not pool or len(s.state) > 4000:
        return None
    target_tokens = min(int(target_tokens), MAX_LONG_TOKENS)
    total = int(target_tokens * CHARS_PER_TOKEN_TRUE)
    need = total - len(s.state)
    if need < 400:
        return None
    r = rng.random()
    frac = 0.0 if r < 0.2 else 1.0 if r < 0.4 else rng.random()  # a fifth at the very start, a fifth at the very end, the rest anywhere
    n_before, n_after = int(need * frac), need - int(need * frac)

    def fill(n: int) -> str:
        parts, got = [], 0
        while got < n:
            p = rng.choice(pool)
            parts.append(p)
            got += len(p) + 2
        return "\n\n".join(parts)

    before = fill(n_before) + "\n\n" if n_before >= 160 else ""
    after = "\n\n" + fill(n_after) if n_after >= 160 else ""
    state = before + s.state + after
    target = s.target
    if s.ptype == "extract" and s.target is not None:
        shift = len(before)
        spans = s.spans()
        shifted = [[a + shift, b + shift] for a, b in spans]
        target = shifted if (isinstance(s.target[0], (list, tuple)) or len(shifted) > 1) else shifted[0]
    meta = dict(s.meta or {})
    meta["max_len"] = int(min(8192, len(state) / CHARS_PER_TOKEN_SAFE + len(s.question) / 2.0 + sum(len(o) for o in (s.options or [])) / 2.0 + 320))
    meta["long"] = True
    out = Sample(s.task, s.family, s.lang, s.ptype, state, s.question, s.options, target, s.levels, s.weight, s.split, meta)
    try:
        out.validate()
    except ValueError:
        return None
    return out


def option_vocab(samples: Iterable[Sample]) -> Dict[Tuple[str, str], List[str]]:
    """(task, question) -> every distinct option string seen, in first-seen order."""
    seen: Dict[Tuple[str, str], Dict[str, None]] = {}
    for s in samples:
        if s.ptype == "choice" and s.options and not _skip(s.task):
            d = seen.setdefault((s.task, s.question), {})
            for o in s.options:
                d[o] = None
    return {k: list(v) for k, v in seen.items()}


def inflate_options(s: Sample, vocab: Dict[Tuple[str, str], List[str]], rng: random.Random, k_target: int) -> Optional[Sample]:
    if s.ptype != "choice" or not s.options or _skip(s.task):
        return None
    pool = vocab.get((s.task, s.question))
    have = len(s.options)
    if not pool or len(pool) < max(have * 3 // 2 + 8, 24):
        return None
    k = min(MAX_OPTIONS, k_target, len(pool))
    if k <= have + 8:
        return None
    present = {o.strip().lower() for o in s.options}
    cand = [o for o in pool if o.strip().lower() not in present]
    rng.shuffle(cand)
    extra: List[str] = []
    for o in cand:
        if len(extra) >= k - have:
            break
        if o.strip().lower() not in {e.strip().lower() for e in extra[-3:]}:
            extra.append(o)
    options = list(s.options) + extra
    if len({o.strip().lower() for o in options}) != len(options):
        return None
    target = s.target
    if isinstance(target, (list, tuple)):
        target = list(target) + [0.0] * len(extra)
    meta = dict(s.meta or {})
    approx_tokens = sum(len(o) for o in options) / 2.6 + len(options) + len(s.state) / CHARS_PER_TOKEN_SAFE + len(s.question) / 2.0
    meta["max_len"] = int(min(8192, max(4096, approx_tokens + 200)))  # >= 4096: with 255 options that is still 7 tokens each, as at run time
    meta["big"] = len(options)
    out = Sample(s.task, s.family, s.lang, s.ptype, s.state, s.question, options, target, s.levels, s.weight, s.split, meta)
    try:
        out.validate()
    except ValueError:
        return None
    return out


def inflate_rows(lines: Sequence[str], n_long: int, n_big: int, seed: int = 7, long_tokens: Sequence[int] = (1500, 3000, 4500, 6000), long_weights: Sequence[float] = (0.4, 0.3, 0.2, 0.1),
                 big_sizes: Sequence[int] = (64, 128, 200, 255), big_weights: Sequence[float] = (0.3, 0.3, 0.2, 0.2),
                 pool_lines: Optional[Sequence[str]] = None) -> Tuple[List[str], Dict[str, int]]:
    """Pick rows of ``lines`` (JSONL) and return their inflated copies (as JSONL strings) plus counts.
    ``pool_lines``: where the filler paragraphs come from (default: ``lines`` itself); needed when ``lines`` are all one kind of row
    (command-safety rows have no paragraph that is not itself a command)."""
    rng = random.Random(seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    pool_src: List[str] = []
    parsed: Dict[int, Sample] = {}
    for i in order[: min(len(order), 60000)]:
        s = Sample.from_json(lines[i])
        parsed[i] = s
        if pool_lines is None and len(pool_src) < 40000 and not _skip(s.task) and len(s.state) >= 160:
            pool_src.append(s.state)
    if pool_lines is not None:
        for j in random.Random(seed + 1).sample(range(len(pool_lines)), min(len(pool_lines), 60000)):
            s = Sample.from_json(pool_lines[j])
            if len(pool_src) < 40000 and not _skip(s.task) and len(s.state) >= 160:
                pool_src.append(s.state)
    pool = paragraphs(pool_src)
    vocab = option_vocab(parsed.values())
    out: List[str] = []
    counts = {"long": 0, "big": 0}
    for i in order:
        if counts["long"] >= n_long and counts["big"] >= n_big:
            break
        s = parsed.get(i) or Sample.from_json(lines[i])
        if counts["big"] < n_big and s.ptype == "choice":
            k = rng.choices(list(big_sizes), list(big_weights))[0]
            b = inflate_options(s, vocab, rng, k)
            if b is not None:
                out.append(b.to_json())
                counts["big"] += 1
                continue
        if counts["long"] < n_long:
            t = rng.choices(list(long_tokens), list(long_weights))[0]
            b = inflate_state(s, pool, rng, t)
            if b is not None:
                out.append(b.to_json())
                counts["long"] += 1
    return out, counts
