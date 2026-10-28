"""Near-duplicate detection for teacher data and exams (MinHash over word shingles + banded LSH), no extra dependencies.

Used twice by the ingest: within the training data (a generator repeating itself teaches the model one example many times) and
between training data and every exam (the leak that made past exams flatter the model). Exact duplicates are caught by a hash
of the normalised text; near duplicates by an estimated Jaccard similarity of word 3-gram sets.
"""

from __future__ import annotations

import hashlib
import re
import struct
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

_WORD = re.compile(r"\w+", re.UNICODE)
_MASK = (1 << 61) - 1                              # Mersenne prime modulus for the permutation hashes


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def exact_key(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 3) -> Set[str]:
    w = _WORD.findall(text.lower())
    if len(w) <= k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def _h64(s: str) -> int:
    return struct.unpack("<Q", hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest())[0]


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 1):
        import random

        rng = random.Random(seed)
        self.num_perm = num_perm
        self.a = [rng.randrange(1, _MASK) for _ in range(num_perm)]
        self.b = [rng.randrange(0, _MASK) for _ in range(num_perm)]

    def signature(self, items: Iterable[str]) -> Tuple[int, ...]:
        hs = [_h64(x) for x in items]
        if not hs:
            return tuple([_MASK] * self.num_perm)
        return tuple(min(((a * h + b) & _MASK) for h in hs) for a, b in zip(self.a, self.b))

    @staticmethod
    def similarity(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> float:
        return sum(x == y for x, y in zip(s1, s2)) / len(s1)


class NearDupIndex:
    """Banded LSH: two texts land in a common bucket when one band of their signatures is equal; candidates are then checked
    with the full signature. 16 bands x 4 rows finds pairs above ~0.5 Jaccard almost surely."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 64, bands: int = 16, k: int = 3):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.threshold, self.bands, self.rows, self.k = threshold, bands, num_perm // bands, k
        self.hasher = MinHasher(num_perm)
        self.buckets: List[Dict[Tuple[int, ...], List[int]]] = [dict() for _ in range(bands)]
        self.sigs: List[Tuple[int, ...]] = []
        self.keys: List[str] = []

    def _bands(self, sig: Tuple[int, ...]):
        for b in range(self.bands):
            yield b, sig[b * self.rows:(b + 1) * self.rows]

    def query(self, text: str, ignore: Optional[Callable[[str], bool]] = None) -> Optional[Tuple[str, float]]:
        """The most similar indexed text above the threshold: (its key, similarity), or None. ``ignore(key)`` skips entries that
        are allowed to be similar (several questions asked about one text on purpose)."""
        sig = self.hasher.signature(shingles(text, self.k))
        best: Optional[Tuple[str, float]] = None
        seen: Set[int] = set()
        for b, band in self._bands(sig):
            for i in self.buckets[b].get(band, ()):
                if i in seen:
                    continue
                seen.add(i)
                if ignore is not None and ignore(self.keys[i]):
                    continue
                s = MinHasher.similarity(sig, self.sigs[i])
                if s >= self.threshold and (best is None or s > best[1]):
                    best = (self.keys[i], s)
        return best

    def add(self, key: str, text: str) -> None:
        sig = self.hasher.signature(shingles(text, self.k))
        i = len(self.sigs)
        self.sigs.append(sig)
        self.keys.append(key)
        for b, band in self._bands(sig):
            self.buckets[b].setdefault(band, []).append(i)
