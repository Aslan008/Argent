"""Random access to a big JSONL corpus without loading it, and length-bucketed batch planning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import Sample


def build_index(path: str | Path) -> Path:
    """Write ``<path>.idx.npy``: int64 array [N, 3] of (byte offset, char length of state+question+options, per-sample max_len or 0)."""
    import json

    path = Path(path)
    rows: List[tuple] = []
    with open(path, "rb") as f:
        off = 0
        for line in f:
            if line.strip():
                d = json.loads(line)
                n = len(d.get("state", "")) + len(d.get("question", "")) + sum(len(o) + 1 for o in (d.get("options") or []))
                rows.append((off, n, int((d.get("meta") or {}).get("max_len", 0))))
            off += len(line)
    idx_path = Path(str(path) + ".idx.npy")
    np.save(idx_path, np.asarray(rows, dtype=np.int64).reshape(-1, 3))
    return idx_path


class JsonlReader:
    """Lazy, fork-safe reader: the file handle is opened on first use inside each worker process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        idx = Path(str(self.path) + ".idx.npy")
        if not idx.exists():
            build_index(self.path)
        arr = np.load(idx)
        self.offsets = arr[:, 0]
        self.lens = arr[:, 1]
        self.caps = arr[:, 2] if arr.shape[1] > 2 else np.zeros(len(arr), dtype=np.int64)
        self._fh = None
        self._pid = None

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, i: int) -> Sample:
        if self._fh is None or self._pid != os.getpid():
            self._fh = open(self.path, "rb")
            self._pid = os.getpid()
        self._fh.seek(int(self.offsets[i]))
        return Sample.from_json(self._fh.readline().decode("utf-8"))

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_fh"] = None
        d["_pid"] = None
        return d


def read_all(path: str | Path) -> List[Sample]:
    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Sample.from_json(line))
    return out


_WORKER: Dict[str, Any] = {}


def _len_chunk(args: Tuple[str, str, int, int, int, int]) -> Tuple[int, np.ndarray]:
    path, tok_path, vocab, max_len, lo, hi = args
    from .encoding import Leeth2Encoder

    if "enc" not in _WORKER:
        from transformers import AutoTokenizer

        _WORKER["enc"] = Leeth2Encoder(AutoTokenizer.from_pretrained(tok_path), vocab, max_len=max_len)
        _WORKER["rd"] = JsonlReader(path)
    enc, rd = _WORKER["enc"], _WORKER["rd"]
    import random

    out = np.empty(hi - lo, dtype=np.int32)
    for k, i in enumerate(range(lo, hi)):
        out[k] = len(enc.encode(rd.get(i), rng=random.Random(i)))
    return lo, out


def exact_token_lengths(path: str | Path, tok_path: str, vocab: int, max_len: int, workers: int = 4, chunk: int = 20000, cache: Optional[Path] = None) -> np.ndarray:
    """Real encoded length of every sample. Packing batches by true length instead of a character-based guess takes
    padding from ~31% to <1% and removes the memory spikes that came from batches the guess had under-measured."""
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    n = len(JsonlReader(path))
    jobs = [(str(path), tok_path, vocab, max_len, lo, min(n, lo + chunk)) for lo in range(0, n, chunk)]
    lens = np.empty(n, dtype=np.int32)
    if workers <= 1:
        for j in jobs:
            lo, arr = _len_chunk(j)
            lens[lo : lo + len(arr)] = arr
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            for lo, arr in pool.imap_unordered(_len_chunk, jobs):
                lens[lo : lo + len(arr)] = arr
    if cache is not None:
        np.save(cache, lens)
    return lens


def estimate_tokens(char_lens: np.ndarray, max_len: int, caps: Optional[np.ndarray] = None) -> np.ndarray:
    """Cheap token estimate from characters (mixed EN/RU/code ~ 3.2 chars per token). Ordinary samples are cut to
    ``max_len`` by the encoder (a little headroom for option lists); long-context samples carry their own cap."""
    est = np.clip(char_lens / 3.2 + 14.0, 16.0, 4096.0)
    limit = np.full(len(est), 1.5 * max_len)
    if caps is not None:
        limit = np.where(caps > 0, caps, limit)
    return np.minimum(est, limit)


def plan_batches(
    char_lens: np.ndarray,
    tokens_per_batch: int,
    max_batch: int,
    seed: int,
    epoch: int = 0,
    world: int = 1,
    max_len: int = 512,
    chunk: int = 20000,
    caps: Optional[np.ndarray] = None,
    token_lens: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Shuffle, sort by length inside chunks (little padding), cut at a padded-token budget, shuffle batches.
    The number of batches is a multiple of ``world`` so every rank takes exactly the same count."""
    rng = np.random.default_rng(seed + 7919 * epoch)
    order = rng.permutation(len(char_lens))
    est = token_lens.astype(np.float64) if token_lens is not None else estimate_tokens(char_lens, max_len, caps)
    batches: List[List[int]] = []
    for c in range(0, len(order), chunk):
        idx = order[c : c + chunk]
        idx = idx[np.argsort(est[idx], kind="stable")]
        cur: List[int] = []
        cur_max = 0.0
        for i in idx:
            m = max(cur_max, float(est[i]))
            if cur and (m * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_batch):
                batches.append(cur)
                cur, m = [], float(est[i])
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
    perm = rng.permutation(len(batches))
    batches = [batches[i] for i in perm]
    n = (len(batches) // world) * world
    return batches[:n]
