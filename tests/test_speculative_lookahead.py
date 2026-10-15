"""
LMTrust Test Suite for Speculative CoT Lookahead & SpeculativeCache.

Validates:
1. In-memory LRU cache mechanics (put, get, hit, miss, LRU eviction).
2. TTL expiration and file modification (mtime) freshness invalidation.
3. Strict Read-Only Invariant (hardware/software ban on mutative tools).
4. Refutation FSM: cancellations vs affirmative exceptions ("хотя бы").
5. Streaming thought observation and background pre-fetching.
6. Refutation cancellation in streaming thoughts.
7. File size limits (512 KB threshold).
8. Thread safety under concurrent load.
9. Turn boundary isolation (reset_turn).
"""

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import pytest

from src.micromodels.lookahead_observer import SpeculativeLookaheadObserver
from src.micromodels.speculative_cache import (
    MAX_SPECULATIVE_FILE_SIZE_BYTES,
    SAFE_READONLY_TOOLS,
    SpeculativeCache,
    speculative_cache,
)


@pytest.fixture
def clean_cache():
    cache = SpeculativeCache(max_entries=4, ttl_seconds=1.0)
    return cache


@pytest.fixture
def observer():
    obs = SpeculativeLookaheadObserver(debounce_interval_sec=0.01)
    obs.reset_turn()
    yield obs
    obs.reset_turn()


# ── 1. Cache Basic Operations & LRU ───────────────────────────
def test_cache_put_get_hit(clean_cache):
    clean_cache.put("read_file", {"file_path": "test.txt"}, "file content")
    res = clean_cache.get("read_file", {"file_path": "test.txt"})
    assert res == "file content"
    assert clean_cache.stats["hits"] == 1

    # Miss on different args
    miss = clean_cache.get("read_file", {"file_path": "other.txt"})
    assert miss is None
    assert clean_cache.stats["misses"] == 1


def test_cache_lru_eviction(clean_cache):
    # Max entries is 4
    for i in range(5):
        clean_cache.put("read_file", {"file_path": f"file_{i}.txt"}, f"content_{i}")

    # file_0 should be evicted
    assert clean_cache.get("read_file", {"file_path": "file_0.txt"}) is None
    # file_1 .. file_4 should be present
    for i in range(1, 5):
        assert clean_cache.get("read_file", {"file_path": f"file_{i}.txt"}) == f"content_{i}"


# ── 2. TTL Expiration ────────────────────────────────────────
def test_cache_ttl_expiration(clean_cache):
    # ttl_seconds is 1.0s
    clean_cache.put("read_file", {"file_path": "quick.txt"}, "data")
    assert clean_cache.get("read_file", {"file_path": "quick.txt"}) == "data"

    time.sleep(1.1)
    # Now should be expired
    assert clean_cache.get("read_file", {"file_path": "quick.txt"}) is None
    assert clean_cache.stats["misses"] >= 1


# ── 3. MTime Freshness Validation ─────────────────────────────
def test_cache_mtime_invalidation(clean_cache):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write("initial content")
        tf_path = tf.name

    try:
        norm_path = tf_path.replace("\\", "/")
        clean_cache.put("read_file", {"file_path": norm_path}, "initial content", target_path=tf_path)

        # Fresh hit
        assert clean_cache.get("read_file", {"file_path": norm_path}) == "initial content"

        # Modify file on disk
        time.sleep(0.05)
        with open(tf_path, "w") as f:
            f.write("modified content")
        # Ensure mtime is strictly greater
        new_mtime = os.path.getmtime(tf_path) + 1.0
        os.utime(tf_path, (new_mtime, new_mtime))

        # Stale cache entry must be detected and evicted!
        assert clean_cache.get("read_file", {"file_path": norm_path}) is None
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)


# ── 4. Cache Invalidation on Mutation ────────────────────────
def test_cache_explicit_invalidation(clean_cache):
    clean_cache.put("read_file", {"file_path": "src/agent.py"}, "agent code", target_path="src/agent.py")
    clean_cache.put("read_file", {"file_path": "src/config.py"}, "config code", target_path="src/config.py")

    assert clean_cache.get("read_file", {"file_path": "src/agent.py"}) == "agent code"

    # Invalidate agent.py
    clean_cache.invalidate("src/agent.py")

    assert clean_cache.get("read_file", {"file_path": "src/agent.py"}) is None
    assert clean_cache.get("read_file", {"file_path": "src/config.py"}) == "config code"
    assert clean_cache.stats["invalidations"] == 1


# ── 5. Strict Read-Only Invariant ────────────────────────────
def test_strict_readonly_invariant(clean_cache):
    """Mutative or dangerous tools must NEVER be accepted into SpeculativeCache."""
    forbidden_tools = ["write_file", "delete_file", "run_command", "git_rollback", "replace_in_file"]
    for tool in forbidden_tools:
        clean_cache.put(tool, {"cmd": "rm -rf"}, "destructive result")
        assert clean_cache.get(tool, {"cmd": "rm -rf"}) is None

    # Only safe tools permitted
    for tool in SAFE_READONLY_TOOLS:
        clean_cache.put(tool, {"param": "val"}, "safe result")
        assert clean_cache.get(tool, {"param": "val"}) == "safe result"


# ── 6. Refutation FSM & Affirmation Exceptions ───────────────
def test_refutation_fsm(observer):
    # Russian refutations
    assert observer.is_refutation("Хотя нет, этот файл открывать не нужно.") is True
    assert observer.is_refutation("Не буду читать agent.py, там всё ясно.") is True
    assert observer.is_refutation("Отставить, проверим другую ветку.") is True
    assert observer.is_refutation("Стоп, не надо запускать поиск.") is True
    assert observer.is_refutation("Я передумал, файл смотреть не стоит.") is True

    # English refutations
    assert observer.is_refutation("Wait, no need to read that file.") is True
    assert observer.is_refutation("Nevermind, skip grep search.") is True

    # Affirmative exception: "хотя бы" / "at least" is NOT a refutation!
    assert observer.is_refutation("Открою хотя бы config.py чтобы проверить порт.") is False
    assert observer.is_refutation("Need to check at least main.py before testing.") is False

    # Normal intent thoughts are NOT refutations
    assert observer.is_refutation("Сначала открою файл src/agent.py.") is False
    assert observer.is_refutation("Посмотрим содержимое папки src.") is False


# ── 7. Streaming Observation & Background Pre-fetch ───────────
def test_streaming_thought_observation_prefetch(observer):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write("hello from speculative prefetch test")
        tf_path = tf.name

    norm_path = tf_path.replace("\\", "/")
    try:
        # Feed thought stream chunk by chunk
        stream_chunks = [
            "Сейчас ",
            "посмотрим ",
            f"файл '{norm_path}'. ",
            "Затем проанализируем его содержимое.",
        ]
        for chunk in stream_chunks:
            observer.feed(chunk)
            time.sleep(0.02)

        observer.flush()

        # Allow background prefetch thread to execute
        time.sleep(0.4)

        # Check SpeculativeCache for instant hit
        hit = speculative_cache.get("read_file", {"file_path": norm_path})
        assert hit is not None
        assert "hello from speculative prefetch test" in hit
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)
        observer.reset_turn()


# ── 8. Refutation Cancelling in Stream ─────────────────────────
def test_streaming_refutation_cancels_cache(observer):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write("temporary content")
        tf_path = tf.name

    norm_path = tf_path.replace("\\", "/")
    try:
        # Pre-seed cache
        speculative_cache.put("read_file", {"file_path": norm_path}, "temporary content", target_path=tf_path)
        assert speculative_cache.get("read_file", {"file_path": norm_path}) is not None

        # Feed refutation stream specifically mentioning the file
        observer.feed(f"Хотя нет, не надо открывать '{norm_path}'.\n")
        observer.flush()

        # Cache must have been invalidated on retraction!
        assert speculative_cache.get("read_file", {"file_path": norm_path}) is None
        assert speculative_cache.stats["cancels"] >= 1
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)
        observer.reset_turn()


# ── 9. File Size Limit (512 KB Threshold) ─────────────────────
def test_large_file_size_threshold(observer):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as tf:
        # Write 600 KB (exceeds MAX_SPECULATIVE_FILE_SIZE_BYTES = 512 KB)
        tf.write(b"0" * (MAX_SPECULATIVE_FILE_SIZE_BYTES + 50 * 1024))
        tf_path = tf.name

    norm_path = tf_path.replace("\\", "/")
    try:
        observer.feed(f"Прочитай файл '{norm_path}'.\n")
        observer.flush()
        time.sleep(0.3)

        # Large file must NOT be prefetched speculatively
        hit = speculative_cache.get("read_file", {"file_path": norm_path})
        assert hit is None
    finally:
        if os.path.exists(tf_path):
            os.remove(tf_path)
        observer.reset_turn()


# ── 10. Thread Safety & Concurrency ───────────────────────────
def test_cache_concurrent_access():
    cache = SpeculativeCache(max_entries=32, ttl_seconds=10.0)

    def worker(worker_id: int):
        for i in range(50):
            fname = f"file_{worker_id}_{i % 10}.txt"
            cache.put("read_file", {"file_path": fname}, f"data_{worker_id}_{i}")
            res = cache.get("read_file", {"file_path": fname})
            if i % 5 == 0:
                cache.invalidate(fname)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in futures:
            f.result()

    # If no race condition / deadlocks, test passes cleanly
    assert cache.stats["prefetches"] > 0


# ── 11. Turn Boundary Isolation ───────────────────────────────
def test_turn_boundary_isolation(observer):
    speculative_cache.put("read_file", {"file_path": "dummy.py"}, "code")
    assert speculative_cache.get("read_file", {"file_path": "dummy.py"}) == "code"

    observer.reset_turn()

    # Everything purged for next turn
    assert speculative_cache.get("read_file", {"file_path": "dummy.py"}) is None
    assert observer._buffer == ""
    assert len(observer._processed_sentences) == 0
