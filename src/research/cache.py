"""On-disk fetch cache (SQLite).

A research run reads the same URL repeatedly (across sub-queries, reranking and
synthesis). Caching the extracted text means a page is downloaded and parsed
once, which removes redundant network round-trips and softens rate limits. The
cache is keyed by (url, raw) because raw vs. clean extraction yield different
text for the same URL. All errors are swallowed — a broken cache must never
take down a fetch; it just degrades to always-miss.
"""

import sqlite3
import threading
import time
from pathlib import Path

from logger import get_logger

log = get_logger("research")

# Lives beside the project's other .argent state; gitignored.
_DB_PATH = Path(".argent") / "web_cache.db"
_DEFAULT_TTL = 24 * 3600  # a day: fresh enough for research, avoids re-fetching
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pages ("
        "url TEXT NOT NULL, raw INTEGER NOT NULL, text TEXT NOT NULL, "
        "ts REAL NOT NULL, PRIMARY KEY (url, raw))"
    )
    return conn


def get_cached(url: str, raw: bool = False, ttl: float = _DEFAULT_TTL):
    """Return cached page text if present and younger than ttl, else None."""
    try:
        with _lock:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT text, ts FROM pages WHERE url=? AND raw=?",
                    (url, int(raw)),
                ).fetchone()
            finally:
                conn.close()
        if row and (time.time() - row[1]) < ttl:
            return row[0]
    except Exception as e:
        log.debug("web cache read failed for %s: %s", url, e)
    return None


def set_cached(url: str, raw: bool, text: str) -> None:
    """Store extracted page text. No-op on any error."""
    if not text:
        return
    try:
        with _lock:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO pages (url, raw, text, ts) VALUES (?,?,?,?)",
                    (url, int(raw), text, time.time()),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        log.debug("web cache write failed for %s: %s", url, e)


def clear_cache() -> None:
    """Drop all cached pages (used by tests and a future /research --fresh)."""
    try:
        with _lock:
            conn = _connect()
            try:
                conn.execute("DELETE FROM pages")
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        log.debug("web cache clear failed: %s", e)
