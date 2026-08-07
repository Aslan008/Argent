"""What a scheduled run has already seen.

A monitoring automation — new job postings, price drops, releases, mentions —
is worthless without this: every run finds the same twenty items and reports
all twenty as news. The only way to say "here is what changed" is to remember
what was there last time.

Two design decisions carry the module:

* **Transactional.** ``filter_new`` stages the keys it hands out; the runner
  commits them only when the run actually finished. Marking on read would mean
  a crash mid-run silently swallows items that were never reported — losing
  exactly the thing the automation exists to catch. Duplicates are annoying;
  a missed item is a bug you never find out about.
* **Scoped per automation and thread-local.** Runs happen on worker threads and
  two automations can be in flight at once, so the current scope travels with
  the thread, like the approval backend.

Interactive use (no automation active) commits immediately: the human is
looking at the answer, so there is no window to lose anything in.
"""

import hashlib
import json
import threading
from datetime import datetime, timedelta

from logger import get_logger

log = get_logger("automation")

_STORE_NAME = "automation_memory.json"

# Old keys are dead weight: a vacancy from last spring will never come back as
# "new", and an unbounded file is a slow leak that nobody notices.
DEFAULT_TTL_DAYS = 30
MAX_KEYS_PER_SCOPE = 5000

# Anything longer is a page of text, not an identifier; hash it so one runaway
# item cannot bloat the file.
MAX_KEY_LENGTH = 200

MANUAL_SCOPE = "manual"

_local = threading.local()
_file_lock = threading.Lock()


def store_path():
    from project_paths import project_root_or_cwd
    return project_root_or_cwd() / ".argent" / _STORE_NAME


# --- current run -----------------------------------------------------------

def set_scope(name: str) -> None:
    """Bind this thread's memory to one automation and start a transaction."""
    _local.scope = name
    _local.pending = []


def reset_scope() -> None:
    _local.scope = None
    _local.pending = []


def current_scope():
    return getattr(_local, "scope", None)


def pending_count() -> int:
    """How many new items this run has found but not yet committed."""
    return len(getattr(_local, "pending", []) or [])


# --- key handling ----------------------------------------------------------

def normalize_key(item) -> str:
    """One identifier, stable across runs.

    Whitespace is collapsed because the same URL scraped twice often differs
    only by a stray newline, and two spellings of one item defeat the whole
    point of remembering it.
    """
    text = " ".join(str(item).split())
    if len(text) > MAX_KEY_LENGTH:
        return "sha1:" + hashlib.sha1(text.encode("utf-8")).hexdigest()
    return text


# --- persistence -----------------------------------------------------------

def _load() -> dict:
    path = store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("automation memory unreadable (%s); starting empty", e)
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    from atomic_io import atomic_write_text
    atomic_write_text(store_path(),
                      json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def _prune(entries: dict, now: datetime, ttl_days: int) -> dict:
    cutoff = now - timedelta(days=max(1, ttl_days))
    fresh = {}
    for key, stamp in entries.items():
        try:
            if datetime.fromisoformat(stamp) >= cutoff:
                fresh[key] = stamp
        except (TypeError, ValueError):
            continue                      # unparseable stamp = forget it
    if len(fresh) > MAX_KEYS_PER_SCOPE:
        newest = sorted(fresh.items(), key=lambda kv: kv[1], reverse=True)
        fresh = dict(newest[:MAX_KEYS_PER_SCOPE])
    return fresh


# --- the operations --------------------------------------------------------

def filter_new(items, scope: str = None, ttl_days: int = DEFAULT_TTL_DAYS,
               now: datetime = None) -> list:
    """Return only the items this scope has not seen, staged for commit.

    Duplicates inside one call collapse: a scrape that lists the same posting
    twice should report it once.
    """
    scope = scope or current_scope() or MANUAL_SCOPE
    now = now or datetime.now()

    seen_before = set()
    fresh = []
    with _file_lock:
        data = _load()
        # Expiry has to apply on READ, not only when the file is next written:
        # a scope that stops producing new keys would otherwise never prune, and
        # an item last seen a year ago would still count as old news forever.
        entries = _prune(data.get(scope) or {}, now, ttl_days)
        staged = set(getattr(_local, "pending", []) or [])
        for item in items or []:
            key = normalize_key(item)
            if not key or key in seen_before:
                continue
            seen_before.add(key)
            if key in entries or key in staged:
                continue
            fresh.append((key, item))

    keys = [k for k, _ in fresh]
    if current_scope() is None:
        # Nobody will commit for an interactive call, so do it now.
        _persist(scope, keys, now, ttl_days)
    else:
        _local.pending = list(getattr(_local, "pending", []) or []) + keys

    return [original for _, original in fresh]


def _persist(scope: str, keys, now: datetime, ttl_days: int = DEFAULT_TTL_DAYS) -> None:
    if not keys:
        return
    stamp = now.isoformat(timespec="seconds")
    with _file_lock:
        data = _load()
        entries = dict(data.get(scope) or {})
        for key in keys:
            entries[key] = stamp
        data[scope] = _prune(entries, now, ttl_days)
        try:
            _save(data)
        except Exception as e:
            log.warning("could not save automation memory: %s", e)


def commit(now: datetime = None) -> int:
    """Persist what this run reported. Returns the number of keys written."""
    keys = list(getattr(_local, "pending", []) or [])
    scope = current_scope() or MANUAL_SCOPE
    _local.pending = []
    _persist(scope, keys, now or datetime.now())
    return len(keys)


def rollback() -> int:
    """Drop the staged keys, so a failed run re-reports its items next time."""
    count = pending_count()
    _local.pending = []
    return count


def forget(scope: str) -> int:
    """Wipe one scope's memory. Returns how many keys were dropped."""
    with _file_lock:
        data = _load()
        entries = data.pop(scope, None)
        if entries is None:
            return 0
        try:
            _save(data)
        except Exception as e:
            log.warning("could not save automation memory: %s", e)
            return 0
    return len(entries)


def stats(scope: str = None) -> dict:
    """{scope: number of remembered keys} — for /tasks memory."""
    data = _load()
    if scope is not None:
        return {scope: len(data.get(scope) or {})}
    return {name: len(entries or {}) for name, entries in data.items()}
