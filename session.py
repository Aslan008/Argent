"""Session persistence for Argent.

Saves and loads conversation history so a restart doesn't cost the context.

Three things here are less obvious than they look:

* **A session is updated, not re-snapshotted.** The id used to come from
  datetime.now(), so autosave-every-5-turns filed a fresh copy of the same
  conversation each time and six snapshots of one chat competed for the 50
  slots. Now the caller keeps the id and saves back over it — which is why the
  write has to be atomic: overwriting the only copy is a new risk the old
  append-only scheme did not have.
* **Where it was saved is part of the session.** The system prompt is rebuilt
  for the CURRENT directory on load, but the conversation still talks about the
  old project's paths, and the working memory (.argent/memory.json) belongs to
  whichever project you are standing in. Without the recorded cwd nothing can
  notice the mismatch, so nobody warns you.
* **Listing does not decompress anything.** The metadata lives in a separate
  index; the old list_sessions unpacked every full message array — megabytes —
  to read six fields, on every save, because cleanup called it.
"""

import gzip
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

from logger import get_logger

log = get_logger("session")

SESSIONS_DIR = Path.home() / ".argent" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = SESSIONS_DIR / "index.json"

MAX_SESSIONS = 50
AUTO_SAVE_INTERVAL = 5

_SUFFIX = ".json.gz"

# Fields the index carries. Everything a listing shows, and nothing that would
# make the index as big as the sessions it indexes.
_META_FIELDS = ("id", "saved_at", "model", "provider", "message_count",
                "preview", "cwd", "label")

_SESSION_LOCK = threading.RLock()


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}{_SUFFIX}"


def _session_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(_SUFFIX):
        return name[:-len(_SUFFIX)]
    return name


def _sanitize_for_filename(name: str) -> str:
    """Replace characters that are problematic in filenames."""
    out = "".join("-" if c in '.\\/:*?"<>| ' else c for c in str(name))
    return out.strip("-") or "unknown"


def _new_session_id(model: str) -> str:
    """A fresh id that is not already taken."""
    base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_sanitize_for_filename(model)}"
    with _SESSION_LOCK:
        if not _session_path(base).exists():
            return base
        # Fallback if multiple generated in the same second
        return f"{base}-{uuid.uuid4().hex[:6]}"


def _make_serializable(messages) -> list:
    """Deep-copy messages and ensure all values are JSON-serializable."""
    if not messages:
        return []
    clean = []
    for m in list(messages):
        entry = {}
        for k, v in m.items():
            try:
                json.dumps(v, ensure_ascii=False)
                entry[k] = v
            except (TypeError, ValueError):
                entry[k] = str(v)
        clean.append(entry)
    return clean


def _first_user_message(messages: list) -> str | None:
    """The first thing the human actually said, or None if they never did."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        # Check against common injected system/tool messages masquerading as user
        lower_content = content.lower().strip()
        if lower_content.startswith((
            "you are", 
            "please act", 
            "you stopped after",
            "your file write was cut off",
            "your tool call was cut off"
        )):
            continue
        return content.strip()
    return None


# --- index ------------------------------------------------------------------

def _read_index() -> dict:
    if not INDEX_PATH.exists():
        return {}
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as e:
        log.warning("session index unreadable (%s); rebuilding", e)
    return {}


def _write_index(index: dict) -> None:
    try:
        from atomic_io import atomic_write_text
        atomic_write_text(INDEX_PATH, json.dumps(index, ensure_ascii=False, indent=1))
    except Exception as e:
        log.warning("could not write session index: %s", e)


def _meta_from_file(path: Path) -> dict:
    """Read one session's metadata off disk. Only for files the index misses."""
    session_id = _session_id_from_path(path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return {k: data.get(k) for k in _META_FIELDS} | {"id": data.get("id", session_id)}
    except Exception:
        # If the file is corrupt or empty, use its creation time
        try:
            mtime = os.path.getmtime(path)
            saved_at = datetime.fromtimestamp(mtime).isoformat()
        except OSError:
            saved_at = ""
            
        return {"id": session_id, "saved_at": saved_at, "model": "", "provider": "",
                "message_count": 0, "preview": "[corrupted session]", "cwd": "",
                "label": ""}


def _sync_index() -> dict:
    """Reconcile the index with what is on disk, decompressing only newcomers."""
    with _SESSION_LOCK:
        index = _read_index()
        on_disk = {_session_id_from_path(p): p for p in SESSIONS_DIR.glob(f"*{_SUFFIX}")}

        changed = False
        for stale in [k for k in index if k not in on_disk]:
            index.pop(stale)
            changed = True
        for new_id, path in on_disk.items():
            if new_id not in index:
                index[new_id] = _meta_from_file(path)
                changed = True

        if changed:
            _write_index(index)
        return index


# --- the operations ---------------------------------------------------------

def save_session(messages: list, metadata: dict = None, session_id: str = None,
                 label: str = None) -> str | None:
    """Save a session, returning its id — or None if there was nothing to save."""
    metadata = metadata or {}

    first_user = _first_user_message(messages)
    if first_user is None:
        log.info("not saving a session with no user message")
        return None

    with _SESSION_LOCK:
        if not session_id:
            session_id = _new_session_id(metadata.get("model", "unknown"))

        safe_messages = _make_serializable(messages)
        
        try:
            cwd = metadata.get("cwd") or os.getcwd()
        except OSError:
            cwd = ""
            
        clean_label = str(label or metadata.get("label") or "")[:500]
        
        data = {
            "id": session_id,
            "saved_at": datetime.now().isoformat(),
            "model": metadata.get("model", "unknown"),
            "provider": metadata.get("provider", "unknown"),
            "cwd": cwd,
            "label": clean_label,
            "message_count": len(safe_messages),
            "preview": first_user[:80],
            "messages": safe_messages,
        }

        path = _session_path(session_id)
        try:
            from atomic_io import atomic_write_bytes

            def _write(fh):
                from text_safety import repair_surrogates
                payload = repair_surrogates(
                    json.dumps(data, ensure_ascii=False, default=str))
                with gzip.GzipFile(fileobj=fh, mode="wb") as gz:
                    gz.write(payload.encode("utf-8"))

            atomic_write_bytes(path, _write)
        except Exception as e:
            log.error("Failed to save session %s: %s", session_id, e)
            return None

        try:
            # Sync index instead of trusting the read, to prevent losing other entries if corrupt
            index = _sync_index()
            index[session_id] = {k: data.get(k) for k in _META_FIELDS}
            _write_index(index)
        except Exception as e:
            log.error("Failed to update index for session %s: %s", session_id, e)

        _cleanup_old_sessions()
        log.info("Session saved: %s (%d messages)", session_id, len(safe_messages))
        return session_id


def load_session(session_id: str) -> dict | None:
    """Load a session by ID. Returns dict with messages and metadata."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to load session %s: %s", session_id, e)
        return None


def list_sessions(query: str = None) -> list[dict]:
    """Saved sessions, newest first, optionally filtered."""
    sessions = list(_sync_index().values())
    if query and query.strip():
        needle = query.strip().lower()
        sessions = [s for s in sessions
                    if needle in " ".join(str(s.get(k) or "")
                                          for k in ("id", "label", "preview",
                                                    "model", "cwd")).lower()]
    sessions.sort(key=lambda s: s.get("saved_at") or "", reverse=True)
    return sessions


def find_session(reference: str, sessions: list = None) -> dict | None:
    """Resolve what the user typed after /load: a position, or any substring."""
    reference = (reference or "").strip()
    if not reference:
        return None

    if sessions is None:
        matches = list_sessions(reference)
        # For positional lookup
        sorted_sessions = list_sessions()
    else:
        needle = reference.lower()
        matches = [s for s in sessions
                   if needle in " ".join(str(s.get(k) or "")
                                         for k in ("id", "label", "preview",
                                                   "model", "cwd")).lower()]
        # We assume `sessions` is already sorted newest-first by the caller.
        sorted_sessions = sessions

    if reference.isdigit():
        idx = int(reference) - 1
        return sorted_sessions[idx] if 0 <= idx < len(sorted_sessions) else None

    if not matches:
        return None
    if len(matches) > 1:
        exact = [m for m in matches if (m.get("label") or "").lower() == reference.lower()]
        if exact:
            return exact[0]
    return matches[0]


def delete_session(session_id: str) -> bool:
    """Delete a session by ID."""
    with _SESSION_LOCK:
        path = _session_path(session_id)
        existed = path.exists()
        if existed:
            try:
                path.unlink()
            except OSError as e:
                log.warning("could not delete session file: %s", e)
                return False
        
        # Always try to remove from index even if file didn't exist (cleanup zombie index entry)
        index = _read_index()
        if index.pop(session_id, None) is not None:
            _write_index(index)
        return existed


def get_last_session() -> dict | None:
    """Get the most recent session."""
    sessions = list_sessions()
    return sessions[0] if sessions else None


def _cleanup_old_sessions():
    """Remove oldest sessions if exceeding MAX_SESSIONS."""
    max_count = max(1, MAX_SESSIONS)
    with _SESSION_LOCK:
        for s in list_sessions()[max_count:]:
            delete_session(s["id"])
