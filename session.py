"""
Session persistence for Argent.
Saves and loads conversation history so users don't lose context on restart.
"""

import json
import gzip
import time
from logger import get_logger

log = get_logger("session")
from pathlib import Path
from datetime import datetime

SESSIONS_DIR = Path.home() / ".argent" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SESSIONS = 50
AUTO_SAVE_INTERVAL = 5


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json.gz"


def _sanitize_for_filename(name: str) -> str:
    """Replace characters that are problematic in filenames."""
    return name.replace(".", "-").replace("/", "-").replace("\\", "-").replace(":", "-")


def _make_serializable(messages: list) -> list:
    """Deep-copy messages and ensure all values are JSON-serializable."""
    clean = []
    for m in messages:
        entry = {}
        for k, v in m.items():
            try:
                json.dumps(v, ensure_ascii=False)
                entry[k] = v
            except (TypeError, ValueError):
                entry[k] = str(v)
        clean.append(entry)
    return clean


def save_session(messages: list, metadata: dict = None) -> str:
    """Save a session and return its ID."""
    if metadata is None:
        metadata = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = _sanitize_for_filename(metadata.get("model", "unknown"))
    session_id = f"{timestamp}_{model}"

    preview = ""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            content = m["content"]
            # Skip system-generated prompts
            if not content.startswith("You are") and not content.startswith("Please act"):
                preview = content[:80]
                break

    safe_messages = _make_serializable(messages)

    data = {
        "id": session_id,
        "saved_at": datetime.now().isoformat(),
        "model": metadata.get("model", "unknown"),
        "provider": metadata.get("provider", "ollama"),
        "message_count": len(safe_messages),
        "preview": preview,
        "messages": safe_messages,
    }

    path = _session_path(session_id)
    try:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception as e:
        log.error("Failed to save session %s: %s", session_id, e)
        # Clean up the broken file
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

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


def list_sessions() -> list[dict]:
    """List all saved sessions, newest first."""
    sessions = []
    for path in SESSIONS_DIR.glob("*.json.gz"):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id", path.stem),
                "saved_at": data.get("saved_at", ""),
                "model": data.get("model", ""),
                "provider": data.get("provider", ""),
                "message_count": data.get("message_count", 0),
                "preview": data.get("preview", ""),
            })
        except Exception:
            sessions.append({
                "id": path.stem,
                "saved_at": "",
                "model": "",
                "provider": "",
                "message_count": 0,
                "preview": "[corrupted session]",
            })
    sessions.sort(key=lambda s: s.get("saved_at", ""), reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session by ID."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def get_last_session() -> dict | None:
    """Get the most recent session."""
    sessions = list_sessions()
    return sessions[0] if sessions else None


def _cleanup_old_sessions():
    """Remove oldest sessions if exceeding MAX_SESSIONS."""
    sessions = list_sessions()
    for s in sessions[MAX_SESSIONS:]:
        delete_session(s["id"])
