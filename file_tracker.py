"""
File change tracker for Argent.
Enables /diff, /undo, and /undo_all commands by snapshotting files before modification.

Snapshots are keyed by a short hash of the file's absolute path — NOT by the
path itself. The old scheme turned the whole path into the filename
(C__Users_..._Deep_Script.cs__TS), which on Windows blew past the 255-char
filename limit / 260-char MAX_PATH for deeply nested files (e.g. Unity trees)
and made the backup silently fail. The real path lives in a small `<key>.path`
sidecar so /undo_all can still restore to the right place and the UI can show
a readable name. Each file keeps at most MAX_SNAPSHOTS_PER_FILE recent
snapshots so file_history can't grow without bound.
"""

import os
import difflib
import hashlib
import shutil
from pathlib import Path
from datetime import datetime

from logger import get_logger

log = get_logger("file_tracker")

HISTORY_DIR = Path.home() / ".argent" / "file_history"

# Rotation: keep only the most recent N snapshots per file. Undo restores the
# latest; older ones are just depth we don't need to hoard forever.
MAX_SNAPSHOTS_PER_FILE = 10


def _session_dir(session_id: str = "default") -> Path:
    d = HISTORY_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(src: Path) -> str:
    """Short, fixed-length, path-length-safe key for a file's absolute path."""
    return hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:16]


def _rotate(dest_dir: Path, key: str) -> None:
    """Drop snapshots beyond MAX_SNAPSHOTS_PER_FILE (oldest first)."""
    snaps = sorted(dest_dir.glob(f"{key}__*"))
    for old in snaps[:-MAX_SNAPSHOTS_PER_FILE]:
        try:
            old.unlink()
        except OSError:
            pass


def snapshot(file_path: str, session_id: str = "default") -> bool:
    """Save a copy of the file before it gets modified."""
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        return False

    dest_dir = _session_dir(session_id)
    key = _key(src)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = dest_dir / f"{key}__{ts}"

    try:
        shutil.copy2(str(src), str(dest))
        # Record the real path so undo_all / listings can recover it.
        (dest_dir / f"{key}.path").write_text(str(src), encoding="utf-8")
        _rotate(dest_dir, key)
        log.debug("Snapshot saved: %s -> %s", src, dest.name)
        return True
    except Exception as e:
        log.warning("Failed to snapshot %s: %s", src, e)
        return False


def get_diff(file_path: str, session_id: str = "default") -> str:
    """Return a unified diff between the latest snapshot and the current file."""
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        return f"File '{file_path}' does not exist."

    dest_dir = _session_dir(session_id)
    snapshots = sorted(dest_dir.glob(f"{_key(src)}__*"))
    if not snapshots:
        return f"No previous snapshots found for '{file_path}'."

    latest = snapshots[-1]
    try:
        old_lines = latest.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"{src.name} (before)",
            tofile=f"{src.name} (current)",
        )
        result = "".join(diff)
        return result if result else f"No changes detected in '{file_path}'."
    except Exception as e:
        return f"Error generating diff: {e}"


def undo(file_path: str, session_id: str = "default") -> str:
    """Restore a file to its latest snapshot."""
    src = Path(file_path).expanduser().resolve()
    dest_dir = _session_dir(session_id)
    snapshots = sorted(dest_dir.glob(f"{_key(src)}__*"))
    if not snapshots:
        return f"No snapshots found for '{file_path}'. Cannot undo."

    latest = snapshots[-1]
    try:
        shutil.copy2(str(latest), str(src))
        latest.unlink()
        return f"Restored '{file_path}' to previous version."
    except Exception as e:
        return f"Error restoring file: {e}"


def get_pending_changes(session_id: str = "default") -> list[dict]:
    """List all files that have snapshots (i.e., were modified)."""
    dest_dir = _session_dir(session_id)
    result = []
    for meta in dest_dir.glob("*.path"):
        key = meta.stem
        snaps = list(dest_dir.glob(f"{key}__*"))
        if not snaps:
            continue
        try:
            path = meta.read_text(encoding="utf-8").strip()
        except OSError:
            path = key
        result.append({
            "key": path,                 # human-readable: the real file path
            "path": path,
            "snapshot_count": len(snaps),
        })
    return result


def undo_all(session_id: str = "default") -> str:
    """Restore all files to their latest snapshots."""
    dest_dir = _session_dir(session_id)
    metas = list(dest_dir.glob("*.path"))
    if not metas:
        return "No pending changes to undo."

    results = []
    for meta in metas:
        key = meta.stem
        snapshots = sorted(dest_dir.glob(f"{key}__*"))
        if not snapshots:
            continue
        try:
            target = meta.read_text(encoding="utf-8").strip()
            shutil.copy2(str(snapshots[-1]), target)
            results.append(f"  Restored: {target}")
        except Exception:
            results.append(f"  Failed: {key}")

    if not results:
        return "No pending changes to undo."
    return "Undo all results:\n" + "\n".join(results)
