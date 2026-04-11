"""
File change tracker for Argent.
Enables /diff, /undo, and /undo_all commands by snapshotting files before modification.
"""

import os
import difflib
import shutil
from pathlib import Path
from datetime import datetime

from logger import get_logger

log = get_logger("file_tracker")

HISTORY_DIR = Path.home() / ".argent" / "file_history"


def _session_dir(session_id: str = "default") -> Path:
    d = HISTORY_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot(file_path: str, session_id: str = "default") -> bool:
    """Save a copy of the file before it gets modified."""
    src = Path(file_path).expanduser().resolve()
    if not src.exists():
        return False

    dest_dir = _session_dir(session_id)
    rel = str(src).replace(os.sep, "_").replace(":", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = dest_dir / f"{rel}__{ts}"

    try:
        shutil.copy2(str(src), str(dest))
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

    rel = str(src).replace(os.sep, "_").replace(":", "_")
    dest_dir = _session_dir(session_id)
    snapshots = sorted(dest_dir.glob(f"{rel}__*"))
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
    rel = str(src).replace(os.sep, "_").replace(":", "_")
    dest_dir = _session_dir(session_id)
    snapshots = sorted(dest_dir.glob(f"{rel}__*"))
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
    changes = {}
    for snap in dest_dir.glob("*__*"):
        parts = snap.name.rsplit("__", 1)
        if len(parts) == 2:
            key = parts[0]
            if key not in changes:
                changes[key] = {"snapshots": 0, "latest": snap}
            changes[key]["snapshots"] += 1
            changes[key]["latest"] = snap

    result = []
    for key, info in changes.items():
        result.append({
            "key": key,
            "snapshot_count": info["snapshots"],
        })
    return result


def undo_all(session_id: str = "default") -> str:
    """Restore all files to their latest snapshots."""
    pending = get_pending_changes(session_id)
    if not pending:
        return "No pending changes to undo."

    results = []
    dest_dir = _session_dir(session_id)
    for item in pending:
        key = item["key"]
        snapshots = sorted(dest_dir.glob(f"{key}__*"))
        if snapshots:
            latest = snapshots[-1]
            try:
                parts = key.replace("_", os.sep, 1) if os.sep == "/" else key.replace("_", ":\\", 1)
                shutil.copy2(str(latest), parts)
                results.append(f"  Restored: {parts}")
            except Exception:
                results.append(f"  Failed: {key}")

    return "Undo all results:\n" + "\n".join(results)
