"""
src/leeth/trajectory.py — Phase 4 trajectory infrastructure.

Links every Leeth decision of one agent turn into a trajectory and captures
OUTCOMES (tool results, approvals, command exit codes, LLM delegation) that
the decision-level feedback loop cannot see. This is the data layer for
trajectory-level RL (plan Phase 5) and richer RLCD verdicts (Phase 2).

Design:
- One trajectory per agent turn (process_user_input call).
- Thread-local "current trajectory" so deep call sites (approval.py,
tools/command_ops.py) can attach events without threading a parameter
through every signature.
- Append-only JSONL: a turn writes ONE record line, everything after it is a
small patch line ({"patch": id, "set": {...}, "append": {...}}). Updating a
status used to load and rewrite the whole file — at 8 MB and several events a
turn that was hundreds of megabytes of I/O per turn, and it grows.
- Compaction keeps the folder bounded: the newest records stay, the rest move
  to `archive/` (at most ARCHIVE_KEEP files).
- Failure-tolerant: every function swallows its own errors — telemetry must
never break the agent turn."""

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from argent.logger import get_logger

logger = get_logger("leeth_trajectory")

DEFAULT_STORE_PATH = "src/leeth/data/leeth_trajectories.jsonl"

#: Data, not source: the store lives beside the other generated state of the
#: project, not inside the package it belongs to.
STORE_DIR = (".argent", "leeth")
STORE_NAME = "leeth_trajectories.jsonl"

#: Fold untouched for a long time before compaction kicks in, and how much of it
#: survives. The point is a bound, not a policy: 64 MB is roughly eight thousand
#: trajectories of today's average size.
COMPACT_AT_BYTES = 64 * 1024 * 1024
KEEP_RECORDS = 8000
ARCHIVE_KEEP = 3

_lock = threading.RLock()
_local = threading.local()
_known_ids: set = set()


def _project_root() -> Path:
    try:
        from argent.project_paths import project_root_or_cwd
        return Path(project_root_or_cwd())
    except Exception:
        return Path.cwd()


def _store_path() -> Path:
    override = os.environ.get("LEETH_TRAJECTORY_PATH")
    if override:
        return Path(override)
    path = _project_root().joinpath(*STORE_DIR, STORE_NAME)
    _migrate_legacy_store(path)
    return path


def _migrate_legacy_store(target: Path) -> None:
    """Move the old in-tree store to the data folder, once.

    It used to live at `src/leeth/data/leeth_trajectories.jsonl` — generated data
    inside the package. An explicit LEETH_TRAJECTORY_PATH is never second-guessed.
    """
    try:
        if target.exists():
            return
        legacy = _project_root() / DEFAULT_STORE_PATH
        if not legacy.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy, target)
        logger.info("Leeth trajectories moved: %s -> %s", legacy, target)
    except Exception as exc:
        logger.debug("legacy store migration failed: %s", exc)


def _is_enabled() -> bool:
    if os.environ.get("LEETH_TRAJECTORY_DISABLED") == "1":
        return False
    if os.environ.get("LEETH_TRAJECTORY_FORCE") == "1":
        return True
    if "pytest" in sys.modules:
        return False
    try:
        import argent.config as config
        return config.get_leeth_feedback_enabled()
    except Exception:
        return True


def _read_records(path: Path):
    """Parse the store into (records-by-id, order, patch lines).

    Tolerant on purpose: a half-written last line (the process died mid-append) must
    cost one record, not the whole history.
    """
    records: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    patches: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt trajectory line in %s", path)
                    continue
                if not isinstance(item, dict):
                    continue
                if "patch" in item:
                    patches.append(item)
                    continue
                tid = item.get("id")
                if not tid:
                    continue
                if tid not in records:
                    order.append(tid)
                records[tid] = item
    except OSError as exc:
        logger.debug("trajectory store unreadable: %s", exc)
        return {}, [], []
    return records, order, patches


def _apply_patch(rec: Dict[str, Any], patch: Dict[str, Any]) -> None:
    for key, value in (patch.get("set") or {}).items():
        rec[key] = value
    for key, items in (patch.get("append") or {}).items():
        bucket = rec.get(key)
        if not isinstance(bucket, list):
            bucket = rec[key] = []
        bucket.extend(items)


def _load_all() -> List[Dict[str, Any]]:
    """Every record with its patches applied, in the order the records appeared."""
    path = _store_path()
    if not path.exists():
        return []
    records, order, patches = _read_records(path)
    orphans = 0
    for patch in patches:
        rec = records.get(patch.get("patch"))
        if rec is None:
            orphans += 1
            continue
        _apply_patch(rec, patch)
    if orphans:
        logger.debug("%d trajectory patch(es) had no record", orphans)
    return [records[tid] for tid in order if tid in records]


def _append(record: Dict[str, Any]) -> bool:
    """Append one line. No full-file rewrite — that is the whole point."""
    try:
        path = _store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as exc:
        logger.debug("trajectory append failed: %s", exc)
        return False


def _patch(traj_id: str, set_fields: Optional[Dict[str, Any]] = None,
           append: Optional[Dict[str, List[Any]]] = None) -> bool:
    body: Dict[str, Any] = {"patch": traj_id}
    if set_fields:
        body["set"] = set_fields
    if append:
        body["append"] = append
    if len(body) == 1:
        return True
    return _append(body)


def _has_record(traj_id: str) -> bool:
    """Cheap existence check: ids created by this process are trusted, others cost one scan."""
    if traj_id in _known_ids:
        return True
    for rec in _load_all():
        _known_ids.add(rec.get("id"))
    return traj_id in _known_ids


def _save_all(records: List[Dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _diff_patch(before: Dict[str, Any], after: Dict[str, Any]):
    """The smallest patch that turns `before` into `after`."""
    set_fields: Dict[str, Any] = {}
    append: Dict[str, List[Any]] = {}
    for key, value in after.items():
        old = before.get(key)
        if value == old:
            continue
        grown = (isinstance(value, list) and isinstance(old, list)
                 and len(value) >= len(old) and value[:len(old)] == old)
        if grown:
            append[key] = value[len(old):]
        else:
            set_fields[key] = value
    for key in before:
        if key not in after:
            set_fields[key] = None
    return set_fields, append


def _update(traj_id: str, mutate) -> bool:
    """Generic read-modify-write, kept for callers that change a computed field.

    Hot paths (an event, a decision link, closing a turn) use `_patch` directly and
    never read the file at all.
    """
    with _lock:
        for rec in _load_all():
            if rec.get("id") != traj_id:
                continue
            before = json.loads(json.dumps(rec))
            mutate(rec)
            set_fields, append = _diff_patch(before, rec)
            return _patch(traj_id, set_fields, append)
    return False


def _maybe_compact() -> None:
    """Keep the store and its archive bounded. Called once per turn; a stat when idle."""
    path = _store_path()
    try:
        if path.stat().st_size < COMPACT_AT_BYTES:
            return
    except OSError:
        return
    try:
        records = _load_all()
        records.sort(key=lambda r: r.get("ts") or 0)
        kept = records[-KEEP_RECORDS:]
        archive_dir = path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with _lock:
            os.replace(path, archive_dir / f"{path.stem}-{stamp}{path.suffix}")
            _save_all(kept)
        for old in sorted(archive_dir.glob(f"{path.stem}-*{path.suffix}"))[:-ARCHIVE_KEEP]:
            old.unlink(missing_ok=True)
        logger.info("Leeth trajectories compacted: %d of %d kept", len(kept), len(records))
    except Exception as exc:
        logger.debug("trajectory compaction failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Turn lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start_trajectory(query: str) -> Optional[str]:
    """Opens a trajectory for one agent turn and makes it current (thread-local)."""
    try:
        if not _is_enabled():
            return None
        traj_id = f"tr_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        record = {
            "id": traj_id,
            "ts": time.time(),
            "query": (query or "")[:300],
            "status": "open",
            "decision_ids": [],
            "events": [],
            "end_ts": None,
            "end_reason": None,
        }
        _maybe_compact()
        if not _append(record):
            return None
        _known_ids.add(traj_id)
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        stack.append(traj_id)
        return traj_id
    except Exception as exc:
        logger.debug("start_trajectory failed: %s", exc)
        return None


def current_trajectory_id() -> Optional[str]:
    """Innermost open trajectory on this thread (stack top) — survives nesting
    (e.g. delegate_local spawning a sub-turn on the same thread)."""
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


def end_trajectory(reason: str = "completed") -> None:
    """Closes the innermost open trajectory and restores the outer one (if any)."""
    stack = getattr(_local, "stack", None)
    traj_id = stack.pop() if stack else None
    if not traj_id:
        return

    def _mutate(rec: Dict[str, Any]) -> None:
        rec["status"] = reason
        rec["end_ts"] = time.time()

    try:
        # A patch instead of a rewrite: closing a turn used to cost a read and a
        # write of the entire file, which at 8 MB and several events per turn was
        # hundreds of megabytes of I/O for one telemetry record.
        if not _has_record(traj_id):
            return
        _patch(traj_id, set_fields={"status": reason, "end_ts": time.time()})
    except Exception as exc:
        logger.debug("end_trajectory failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Attachments
# ─────────────────────────────────────────────────────────────────────────────

def attach_decision(decision_id: Optional[str], traj_id: Optional[str] = None) -> bool:
    """Links a feedback decision record to the trajectory (both sides)."""
    if not decision_id:
        return False
    traj_id = traj_id or current_trajectory_id()
    if not traj_id:
        return False

    if not _has_record(traj_id):
        return False
    ok = _patch(traj_id, append={"decision_ids": [decision_id]})
    if ok:
        try:
            from src.leeth import feedback as leeth_feedback
            leeth_feedback.attach_trajectory(decision_id, traj_id)
        except Exception as exc:
            logger.debug("feedback.attach_trajectory failed: %s", exc)
    return ok


def record_event(event_type: str, **fields: Any) -> bool:
    """Appends a structured event to the current trajectory.

    Event types: tool_exec (tool, ok, detail), approval (action, approved),
    command_exit (command, code), delegated_llm (reason), review (verdict, tool).
    """
    traj_id = current_trajectory_id()
    if not traj_id:
        return False

    event = {"ts": time.time(), "type": event_type, **_sanitize(fields)}

    try:
        if not _has_record(traj_id):
            return False
        return _patch(traj_id, append={"events": [event]})
    except Exception as exc:
        logger.debug("record_event failed: %s", exc)
        return False


def _sanitize(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Small AND safe: secrets are replaced before the payload is truncated.

    Order matters: truncating first would cut the tail off a key and leave the rest
    unrecognisable to the patterns, so the log would keep half a secret forever.
    """
    clean: Dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, str):
            clean[k] = _redact(v)[:300]
        elif isinstance(v, (int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = _redact(str(v))[:200]
    return clean


def _redact(text: str) -> str:
    try:
        from src.agent.secret_redactor import redact
        return redact(text)
    except Exception:
        return text


def get_trajectory(traj_id: str) -> Optional[Dict[str, Any]]:
    for rec in _load_all():
        if rec.get("id") == traj_id:
            return rec
    return None


def load_trajectories() -> List[Dict[str, Any]]:
    return _load_all()


def stats() -> Dict[str, Any]:
    records = _load_all()
    by_status: Dict[str, int] = {}
    n_decisions = 0
    n_events = 0
    for rec in records:
        by_status[rec.get("status") or "open"] = by_status.get(rec.get("status") or "open", 0) + 1
        n_decisions += len(rec.get("decision_ids") or [])
        n_events += len(rec.get("events") or [])
    return {
        "total": len(records),
        "by_status": by_status,
        "decisions_linked": n_decisions,
        "events": n_events,
        "store": str(_store_path()),
    }