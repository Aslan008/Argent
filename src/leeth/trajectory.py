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
- JSONL store, append-only events, atomic rewrite for status updates.
- Failure-tolerant: every function swallows its own errors — telemetry must
  never break the agent turn.
"""

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from logger import get_logger

logger = get_logger("leeth_trajectory")

DEFAULT_STORE_PATH = "src/leeth/data/leeth_trajectories.jsonl"

_lock = threading.Lock()
_local = threading.local()


def _store_path() -> Path:
    return Path(os.environ.get("LEETH_TRAJECTORY_PATH", DEFAULT_STORE_PATH))


def _is_enabled() -> bool:
    if os.environ.get("LEETH_TRAJECTORY_DISABLED") == "1":
        return False
    if os.environ.get("LEETH_TRAJECTORY_FORCE") == "1":
        return True
    if "pytest" in sys.modules:
        return False
    try:
        import config
        return config.get_leeth_feedback_enabled()
    except Exception:
        return True


def _load_all() -> List[Dict[str, Any]]:
    path = _store_path()
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt trajectory line in %s", path)
    return records


def _save_all(records: List[Dict[str, Any]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _update(traj_id: str, mutate) -> bool:
    with _lock:
        records = _load_all()
        for rec in records:
            if rec.get("id") == traj_id:
                mutate(rec)
                _save_all(records)
                return True
    return False


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
        with _lock:
            path = _store_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        _update(traj_id, _mutate)
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

    def _mutate(rec: Dict[str, Any]) -> None:
        ids = rec.setdefault("decision_ids", [])
        if decision_id not in ids:
            ids.append(decision_id)

    ok = _update(traj_id, _mutate)
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

    def _mutate(rec: Dict[str, Any]) -> None:
        rec.setdefault("events", []).append({"ts": time.time(), "type": event_type, **_sanitize(fields)})

    try:
        return _update(traj_id, _mutate)
    except Exception as exc:
        logger.debug("record_event failed: %s", exc)
        return False


def _sanitize(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps event payloads small: strings truncated, non-scalar values stringified."""
    clean: Dict[str, Any] = {}
    for k, v in fields.items():
        if isinstance(v, str):
            clean[k] = v[:300]
        elif isinstance(v, (int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)[:200]
    return clean


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