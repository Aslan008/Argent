"""Append-only, best-effort trace writer.

The recorder is deliberately boring: one JSON object per line, flushed as it is
written, and every failure swallowed after a warning. A trace problem must never
change a turn's behaviour or break the UI.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from argent.logger import get_logger
from argent.replay.schema import normalize_event

log = get_logger("replay")


def default_trace_dir() -> Path:
    override = os.environ.get("ARGENT_REPLAY_DIR")
    return Path(override) if override else Path.home() / ".argent" / "replay"


def default_trace_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return default_trace_dir() / f"trace_{stamp}_{os.getpid()}.jsonl"


class TraceRecorder:
    """Writes a trace; disabled instances are cheap no-ops."""

    def __init__(self, path: Optional[Path] = None, meta: Optional[Dict[str, Any]] = None):
        self.path = Path(path) if path else default_trace_path()
        self._lock = threading.RLock()
        self._seq = 0
        self._fh = None
        self._closed = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        except OSError as e:
            log.warning("replay trace unavailable at %s: %s", self.path, e)
        self.record({"type": "session_start", **(meta or {})})
        atexit.register(self.close)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def record(self, event: Dict[str, Any]) -> None:
        if self._fh is None or self._closed:
            return
        try:
            row = normalize_event(event)
            if row is None:
                return
            with self._lock:
                if self._fh is None or self._closed:
                    return
                self._seq += 1
                row["seq"] = self._seq
                self._fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                self._fh.flush()
        except Exception as e:
            # Never let tracing break the turn; one warning is enough.
            log.warning("could not write replay trace: %s", e)

    def link_session(self, session_id: str) -> None:
        if session_id:
            self.record({"type": "session_link", "session_id": str(session_id)})

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None
