"""Reading traces and (for old sessions) reconstructing a display timeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List, Optional

from argent.replay.recorder import default_trace_dir


def load_trace(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def list_traces(directory: Optional[Path] = None) -> List[Path]:
    root = Path(directory) if directory else default_trace_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_session(session_id: str, sessions_dir: Optional[Path] = None) -> Optional[Path]:
    root = Path(sessions_dir) if sessions_dir else Path.home() / ".argent" / "sessions"
    direct = root / f"{session_id}.json.gz"
    if direct.exists():
        return direct
    matches = list(root.glob(f"{session_id}*.json.gz"))
    return matches[0] if matches else None


def load_session_timeline(path: Path) -> List[dict]:
    """A trace-shaped timeline for a session saved before tracing existed."""
    try:
        from src.agent.history_replay import build_items

        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        items = build_items(data.get("messages") or [])
    except Exception:
        return []
    rows: List[dict] = []
    turn = 0
    for item in items:
        role = item.get("role")
        if role == "user":
            turn += 1
            rows.append({"kind": "user_message", "turn": turn, "text": item.get("text", "")})
        elif role == "assistant":
            rows.append({
                "kind": "assistant_message",
                "turn": turn,
                "text": item.get("text", ""),
                "thinking": item.get("thinking", ""),
            })
        elif role == "tool":
            rows.append({
                "kind": "tool_call",
                "turn": turn,
                "name": item.get("name", "tool"),
                "args": item.get("args", {}),
                "result": item.get("result", ""),
            })
        elif role == "note":
            rows.append({"kind": "notice", "turn": turn, "text": item.get("text", "")})
    return rows


def load_source(source: Path) -> List[dict]:
    path = Path(source)
    if path.name.endswith(".json.gz"):
        return load_session_timeline(path)
    return load_trace(path)


def latest_source(directory: Optional[Path] = None) -> Optional[Path]:
    traces = list_traces(directory)
    if traces:
        return traces[0]
    return None


def normalize_source(value: str, sessions_dir: Optional[Path] = None) -> Optional[Path]:
    path = Path(value)
    if path.exists():
        return path
    return find_session(value, sessions_dir)
