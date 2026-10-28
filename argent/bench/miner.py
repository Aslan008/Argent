"""Mine real traces/sessions for failures and write draft cases.

Drafts are never runnable: they have no checks. The miner only points at
episodes worth turning into a regression case.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

from argent.bench.case import CaseError
from argent.bench.promote import _slug
from argent.replay.reader import list_traces, load_trace

_FAIL = re.compile(
    r"(?i)(\berror\b|\bfailed\b|exit code:\s*[1-9]|traceback|permission denied|"
    r"not recognized|no such file|command not found)"
)
_MAX_TASK = 4000


def _draft(out_dir: Path, task: str, start: str, source: str, signal: str) -> Optional[Path]:
    task = " ".join(task.split())[:_MAX_TASK]
    if not task:
        return None
    ident = f"mined-{_slug(task)}-{hashlib.sha1((task + source).encode('utf-8')).hexdigest()[:6]}"
    path = Path(out_dir) / f"{ident}.yaml"
    if path.exists():
        return None
    payload = {
        "id": ident,
        "task": task,
        "start": start or "HEAD",
        "checks": [],
        "tags": ["mined"],
        "source": source,
        "signal": signal[:300],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def mine_trace(trace_path: Path, out_dir: Path, limit: int = 20) -> List[Path]:
    try:
        rows = load_trace(trace_path)
    except Exception:
        return []
    drafts: List[Path] = []
    task = ""
    start = "HEAD"
    drafted_turns = set()
    for row in rows:
        kind = row.get("kind")
        turn = row.get("turn")
        if kind == "user_message":
            task = str(row.get("text") or "")
            continue
        if kind == "checkpoint" and row.get("sha"):
            start = str(row["sha"])
            continue
        if kind == "error":
            signal = str(row.get("text") or "")
        elif kind == "tool_end":
            signal = str(row.get("result") or "")
        else:
            continue
        if not _FAIL.search(signal):
            continue
        if turn in drafted_turns:
            continue
        path = _draft(out_dir, task, start, str(trace_path), signal)
        if path is not None:
            drafts.append(path)
        if turn is not None:
            drafted_turns.add(turn)
        if len(drafts) >= limit:
            break
    return drafts


def mine_session(session_timeline: list, out_dir: Path, source: str = "") -> List[Path]:
    drafts: List[Path] = []
    task = ""
    for row in session_timeline:
        kind = row.get("kind")
        if kind == "user_message":
            task = str(row.get("text") or "")
        elif kind in ("tool_call", "tool_end"):
            result = str(row.get("result") or "")
            if _FAIL.search(result):
                path = _draft(out_dir, task, "HEAD", source, result)
                if path is not None:
                    drafts.append(path)
    return drafts


def mine(traces_dir: Optional[Path] = None, sessions_dir: Optional[Path] = None,
         out_dir: Path = Path("bench/cases/_drafts"), limit: int = 20) -> List[Path]:
    drafts: List[Path] = []
    if traces_dir is not None:
        for trace in list_traces(traces_dir):
            drafts.extend(mine_trace(trace, out_dir, limit - len(drafts)))
            if len(drafts) >= limit:
                return drafts
    if sessions_dir is not None:
        from argent.replay.reader import load_session_timeline

        for session in sorted(Path(sessions_dir).glob("*.json.gz")):
            drafts.extend(mine_session(load_session_timeline(session), out_dir, str(session)))
            if len(drafts) >= limit:
                return drafts
    return drafts
