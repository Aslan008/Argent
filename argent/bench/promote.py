"""Turn a saved session into a draft bench case.

The case is deliberately incomplete: it has the task and a start state, but no
checks. A person (or a later step) must write what "passed" means; otherwise the
bench would measure nothing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from argent.bench.case import CaseError


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(block.get("text") or "") if isinstance(block, dict) else str(block)
            for block in value
        )
    return "" if value is None else str(value)


def _slug(text: str, limit: int = 36) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:limit].strip("-") or "case")


def _load(path: Path) -> Dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        raise CaseError(f"cannot read session {path}: {e}") from e
    if not isinstance(data, dict):
        raise CaseError(f"session {path} is not an object")
    return data


def user_turns(messages: List[dict]) -> List[str]:
    out = []
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "user":
            text = _text(message.get("content")).strip()
            if text:
                out.append(text)
    return out


def promote_session(session_path: Path, *, turn: Optional[int] = None,
                    out_dir: Path = Path("bench/cases"), start: str = "HEAD",
                    force: bool = False) -> Path:
    data = _load(Path(session_path))
    turns = user_turns(data.get("messages") or [])
    if not turns:
        raise CaseError("session has no user message to promote")
    if turn is None:
        task = turns[-1]
    else:
        if turn < 1 or turn > len(turns):
            raise CaseError(f"turn {turn} is out of range (1..{len(turns)})")
        task = turns[turn - 1]

    ident = f"promoted-{_slug(task)}-{hashlib.sha1(task.encode('utf-8')).hexdigest()[:6]}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ident}.yaml"
    if path.exists() and not force:
        raise CaseError(f"{path} already exists (use --force to overwrite)")
    payload = {
        "id": ident,
        "task": task,
        "start": start,
        "checks": [],
        "tags": ["promoted"],
        "source": str(session_path),
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
