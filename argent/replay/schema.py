"""The trace event format.

A trace is a JSONL file of small, redacted facts about a run. Full answers and
tool outputs stay in the saved session; the trace stores enough to reconstruct
the timeline (names, arguments, result hashes/sizes, timings, usage, checkpoints)
without duplicating a whole conversation or leaking a secret.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1

MAX_TEXT = 2_000
MAX_ARG = 600
MAX_RESULT = 4_000
MAX_DIFF = 6_000
MAX_ITEMS = 50
MAX_DEPTH = 6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _redact(text: str) -> str:
    try:
        from src.agent.secret_redactor import redact

        return redact(text)
    except Exception:
        return text


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"… (+{len(text) - limit} chars)"


def _sanitize(value: Any, depth: int = 0) -> Any:
    """JSON-safe, redacted, bounded representation of an event field."""
    if depth > MAX_DEPTH:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(_redact(value), MAX_TEXT)
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for i, (key, item) in enumerate(value.items()):
            if i >= MAX_ITEMS:
                out["…"] = f"<{len(value) - MAX_ITEMS} more>"
                break
            out[str(key)] = _sanitize(item, depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [_sanitize(item, depth + 1) for item in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            out.append(f"<{len(items) - MAX_ITEMS} more>")
        return out
    return _clip(_redact(str(value)), MAX_TEXT)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def make_event(kind: str, **fields: Any) -> Dict[str, Any]:
    """One trace row with the schema version and a UTC timestamp."""
    event: Dict[str, Any] = {"v": SCHEMA_VERSION, "ts": _now(), "kind": str(kind)}
    for key, value in fields.items():
        if key in ("v", "ts", "kind", "seq"):
            continue
        event[str(key)] = _sanitize(value)
    return event


def normalize_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turn an Argent event (``type``) into a trace row (``kind``)."""
    if not isinstance(event, dict):
        return None
    kind = str(event.get("type") or "event")
    fields = {key: value for key, value in event.items() if key != "type"}
    if kind == "tool_end":
        result = fields.get("result")
        if isinstance(result, str):
            # Keep the size and a digest for correlation; the full text is in
            # the saved session, and clipping it here keeps traces small.
            fields["result_chars"] = len(result)
            fields["result_sha256"] = _sha(_redact(result))
            fields["result"] = _clip(result, MAX_RESULT)
    elif kind == "diff":
        fields["diff"] = _clip(str(fields.get("diff") or ""), MAX_DIFF)
    elif kind in ("tool_start", "approval_request"):
        args = fields.get("args")
        if isinstance(args, dict):
            fields["args"] = {
                str(key): _clip(_redact(str(value)), MAX_ARG)
                for key, value in list(args.items())[:MAX_ITEMS]
            }
    return make_event(kind, **fields)
