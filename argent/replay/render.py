"""Plain-text rendering of a trace timeline (the TUI view comes later)."""

from __future__ import annotations

import json
from typing import Iterable, List

_HEAD = {
    "session_start": "◆ сессия",
    "turn_start": "── ход",
    "user_message": "› вы",
    "assistant_message": "‹ агент",
    "content": "‹ агент",
    "content_replace": "‹ агент",
    "thinking": "◇ размышление",
    "tool_start": "→ инструмент",
    "tool_call": "→ инструмент",
    "tool_end": "← результат",
    "approval_request": "! подтверждение",
    "checkpoint": "◆ чекпоинт",
    "diff": "± diff",
    "usage": "· токены",
    "turn_snapshot": "· итог хода",
    "error": "✗ ошибка",
    "notice": "· заметка",
    "done": "✓ ход завершён",
    "session_link": "◆ сессия сохранена",
}


def _one_line(text: object, limit: int = 180) -> str:
    s = " ".join(str(text or "").split())
    return s[:limit] + ("…" if len(s) > limit else "")


def render(rows: Iterable[dict], content: bool = False, out=None) -> str:
    lines: List[str] = []
    last_turn = None
    for row in rows:
        kind = str(row.get("kind") or "event")
        turn = row.get("turn")
        if turn is not None and turn != last_turn:
            lines.append("")
            lines.append(f"── ход {turn} ──")
            last_turn = turn
        head = _HEAD.get(kind, kind)

        if kind == "session_start":
            bits = [str(row.get("model") or "?"), str(row.get("provider") or "?")]
            if row.get("cwd"):
                bits.append(str(row["cwd"]))
            lines.append(f"{head}: {' · '.join(bits)}")
        elif kind == "user_message":
            lines.append(f"{head}: {_one_line(row.get('text'))}")
        elif kind in ("assistant_message", "content", "content_replace"):
            if content:
                lines.append(f"{head}: {_one_line(row.get('text'))}")
            else:
                text = str(row.get("text") or "")
                lines.append(f"{head}: {len(text)} символов")
        elif kind == "thinking":
            if content:
                lines.append(f"{head}: {_one_line(row.get('text'))}")
        elif kind in ("tool_start", "tool_call"):
            args = row.get("args")
            rendered = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args or "")
            lines.append(f"{head}: {row.get('name') or '?'} {_one_line(rendered)}")
            if row.get("result"):
                lines.append(f"  ← {_one_line(row.get('result'))}")
        elif kind == "tool_end":
            bits = [str(row.get("name") or "?")]
            if row.get("duration_secs") is not None:
                bits.append(f"{row['duration_secs']:.2f}s")
            if row.get("result_chars") is not None:
                bits.append(f"{row['result_chars']} chars")
            lines.append(f"{head}: {' · '.join(bits)}")
            if content and row.get("result"):
                lines.append(f"  {_one_line(row.get('result'), 240)}")
        elif kind == "diff":
            lines.append(f"{head}: {row.get('file') or '?'} ({len(str(row.get('diff') or ''))} символов)")
        elif kind == "checkpoint":
            lines.append(f"{head}: {row.get('label') or ''} {row.get('sha') or ''}".rstrip())
        elif kind == "usage":
            data = row.get("data") if isinstance(row.get("data"), dict) else row
            bits = []
            for key in ("prompt", "completion", "cost"):
                if data.get(key) is not None:
                    bits.append(f"{key}={data[key]}")
            lines.append(f"{head}: {' · '.join(bits) or '—'}")
        elif kind in ("error", "notice", "approval_request", "turn_snapshot", "done", "session_link"):
            text = row.get("text") or row.get("action") or row.get("session_id") or ""
            lines.append(f"{head}: {_one_line(text)}" if text else head)
        else:
            lines.append(f"{head}: {_one_line(json.dumps(row, ensure_ascii=False))}")
    return "\n".join(lines).lstrip("\n")
