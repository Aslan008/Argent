"""A restored conversation, put back on the screen.

The model remembers a restored session — `agent.messages` is the whole of it — but a full-screen UI starts with an
empty transcript, so the person was looking at a blank screen above a conversation that was still going. Only the
last question and a clipped answer used to be shown.

This module turns the saved messages into display blocks:

    user       -> {"role": "user", "text"}
    assistant  -> {"role": "assistant", "text", "thinking"}
    tool       -> {"role": "tool", "name", "args", "result"}   (the call and its result, one block)
    system     -> a muted note for a compaction summary; every other system message is not shown

and sends them as ONE `history` event. It is deliberately not a replay of live events: `content` / `tool_start`
would mark the UI busy, start a turn clock and arm the prompt queue for something that happened last week.

Long sessions are shown from the last MAX_TURNS requests on; what is above is counted in a note. Long tool results and
long argument values (a whole file written by `write_file`) are clipped: the transcript is for finding one's place,
the full text is in the session.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, List, Optional

MAX_TURNS = 40
MAX_RESULT_CHARS = 1500
MAX_ARG_CHARS = 300
MAX_TEXT_CHARS = 20000
MAX_THINKING_CHARS = 4000

_SINK: Optional[Callable[[dict], None]] = None
_LOCK = threading.Lock()


def set_sink(sink: Optional[Callable[[dict], None]]) -> None:
    """Where `history` events go: the bridge registers its emitter, tests register a list's append."""
    global _SINK
    with _LOCK:
        _SINK = sink


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                kind = block.get("type")
                if kind in (None, "text") and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif kind in ("image", "image_url"):
                    parts.append("[изображение]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _clip(text: str, limit: int, note: str = "…") -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n{note} (+{len(text) - limit} симв.)"


def _clip_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {"arguments": _clip(args, MAX_ARG_CHARS)}
    if not isinstance(args, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str):
            out[str(key)] = _clip(value, MAX_ARG_CHARS)
        elif isinstance(value, (int, float, bool)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = _clip(json.dumps(value, ensure_ascii=False, default=str), MAX_ARG_CHARS)
    return out


def _is_distillate(message: dict) -> bool:
    return message.get("role") == "system" and "CONTEXT DISTILLATE" in str(message.get("content") or "")


def build_items(messages: List[dict], max_turns: int = MAX_TURNS) -> List[dict]:
    """Display blocks for the conversation in `messages` (the system prompt excluded)."""
    rows = [m for m in (messages or []) if isinstance(m, dict)]
    user_at = [i for i, m in enumerate(rows)
               if m.get("role") == "user" and _text_of(m.get("content")).strip()]
    start = 0
    hidden = 0
    if max_turns and len(user_at) > max_turns:
        start = user_at[-max_turns]
        hidden = len(user_at) - max_turns

    items: List[dict] = []
    if hidden:
        items.append({"role": "note", "text": f"[dim]… ещё {hidden} ранних запросов выше не показаны "
                                                f"(они в сессии, модель их помнит)[/dim]"})
    names_by_id: Dict[str, str] = {}
    pending: List[dict] = []              # tool blocks still waiting for their result, in call order

    for message in rows[start:]:
        role = message.get("role")
        if role == "user":
            text = _text_of(message.get("content")).strip()
            if text:
                items.append({"role": "user", "text": _clip(text, MAX_TEXT_CHARS)})
        elif role == "assistant":
            text = _text_of(message.get("content")).strip()
            thinking = str(message.get("thinking") or "").strip()
            if text or thinking:
                items.append({"role": "assistant", "text": _clip(text, MAX_TEXT_CHARS),
                              "thinking": _clip(thinking, MAX_THINKING_CHARS)})
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                name = str(fn.get("name") or call.get("name") or "tool")
                if call.get("id"):
                    names_by_id[str(call["id"])] = name
                block = {"role": "tool", "name": name,
                         "args": _clip_args(fn.get("arguments", call.get("arguments"))),
                         "result": "", "_id": str(call.get("id") or "")}
                items.append(block)
                pending.append(block)
        elif role == "tool":
            result = _clip(_text_of(message.get("content")), MAX_RESULT_CHARS)
            tid = str(message.get("tool_call_id") or "")
            target = None
            if tid:
                target = next((b for b in pending if b["_id"] == tid), None)
            if target is None and pending:
                target = pending[0]                 # providers without call ids answer in call order
            if target is not None:
                pending.remove(target)
            if target is None:
                items.append({"role": "tool", "name": names_by_id.get(tid, "tool"), "args": {}, "result": result,
                              "_id": tid})
            else:
                target["result"] = result or "(пусто)"
        elif _is_distillate(message):
            first = str(message.get("content") or "").splitlines()[0].strip("= ").strip()
            items.append({"role": "note", "text": f"[dim]── {first or 'сжатая часть диалога'}: "
                                                  f"ранние сообщения сведены в сводку ──[/dim]"})
    for item in items:
        item.pop("_id", None)
        if item["role"] == "tool" and not item["result"]:
            item["result"] = "не завершено"
    return items


def show(messages: List[dict], replace: bool = False, saved_at: str = "") -> bool:
    """Put the conversation on the screen. True when something was shown."""
    try:
        items = build_items(messages)
    except Exception:
        return False
    if not items:
        return False
    header = f"[dim]── восстановленный диалог{(' · ' + saved_at) if saved_at else ''} ──[/dim]"
    items = [{"role": "note", "text": header}] + items + [{"role": "note", "text": "[dim]── продолжение ──[/dim]"}]
    with _LOCK:
        sink = _SINK
    if sink is not None:
        sink({"type": "history", "items": items, "replace": bool(replace)})
        return True
    return _print_to_console(items)


def _print_to_console(items: List[dict], last: int = 12) -> bool:
    """The terminal REPL: its scrollback is the transcript, so only the tail is repeated."""
    try:
        from ui import console
        shown = items[-last:]
        for item in shown:
            role = item["role"]
            if role == "user":
                console.print(f"[bold]> {item['text']}[/bold]", markup=True, highlight=False)
            elif role == "assistant" and item["text"]:
                console.print(item["text"], markup=False, highlight=False)
            elif role == "tool":
                console.print(f"[dim]· {item['name']}[/dim]", markup=True)
            elif role == "note":
                console.print(item["text"], markup=True)
        return True
    except Exception:
        return False
