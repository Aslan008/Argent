"""Normalise ArgentAgent's internal chunks into a stable JSON event schema.

The agent already yields structured chunks; this maps them to a small, documented
contract any client (Telegram, automation runner, background worker) can rely on:

  {"type": "content",        "text": str}         assistant output
  {"type": "content_replace","text": str}         replace the current answer
  {"type": "thinking",       "text": str}         reasoning stream
  {"type": "tool_start",     "name": str, "args": dict}
  {"type": "tool_end",       "name": str, "result": str}
  {"type": "notice",         "text": str}         self-healing / system notice
  {"type": "error",          "text": str}         a genuine failure
  {"type": "usage",          "data": dict}
  {"type": "checkpoint",     "sha": str, "label": str}   time-machine snapshot
  {"type": "diff",           "file": str, "diff": str}   unified diff of an edit
  {"type": "done"}                                turn finished (added by session)
"""

_NOTICE_PREFIXES = ("[System:", "[Loop Guard]", "[Argent:")


def to_event(chunk: dict):
    kind = chunk.get("type")

    if kind in ("content_stream", "content"):
        return {"type": "content", "text": chunk.get("content", "")}
    if kind == "content_replace":
        return {"type": "content_replace", "text": chunk.get("content", "")}
    if kind == "thinking_stream":
        return {"type": "thinking", "text": chunk.get("content", "")}
    if kind == "rethink_stream":
        # Same wire type so existing consumers (Telegram, GUI) keep working; the
        # flag lets a frontend that distinguishes the two (the TUI draws
        # "◈ Перепроверка" apart from "◇ Анализ") route it.
        return {"type": "thinking", "text": chunk.get("content", ""), "rethink": True}
    if kind == "tool_start":
        return {"type": "tool_start", "name": chunk.get("name", ""),
                "args": chunk.get("args", {})}
    if kind == "tool_end":
        ev = {"type": "tool_end", "name": chunk.get("name", ""),
              "result": str(chunk.get("result", ""))}
        if "duration_secs" in chunk:
            ev["duration_secs"] = chunk["duration_secs"]
        return ev
    if kind == "tool_generating":
        # Raw tool-argument JSON still streaming: the frontend shows a live
        # byte counter instead of pretending the call has started.
        # `delta` is the newly streamed slice of that JSON, so a frontend can show
        # what is being written (the REPL's "Генерация" box does).
        return {"type": "tool_generating", "name": chunk.get("name", ""),
                "bytes": int(chunk.get("bytes", 0) or 0),
                "delta": str(chunk.get("delta", "") or "")}
    if kind == "usage":
        return {"type": "usage", "data": chunk.get("data", {})}
    if kind == "checkpoint":
        return {"type": "checkpoint", "sha": chunk.get("sha", ""),
                "label": chunk.get("label", "")}
    if kind == "diff":
        return {"type": "diff", "file": chunk.get("file", ""),
                "diff": chunk.get("diff", "")}
    if kind == "error":
        content = str(chunk.get("content", ""))
        etype = "notice" if content.lstrip().startswith(_NOTICE_PREFIXES) else "error"
        return {"type": etype, "text": content}

    return None
