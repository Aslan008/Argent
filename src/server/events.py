"""Normalise ArgentAgent's internal chunks into a stable JSON event schema.

The agent already yields structured chunks; this maps them to a small, documented
contract the GUI can rely on, independent of the terminal renderer:

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

Returns None for chunks the GUI stream should ignore (e.g. tool_generating).
"""

# System notices travel on the "error" channel but are not failures — same
# bracket convention the terminal renderer uses.
_NOTICE_PREFIXES = ("[System:", "[Loop Guard]", "[Argent:")


def to_event(chunk: dict):
    kind = chunk.get("type")

    if kind in ("content_stream", "content"):
        return {"type": "content", "text": chunk.get("content", "")}
    if kind == "content_replace":
        return {"type": "content_replace", "text": chunk.get("content", "")}
    if kind == "thinking_stream":
        return {"type": "thinking", "text": chunk.get("content", "")}
    if kind == "tool_start":
        return {"type": "tool_start", "name": chunk.get("name", ""),
                "args": chunk.get("args", {})}
    if kind == "tool_end":
        return {"type": "tool_end", "name": chunk.get("name", ""),
                "result": str(chunk.get("result", ""))}
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

    # tool_generating and anything else: not part of the GUI stream.
    return None
