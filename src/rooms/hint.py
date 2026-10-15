"""A nudge toward the right room, only when one clearly fits.

A recipe nobody remembers is not used. Each model call, the router (deterministic,
no model call, no network) reads the person's latest message; when exactly one room
clearly fits it, one short line is added to the volatile tail of the request:
"a room fits — rooms(action='run', ...) would do it as a bounded recipe". Nothing
is added for anything else, so ordinary chat pays nothing. The model still decides,
and every run is put to the person first.
"""

from pathlib import Path
from typing import Optional

from src.rooms.router import pick_room

_cache = {"key": None, "lib": None}


def _library():
    from src.rooms.session import build_library, user_rooms_dir
    udir = user_rooms_dir()
    try:
        stamp = udir.stat().st_mtime if udir.exists() else 0
    except OSError:
        stamp = 0
    key = (str(Path.cwd()), stamp)
    if _cache["key"] != key:
        _cache["lib"] = build_library()
        _cache["key"] = key
    return _cache["lib"]


def last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):          # multimodal: take the text parts
                content = " ".join(p["text"] for p in content
                                   if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"])
            return content if isinstance(content, str) else ""
    return ""


def room_hint(messages, library=None) -> Optional[str]:
    """The line to add to this call's tail, or None."""
    try:
        from config import get_rooms_hint
        if not get_rooms_hint():
            return None
        text = last_user_text(messages).strip()
        if len(text) < 12 or text.startswith("/"):
            return None
        lib = library if library is not None else _library()
        pick = pick_room(text, lib)
        if not pick.clear:
            return None
        room = lib.get(pick.name)
        from src.rooms.library import short_description
        what = short_description(room)
        return (
            "## A ROOM FITS THIS REQUEST\n"
            f"`{pick.name}` — {what}.\n"
            f"If the person wants this done, rooms(action=\"run\", room=\"{pick.name}\", task=<their words, complete>) "
            "runs it as a bounded recipe with checks that a process decides (the person is asked first). "
            "Skip it for questions, explanations and small edits."
        )
    except Exception:
        return None
