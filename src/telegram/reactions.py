"""The reaction the bot chooses for the message it answers.

A model may end its reply with ``[[react:🔥]]``. That tag is never shown: it is cut out
before the text is sent and becomes a reaction on the person's message. No tag means
"no reaction" — silence is a valid choice here, not a missing one.

Kept out of the bot class so the parsing can be tested without a socket, and out of the
prompt so the format lives in exactly one place.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

#: ``[[react:🔥]]``, tolerating spaces and case around the emoji.
TAG = re.compile(r"\[\[\s*react\s*:\s*([^\[\]]{1,24}?)\s*\]\]", re.IGNORECASE)

#: A single emoji is a few code points (skin tone, variation selector, ZWJ sequence).
MAX_EMOJI_LEN = 8


def extract(text: str) -> Tuple[str, Optional[str]]:
    """Return ``(text without reaction tags, the emoji asked for or None)``.

    Several tags are tolerated — the last usable one wins, all of them are removed, so a
    leaking tag can never reach the chat.
    """
    if not text:
        return text, None

    found = TAG.findall(text)
    cleaned = TAG.sub("", text).rstrip()
    if not found:
        return cleaned, None

    for raw in reversed(found):
        candidate = raw.strip()
        if not candidate or len(candidate) > MAX_EMOJI_LEN:
            continue
        if any(ch.isalnum() for ch in candidate):
            continue        # a word, not a reaction: refuse instead of asking Telegram and getting a 400
        return cleaned, candidate
    return cleaned, None
