"""Text that can actually be written as UTF-8.

Python strings may hold lone surrogates (U+D800-U+DFFF) — half of a UTF-16
pair, valid inside the interpreter, impossible to encode. They arrive from JSON
escapes decoded one at a time, from providers, from web pages and from
filesystem names, and they are inert until something tries to encode:

    UnicodeEncodeError: 'utf-8' codec can't encode characters in
    position 53000-53001: surrogates not allowed

That was a real failure. Two ADJACENT surrogates at 53000-53001 is a decomposed
emoji, and the damage was not cosmetic: once one is in the conversation
history, every following request fails to encode, so the chat is bricked until
/clear — for a character the user never even sees.

So the repair puts pairs back together first (the emoji survives) and only
replaces what is genuinely orphaned. Dropping everything would be simpler and
would silently eat one character out of the middle of somebody's file.
"""

_HIGH = range(0xD800, 0xDC00)
_LOW = range(0xDC00, 0xE000)
REPLACEMENT = "�"


def has_lone_surrogates(text: str) -> bool:
    """Cheap check — the common case is that there are none."""
    return isinstance(text, str) and any(0xD800 <= ord(c) < 0xE000 for c in text)


def repair_surrogates(text: str) -> str:
    """Recombine surrogate pairs; replace the orphans. Returns encodable text."""
    if not has_lone_surrogates(text):
        return text

    out = []
    i, n = 0, len(text)
    while i < n:
        code = ord(text[i])
        if code in _HIGH and i + 1 < n and ord(text[i + 1]) in _LOW:
            # A split emoji, not corruption: put it back.
            out.append(chr(0x10000 + ((code - 0xD800) << 10) + (ord(text[i + 1]) - 0xDC00)))
            i += 2
            continue
        out.append(REPLACEMENT if 0xD800 <= code < 0xE000 else text[i])
        i += 1
    return "".join(out)


def repair_structure(value):
    """repair_surrogates over a nested dict/list, copying only what changes.

    Message histories are mostly clean, and rebuilding every dict on every turn
    to fix nothing would be a per-request cost paid forever.
    """
    if isinstance(value, str):
        return repair_surrogates(value)
    if isinstance(value, dict):
        repaired = {k: repair_structure(v) for k, v in value.items()}
        return value if all(repaired[k] is value[k] for k in value) else repaired
    if isinstance(value, list):
        repaired = [repair_structure(v) for v in value]
        return value if all(a is b for a, b in zip(repaired, value)) else repaired
    return value
