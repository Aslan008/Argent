"""`{{field}}` substitution for the text of a room.

A room is a reusable recipe, so the words it hands to an agent or a tool must be
able to carry the person's task and the run's parameters: ``"Fix this: {{task}}"``,
``"python -m pytest {{test_path}}"``. Substitution is a plain lookup in the run
state — never evaluated, no attribute access, no calls — so a placeholder cannot
reach anything the state does not already hold. An unknown field becomes an empty
string; a long value is cut so one huge tool output cannot swamp a prompt.
"""

import re
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][\w.\-]*)\s*\}\}")

#: A value longer than this is cut in the middle (head + tail keep what matters of a log).
MAX_VALUE_CHARS = 4000


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n… [{len(text) - 2 * half} знаков пропущено] …\n{text[-half:]}"


def lookup(name: str, data: dict) -> Any:
    """A field of the state; `item.title` / `rows.0` reach inside a dict or a list the state holds
    (plain lookups, nothing is evaluated)."""
    key = name[len("state."):] if name.startswith("state.") else name
    if key in data:
        return data.get(key)
    if "." not in key:
        return None
    # The longest field name that exists, then keys / indices inside its value.
    parts = key.split(".")
    for i in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:i])
        if head in data:
            cur = data[head]
            for part in parts[i:]:
                if isinstance(cur, dict):
                    cur = cur.get(part)
                elif isinstance(cur, list):
                    try:
                        cur = cur[int(part)]
                    except (ValueError, IndexError):
                        return None
                else:
                    return None
                if cur is None:
                    return None
            return cur
    return None


def render(text: str, data: dict, limit: int = MAX_VALUE_CHARS) -> str:
    """Replace every `{{name}}` in `text` with the state value (str) or ''."""
    if not text or "{{" not in text:
        return text

    def sub(match: "re.Match") -> str:
        value = lookup(match.group(1), data)
        if value is None:
            return ""
        return _cut(str(value), limit)

    return _PLACEHOLDER.sub(sub, text)


def render_value(value: Any, data: dict, limit: int = MAX_VALUE_CHARS) -> Any:
    """`render` through strings nested in lists and dicts; other values pass as they are."""
    if isinstance(value, str):
        return render(value, data, limit)
    if isinstance(value, list):
        return [render_value(v, data, limit) for v in value]
    if isinstance(value, dict):
        return {k: render_value(v, data, limit) for k, v in value.items()}
    return value


def placeholders(text: str) -> list:
    """The field names a text refers to (for the validator and `/rooms show`)."""
    return _PLACEHOLDER.findall(text or "")
