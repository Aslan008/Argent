"""One-off: shorten the three fattest tool descriptions in place.

The budget test caps the sum of all tool descriptions, and the gateway needs room
for its own schema. Rewritten as a script because the literals are split across
lines by implicit concatenation.
"""
import re

PATH = "argent/tools/schemas.py"

NEW = {
    "grep_search": (
        "Universal search and semantic code navigation. Default target='all': fast ripgrep "
        "across any file type; the pattern is a regex, case ignored. whole_word=true for an "
        "exact word; literal=true for text with regex characters; case_sensitive=true for "
        "exact case. target='definition'|'references'|'implementations'|'call_hierarchy' "
        "uses LSP, no coordinates needed. To find a file BY NAME use search_files."
    ),
    "manage_automations": (
        "Scheduled work while the person is away: monitoring, a digest, a recurring check, a "
        "reminder ('every day', 'every hour', 'remind me'). Modes: agent (a model runs `task` "
        "with `tools`), command (shell, cheapest), remind (a message). Unattended runs cannot "
        "ask or approve: list allowed `commands` exactly. Actions: list, show, create, update, "
        "remove, enable, disable, run, runs, templates. Details: read_skill('argent-tools-reference')."
    ),
    "desktop_act": (
        "Act in another program's window: click, type, set a value, press keys, scroll, drag, "
        "bring a window forward. Use element ids from the LATEST desktop_inspect; the answer "
        "shows the window after the action, so check it before the next step. Every action "
        "goes through the approval gate: in manual mode the person confirms each, in /vibe an "
        "ordinary control runs while one named like Delete, Pay, Send or Удалить is refused. "
        "Never for passwords, payments or security dialogs. Does nothing unless desktop "
        "control is on for that program."
    ),
}


def rewrite(src: str, name: str, text: str) -> str:
    anchor = f'"name": "{name}",'
    i = src.index(anchor)
    start = src.index('"description": (', i)
    # The block ends at the first closing paren that sits on its own indentation line.
    end = src.index("\n            ),", start) + len("\n            ),")
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 88:
            lines.append(cur + " ")
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    body = "".join(f'                "{ln}"\n' for ln in lines)
    return src[:start] + '"description": (\n' + body + "            )," + src[end:]


def main() -> None:
    src = open(PATH, encoding="utf-8").read()
    before = sum(len(v) for v in NEW.values())
    for name, text in NEW.items():
        src = rewrite(src, name, text)
    open(PATH, "w", encoding="utf-8").write(src)
    print("rewritten:", ", ".join(NEW))


if __name__ == "__main__":
    main()
