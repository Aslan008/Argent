"""Generate `docs/TOOLS.md` from the schemas — never hand-edit that file.

The tool surface used to be described in three places that drifted: the schema the
model reads, the README, and whatever a person remembered. Now there is one source
(`argent/tools/schemas.py`), one taxonomy (`argent/tools/TOOL_GROUPS`), and this
script turns them into a page. `tests/test_tools_doc.py` fails when the page is out
of date, so the drift is not a matter of discipline.

Run:
    .venv\\Scripts\\python.exe scripts/gen_tools_doc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "TOOLS.md"

ONE_LINE = 118          # how much of a description the table shows


def _one_line(text: str) -> str:
    """First sentence of a description, on one line, no markdown breaks."""
    flat = " ".join((text or "").split())
    flat = flat.replace("|", "\\|")
    for stop in (". ", ".\n", " — "):
        cut = flat.find(stop)
        if 0 < cut <= ONE_LINE:
            return flat[:cut + 1].strip()
    if len(flat) <= ONE_LINE:
        return flat
    return flat[:ONE_LINE].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _required(fn: dict) -> str:
    params = fn.get("parameters") or {}
    required = params.get("required") or []
    return ", ".join(f"`{name}`" for name in required) if required else "—"


def render() -> str:
    from argent.tools import OPTIONAL_TOOLS, TOOL_GROUPS
    from argent.tools.schemas import TOOL_SCHEMAS

    by_name = {t["function"]["name"]: t["function"] for t in TOOL_SCHEMAS}
    total = sum(len(fn.get("description") or "") for fn in by_name.values())
    advertised = sum(1 for name in by_name if name not in OPTIONAL_TOOLS)

    lines = [
        "# Argent tools",
        "",
        "<!-- GENERATED FILE — do not edit by hand.",
        "     Regenerate: .venv\\Scripts\\python.exe scripts/gen_tools_doc.py",
        "     Source: argent/tools/schemas.py + argent/tools/TOOL_GROUPS. -->",
        "",
        f"{advertised} tools in {len(TOOL_GROUPS)} groups. The model is given one line per "
        "tool; the long form — `ops` for rooms, the checks `wait_heartbeat` accepts, the "
        "operators `search_web` understands — lives in "
        "`skills/argent-tools-reference/SKILL.md` and is read with `read_skill`.",
        "",
        f"Total description budget: **{total} characters** (paid in every request; "
        "`tests/test_tool_contracts.py` keeps it from growing).",
        "",
    ]

    assigned = set()
    for group, members in TOOL_GROUPS.items():
        names = sorted(n for n in members if n in by_name)
        assigned.update(names)
        if not names:
            continue
        lines.append(f"## {group} ({len(names)})")
        lines.append("")
        lines.append("| tool | what it does | required |")
        lines.append("|---|---|---|")
        for name in names:
            fn = by_name[name]
            lines.append(f"| `{name}` | {_one_line(fn.get('description'))} | {_required(fn)} |")
        optional = sorted(n for n in names if n in OPTIONAL_TOOLS)
        lines.append("")
        if optional:
            lines.append(f"_Optional — present only while its feature is installed: "
                         f"{', '.join('`' + n + '`' for n in optional)}._")
            lines.append("")

    orphans = sorted(set(by_name) - assigned - set(OPTIONAL_TOOLS))
    if orphans:
        lines.append("## Ungrouped (!)")
        lines.append("")
        lines.append("These tools are advertised but missing from `TOOL_GROUPS`: "
                     + ", ".join(f"`{n}`" for n in orphans))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("See also: `docs/TOOLS.md` is generated; behaviour rules live in "
                 "`ARGENT_BACKLOG.md`, the plan in `.argent/artifacts/tools-plan.md`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = render()
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
