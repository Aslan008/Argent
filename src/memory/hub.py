"""The two memory levels seen as one: this project's notes, and the user's own that hold everywhere.

A name without a prefix means the project's note, or, when the project has none, the global one with that name.
`global:name` and `project:name` say which one exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from src.memory.store import MemoryRefused, Node, NodeStore, normalize_name

SCOPES = ("project", "global")


def enabled() -> bool:
    try:
        import config
        return bool(config.is_memory_enabled())
    except Exception:
        return True


def project_dir() -> Path:
    from project_paths import project_root_or_cwd
    return Path(project_root_or_cwd()) / ".argent" / "memory"


def global_dir() -> Path:
    return Path.home() / ".argent" / "memory"


def store_for(scope: str) -> NodeStore:
    scope = (scope or "project").strip().lower()
    if scope in ("global", "глобальная", "глобально"):
        return NodeStore(global_dir(), "global")
    if scope in ("project", "проект", "проектная"):
        return NodeStore(project_dir(), "project")
    raise MemoryRefused("scope must be 'project' or 'global'")


def stores() -> List[NodeStore]:
    return [NodeStore(project_dir(), "project"), NodeStore(global_dir(), "global")]


def split_scope(name: str) -> Tuple[Optional[str], str]:
    head, sep, tail = str(name or "").partition(":")
    if sep and head.strip().lower() in SCOPES:
        return head.strip().lower(), tail
    return None, str(name or "")


def find(name: str) -> Optional[Node]:
    scope, bare = split_scope(name)
    if scope:
        return store_for(scope).get(bare)
    for store in stores():
        node = store.get(bare)
        if node is not None:
            return node
    return None


def search(query: str, scope: str = "all", limit: int = 5) -> List[Node]:
    found = []
    for store in stores():
        if scope in SCOPES and store.scope != scope:
            continue
        for score, node in store.search(query, limit):
            # a note of the project outweighs a global one of the same strength
            found.append((score + (0.5 if store.scope == "project" else 0.0), node))
    found.sort(key=lambda s: (-s[0], s[1].scope, s[1].name))
    return [n for _, n in found[:limit]]


_INSTRUCTIONS = (
    "## LONG-TERM MEMORY\n"
    "Notes kept between sessions. Open one with memory(action=\"read\", name=...), look for one with "
    "action=\"search\". Save with action=\"write\" what will matter later and is NOT in the code: the user's "
    "preferences and corrections, decisions and their reason, hidden constraints. Not progress, not secrets.\n"
)


def prompt_section(limit: int) -> str:
    """The index for the system prompt, cut to `limit` characters. Empty when there are no notes."""
    if not enabled() or limit <= 0:
        return ""
    blocks = []
    for store in stores():
        lines = store.index_lines()
        if lines:
            title = "Project notes" if store.scope == "project" else "Global notes (all projects)"
            blocks.append((title, lines))
    if not blocks:
        return ""
    out = [_INSTRUCTIONS.rstrip("\n")]
    budget = limit - len(out[0])
    omitted = 0
    for title, lines in blocks:
        header = f"{title}:"
        if budget < len(header) + 60:
            omitted += len(lines)
            continue
        out.append(header)
        budget -= len(header) + 1
        for i, line in enumerate(lines):
            if budget - len(line) - 1 < 40:
                omitted += len(lines) - i
                break
            out.append(line)
            budget -= len(line) + 1
    if omitted:
        out.append(f"(+{omitted} more: use search)")
    return "\n".join(out)
