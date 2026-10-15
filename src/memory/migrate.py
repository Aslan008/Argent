"""One-time import of the working memory (.argent/memory.json) into a note.

The working memory stays what it was — the state of one task. What it holds that is worth keeping beyond the task
(its key facts) becomes the note `imported-key-facts`. Nothing is removed from memory.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.memory.store import MemoryRefused, NodeStore, WriteResult

NOTE_NAME = "imported-key-facts"


def import_working_memory(store: NodeStore, memory_json: Path) -> Optional[WriteResult]:
    """Write the key facts of `memory_json` as a note. None when there is nothing to import."""
    try:
        data = json.loads(Path(memory_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    facts = [str(f).strip() for f in (data.get("key_facts") or []) if str(f).strip()]
    if not facts:
        return None
    lines = [f"- {fact}" for fact in facts]
    objective = str(data.get("objective") or "").strip()
    body = "Facts the agent noted while working (imported from the working memory):\n" + "\n".join(lines)
    if objective:
        body += f"\n\nObjective at the time of the import: {objective}"
    try:
        return store.put(NOTE_NAME, "key facts from the working memory of earlier sessions", body,
                         type="project", source="agent")
    except MemoryRefused:
        return None
