"""
Persistent Memory Manager for Argent.
Maintains a structured record of conversation progress that survives context resets.
Designed for small local models that can't hold long conversations.

Memory structure:
- objective: what the user wants to accomplish
- completed: list of completed actions
- files_modified: files that were changed
- key_facts: important discoveries or decisions
- current_task: what the model is working on right now
- errors_encountered: errors seen so far (to avoid repeating)
"""

import json
import threading
from pathlib import Path
from datetime import datetime

from atomic_io import atomic_write_text
from logger import get_logger

log = get_logger("memory")

MEMORY_FILE = Path(".argent") / "memory.json"


def resolve_memory_file() -> Path:
    """Locate the project's .argent/memory.json via the shared project-root
    lookup, so `cd src` keeps using the same memory instead of starting over."""
    from project_paths import project_root_or_cwd
    return project_root_or_cwd() / ".argent" / "memory.json"


class MemoryManager:
    def __init__(self):
        # Resolve once at construction: a cd mid-session must not silently
        # migrate the memory file out from under the running agent.
        self._memory_file = resolve_memory_file()
        self.data = self._load()
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if self._memory_file.exists():
            try:
                loaded = json.loads(self._memory_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
                raise ValueError(f"expected an object, got {type(loaded).__name__}")
            except Exception as e:
                # This file is what a small model knows about the work after a
                # context reset. Starting over from blank without a word makes
                # the model look like it forgot, with no way to tell that from
                # a damaged file — and the next _save() overwrites the evidence.
                self._quarantine(e)
        return {
            "objective": "",
            "current_task": "",
            "completed": [],
            "files_modified": [],
            "key_facts": [],
            "errors_encountered": [],
            "updated_at": "",
        }

    def _quarantine(self, reason) -> None:
        """Keep a damaged memory file instead of writing over it."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = self._memory_file.with_name(f"{self._memory_file.name}.broken-{stamp}")
        try:
            self._memory_file.replace(backup)
            log.warning("memory.json unreadable (%s) — kept as %s", reason, backup.name)
        except OSError as e:
            log.error("memory.json unreadable (%s) and could not be set aside: %s",
                      reason, e)

    def _save(self):
        with self._lock:
            self.data["updated_at"] = datetime.now().isoformat()
            # write_text truncates first: a crash mid-write left the file
            # half-written, and the project already had atomic_write_text for
            # exactly this (config.py and session.py use it).
            atomic_write_text(
                self._memory_file,
                json.dumps(self.data, indent=2, ensure_ascii=False),
            )

    def set_objective(self, text: str):
        self.data["objective"] = text[:500]
        self._save()
        log.info("Objective set: %s", text[:80])

    def set_current_task(self, text: str):
        self.data["current_task"] = text[:300]
        self._save()

    def add_completed(self, action: str):
        if action and action not in self.data["completed"][-3:]:
            self.data["completed"].append(action[:200])
            if len(self.data["completed"]) > 20:
                self.data["completed"] = self.data["completed"][-20:]
            self._save()

    def add_file_modified(self, filepath: str):
        if filepath and filepath not in self.data["files_modified"]:
            self.data["files_modified"].append(filepath)
            if len(self.data["files_modified"]) > 30:
                self.data["files_modified"] = self.data["files_modified"][-30:]
            self._save()

    def add_fact(self, fact: str):
        if fact and fact not in self.data["key_facts"]:
            self.data["key_facts"].append(fact[:300])
            if len(self.data["key_facts"]) > 15:
                self.data["key_facts"] = self.data["key_facts"][-15:]
            self._save()

    def add_error(self, error: str):
        if error and error not in self.data["errors_encountered"][-5:]:
            self.data["errors_encountered"].append(error[:200])
            if len(self.data["errors_encountered"]) > 10:
                self.data["errors_encountered"] = self.data["errors_encountered"][-10:]
            self._save()

    def build_context_note(self) -> str:
        """Build a concise memory note to inject into system prompt after context reset."""
        d = self.data
        if not d["objective"] and not d["completed"] and not d["current_task"]:
            return ""

        parts = []

        if d["objective"]:
            parts.append(f"OBJECTIVE: {d['objective']}")

        if d["current_task"]:
            parts.append(f"CURRENT TASK: {d['current_task']}")

        if d["completed"]:
            recent = d["completed"][-8:]
            parts.append("COMPLETED ACTIONS:")
            for i, action in enumerate(recent, 1):
                parts.append(f"  {i}. {action}")

        if d["files_modified"]:
            parts.append(f"FILES MODIFIED: {', '.join(d['files_modified'][-10:])}")

        if d["key_facts"]:
            parts.append("KEY FACTS:")
            for fact in d["key_facts"][-5:]:
                parts.append(f"  - {fact}")

        if d["errors_encountered"]:
            parts.append("KNOWN ERRORS (do not repeat these approaches):")
            for err in d["errors_encountered"][-3:]:
                parts.append(f"  - {err}")

        return "\n".join(parts)

    def clear(self):
        self.data = {
            "objective": "",
            "current_task": "",
            "completed": [],
            "files_modified": [],
            "key_facts": [],
            "errors_encountered": [],
            "updated_at": "",
        }
        self._save()


memory = MemoryManager()
