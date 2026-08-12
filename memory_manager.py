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


def resolve_memory_file() -> Path:
    """Locate the project's .argent/memory.json via the shared project-root
    lookup, so `cd src` keeps using the same memory instead of starting over."""
    from project_paths import project_root_or_cwd
    root = project_root_or_cwd()
    if root is None:
        root = Path.cwd()
    return root / ".argent" / "memory.json"


class MemoryManager:
    def __init__(self):
        # Initialize lock BEFORE load, in case load/quarantine needs to sync
        self._lock = threading.RLock()
        
        # Resolve once at construction: a cd mid-session must not silently
        # migrate the memory file out from under the running agent.
        self._memory_file = resolve_memory_file()
        self.data = self._load()

    def _get_template(self) -> dict:
        return {
            "objective": "",
            "current_task": "",
            "completed": [],
            "files_modified": [],
            "key_facts": [],
            "errors_encountered": [],
            "updated_at": "",
        }

    def _load(self) -> dict:
        template = self._get_template()
        if self._memory_file.exists():
            try:
                loaded = json.loads(self._memory_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    # Robust merge to ensure missing keys are present
                    for k, v in template.items():
                        if k not in loaded:
                            loaded[k] = v
                        # Coerce lists if they were corrupted into strings/dicts
                        elif isinstance(v, list) and not isinstance(loaded[k], list):
                            loaded[k] = [str(loaded[k])] if loaded[k] else []
                        # Ensure lists only contain strings
                        elif isinstance(v, list):
                            loaded[k] = [str(item) for item in loaded[k]]
                    return loaded
                raise ValueError(f"expected an object, got {type(loaded).__name__}")
            except Exception as e:
                # This file is what a small model knows about the work after a
                # context reset. Starting over from blank without a word makes
                # the model look like it forgot, with no way to tell that from
                # a damaged file — and the next _save() overwrites the evidence.
                self._quarantine(e)
        return template

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
            now_iso = datetime.now().isoformat()
            if not self.data.get("updated_at") or now_iso > self.data["updated_at"]:
                self.data["updated_at"] = now_iso
                
            self._memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            # write_text truncates first: a crash mid-write left the file
            # half-written, and the project already had atomic_write_text for
            # exactly this (config.py and session.py use it).
            atomic_write_text(
                self._memory_file,
                json.dumps(self.data, indent=2, ensure_ascii=False),
            )

    def set_objective(self, text: str):
        text = str(text or "")
        with self._lock:
            self.data["objective"] = text[:500]
            self._save()
        log.info("Objective set: %s", text[:80])

    def set_current_task(self, text: str):
        text = str(text or "")
        with self._lock:
            self.data["current_task"] = text[:300]
            self._save()

    def add_completed(self, action: str):
        action = str(action or "")
        with self._lock:
            if action and action not in self.data["completed"][-3:]:
                self.data["completed"].append(action[:200])
                if len(self.data["completed"]) > 20:
                    self.data["completed"] = self.data["completed"][-20:]
                self._save()

    def add_file_modified(self, filepath: str):
        filepath = str(filepath or "")
        with self._lock:
            if filepath and filepath not in self.data["files_modified"]:
                self.data["files_modified"].append(filepath)
                if len(self.data["files_modified"]) > 30:
                    self.data["files_modified"] = self.data["files_modified"][-30:]
                self._save()

    def add_fact(self, fact: str):
        fact = str(fact or "")
        with self._lock:
            if fact and fact not in self.data["key_facts"]:
                self.data["key_facts"].append(fact[:300])
                if len(self.data["key_facts"]) > 15:
                    self.data["key_facts"] = self.data["key_facts"][-15:]
                self._save()

    def add_error(self, error: str):
        error = str(error or "")
        with self._lock:
            if error and error not in self.data["errors_encountered"][-5:]:
                self.data["errors_encountered"].append(error[:200])
                if len(self.data["errors_encountered"]) > 10:
                    self.data["errors_encountered"] = self.data["errors_encountered"][-10:]
                self._save()

    def build_context_note(self) -> str:
        """Build a concise memory note to inject into system prompt after context reset."""
        with self._lock:
            d = self.data
            if not d.get("objective") and not d.get("completed") and not d.get("current_task"):
                return ""

            parts = []

            if d.get("objective"):
                parts.append(f"OBJECTIVE: {d['objective']}")

            if d.get("current_task"):
                parts.append(f"CURRENT TASK: {d['current_task']}")

            if d.get("completed"):
                recent = d["completed"][-8:]
                parts.append("COMPLETED ACTIONS:")
                for i, action in enumerate(recent, 1):
                    parts.append(f"  {i}. {action}")

            if d.get("files_modified"):
                parts.append(f"FILES MODIFIED: {', '.join(d['files_modified'][-10:])}")

            if d.get("key_facts"):
                parts.append("KEY FACTS:")
                for fact in d["key_facts"][-5:]:
                    parts.append(f"  - {fact}")

            if d.get("errors_encountered"):
                parts.append("KNOWN ERRORS (do not repeat these approaches):")
                for err in d["errors_encountered"][-3:]:
                    parts.append(f"  - {err}")

            return "\n".join(parts)

    def clear(self):
        with self._lock:
            self.data = self._get_template()
            self._save()


class _MemoryProxy:
    """Lazy initialization of MemoryManager to avoid I/O at import time."""
    def __init__(self):
        self._mgr = None
        self._lock = threading.Lock()

    def _get(self):
        if self._mgr is None:
            with self._lock:
                if self._mgr is None:
                    self._mgr = MemoryManager()
        return self._mgr

    def __getattr__(self, name):
        return getattr(self._get(), name)
        
    def __setattr__(self, name, value):
        if name in ("_mgr", "_lock"):
            super().__setattr__(name, value)
        else:
            setattr(self._get(), name, value)


memory = _MemoryProxy()
