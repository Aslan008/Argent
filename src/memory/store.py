"""One folder of memory notes.

A note is a Markdown file: a small header between `---` lines, then the text.

    ---
    name: why-no-orm
    description: the project talks to SQLite by hand, an ORM was tried and dropped
    metadata:
      type: project
      source: agent
      updated: 2026-10-13
    ---
    Free text. Link to other notes as [[other-name]].

The index (MEMORY.md) is generated from the notes, never edited: a note that appears in the folder by hand is in
the index at the next rebuild. Nothing is deleted for good — a forgotten note goes to `.trash/`, and a consolidation
run keeps a copy of the whole folder in `.snapshots/` first.

Who wrote a note (`source`) matters: `user` notes (written by the person, or found in the folder without a source)
are never changed or removed by the model or by the consolidation; they only get suggestions in its report.
"""

from __future__ import annotations

import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logger import get_logger

log = get_logger("memory")

INDEX_NAME = "MEMORY.md"
INDEX_MAX_LINES = 200
MAX_BODY_CHARS = 6000
MAX_DESCRIPTION_CHARS = 200
MAX_NODES = 400
SNAPSHOTS_KEPT = 5
TRASH_KEPT = 50

TYPES = ("user", "feedback", "project", "reference")
SOURCES = ("user", "agent", "dream")

# A name is also a file name and a [[link]]: letters, digits, "-" and "_" only.
_NAME_OK = re.compile(r"^[0-9a-zа-яё][0-9a-zа-яё_-]{0,62}$")
_RESERVED = {"memory", "index", "con", "prn", "aux", "nul"}
_LINK = re.compile(r"\[\[([^\]\n]{1,64})\]\]")
_WORD = re.compile(r"[0-9a-zа-яё]{2,}", re.IGNORECASE)


class MemoryRefused(ValueError):
    """A refusal with a reason the model (or the person) can act on."""


def normalize_name(raw: str) -> str:
    """'Why no ORM' -> 'why-no-orm'. Raises MemoryRefused when nothing usable is left."""
    text = re.sub(r"[\s/\\]+", "-", str(raw or "").strip().lower())
    text = re.sub(r"[^0-9a-zа-яё_-]", "", text).strip("-_")
    text = re.sub(r"-{2,}", "-", text)[:63].strip("-_")
    if not text or not _NAME_OK.match(text):
        raise MemoryRefused(f"'{raw}' cannot be a note name: use a short name made of letters, digits, '-' and '_'")
    if text in _RESERVED:
        raise MemoryRefused(f"'{text}' is a reserved name, choose another one")
    return text


@dataclass
class Node:
    name: str
    description: str = ""
    body: str = ""
    type: str = "project"
    source: str = "user"
    updated: str = ""
    scope: str = "project"

    @property
    def links(self) -> List[str]:
        out = []
        for m in _LINK.finditer(self.body):
            try:
                out.append(normalize_name(m.group(1)))
            except MemoryRefused:
                continue
        return list(dict.fromkeys(out))

    @property
    def is_user_owned(self) -> bool:
        return self.source == "user"

    def render(self) -> str:
        return (f"---\nname: {self.name}\ndescription: {_one_line(self.description)}\nmetadata:\n"
                f"  type: {self.type}\n  source: {self.source}\n  updated: {self.updated}\n---\n"
                f"{self.body.strip()}\n")


@dataclass
class WriteResult:
    node: Node
    created: bool
    changed: bool
    redacted: bool = False
    warnings: List[str] = field(default_factory=list)


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse(text: str, name: str, scope: str = "project") -> Node:
    """A note from file text. Tolerant: a file with no header is a `user` note whose first line is its description."""
    node = Node(name=name, scope=scope)
    body = text.replace("\r\n", "\n")
    if body.startswith("---\n"):
        end = body.find("\n---", 4)
        if end != -1:
            header, body = body[4:end], body[end + 4:].lstrip("\n")
            section = ""
            for line in header.split("\n"):
                if not line.strip():
                    continue
                if not line.startswith((" ", "\t")):
                    key, _, value = line.partition(":")
                    key, value = key.strip().lower(), value.strip()
                    section = key if not value else ""
                    if key == "description":
                        node.description = value.strip("\"'")
                    elif key == "type" and value in TYPES:
                        node.type = value
                    elif key == "source" and value in SOURCES:
                        node.source = value
                    elif key == "updated":
                        node.updated = value
                elif section == "metadata":
                    key, _, value = line.strip().partition(":")
                    key, value = key.strip().lower(), value.strip()
                    if key == "type" and value in TYPES:
                        node.type = value
                    elif key == "source" and value in SOURCES:
                        node.source = value
                    elif key == "updated":
                        node.updated = value
            node.body = body.strip()
            return node
    node.body = body.strip()
    if not node.description:
        first = next((ln.strip(" #\t") for ln in node.body.split("\n") if ln.strip()), "")
        node.description = first[:MAX_DESCRIPTION_CHARS]
    return node


def _stem(word: str) -> str:
    return word[:5]


def _tokens(text: str) -> List[str]:
    return [_stem(w.lower()) for w in _WORD.findall(text or "")]


class NodeStore:
    """The notes of one level (a project, or the user's own home)."""

    def __init__(self, root: Path, scope: str = "project"):
        self.root = Path(root)
        self.scope = scope
        self._lock = threading.RLock()

    # -- paths ---------------------------------------------------------------------------------

    def path_of(self, name: str) -> Path:
        return self.root / f"{name}.md"

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    @property
    def trash_dir(self) -> Path:
        return self.root / ".trash"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / ".snapshots"

    # -- reading -------------------------------------------------------------------------------

    def names(self) -> List[str]:
        try:
            files = [p for p in self.root.glob("*.md") if p.is_file() and p.name != INDEX_NAME]
        except OSError:
            return []
        return sorted(p.stem for p in files if _NAME_OK.match(p.stem))

    def get(self, name: str) -> Optional[Node]:
        try:
            name = normalize_name(name)
            text = self.path_of(name).read_text(encoding="utf-8")
        except (MemoryRefused, OSError, UnicodeDecodeError):
            return None
        node = parse(text, name, self.scope)
        node.name = name
        return node

    def all(self) -> List[Node]:
        return [n for n in (self.get(name) for name in self.names()) if n is not None]

    def __len__(self) -> int:
        return len(self.names())

    # -- writing -------------------------------------------------------------------------------

    def put(self, name: str, description: str, body: str, type: str = "project", source: str = "agent",
            now: Optional[datetime] = None) -> WriteResult:
        """Create or replace a note. Secrets are removed first; a `user` note is not replaced by anyone else."""
        from atomic_io import atomic_write_text
        from src.agent.secret_redactor import redact

        name = normalize_name(name)
        if type not in TYPES:
            raise MemoryRefused(f"type must be one of: {', '.join(TYPES)}")
        if source not in SOURCES:
            raise MemoryRefused(f"source must be one of: {', '.join(SOURCES)}")
        description = _one_line(description)
        body = str(body or "").strip()
        if not description:
            raise MemoryRefused("description is required: one line saying what the note is for, so it can be found")
        if not body:
            raise MemoryRefused("body is empty")
        clean_description = redact(description)[:MAX_DESCRIPTION_CHARS]
        clean_body = redact(body)
        redacted = clean_description != description[:MAX_DESCRIPTION_CHARS] or clean_body != body
        warnings = []
        if len(clean_body) > MAX_BODY_CHARS:
            clean_body = clean_body[:MAX_BODY_CHARS].rstrip() + "\n[…cut: keep a note short, split it into several]"
            warnings.append(f"the text was cut to {MAX_BODY_CHARS} characters")
        if redacted:
            warnings.append("a secret was found in the text and hidden")
        with self._lock:
            old = self.get(name)
            if old is not None and old.is_user_owned and source != "user":
                raise MemoryRefused(f"'{name}' was written by the user: do not change it, suggest the change in chat")
            if old is None and len(self.names()) >= MAX_NODES:
                raise MemoryRefused(f"the memory holds {MAX_NODES} notes already: merge or forget some first")
            stamp = (now or datetime.now()).strftime("%Y-%m-%d")
            node = Node(name=name, description=clean_description, body=clean_body, type=type, source=source,
                        updated=stamp, scope=self.scope)
            changed = (old is None or (old.description, old.body, old.type) !=
                       (node.description, node.body, node.type))
            if changed:
                self.root.mkdir(parents=True, exist_ok=True)
                atomic_write_text(self.path_of(name), node.render())
                self.rebuild_index()
            elif old is not None:
                node.updated = old.updated
            return WriteResult(node=node, created=old is None, changed=changed, redacted=redacted,
                               warnings=warnings)

    def forget(self, name: str, by: str = "agent", now: Optional[datetime] = None) -> bool:
        """Move a note to the trash. A `user` note is only removed by the user (by=user)."""
        name = normalize_name(name)
        with self._lock:
            node = self.get(name)
            if node is None:
                return False
            if node.is_user_owned and by != "user":
                raise MemoryRefused(f"'{name}' was written by the user: only the user removes it")
            self.trash_dir.mkdir(parents=True, exist_ok=True)
            stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
            target = self.trash_dir / f"{stamp}-{name}.md"
            shutil.move(str(self.path_of(name)), str(target))
            self._prune(self.trash_dir, TRASH_KEPT)
            self.rebuild_index()
            return True

    def restore_from_trash(self, name: str) -> Optional[str]:
        """Bring back the newest trashed copy of a note. Returns its name, or None when there is none."""
        name = normalize_name(name)
        with self._lock:
            copies = sorted(self.trash_dir.glob(f"*-{name}.md")) if self.trash_dir.exists() else []
            if not copies or self.path_of(name).exists():
                return None
            self.root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(copies[-1]), str(self.path_of(name)))
            self.rebuild_index()
            return name

    @staticmethod
    def _prune(folder: Path, keep: int) -> None:
        try:
            items = sorted(folder.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for old in items[:-keep] if len(items) > keep else []:
            try:
                shutil.rmtree(old) if old.is_dir() else old.unlink()
            except OSError:
                pass

    # -- the index -----------------------------------------------------------------------------

    def index_lines(self) -> List[str]:
        nodes = sorted(self.all(), key=lambda n: (TYPES.index(n.type) if n.type in TYPES else 9, n.name))
        return [f"- [{n.name}]({n.name}.md) — {n.description}" for n in nodes]

    def rebuild_index(self) -> Path:
        from atomic_io import atomic_write_text
        lines = self.index_lines()
        if len(lines) > INDEX_MAX_LINES:
            extra = len(lines) - INDEX_MAX_LINES + 1
            lines = lines[:INDEX_MAX_LINES - 1] + [f"- … and {extra} more notes: find them with memory search"]
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.index_path, "\n".join(lines) + ("\n" if lines else ""))
        return self.index_path

    # -- search --------------------------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> List[Tuple[float, Node]]:
        words = [w for w in dict.fromkeys(_tokens(query))]
        if not words:
            return []
        scored = []
        for node in self.all():
            head = set(_tokens(f"{node.name.replace('-', ' ').replace('_', ' ')} {node.description}"))
            body_tokens = _tokens(node.body)
            score = 0.0
            for w in words:
                if w in head:
                    score += 3.0
                score += min(3, body_tokens.count(w)) * 1.0
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda s: (-s[0], s[1].name))
        return scored[:limit]

    def snippet(self, node: Node, query: str, width: int = 220) -> str:
        words = set(_tokens(query))
        for line in node.body.split("\n"):
            if words & set(_tokens(line)):
                return _one_line(line)[:width]
        return _one_line(node.body)[:width]

    # -- snapshots (for the consolidation) -----------------------------------------------------

    def snapshot(self, now: Optional[datetime] = None, allow_empty: bool = False) -> Optional[Path]:
        """A copy of every note and the index. Returns its folder (None for an empty memory, unless `allow_empty`:
        a run that is about to write the first notes needs "nothing" to go back to)."""
        with self._lock:
            names = self.names()
            if not names and not allow_empty:
                return None
            stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
            target = self.snapshots_dir / stamp
            n = 1
            while target.exists():
                n += 1
                target = self.snapshots_dir / f"{stamp}-{n}"
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(self.path_of(name), target / f"{name}.md")
            self._prune(self.snapshots_dir, SNAPSHOTS_KEPT)
            return target

    def snapshots(self) -> List[Path]:
        try:
            return sorted(p for p in self.snapshots_dir.iterdir() if p.is_dir())
        except OSError:
            return []

    def restore(self, snapshot: Path) -> Dict[str, int]:
        """Put the folder back as it was in the snapshot. What was added since goes to the trash, not away."""
        with self._lock:
            before = set(self.names())
            kept = {p.stem for p in snapshot.glob("*.md")}
            removed = 0
            for name in sorted(before - kept):
                node = self.get(name)
                if node is not None:
                    self.forget(name, by="user")
                    removed += 1
            restored = 0
            self.root.mkdir(parents=True, exist_ok=True)
            for p in snapshot.glob("*.md"):
                shutil.copy2(p, self.path_of(p.stem))
                restored += 1
            self.rebuild_index()
            return {"restored": restored, "removed": removed}
