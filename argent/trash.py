"""Undo for a deletion an autonomous run was allowed to make.

A mode that refuses every delete is useless for unattended work (it cannot clean
its own temp, rebuild a directory, or remove a probe file), and a mode that
allows every delete is reckless. The honest middle is this module: before a
delete the gate copies the target here, then the delete may proceed. What the
gate needs is not a promise that nothing will be deleted, but the ability to put
it back.

Layout::

    .argent/trash/<id>/manifest.json     what was saved, when, and why
    .argent/trash/<id>/files/<relative>  the saved bytes

The manifest describes the delete that was about to happen; `restore()` puts the
files back where they were. A restore never overwrites: if the original path is
occupied again, the copy lands next to it with a `restored-<stamp>` suffix.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# How much a single delete may cost us. Past this the gate must refuse instead:
# a snapshot that silently skipped half the files would be a lie about safety.
MAX_FILES = 5000
MAX_BYTES = 300 * 1024 * 1024          # 300 MB

# The trash is not allowed to grow without bound either.
KEEP_SNAPSHOTS = 20
KEEP_BYTES = 3 * 1024 * 1024 * 1024    # 3 GB

_ROOT_OVERRIDE: Path | None = None


class TooBig(Exception):
    """The delete is larger than a snapshot may hold; the gate must refuse it."""


class TrashError(Exception):
    """The snapshot could not be written; the delete must not proceed."""


@dataclass
class Entry:
    original: str
    relative: str
    kind: str                    # "file" | "dir"
    files: int = 1
    bytes: int = 0


@dataclass
class Snapshot:
    id: str
    created: float
    command: str
    reason: str
    entries: list[Entry] = field(default_factory=list)

    @property
    def files(self) -> int:
        return sum(e.files for e in self.entries)

    @property
    def bytes(self) -> int:
        return sum(e.bytes for e in self.entries)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "created": self.created,
            "command": self.command,
            "reason": self.reason,
            "files": self.files,
            "bytes": self.bytes,
            "entries": [e.__dict__ for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        return cls(
            id=data["id"],
            created=float(data.get("created", 0)),
            command=data.get("command", ""),
            reason=data.get("reason", ""),
            entries=[Entry(**e) for e in data.get("entries", [])],
        )


def use_root(path: Path | None) -> None:
    """Point the trash elsewhere (tests, or a project on another drive)."""
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = Path(path) if path else None


def root() -> Path:
    if _ROOT_OVERRIDE is not None:
        return _ROOT_OVERRIDE
    from argent.project_paths import project_root_or_cwd
    return Path(project_root_or_cwd()) / ".argent" / "trash"


# --- measuring before copying -------------------------------------------------

def _measure(path: Path) -> tuple[int, int]:
    """(files, bytes) under `path`, following the same rules the copy will."""
    if path.is_symlink() or path.is_file():
        try:
            return 1, path.lstat().st_size
        except OSError:
            return 1, 0
    files = 0
    size = 0
    for current, dirs, names in os.walk(path):
        for name in names:
            files += 1
            try:
                size += (Path(current) / name).lstat().st_size
            except OSError:
                pass
            if files > MAX_FILES or size > MAX_BYTES:
                return files, size
        del dirs  # os.walk pruning is not needed: the caps end the loop
    return files, size


def _relative_to_project(path: Path) -> tuple[str, Path]:
    """Relative name to store under `files/`, and the base it was taken from."""
    from argent.project_paths import project_root_or_cwd
    base = Path(project_root_or_cwd()).resolve()
    try:
        return str(path.resolve().relative_to(base)), base
    except (ValueError, OSError):
        # Inside the project this cannot happen; if it does, keep the path
        # readable instead of losing the file under a mangled name.
        drive = path.drive.replace(":", "") or "root"
        return str(Path(drive) / path.relative_to(path.anchor)), base


# --- taking a snapshot --------------------------------------------------------

def snapshot(paths, command: str = "", reason: str = "") -> Snapshot:
    """Copy `paths` into the trash and return the manifest.

    Raises TooBig when the delete is larger than a snapshot may hold, and
    TrashError when the copy itself failed — both mean "the delete must not
    proceed", never "proceed anyway".
    """
    targets = []
    total_files = 0
    total_bytes = 0
    for raw in paths:
        path = Path(raw)
        if not path.exists() and not path.is_symlink():
            continue
        files, size = _measure(path)
        total_files += files
        total_bytes += size
        if total_files > MAX_FILES or total_bytes > MAX_BYTES:
            raise TooBig(
                f"{total_files} файлов / {total_bytes // (1024 * 1024)} МБ — "
                f"больше, чем корзина готова сохранить")
        targets.append(path)

    if not targets:
        return Snapshot(id="", created=time.time(), command=command, reason=reason)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap_id = f"{stamp}-{uuid.uuid4().hex[:4]}"
    base = root()
    payload = base / snap_id / "files"
    try:
        payload.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise TrashError(f"не удалось создать корзину: {e}") from e

    snap = Snapshot(id=snap_id, created=time.time(), command=command, reason=reason)
    try:
        for path in targets:
            rel, _base = _relative_to_project(path)
            dest = payload / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            files, size = _measure(path)
            if path.is_dir() and not path.is_symlink():
                shutil.copytree(path, dest, symlinks=True, dirs_exist_ok=True)
                kind = "dir"
            else:
                shutil.copy2(path, dest, follow_symlinks=False)
                kind = "file"
            snap.entries.append(Entry(
                original=str(path), relative=rel, kind=kind, files=files, bytes=size))
    except OSError as e:
        shutil.rmtree(base / snap_id, ignore_errors=True)
        raise TrashError(f"не удалось сохранить копию: {e}") from e

    _write_manifest(snap)
    prune()
    return snap


def _write_manifest(snap: Snapshot) -> None:
    folder = root() / snap.id
    try:
        (folder / "manifest.json").write_text(
            json.dumps(snap.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        raise TrashError(f"не удалось записать опись корзины: {e}") from e


# --- reading and restoring ----------------------------------------------------

def list_snapshots() -> list[Snapshot]:
    base = root()
    if not base.is_dir():
        return []
    found = []
    for folder in base.iterdir():
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            found.append(Snapshot.from_dict(json.loads(manifest.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    found.sort(key=lambda s: s.created, reverse=True)
    return found


def get_snapshot(snap_id: str) -> Snapshot | None:
    for snap in list_snapshots():
        if snap.id == snap_id or snap.id.startswith(snap_id):
            return snap
    return None


def restore(snap_id: str) -> list[Path]:
    """Put a snapshot's files back. Returns where each one landed.

    A path that is occupied again is not overwritten: the copy goes next to it
    with a `restored-<stamp>` suffix, because clobbering whatever the agent
    wrote since would be a second deletion dressed as an undo.
    """
    snap = get_snapshot(snap_id)
    if snap is None:
        raise KeyError(f"снимка {snap_id} нет в корзине")
    payload = root() / snap.id / "files"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    restored: list[Path] = []
    for entry in snap.entries:
        source = payload / entry.relative
        if not source.exists() and not source.is_symlink():
            continue
        target = Path(entry.original)
        if target.exists() or target.is_symlink():
            name = target.name if entry.kind == "file" else "restored"
            target = target.with_name(f"{name}.restored-{stamp}")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.kind == "dir":
                shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target, follow_symlinks=False)
        except OSError:
            continue
        restored.append(target)
    return restored


def remove(snap_id: str) -> bool:
    snap = get_snapshot(snap_id)
    if snap is None:
        return False
    shutil.rmtree(root() / snap.id, ignore_errors=True)
    return True


def prune() -> None:
    """Keep the newest snapshots while they fit the budget."""
    snaps = list_snapshots()
    total = 0
    for index, snap in enumerate(snaps):
        total += snap.bytes
        if index >= KEEP_SNAPSHOTS or total > KEEP_BYTES:
            shutil.rmtree(root() / snap.id, ignore_errors=True)
