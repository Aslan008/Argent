"""`/backup`: one zip of what the person made — rooms (with their earlier versions), automations, webhook
hashes, memory notes — and the way back.

What goes in is an explicit allowlist, not "everything under .argent": that folder also holds model
weights, caches, a vector database and logs (tens of GB on a working machine), and the API keys live in
the user config file, which is never read here. Restoring only writes inside the same allowlisted places,
refuses any path that tries to leave them, and takes a safety copy of what it is about to replace.

The archive is a faithful copy: it holds your rooms and notes verbatim, so treat it like your own files
and do not publish it.
"""

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

MANIFEST = "manifest.json"
FORMAT = 1

# label -> what is taken from the root of that label (relative to it). A name ending in "/" is a folder.
_ARGENT_ITEMS = ("rooms/", "rooms_stats.json", "automations.json", "automation_state.json",
                 "webhooks.json", "memory.json")
_MEMORY_SKIP_DIRS = {".snapshots"}               # copies the consolidation makes: they can be rebuilt
_SKIP_NAMES = {"Thumbs.db", ".DS_Store"}


@dataclass
class Root:
    label: str                 # "argent" | "project_memory" | "global_memory"
    path: Path                 # where it lives on this PC
    items: Optional[tuple]     # allowlist inside it; None = the whole folder (minus skips)
    title: str


def roots() -> List[Root]:
    """The places a backup reads from and restores to. A place that does not exist yet is just empty."""
    from src.memory.hub import global_dir, project_dir
    from src.rooms.session import user_rooms_dir
    argent_dir = user_rooms_dir().parent          # `.argent` next to Argent (or next to ARGENT_ROOMS_DIR)
    return [
        Root("argent", argent_dir, _ARGENT_ITEMS, "комнаты, автоматизации, вебхуки"),
        Root("project_memory", project_dir(), None, "память проекта"),
        Root("global_memory", global_dir(), None, "глобальная память"),
    ]


def default_folder() -> Path:
    return roots()[0].path / "backups"


def _files(root: Root) -> List[Path]:
    """Every file of a root that belongs in the backup, absolute."""
    base = root.path
    if not base.is_dir():
        return []
    found: List[Path] = []
    if root.items is None:
        candidates = [base]
    else:
        candidates = [base / item.rstrip("/") for item in root.items]
    for entry in candidates:
        if entry.is_file():
            found.append(entry)
        elif entry.is_dir():
            for p in sorted(entry.rglob("*")):
                if not p.is_file() or p.name in _SKIP_NAMES:
                    continue
                if any(part in _MEMORY_SKIP_DIRS for part in p.relative_to(entry).parts):
                    continue
                found.append(p)
    return found


def _is_inside_backups(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except (ValueError, OSError):
        return False


def preview() -> List[Tuple[str, int, int]]:
    """[(title, files, bytes)] of what a backup would hold now."""
    out = []
    for root in roots():
        files = _files(root)
        out.append((root.title, len(files), sum(f.stat().st_size for f in files if f.exists())))
    return out


def create(target: Optional[Path] = None) -> Tuple[Path, int, int]:
    """Write the archive. Returns (path, files, bytes of content). Raises OSError when it cannot write."""
    folder = default_folder()
    target = Path(target) if target else folder / f"argent-backup-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    if target.is_dir():
        target = target / f"argent-backup-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    count, total = 0, 0
    manifest = {"format": FORMAT, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "roots": {}}
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for root in roots():
                names = []
                for f in _files(root):
                    if _is_inside_backups(f, folder):
                        continue                                  # never a backup inside a backup
                    rel = f.relative_to(root.path).as_posix()
                    zf.write(f, f"{root.label}/{rel}")
                    names.append(rel)
                    count += 1
                    total += f.stat().st_size
                manifest["roots"][root.label] = names
            zf.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1))
        tmp.replace(target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target, count, total


def list_backups() -> List[Path]:
    folder = default_folder()
    return sorted(folder.glob("argent-backup-*.zip"), reverse=True) if folder.is_dir() else []


class BackupError(ValueError):
    pass


def _read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        data = json.loads(zf.read(MANIFEST).decode("utf-8"))
    except (KeyError, ValueError):
        raise BackupError("это не архив Argent (нет manifest.json)")
    if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("roots"), dict):
        raise BackupError("формат архива не поддерживается этой версией Argent")
    return data


def _safe_target(root: Root, rel: str) -> Path:
    """Where `rel` goes, or BackupError when it is not in the allowlist or tries to leave the root."""
    parts = Path(rel).parts
    if not parts or Path(rel).is_absolute() or ".." in parts or ":" in rel:
        raise BackupError(f"недопустимый путь в архиве: {rel!r}")
    if root.items is not None:
        allowed = {item.rstrip("/") for item in root.items}
        if parts[0] not in allowed:
            raise BackupError(f"в архиве лишний файл (вне разрешённых мест): {rel!r}")
    return root.path.joinpath(*parts)


def inspect(archive: Path) -> dict:
    """What an archive holds: {"created": str, "roots": {label: [files]}}. Raises BackupError."""
    try:
        with zipfile.ZipFile(archive) as zf:
            return _read_manifest(zf)
    except zipfile.BadZipFile:
        raise BackupError("файл повреждён или это не zip")


def restore(archive: Path, confirm: Optional[Callable[[str], bool]] = None) -> Tuple[int, Optional[Path]]:
    """Put an archive back. Everything is checked BEFORE anything is written. Returns (files restored,
    the safety copy of what was replaced or None). `confirm(text)` is asked once, with the counts."""
    by_label = {r.label: r for r in roots()}
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError):
        raise BackupError("файл повреждён или это не zip")
    with zf:
        manifest = _read_manifest(zf)
        plan: List[Tuple[str, Path]] = []
        present = set(zf.namelist())
        for label, names in manifest["roots"].items():
            root = by_label.get(label)
            if root is None:
                raise BackupError(f"в архиве неизвестная часть «{label}»")
            for rel in names:
                member = f"{label}/{rel}"
                if member not in present:
                    raise BackupError(f"в архиве нет файла из списка: {member}")
                plan.append((member, _safe_target(root, rel)))
        replaced = [t for _, t in plan if t.exists()]
        if confirm is not None and not confirm(
                f"Вернуть {len(plan)} файлов из архива от {manifest.get('created', '?')}"
                + (f"; {len(replaced)} из них заменят нынешние (их копия сохранится)" if replaced else "") + "?"):
            return 0, None
        safety: Optional[Path] = None
        if replaced:
            safety, _, _ = create(default_folder() / f"argent-backup-before-restore-{time.strftime('%Y%m%d-%H%M%S')}.zip")
        for member, target in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return len(plan), safety
