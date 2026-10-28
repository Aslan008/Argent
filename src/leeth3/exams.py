"""Leeth 3 exams: where they live, how they are frozen, and the transformations of the stress sets (plan section 6).

The exam CONTENT never enters the repository (the repository is pushed to GitHub, and anything public can end up in some model's
training data - including the teacher's). It lives in ``~/.argent/leeth3_exams/`` (``LEETH3_EXAMS`` overrides), outside the folder the
teacher agent can read. The repository keeps only the manifest of hashes (``reports/leeth3_exams_manifest.json``), so a changed exam
is detected.

Layout:   <dir>/<set>.jsonl          teacher-record format (src/leeth3/records.py), one item per line
          <dir>/argent/queries.json  Argent requests (list), run through the real dispatcher by scripts/leeth2_real_exam.py
          <dir>/argent/labels.json   {"tools": {request: tool}, "skip": []} - every other request is "delegate"
          <dir>/MANIFEST.json        sha256 and counts of every file; "frozen": true once the exams are fixed

Freezing: after ``freeze`` the exam files may not change (``verify`` fails on any difference); an exam is replaced only by a NEW set
name with a written reason in the plan, never edited in place after a candidate was measured on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .records import Record, parse_record

REPO_MANIFEST = Path(__file__).resolve().parents[2] / "reports" / "leeth3_exams_manifest.json"


def exams_dir() -> Path:
    return Path(os.environ.get("LEETH3_EXAMS") or (Path.home() / ".argent" / "leeth3_exams"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    p = (root or exams_dir()) / "MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"frozen": False, "files": {}}


def write_set(name: str, items: Iterable[Any], root: Optional[Path] = None) -> Path:
    """Validate every item (a dict or a Record) and write the set; refuses to overwrite a frozen set."""
    root = root or exams_dir()
    man = load_manifest(root)
    rel = f"{name}.jsonl"
    if man.get("frozen") and rel in man["files"]:
        raise PermissionError(f"exam {rel} is frozen; create a new set instead of editing it")
    recs: List[Record] = [it if isinstance(it, Record) else parse_record(it) for it in items]
    ids = [r.id for r in recs]
    if len(set(ids)) != len(ids):
        dup = [k for k, v in Counter(ids).items() if v > 1][:5]
        raise ValueError(f"duplicate ids in {name}: {dup}")
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in recs), encoding="utf-8")
    return path


def read_set(name: str, root: Optional[Path] = None) -> List[Record]:
    path = (root or exams_dir()) / f"{name}.jsonl"
    return [parse_record(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or exams_dir()
    files: Dict[str, Any] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.json":
            continue
        rel = p.relative_to(root).as_posix()
        entry: Dict[str, Any] = {"sha256": _sha(p), "bytes": p.stat().st_size}
        if p.suffix == ".jsonl":
            recs = [parse_record(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            entry.update(items=len(recs), ptype=dict(Counter(r.ptype for r in recs)), families=len({r.family for r in recs}),
                         teachers=dict(Counter(r.teacher for r in recs)))
        elif rel.endswith("queries.json"):
            entry["items"] = len(json.loads(p.read_text(encoding="utf-8")))
        files[rel] = entry
    return {"files": files}


def freeze(root: Optional[Path] = None, repo_manifest: Optional[Path] = REPO_MANIFEST) -> Dict[str, Any]:
    root = root or exams_dir()
    # Freezing again may only ADD sets; it must never bless a frozen set that was edited or removed in the meantime.
    broken = [d for d in verify(root) if not d.startswith(("added after freeze", "exams are not frozen"))]
    if broken:
        raise PermissionError(f"frozen exams differ from their manifest: {broken[:5]}; restore them or create a new set")
    man = build_manifest(root)
    # Who wrote each set is in files[*].teachers (the record field): none of them may be the family that generates training data.
    man.update(frozen=True, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               author="per set, see files[*].teachers; argent/queries.json - claude (Anthropic)")
    (root / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    if repo_manifest is not None:                 # hashes and counts only - never the content
        repo_manifest.parent.mkdir(parents=True, exist_ok=True)
        repo_manifest.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def verify(root: Optional[Path] = None) -> List[str]:
    """Differences between the exam files and their frozen manifest (empty = intact)."""
    root = root or exams_dir()
    man = load_manifest(root)
    if not man.get("frozen"):
        return ["exams are not frozen yet"]
    now = build_manifest(root)["files"]
    out = []
    for rel, e in man["files"].items():
        if rel not in now:
            out.append(f"missing: {rel}")
        elif now[rel]["sha256"] != e["sha256"]:
            out.append(f"changed: {rel}")
    out += [f"added after freeze: {rel}" for rel in now if rel not in man["files"]]
    return out


# ---- stress transformations (deterministic per seed) --------------------------------------------------------------------------------
_KEYBOARD_NEIGHBOURS = {
    **{a: b for a, b in zip("qwertyuiopasdfghjklzxcvbnm", "wertyuiopqsdfghjklaxcvbnmz")},
    **{a: b for a, b in zip("йцукенгшщзхфывапролджэячсмить", "цукенгшщзхйывапролджэфчсмитья")},
}


def typos(text: str, rng: random.Random, rate: float = 0.04) -> str:
    """Human-like typos in words longer than 3 letters: swap of neighbours, a dropped letter, a neighbouring key, a doubled letter.
    Paths, commands with punctuation and numbers are left alone (a typo there changes the meaning, not the wording)."""
    out = []
    for w in text.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate * 3:
            i = rng.randrange(1, len(w) - 1)
            kind = rng.randrange(4)
            if kind == 0:
                w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            elif kind == 1:
                w = w[:i] + w[i + 1:]
            elif kind == 2 and w[i].lower() in _KEYBOARD_NEIGHBOURS:
                w = w[:i] + _KEYBOARD_NEIGHBOURS[w[i].lower()] + w[i + 1:]
            else:
                w = w[:i] + w[i] + w[i:]
        out.append(w)
    return " ".join(out)


def inflate(state: str, filler: Sequence[str], target_words: int, rng: random.Random, position: str = "middle") -> str:
    """Bury ``state`` inside unrelated documents up to ~target_words words: at the start, in the middle or at the end."""
    have = len(state.split())
    docs: List[str] = []
    while have < target_words:
        d = rng.choice(filler)
        docs.append(d)
        have += len(d.split())
    if position == "start":
        parts = [state] + docs
    elif position == "end":
        parts = docs + [state]
    else:
        k = len(docs) // 2
        parts = docs[:k] + [state] + docs[k:]
    return "\n\n".join(parts)
