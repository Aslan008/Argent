"""What a red check actually proves.

A check command is an oracle about *the repository*, not about the patch. In a
repository that is already red somewhere — this project carries ~45 failures that
belong to the environment, not the code — `pytest -k "..."` exits non-zero no
matter what a worker did. The engine used to read that as "the worker failed" and
sent the step back: three attempts, twenty minutes, nothing to fix.

So the verdict here is a comparison, not a boolean: what failed before the patch,
what failed after it, and is anything new. `judge()` answers with one of

* the check is green now;
* it was red before and the red is the same — not this patch's fault;
* it was green before and is red now — this patch broke it;
* it was red before and has *more* red now — this patch made it worse.

The signature is read from the output, because a check's exit code cannot tell
`FAILED tests/a.py::x` from `FAILED tests/b.py::y`. Recognised dialects: pytest
(`FAILED/ERROR id`), cargo (`test name ... FAILED`, `error[E0308]`), jest
(`✕ name`), and a coarse fallback — the set of lines that look like errors, with
numbers and addresses rubbed out so the same failure keeps the same face.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# pytest: "FAILED tests/test_a.py::TestX::test_y - AssertionError: ..."
_PYTEST = re.compile(r"^(FAILED|ERROR)\s+([^\s]+(?:::[^\s]+)*)", re.M)
# cargo: "test engine::tests::drops_it ... FAILED", "error[E0308]: mismatched types"
_CARGO_TEST = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+FAILED", re.M)
_CARGO_ERR = re.compile(r"^(error(?:\[[A-Z]\d+\])?):\s*(.{0,80})", re.M)
# jest/vitest: "  ✕ renders the row (12 ms)"
_JEST = re.compile(r"^\s*[✕×]\s+(.+?)\s*(?:\(\d+\s*ms\))?$", re.M)
# go: "--- FAIL: TestThing (0.00s)"
_GO = re.compile(r"^---\s+FAIL:\s+(\S+)", re.M)

# A line that reads like an error even when the dialect is unknown. Numbers, hex
# addresses and quoted strings are rubbed out: the same breakage must produce the
# same text, and a different one must not.
_LOOKS_LIKE_ERROR = re.compile(
    r"(?i)\b("
    r"traceback|error|failed|failure|panic|exception|assertionerror|" 
    r"module ?not ?found|importerror|cannot import|no module named|"
    r"not recognized|is not recognized|command not found|denied|permission"
    r")\b")
_NOISE = [
    re.compile(r"0x[0-9a-fA-F]+"),             # addresses
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|passed|failed|error[s]?)\b", re.I),
    re.compile(r"/[^\s:'\"]+"),                # absolute paths (/usr/..., /tmp/...)
    re.compile(r"[A-Za-z]:\\[^\s:'\"]+"),      # windows paths
    re.compile(r"'\s*'"),                      # empty quoted fragments
]
MAX_SIGNATURE = 60


@dataclass(frozen=True)
class Result:
    """One run of one check command."""
    ok: bool
    failed: FrozenSet[str] = frozenset()
    output: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "failed": sorted(self.failed), "output": self.output[:2000]}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["Result"]:
        if not data:
            return None
        return cls(ok=bool(data.get("ok")), failed=frozenset(data.get("failed") or []),
                   output=str(data.get("output") or ""))


@dataclass
class Verdict:
    ok: bool
    reason: str
    new: List[str] = field(default_factory=list)      # failures this patch added
    fixed: int = 0                                     # failures it removed
    pre_existing: int = 0                              # failures that were already there
    measured: bool = True                              # False: there was no "before" to compare with


def _normalize(line: str) -> str:
    text = _ANSI.sub("", line).strip().lower()
    for pattern in _NOISE:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .:-")


def failure_signature(output: str) -> FrozenSet[str]:
    """The identities of the failures in `output` (empty when none can be named)."""
    text = _ANSI.sub("", output or "")
    found: List[str] = []
    for pattern in (_PYTEST, _CARGO_TEST, _GO):
        found.extend(m.group(2) if pattern is _PYTEST else m.group(1) for m in pattern.finditer(text))
    found.extend(f"{m.group(1)}: {_normalize(m.group(2))}" for m in _CARGO_ERR.finditer(text))
    found.extend(_normalize(m.group(1)) for m in _JEST.finditer(text))
    if not found:
        # Unknown dialect: keep the lines that read like errors, stripped of the bits
        # that change from run to run.
        for line in text.splitlines():
            if _LOOKS_LIKE_ERROR.search(line):
                cleaned = _normalize(line)
                if cleaned and len(cleaned) > 8:
                    found.append(cleaned)
    unique = sorted({f for f in found if f})
    return frozenset(unique[:MAX_SIGNATURE])


def judge(before: Result, after: Result) -> Verdict:
    """Did *this patch* break something, given how the check behaved without it?"""
    if after.ok and not before.ok:
        return Verdict(ok=True, reason="the check was red before the patch and is green now",
                       fixed=len(before.failed), pre_existing=0)
    if after.ok:
        return Verdict(ok=True, reason="passed")

    new = sorted(set(after.failed) - set(before.failed))[:MAX_SIGNATURE]
    gone = sorted(set(before.failed) - set(after.failed))
    if before.ok:
        return Verdict(ok=False, new=new,
                       reason="the check passed without the patch, so the patch broke it")
    if not new:
        if not after.failed:
            return Verdict(ok=True, reason=("the check was red before the patch and still is, and its output "
                                                  "names no particular failure — nothing here can be blamed on "
                                                  "this patch"))
        return Verdict(ok=True, pre_existing=len(after.failed), fixed=len(gone),
                       reason=(f"{len(after.failed)} failure(s) that were already there before the patch, "
                               f"nothing new"))
    return Verdict(ok=False, new=new, pre_existing=len(after.failed) - len(new),
                   reason=(f"{len(new)} new failure(s) — they were not there before the patch"
                           + (f" ({len(after.failed) - len(new)} others were already red)" if len(after.failed) > len(new) else "")))


def describe(verdict: Verdict, limit: int = 4) -> str:
    """One line for the log and the step report."""
    text = verdict.reason
    if verdict.new:
        text += ": " + "; ".join(verdict.new[:limit])
        if len(verdict.new) > limit:
            text += f" и ещё {len(verdict.new) - limit}"
    return text


# ── checks nobody wrote ───────────────────────────────────────────────────────

_MAX_DERIVED = 3


def derive_checks(files: Sequence[str], root: str, limit: int = _MAX_DERIVED) -> List[str]:
    """Test commands that cover the files a step touched, when the plan named none.

    A step without checks is unverified: the engine has nothing to ask about it. The
    repository usually already knows which test file belongs to a module, so the
    cheapest honest check is "the tests for the files you touched" — never the whole
    suite, which says nothing about this step.
    """
    base = Path(root)
    tests_dir = base / "tests"
    wanted: List[str] = []
    commands: List[str] = []

    for raw in files:
        path = Path(str(raw).replace("\\", "/"))
        if path.suffix == ".rs":
            # A Rust file belongs to the package that contains it: `cargo test -p name`
            # is the same "the tests for what you touched", not the whole workspace.
            parts = path.parts
            if "crates" in parts:
                index = parts.index("crates")
                if index + 1 < len(parts):
                    command = f"cargo test -p {parts[index + 1]} -q"
                    if command not in commands:
                        commands.append(command)
            continue
        if path.suffix != ".py" or not tests_dir.is_dir():
            continue
        stem = path.stem
        if stem.startswith("test_"):
            continue
        patterns = (f"test_{stem}.py", f"test_{stem}_*.py", f"{stem}_test.py")
        for pattern in patterns:
            for found in sorted(tests_dir.rglob(pattern)):
                rel = str(found.relative_to(base)).replace("\\", "/")
                if rel not in wanted:
                    wanted.append(rel)

    if wanted:
        commands.insert(0, "python -m pytest " + " ".join(wanted[:_MAX_DERIVED]) + " -q")
    return commands[:limit]


def summarize(results: Iterable[str]) -> str:
    """'2 проверки: 1 зелёная, 1 была красной до патча' — the short version for a panel."""
    items = list(results)
    if not items:
        return "проверок не было"
    red = sum(1 for i in items if i.startswith("≈"))
    return f"проверок: {len(items)}, из них была красной до патча: {red}"
