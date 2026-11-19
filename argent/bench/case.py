"""A bench case: a task, a start state, and checks that decide pass/fail."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from argent.logger import get_logger

log = get_logger("bench")


class CaseError(ValueError):
    """A case file that cannot be run as written."""


@dataclass
class Check:
    cmd: str
    expect_exit: int = 0
    timeout_s: float = 120.0
    stdout_contains: Optional[str] = None
    stdout_not_contains: Optional[str] = None

    @classmethod
    def from_value(cls, value: Any) -> "Check":
        if isinstance(value, str):
            return cls(cmd=value)
        if not isinstance(value, dict):
            raise CaseError(f"a check must be a string or an object, got {type(value).__name__}")
        cmd = str(value.get("cmd") or "").strip()
        if not cmd:
            raise CaseError("a check needs a non-empty 'cmd'")
        return cls(
            cmd=cmd,
            expect_exit=int(value.get("expect_exit", 0)),
            timeout_s=float(value.get("timeout_s", 120)),
            stdout_contains=value.get("stdout_contains"),
            stdout_not_contains=value.get("stdout_not_contains"),
        )


@dataclass
class Case:
    id: str
    task: str
    start: str = "HEAD"
    setup: List[str] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)
    timeout_s: float = 300.0
    tags: List[str] = field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    source: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Optional[Path] = None) -> "Case":
        if not isinstance(data, dict):
            raise CaseError("a case must be a YAML/JSON object")
        ident = str(data.get("id") or "").strip()
        task = str(data.get("task") or "").strip()
        if not ident:
            raise CaseError("case 'id' is required")
        if not task:
            raise CaseError(f"case {ident!r}: 'task' is required")
        raw_checks = data.get("checks") or []
        if not isinstance(raw_checks, list):
            raise CaseError(f"case {ident!r}: 'checks' must be a list")
        checks = [Check.from_value(item) for item in raw_checks]
        limits = data.get("limits") if isinstance(data.get("limits"), dict) else {}
        setup = data.get("setup") or []
        if not isinstance(setup, list):
            raise CaseError(f"case {ident!r}: 'setup' must be a list")
        allowed = data.get("allowed_tools")
        if allowed is not None and not isinstance(allowed, list):
            raise CaseError(f"case {ident!r}: 'allowed_tools' must be a list")
        return cls(
            id=ident,
            task=task,
            start=str(data.get("start") or "HEAD"),
            setup=[str(x) for x in setup],
            checks=checks,
            timeout_s=float(limits.get("timeout_s", data.get("timeout_s", 300))),
            tags=[str(x) for x in (data.get("tags") or [])],
            allowed_tools=[str(x) for x in allowed] if allowed else None,
            source=source,
        )

    def validate_runnable(self) -> None:
        if not self.checks:
            raise CaseError(
                f"case {self.id!r} has no checks; a case without a check cannot decide pass/fail"
            )


def load_case(path: Path) -> Case:
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        raise CaseError(f"cannot read {path}: {e}") from e
    return Case.from_dict(data, source=Path(path))


def list_cases(directory: Path) -> List[Path]:
    root = Path(directory)
    if not root.is_dir():
        return []
    suffixes = {".yaml", ".yml", ".json"}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in suffixes and p.is_file())
