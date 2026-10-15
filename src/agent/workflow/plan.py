"""The plan of a workflow, and the strictness it is read with.

A model writes the plan; nothing about it is trusted until it has been parsed and validated here. A small model
writes plans that are almost right — a role spelled "developer", a dependency on a step that does not exist, a
fenced block around the JSON — so the reader forgives spelling and refuses structure: unknown dependencies, cycles,
duplicate ids, too many steps. The refusal is a sentence the planner can be shown and asked to fix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROLES = ("coder", "researcher", "reviewer", "docwriter")
WRITER_ROLES = frozenset({"coder", "docwriter"})

PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,23}$")
_ROLE_WORDS = {
    "coder": ("coder", "developer", "dev", "engineer", "implementer", "implementation", "code", "programmer",
              "executor", "writer-code", "fixer"),
    "researcher": ("researcher", "research", "analyst", "explorer", "investigator", "reader", "analysis"),
    "reviewer": ("reviewer", "review", "tester", "qa", "verifier", "auditor", "checker", "test"),
    "docwriter": ("docwriter", "docs", "documentation", "doc", "technical-writer"),
}
_ROLE_LOOKUP = {w: role for role, words in _ROLE_WORDS.items() for w in words}

MAX_TEXT = 600
MAX_CHECKS = 3
MAX_FILES = 12
MAX_CHECK_LEN = 300


class PlanError(ValueError):
    """The plan cannot be used. The message is written for the model that wrote it."""


@dataclass
class Step:
    id: str
    title: str
    goal: str
    role: str = "coder"
    deps: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)
    status: str = PENDING
    attempts: int = 0
    report: str = ""
    verdict: str = ""
    changed: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)        # what went wrong so far: shown to the next attempt
    error: str = ""

    @property
    def writes(self) -> bool:
        return self.role in WRITER_ROLES

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class Plan:
    goal: str
    steps: List[Step]
    final_checks: List[str] = field(default_factory=list)
    summary: str = ""

    def by_id(self) -> Dict[str, Step]:
        return {s.id: s for s in self.steps}

    def done_ids(self) -> set:
        return {s.id for s in self.steps if s.status == DONE}

    def ready(self) -> List[Step]:
        """Pending steps whose dependencies are all done, in plan order."""
        done = self.done_ids()
        return [s for s in self.steps if s.status == PENDING and all(d in done for d in s.deps)]

    def unfinished(self) -> List[Step]:
        return [s for s in self.steps if s.status in (PENDING, RUNNING)]

    def skip_blocked(self) -> List[Step]:
        """Mark pending steps that can never run (a dependency failed or was skipped). Returns them."""
        changed = []
        loop = True
        while loop:
            loop = False
            dead = {s.id for s in self.steps if s.status in (FAILED, SKIPPED)}
            for s in self.steps:
                if s.status == PENDING and any(d in dead for d in s.deps):
                    s.status = SKIPPED
                    s.error = "a step it depends on did not finish"
                    changed.append(s)
                    loop = True
        return changed

    def to_dict(self) -> dict:
        return {"goal": self.goal, "summary": self.summary, "final_checks": list(self.final_checks),
                "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        return cls(goal=data.get("goal", ""), summary=data.get("summary", ""),
                   final_checks=list(data.get("final_checks") or []),
                   steps=[Step.from_dict(s) for s in data.get("steps") or []])

    def outline(self) -> str:
        """The plan as the person reads it before saying yes."""
        lines = []
        if self.summary:
            lines.append(self.summary)
        for s in self.steps:
            dep = f" ← после {', '.join(s.deps)}" if s.deps else ""
            lines.append(f"{s.id}. [{s.role}] {s.title}{dep}")
            for c in s.checks:
                lines.append(f"      проверка: {c}")
        for c in self.final_checks:
            lines.append(f"итоговая проверка: {c}")
        return "\n".join(lines)


# ── reading what the model wrote ───────────────────────────────────────────────

def _first_json_object(text: str) -> str:
    """The first balanced {...} in the text (the model may wrap it in prose or a fence)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    raise PlanError("the answer contains no JSON object")


def _strings(value, limit: int, each: int = MAX_TEXT) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise PlanError("a list of strings was expected")
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text[:each])
    return out[:limit]


def _role(raw, step_id: str) -> str:
    word = str(raw or "coder").strip().lower().replace(" ", "-")
    role = _ROLE_LOOKUP.get(word)
    if role is None:
        raise PlanError(f"step {step_id}: unknown role {raw!r}; use one of {', '.join(ROLES)}")
    return role


def parse_plan(text: str, goal: str, max_steps: int = 8) -> Plan:
    """Parse and validate. Raises PlanError with a message the planner can act on."""
    try:
        data = json.loads(_first_json_object(text or ""))
    except json.JSONDecodeError as e:
        raise PlanError(f"the JSON is invalid: {e.msg} (line {e.lineno})")
    if not isinstance(data, dict):
        raise PlanError("the plan must be a JSON object with a `steps` array")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanError("the plan has no `steps` array (or it is empty)")
    if len(raw_steps) > max_steps:
        raise PlanError(f"the plan has {len(raw_steps)} steps; at most {max_steps} are allowed — merge related ones")

    steps: List[Step] = []
    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            raise PlanError(f"step #{index} is not an object")
        step_id = str(raw.get("id") or f"s{index}").strip()
        if not _ID.match(step_id):
            raise PlanError(f"step #{index}: bad id {step_id!r} (letters, digits, - and _, up to 24 characters)")
        goal_text = str(raw.get("goal") or raw.get("description") or raw.get("task") or "").strip()
        title = str(raw.get("title") or raw.get("name") or goal_text[:60]).strip()
        if not goal_text:
            raise PlanError(f"step {step_id}: `goal` is empty — say what the worker must achieve")
        try:
            steps.append(Step(
                id=step_id, title=title[:120], goal=goal_text[:MAX_TEXT * 2],
                role=_role(raw.get("role"), step_id),
                deps=_strings(raw.get("deps") or raw.get("depends_on"), 8, 24),
                files=_strings(raw.get("files"), MAX_FILES, 200),
                checks=[c[:MAX_CHECK_LEN] for c in _strings(raw.get("checks"), MAX_CHECKS, MAX_CHECK_LEN)],
            ))
        except PlanError as e:
            raise PlanError(f"step {step_id}: {e}")

    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise PlanError(f"duplicate step ids: {', '.join(dup)}")
    known = set(ids)
    for s in steps:
        for d in s.deps:
            if d not in known:
                raise PlanError(f"step {s.id} depends on {d!r}, which is not a step of this plan")
            if d == s.id:
                raise PlanError(f"step {s.id} depends on itself")
    _reject_cycles(steps)

    return Plan(goal=goal, steps=steps,
                final_checks=[c[:MAX_CHECK_LEN] for c in _strings(data.get("final_checks"), MAX_CHECKS, MAX_CHECK_LEN)],
                summary=str(data.get("summary") or "").strip()[:MAX_TEXT])


def _reject_cycles(steps: List[Step]) -> None:
    remaining = {s.id: set(s.deps) for s in steps}
    while remaining:
        free = [i for i, deps in remaining.items() if not deps]
        if not free:
            raise PlanError("the dependencies form a cycle: " + ", ".join(sorted(remaining)))
        for i in free:
            remaining.pop(i)
        for deps in remaining.values():
            deps.difference_update(free)


def layers(plan: Plan) -> List[List[str]]:
    """Step ids grouped by how many steps must come before them (what could run side by side)."""
    depth: Dict[str, int] = {}
    by_id = plan.by_id()

    def level(step_id: str) -> int:
        if step_id not in depth:
            depth[step_id] = 1 + max((level(d) for d in by_id[step_id].deps), default=-1)
        return depth[step_id]

    out: Dict[int, List[str]] = {}
    for s in plan.steps:
        out.setdefault(level(s.id), []).append(s.id)
    return [out[k] for k in sorted(out)]


def overlaps(a: Step, b: Step) -> bool:
    """Could two writing steps touch the same file? Unknown scope counts as "yes"."""
    if not (a.writes and b.writes):
        return False
    if not a.files or not b.files:
        return True

    def norm(p: str) -> str:
        return p.replace("\\", "/").strip("/").lower()

    for x in map(norm, a.files):
        for y in map(norm, b.files):
            if x == y or x.startswith(y + "/") or y.startswith(x + "/"):
                return True
    return False


def find_step(plan: Optional[Plan], step_id: str) -> Optional[Step]:
    return plan.by_id().get(step_id) if plan else None
