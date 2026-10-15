"""A workflow run on disk.

Every change of state is written at once (a temp file, then a rename), so a crash, a closed window or a power cut
leaves a file that is either the state before or the state after — never half. `resume` reads it, puts the steps
that were running back to pending, and goes on from the last finished step.

The folder is the project's `.argent/workflows/`: the run belongs to the tree it changes.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.agent.workflow.plan import PENDING, RUNNING, Plan

#: Terminal states of a run. A run in any other state can be resumed.
FINAL = ("done", "failed", "cancelled")
MAX_EVENTS = 200


@dataclass
class RunState:
    id: str
    goal: str
    level: int
    cwd: str
    status: str = "planning"          # planning | scheduled | running | stopped | done | failed | cancelled
    plan: Optional[dict] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    spent_seconds: float = 0.0        # active time over all sessions (the budget counts it)
    advisor_calls: int = 0
    advisor: dict = field(default_factory=dict)       # provider, model, cloud, note
    events: List[dict] = field(default_factory=list)
    result: str = ""
    replans: int = 0

    def event(self, text: str) -> None:
        self.events.append({"t": round(time.time(), 1), "text": text})
        del self.events[:-MAX_EVENTS]

    def get_plan(self) -> Optional[Plan]:
        return Plan.from_dict(self.plan) if self.plan else None

    def set_plan(self, plan: Plan) -> None:
        self.plan = plan.to_dict()

    @property
    def resumable(self) -> bool:
        return self.status not in FINAL

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "RunState":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


def folder(cwd: Optional[str] = None) -> Path:
    return Path(cwd or os.getcwd()) / ".argent" / "workflows"


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid() % 1000:03d}"


def _safe(run_id: str) -> str:
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,40}", run_id or ""):
        raise ValueError(f"bad run id: {run_id!r}")
    return run_id


def save(state: RunState) -> Path:
    state.updated = time.time()
    root = folder(state.cwd)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_safe(state.id)}.json"
    tmp = root / f"{state.id}.json.tmp"
    data = json.dumps(state.to_dict(), ensure_ascii=False, indent=1)
    last: Optional[Exception] = None
    for attempt in range(6):                       # an antivirus or the indexer may hold the file for a moment
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
            return path
        except OSError as e:
            last = e
            time.sleep(0.05 * (attempt + 1))
    raise last if last else OSError("could not save the workflow state")


def load(run_id: str, cwd: Optional[str] = None) -> Optional[RunState]:
    path = folder(cwd) / f"{_safe(run_id)}.json"
    try:
        return RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def list_runs(cwd: Optional[str] = None) -> List[RunState]:
    root = folder(cwd)
    out = []
    try:
        names = sorted(root.glob("*.json"))
    except OSError:
        return []
    for path in names:
        try:
            out.append(RunState.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda s: s.created)


def latest(cwd: Optional[str] = None, resumable_only: bool = False) -> Optional[RunState]:
    runs = list_runs(cwd)
    if resumable_only:
        runs = [r for r in runs if r.resumable]
    return runs[-1] if runs else None


def revive(state: RunState) -> RunState:
    """After a crash or a stop: steps that were mid-flight start over; finished ones stay finished."""
    plan = state.get_plan()
    if plan is not None:
        for step in plan.steps:
            if step.status == RUNNING:
                step.status = PENDING
        state.set_plan(plan)
    if state.status in ("running", "stopped"):
        state.status = "running"
    return state
