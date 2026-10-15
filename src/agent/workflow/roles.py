"""What each role of a workflow is told. Kept apart from the engine so the wording can be read and tested alone.

Four voices:

* the PLANNER — one request, JSON out; it never touches the tree;
* the WORKER — a sub-agent (src/agent/subagents.py) with a task made here: its step, the results of the steps it
  builds on, what failed last time;
* the CRITIC — src/agent/critic.py, fed a patch and the output of its checks;
* the ADVISOR — a stronger second opinion at three decision points: the plan, a step that keeps failing, the whole.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.agent.workflow.plan import Plan, Step

PLANNER_SYSTEM = """You are the PLANNER of an autonomous software workflow. You do not write code. You split ONE goal
into the fewest steps that can be carried out and checked independently, and answer with a single JSON object.

Schema:
{
  "summary": "one sentence: the approach",
  "steps": [
    {"id": "s1", "title": "short title", "goal": "what the worker must achieve, concrete and checkable",
     "role": "coder | researcher | reviewer | docwriter",
     "deps": ["ids of steps whose RESULT this step needs"],
     "files": ["paths the step will change or must read"],
     "checks": ["a non-interactive shell command that exits non-zero when the step is wrong"]}
  ],
  "final_checks": ["commands that prove the whole goal is met"]
}

Rules:
- At most {max_steps} steps. Fewer is better: a step is a unit of work a worker can finish and a command can verify.
- Steps that change DIFFERENT files and do not need each other's result have no deps — they may run in parallel.
- A "coder" step changes files; "researcher" and "reviewer" only read; "docwriter" writes documentation.
- `files` is a promise: a coder step may not touch files outside it. Put every file it will change there.
- `checks` must be real commands for THIS project (look at what the repository uses). If you do not know a
  command that exists, leave `checks` empty rather than invent one.
- Do not plan steps like "commit", "deploy" or "push".
- Output the JSON object and nothing else. Write titles and goals in the language of the user's goal."""


def planner_messages(goal: str, max_steps: int, repo_hint: str = "", feedback: str = "") -> List[Dict[str, str]]:
    user = f"GOAL:\n{goal.strip()}\n"
    if repo_hint.strip():
        user += f"\nWHAT IS KNOWN ABOUT THE PROJECT:\n{repo_hint.strip()}\n"
    if feedback.strip():
        user += f"\nYOUR PREVIOUS ANSWER WAS REJECTED:\n{feedback.strip()}\nFix exactly that and answer again.\n"
    return [{"role": "system", "content": PLANNER_SYSTEM.replace("{max_steps}", str(max_steps))},
            {"role": "user", "content": user}]


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit // 2] + "\n[…]\n" + text[-limit // 2:]


def executor_task(step: Step, plan: Plan, level: int) -> str:
    """The task text a worker sub-agent starts from. It sees this and nothing of the supervisor's dialogue."""
    by_id = plan.by_id()
    parts = [
        f"You are one worker of a larger job. OVERALL GOAL: {plan.goal.strip()}",
        f"YOUR STEP [{step.id}] {step.title}\n{step.goal.strip()}",
    ]
    if step.files:
        parts.append("FILES in scope: " + ", ".join(step.files) +
                     ("\nChange only these files; if the job needs another one, say so in your report "
                      "instead of changing it." if step.writes else ""))
    for dep_id in step.deps:
        dep = by_id.get(dep_id)
        if dep is not None and dep.report:
            parts.append(f"RESULT OF STEP [{dep.id}] {dep.title} (already done and in the tree):\n"
                         f"{_clip(dep.report, 1500)}")
    if step.checks:
        parts.append("YOUR WORK IS CHECKED BY: " + "; ".join(f"`{c}`" for c in step.checks) +
                     "\nRun them yourself before you report, and fix what they show.")
    if step.notes:
        parts.append("PREVIOUS ATTEMPT(S) FAILED. What went wrong — do not repeat it:\n" +
                     "\n".join(f"- {_clip(n, 1500)}" for n in step.notes[-3:]))
    parts.append(
        "RULES: work only inside your step; do not ask the user anything (nobody can answer); make the smallest "
        "change that meets the goal; finish with a short report: what you changed (file by file), what you "
        "checked and its result, and anything the next steps must know.")
    return "\n\n".join(parts)


def critic_target(step: Step, diff: str, check_log: str = "") -> str:
    head = f"STEP [{step.id}] {step.title}\nGOAL OF THE STEP: {step.goal.strip()}\n"
    if step.files:
        head += "PLANNED SCOPE: " + ", ".join(step.files) + "\n"
    out = head + "\nPATCH:\n" + (diff.strip() or "(empty)")
    if check_log.strip():
        out += "\n\nCHECKS THAT WERE RUN (all passed):\n" + _clip(check_log, 2000)
    return out


# ── advisor ───────────────────────────────────────────────────────────────────

ADVISOR_SYSTEM = (
    "You are the ADVISOR of an autonomous coding workflow: a senior engineer consulted at decision points. "
    "You execute nothing and see only what is written below. Be decisive and brief; name concrete files, "
    "commands and failure causes. Do not restate the question. Answer in the language of the goal.\n\n"
)

_ADVISOR_FORMATS = {
    "plan": (
        "TASK: review the plan before any work starts. Look for: a step that cannot be verified, a missing step, "
        "two parallel steps that touch the same files, a dependency that is wrong, a check command that likely "
        "does not exist, scope that is too big for one worker.\n"
        "Answer: up to 6 short bullet points of concrete changes, then the last line exactly "
        "`VERDICT: APPROVE`, `VERDICT: REVISE` (the plan needs the changes above) or `VERDICT: REJECT` "
        "(the approach is wrong)."),
    "stuck": (
        "TASK: a step has failed repeatedly. Read the attempts and say what is really wrong and what the next "
        "attempt must do differently. If the step itself is mis-specified, say how to restate it.\n"
        "Answer: a short diagnosis, then `GUIDANCE:` followed by instructions for the worker (they are pasted "
        "into its task verbatim), then the last line `VERDICT: RETRY` or `VERDICT: GIVE_UP` (more tries are "
        "pointless)."),
    "final": (
        "TASK: judge whether the finished work meets the goal. Use the step reports and check results. Name what "
        "is missing or risky, concretely.\n"
        "Answer: up to 6 bullet points, then the last line `VERDICT: ACCEPT` or `VERDICT: FIX` "
        "(and say what must still be done)."),
}


def advisor_messages(kind: str, goal: str, body: str) -> List[Dict[str, str]]:
    task = _ADVISOR_FORMATS.get(kind, _ADVISOR_FORMATS["final"])
    return [{"role": "system", "content": ADVISOR_SYSTEM + task},
            {"role": "user", "content": f"GOAL:\n{goal.strip()}\n\n{body.strip()}"}]


_VERDICT = re.compile(r"VERDICT\s*:\s*\**\s*(APPROVE|REVISE|REJECT|RETRY|GIVE_UP|ACCEPT|FIX)\b", re.I)


def parse_advisor_verdict(text: str) -> str:
    found = _VERDICT.findall(text or "")
    return found[-1].upper() if found else ""


def advisor_guidance(text: str) -> str:
    """The part of a 'stuck' answer meant for the worker."""
    m = re.search(r"GUIDANCE\s*:\s*(.*?)(?:\n\s*\**VERDICT\s*:|\Z)", text or "", re.S | re.I)
    return (m.group(1).strip() if m else (text or "").strip())


def describe_steps(plan: Plan) -> str:
    """The steps and their outcomes, compactly, for the advisor."""
    lines = []
    for s in plan.steps:
        row = f"[{s.id}] {s.title} — {s.role}, {s.status}, attempts {s.attempts}"
        if s.changed:
            row += f", changed: {', '.join(s.changed[:8])}"
        if s.verdict:
            row += f", critic: {s.verdict}"
        lines.append(row)
        if s.report:
            lines.append("    report: " + _clip(s.report, 500).replace("\n", " "))
        if s.error:
            lines.append("    problem: " + _clip(s.error, 400).replace("\n", " "))
    return "\n".join(lines)
