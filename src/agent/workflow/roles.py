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
     "checks": ["a non-interactive shell command that exits non-zero when the step is wrong"],
     "verify": ["an instruction a second agent can carry out to see the result with its own eyes"],
     "thinking": "optional: off | low | medium | high | xhigh — how hard this step has to think",
     "model": "optional: a model id from WHAT THE WORKERS MAY RUN ON",
     "provider": "optional: the provider of that model"}
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
- A check must be SCOPED to the files of its step — `python -m pytest tests/test_that_module.py -q`, not the
  whole suite. "Run everything and make it green" is not a check: a repository that is already red somewhere
  (environment, missing packages) can never satisfy it, and the step cycles on it for nothing.
- A check that a step cannot influence is worse than no check: it costs three attempts and teaches nothing.
- `verify` is for what an exit code cannot prove: "open http://localhost:3000 and confirm the form submits",
  "read the new report and confirm every section has numbers". A verifier agent carries it out for real and
  answers PASS or FAIL with evidence. Use it for user-visible behaviour only — a unit test needs no verifier.
- `thinking` and `model` are how the person's money is saved, and they are OPTIONAL. Use "thinking": "low"
  for routine steps (renaming, small edits, documentation, listing things), "off" for purely mechanical
  ones, and empty for anything where a wrong move is expensive (design, tricky debugging, security, data).
  Set `model`/`provider` only to a target listed under WHAT THE WORKERS MAY RUN ON — never invent one.
  Empty means the default, which is the main agent itself.
- Do not plan steps like "commit", "deploy" or "push".
- Output the JSON object and nothing else. Write titles and goals in the language of the user's goal."""


def planner_messages(goal: str, max_steps: int, repo_hint: str = "", feedback: str = "",
                     targets: str = "") -> List[Dict[str, str]]:
    user = f"GOAL:\n{goal.strip()}\n"
    if repo_hint.strip():
        user += f"\nWHAT IS KNOWN ABOUT THE PROJECT:\n{repo_hint.strip()}\n"
    if targets.strip():
        user += f"\nWHAT THE WORKERS MAY RUN ON:\n{targets.strip()}\n"
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
        parts.append("If a check is already red BEFORE your change — an unrelated failure, a missing package — say so "
                     "in your report instead of \"fixing\" tests you were not asked to touch. The engine compares the "
                     "check with the unpatched tree, so an honest \"this was red already\" is believed, and a patch "
                     "that repairs someone else's test will be read as scope it was not given.")
    if step.notes:
        parts.append("PREVIOUS ATTEMPT(S) FAILED. What went wrong — do not repeat it:\n" +
                     "\n".join(f"- {_clip(n, 1500)}" for n in step.notes[-3:]))
    parts.append(
        "RULES: work only inside your step; do not ask the user anything (nobody can answer); make the smallest "
        "change that meets the goal; finish with a short report: what you changed (file by file), what you "
        "checked and its result, and anything the next steps must know.")
def history_of(plan: Plan, step: Step, limit: int = 8) -> str:
    """What the run has already done, compactly, for the worker that comes next.

    A worker used to start from its own step alone: it did not know which files the run
    had already changed or what the previous workers learned. Repeating that work is the
    most expensive way to be wrong, so the finished steps are handed over as a brief.
    """
    lines: List[str] = []
    for other in plan.steps:
        if other.id == step.id or other.status != "done":
            continue
        row = f"[{other.id}] {other.title}"
        if other.changed:
            row += " — changed " + ", ".join(other.changed[:6])
        if other.report:
            row += "\n    said: " + _clip(other.report, 400).replace("\n", " ")
        lines.append(row)
    return "\n".join(lines[-limit:])


def executor_task(step: Step, plan: Plan, level: int, repo_hint: str = "", history: str = "") -> str:
    """The task text a worker sub-agent starts from.

    It sees no supervisor dialogue — that is the point of a separate process — so
    everything it could need is written down here: the step, the files, what the run
    already did, what the project looks like, and what the last attempt of THIS step
    tried before it failed. A worker that has to rediscover all of that is a worker
    that spends its budget on archaeology.
    """
    by_id = plan.by_id()
    parts = [
        f"You are one worker of a larger job. OVERALL GOAL: {plan.goal.strip()}",
        f"YOUR STEP [{step.id}] {step.title}\n{step.goal.strip()}",
    ]
    if repo_hint.strip():
        parts.append("WHAT IS KNOWN ABOUT THE PROJECT:\n" + _clip(repo_hint, 1500))
    if step.files:
        parts.append("FILES in scope: " + ", ".join(step.files) +
                     ("\nChange only these files; if the job needs another one, say so in your report "
                      "instead of changing it." if step.writes else ""))
    for dep_id in step.deps:
        dep = by_id.get(dep_id)
        if dep is not None and dep.report:
            parts.append(f"RESULT OF STEP [{dep.id}] {dep.title} (already done and in the tree):\n"
                         f"{_clip(dep.report, 1500)}")
    if history.strip():
        parts.append("WHAT THIS RUN HAS ALREADY DONE — do not redo it, build on it:\n"
                     + _clip(history, 2000))
    if step.checks:
        parts.append("YOUR WORK IS CHECKED BY: " + "; ".join(f"`{c}`" for c in step.checks) +
                     "\nRun them yourself before you report, and fix what they show.")
        parts.append("If a check is already red BEFORE your change — an unrelated failure, a missing package — say so "
                     "in your report instead of \"fixing\" tests you were not asked to touch. The engine compares the "
                     "check with the unpatched tree, so an honest \"this was red already\" is believed, and a patch "
                     "that repairs someone else's test will be read as scope it was not given.")
    if step.verify:
        parts.append("AFTER YOU, A VERIFIER ACTS THESE OUT AND ANSWERS PASS OR FAIL:\n"
                     + "\n".join(f"- {v}" for v in step.verify)
                     + "\nMake each of them true by actually working, and say in your report what you did for each.")
    if step.attempt_reports:
        parts.append("THE PREVIOUS ATTEMPT OF THIS STEP ALREADY TRIED THIS (its changes are NOT in your tree — "
                     "redo what is still missing, keep what was right):\n"
                     + "\n".join(f"- {_clip(r, 1500)}" for r in step.attempt_reports[-2:]))
    if step.notes:
        parts.append("PREVIOUS ATTEMPT(S) FAILED. What went wrong — do not repeat it:\n" +
                     "\n".join(f"- {_clip(n, 1500)}" for n in step.notes[-3:]))
    parts.append(
        "RULES: work only inside your step; do not ask the user anything (nobody can answer); make the smallest "
        "change that meets the goal; finish with a short report: what you changed (file by file), what you "
        "checked and its result, and anything the next steps must know.")
    return "\n\n".join(parts)


VERIFIER_TASK = """You are the VERIFIER of one step of an autonomous job. Another agent did the work; you decide
whether it is really done, and your answer is the only one the engine will accept.

Do not judge from the patch alone. Carry out each instruction below for real — run the command, open the
page, read the file, press the button — and report what you saw. A verifier that only re-reads the diff is
worth nothing: the diff is what we are sceptical about. You may read anything, run commands and use the
browser; do not change the work itself.

Answer with one line per instruction: the instruction, then OK or FAILED, then the evidence (the command you
ran and its result, or what you saw). The last line must be exactly `VERDICT: PASS` (every instruction held)
or `VERDICT: FAIL` (at least one did not)."""


def verifier_task(step: Step, diff: str, check_log: str = "", instruction: str = "") -> str:
    """What the verifying agent is told: the instruction, the step, and the evidence to distrust."""
    checks = [instruction] if instruction else list(step.verify)
    lines = [
        f"STEP [{step.id}] {step.title}",
        f"WHAT THE STEP HAD TO ACHIEVE: {step.goal.strip()}",
        "",
        "WHAT MUST BE TRUE, CHECKED BY DOING IT:",
    ]
    lines += [f"- {c}" for c in checks]
    body = "PATCH:\n" + (diff.strip() or "(empty)")
    if check_log.strip():
        body += "\n\nMECHANICAL CHECKS THAT ALREADY RAN:\n" + _clip(check_log, 2000)
    lines += ["", "THE WORK UNDER SCRUTINY:", _clip(body, 12000)]
    return "\n".join(lines)


_VERIFIER_VERDICT = re.compile(r"VERDICT\s*:\s*\**\s*(PASS|FAIL)\b", re.I)


def parse_verifier_verdict(text: str) -> str:
    """`PASS`, `FAIL`, or `""` when the verifier did not answer in the agreed form."""
    found = _VERIFIER_VERDICT.findall(text or "")
    return found[-1].upper() if found else ""


def critic_target(step: Step, diff: str, check_log: str = "") -> str:
    head = f"STEP [{step.id}] {step.title}\nGOAL OF THE STEP: {step.goal.strip()}\n"
    if step.files:
        head += "PLANNED SCOPE: " + ", ".join(step.files) + "\n"
    out = head + "\nPATCH:\n" + (diff.strip() or "(empty)")
    if check_log.strip():
        out += ("\n\nCHECKS THAT WERE RUN (each line says whether it passed, or why a red "
                "one is not this patch's fault):\n" + _clip(check_log, 2000))
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
        "does not exist, a check that covers the whole test suite (a repository that is already red somewhere can "
        "never satisfy it — ask for the tests of the files the step touches instead), scope that is too big for "
        "one worker.\n"
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
