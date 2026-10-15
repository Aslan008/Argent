"""The child process of one sub-agent.

    python src/agent/subagent_worker.py <spec.json>

A sub-agent that runs in its own interpreter has what a thread cannot give it: its own
working directory (so it can live in a git worktree while you keep working in the real
tree), its own copy of every process-wide setting, no way to scribble over the
supervisor's screen, and a death that costs only itself.

It also has nobody to ask. The person is talking to the supervisor, so the approval
gate here is the unattended one: a gated action is REFUSED and written down, except a
shell command the person had already allowed (the supervisor passes those along from
their permission rules and session grants — a sub-agent never gets more authority than
the supervisor was given). The refusal reaches the model as an ordinary tool result and
the report tells the supervisor what was refused.

The verdict travels on stdout as one marker line, because the rest of stdout is the
agent talking to a console.
"""

import json
import os
import sys
import traceback
from pathlib import Path

_ARGENT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(_ARGENT_ROOT))

from src.automation.runner import UnattendedGate  # noqa: E402

RESULT_MARKER = "ARGENT_SUBAGENT_RESULT "


def _emit(payload: dict) -> None:
    sys.stdout.write("\n" + RESULT_MARKER + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class ChildGate(UnattendedGate):
    """The unattended gate, plus the person's `ask` rules.

    The commands a child may run come from the person's allow rules and session grants, and a
    grant such as "git" is wider than a rule such as "ask before git push". The parent would
    have stopped to ask; the child has nobody to ask, so it must not take the broader grant."""

    def __call__(self, action, destructive, grant_key):
        from approval import command_from_action
        from src.agent import permissions
        command = command_from_action(action)
        if command is not None and permissions.PERMISSIONS.command_rule("ask", command) is not None:
            self.denied.append({"action": action, "destructive": bool(destructive),
                                "note": "the user's rules ask before this one"})
            return "deny"
        return super().__call__(action, destructive, grant_key)


def run(spec: dict) -> dict:
    import approval
    from src.agent import permissions, todos
    from src.agent.permissions import Permissions, parse_rule
    from src.agent.plan_mode import PLAN
    cwd = spec.get("cwd") or os.getcwd()
    os.chdir(cwd)

    rules = []
    for kind in ("deny", "ask"):
        for raw in spec.get(kind) or []:
            try:
                rules.append(parse_rule(raw, kind, "parent"))
            except ValueError:
                continue
    permissions.PERMISSIONS = Permissions.fixed(rules, cwd)

    gate = ChildGate(spec.get("allowed_commands") or [])
    approval.set_approval_backend(gate)

    if spec.get("plan_mode"):
        PLAN.enter()

    from agent import ArgentSubAgent
    agent = ArgentSubAgent(
        spec["role"], spec["task"], tools_override=spec.get("tools"),
        model_override=spec.get("model") or None, provider_override=spec.get("provider") or None)
    with todos.use_scope("worker"):
        report = agent.execute()
    return {"status": "ok", "report": report, "denied": gate.denied, "ran": gate.ran}


def main(argv: list) -> int:
    try:
        spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    except (IndexError, OSError, ValueError) as e:
        sys.stderr.write(f"subagent_worker: cannot read the spec: {e}\n")
        return 2
    try:
        _emit(run(spec))
    except BaseException as e:                       # the supervisor must hear how it died
        _emit({"status": "error", "report": "", "denied": [],
               "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
