"""The `ultracode` tool: a big job as a planned, checked workflow. See src/agent/workflow."""

from argent.logger import get_logger

log = get_logger("tools")


def ultracode(action: str = "start", goal: str = "", run_id: str = "", role: str = "coder",
             files: str = "", checks: str = "", verify: str = "", title: str = "") -> str:
    from src.agent.workflow import runtime

    action = (action or "start").strip().lower()
    if action == "start":
        return runtime.start(goal)
    if action == "resume":
        return runtime.resume(run_id)
    if action == "status":
        return runtime.status()
    if action == "stop":
        return runtime.stop()
    if action == "add_step":
        return runtime.add_step(goal, role=role, title=title, files=files, checks=checks, verify=verify)
    return "Error: action must be one of start, resume, status, stop, add_step."
