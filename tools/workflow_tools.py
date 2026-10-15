"""The `ultracode` tool: a big job as a planned, checked workflow. See src/agent/workflow."""

from logger import get_logger

log = get_logger("tools")


def ultracode(action: str = "start", goal: str = "", run_id: str = "") -> str:
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
    return "Error: action must be one of start, resume, status, stop."
