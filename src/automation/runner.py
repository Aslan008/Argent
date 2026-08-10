"""Run one automation with nobody watching.

The interactive agent assumes a human is present: gated actions raise a prompt
and wait. Neither half of that works unattended — waiting hangs the scheduler
forever, and auto-approving hands an unsupervised model the authority to delete
files or publish things.

So an unattended run REFUSES every gated action and records what it wanted to
do. The refusal is not a failure of the run: the model gets "Execution aborted"
back as a normal tool result and can finish the safe part of its job, and the
attempt shows up in the run log for you to read afterwards. Anything that
genuinely needs to act on the world belongs behind an explicit review step, not
behind a policy flag.

Three more bounds, all of which exist because nobody will notice a runaway:
  * allowed_tools — the task gets only the tools it needs;
  * max_turns — a hard cap on tool round-trips;
  * a single run at a time per automation (the scheduler enforces overlap).
"""

import threading
from datetime import datetime

from logger import get_logger

log = get_logger("automation")

# Tools an unattended run may never use, whatever the automation declares.
# These either need a human (they ask questions), or hand control to something
# that outlives the run (background processes, sub-agents with their own loops).
UNATTENDED_TOOL_DENYLIST = {
    "ask_user_questions", "request_user_approval",
    "run_admin_command", "delete_plugin", "create_plugin",
    "wait_heartbeat", "end_auto_mode",
}


class UnattendedGate:
    """Approval backend for a run with no human: deny, and remember."""

    # Read by approval.request_approval: suppresses the session grants and the
    # POLICY_AUTO shortcut, both of which are consent a present human gave for
    # their own work. Without it a user who had run /vibe, or who had once
    # answered "always allow git", handed every later scheduled run the same
    # authority — and the refusal log stayed empty because nothing was refused.
    unattended = True

    def __init__(self):
        self.denied = []

    def __call__(self, action: str, destructive: bool, grant_key) -> str:
        self.denied.append({"action": action, "destructive": bool(destructive)})
        log.info("automation refused a gated action: %s", action)
        return "deny"


# Granted to every run whatever it declared. Remembering what you already
# reported is not authority over anything, and a narrow toolset that leaves it
# out turns a monitoring job into a machine for repeating itself.
UNATTENDED_ALWAYS_ALLOWED = ("filter_new_items",)


def resolve_tools(allowed_tools: list) -> list | None:
    """The toolset for a run: the declared list minus the denylist.

    None means "the agent's default set" and is only returned when the
    automation declared nothing — the denylist still applies at the approval
    layer, since those tools all route through it or need a human.
    """
    if not allowed_tools:
        return None
    tools = [t for t in allowed_tools if t not in UNATTENDED_TOOL_DENYLIST]
    return tools + [t for t in UNATTENDED_ALWAYS_ALLOWED if t not in tools]


def run_automation(automation, agent=None, now=None) -> dict:
    """Execute one automation. Never raises — a scheduler must survive its jobs.

    Returns {status, summary, denied_actions, new_items, started, finished}
    where status is ok | error | denied. 'denied' means the task could not do
    its job without an action that requires a human.
    """
    import approval
    from src.automation import memory

    started = now or datetime.now()
    gate = UnattendedGate()
    tools = resolve_tools(automation.allowed_tools)
    text_parts = []
    status = "ok"

    if agent is None:
        from agent import ArgentAgent
        agent = ArgentAgent()

    approval.set_approval_backend(gate)
    memory.set_scope(automation.name)
    try:
        turns = 0
        for chunk in agent.process_user_input(automation.task, allowed_tools=tools):
            kind = chunk.get("type")
            if kind in ("content_stream", "content", "content_replace"):
                text_parts.append(chunk.get("content", ""))
            elif kind == "tool_end":
                turns += 1
                if turns >= max(1, automation.max_turns):
                    text_parts.append(
                        f"\n[stopped: reached the {automation.max_turns}-turn budget]")
                    break
            elif kind == "error":
                text_parts.append(f"\n[{chunk.get('content', '')}]")
    except Exception as e:
        status = "error"
        text_parts.append(f"\n[run failed: {e}]")
        log.warning("automation %r failed: %s", automation.name, e)
    finally:
        approval.reset_approval_backend()
        # A run that crashed never delivered its findings, so releasing them
        # keeps the items unseen and they come back next time. Committing here
        # would lose exactly what the automation exists to catch.
        if status == "ok":
            new_items = memory.commit()
        else:
            memory.rollback()
            new_items = 0       # nothing was delivered, so nothing was news
        memory.reset_scope()

    finished = datetime.now()
    summary = "".join(text_parts).strip()
    if status == "ok" and gate.denied and not summary:
        # It needed a human and produced nothing else — that is the whole story
        # of the run, so don't report it as a success.
        status = "denied"

    return {
        "status": status,
        "summary": summary,
        "denied_actions": gate.denied,
        "new_items": new_items,
        "started": started,
        "finished": finished,
    }


def run_and_record(automation, agent=None, notify: bool = True) -> dict:
    """run_automation + persist the outcome + tell the user if it matters.

    ``notify`` is off for a manual /tasks run: you are already looking at the
    output, and a toast about something on your screen is pure noise.
    """
    from src.automation.notify import notify_run
    from src.automation.store import record_run

    result = run_automation(automation, agent=agent)
    record_run(automation.name, result["status"], result["summary"],
               result["started"], result["finished"], result["denied_actions"],
               result.get("new_items", 0))
    if notify:
        try:
            notify_run(automation.name, result)
        except Exception as e:      # never let delivery break a good run
            log.info("could not notify about %r: %s", automation.name, e)
    return result
