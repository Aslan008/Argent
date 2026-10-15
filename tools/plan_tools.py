"""Tools for working in steps: the todo checklist and the way out of plan mode."""

from datetime import datetime
from pathlib import Path

from logger import get_logger

log = get_logger("tools")


def todo_write(todos=None) -> str:
    """Replace the checklist of the current task. See src.agent.todos."""
    from src.agent import todos as todo_store
    return todo_store.write(todos)


def exit_plan_mode(plan: str = "") -> str:
    """Show the plan to the person and, if they approve, leave plan mode."""
    from src.agent.plan_mode import PLAN

    if not PLAN.active:
        return "Plan mode is not on — nothing to leave. Carry on with the task."
    plan = (plan or "").strip()
    if len(plan) < 40:
        return ("Error: `plan` is the COMPLETE plan the person will read and approve — goal, the files to change "
                "and how, the order of steps, how you will verify, the risks. Write it in full and call again.")

    path = Path(".argent") / "artifacts" / f"plan-{datetime.now():%Y%m%d-%H%M%S}.md"
    saved = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plan + "\n", encoding="utf-8")
        saved = path.as_posix()
    except OSError as e:
        log.warning("could not save the plan: %s", e)

    from ui import console
    import questionary
    try:
        from rich.markdown import Markdown
        from rich.panel import Panel
        console.print(Panel(Markdown(plan), title="План", border_style="cyan"))
    except Exception:
        console.print(plan)
    if saved:
        console.print(f"[dim]План сохранён: {saved}[/dim]")
    # The plan itself was printed above; the question repeats where to find it for a frontend that
    # draws the question apart from the output.
    question = f"Утвердить план ({len(plan.splitlines())} строк) и перейти к работе?"
    if saved:
        question += f" Полный текст: {saved}"
    try:
        approved = questionary.confirm(question, default=False).ask()
    except (KeyboardInterrupt, EOFError):
        approved = None

    if approved is None:
        return ("The person dismissed the question without answering. Treat it as NOT approved: stay in plan mode "
                "and ask what they want changed.")
    if not approved:
        return ("The person did NOT approve the plan. Stay in plan mode: ask what to change (ask_user_questions), "
                "revise the plan, and call exit_plan_mode again.")
    PLAN.leave()
    return ("Plan APPROVED. Plan mode is off — every tool works again. Write the steps into todo_write and carry the "
            "plan out one step at a time" + (f" (the plan is saved in {saved})." if saved else "."))
