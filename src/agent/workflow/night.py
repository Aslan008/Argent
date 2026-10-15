"""Night mode: a big job planned and approved now, carried out while nobody is at the keyboard.

Two halves with a clear line between them:

* `prepare` (interactive, the person is here): makes the plan, has the advisor look at it, shows it WITH the
  commands that will run unattended, asks for approval, saves the run as `scheduled` and creates a one-shot
  automation for it. Nothing is approved later, so everything that needs a person happens here.
* `run` (unattended, started by the automation): continues the saved run through the same engine as a daytime
  one. Approval requests are answered by the unattended gate (src/automation/runner.py): refused and recorded,
  except the exact commands the person approved with the plan. The cloud advisor is used only when the
  person had already consented to it; otherwise the main model advises, and the report says so.

Whatever the night does lands the way a daytime run's work does: a patch is applied only after its checks passed
and the critic read it, and the report waits for the person (inbox / Telegram).
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from logger import get_logger
from src.agent.workflow import calibration, runtime, store
from src.agent.workflow.engine import Engine
from src.agent.workflow.plan import Plan

log = get_logger("ultracode")

TASK_PREFIX = "ultracode-night-"
MIN_LEVEL = 2          # level 1 is "a plan the model carries out itself": nothing to run at night


def task_name(run_id: str) -> str:
    return TASK_PREFIX + run_id


def command_for(run_id: str) -> str:
    return f'"{sys.executable}" -m src.agent.workflow.night {run_id}'


def plan_commands(plan: Plan) -> List[str]:
    """Every shell command a run of this plan executes by itself: the checks of the steps and the final ones."""
    out: List[str] = []
    for step in plan.steps:
        out.extend(c for c in step.checks if c and c not in out)
    out.extend(c for c in plan.final_checks if c and c not in out)
    return out


def _when(text: str) -> Tuple[Optional[str], str]:
    """The canonical one-shot schedule for what the person typed, or (None, why not)."""
    from src.automation.schedule import ScheduleError, normalize_schedule
    try:
        canonical = normalize_schedule((text or "").strip())
    except ScheduleError as e:
        return None, str(e)
    if not canonical.startswith("once at"):
        return None, "ночной запуск — одноразовый: «завтра в 2:30», «через 3 часа», «once at 2026-10-14 02:30»"
    return canonical, ""


def prepare(goal: str, when: str) -> str:
    """Plan, show, ask, save, schedule. Returns what to tell the person."""
    from src.agent import plan_mode
    from src.automation import store as astore

    if plan_mode.PLAN.active:
        return "Ночной запуск недоступен в режиме планирования: сначала утвердите план."
    goal = (goal or "").strip()
    if len(goal) < 8:
        return "Нужна полная формулировка цели: что именно сделать и как проверить."
    schedule, why = _when(when)
    if schedule is None:
        return f"Время запуска не понято: {why}"

    level, reason, profile = runtime._decide_level()
    if level < MIN_LEVEL:
        return (f"Ночной запуск здесь не имеет смысла: {reason} (уровень {level}). Нужен уровень "
                f"{MIN_LEVEL} или выше (/ultracode level, /ultracode calibrate).")

    state = store.RunState(id=store.new_id(), goal=goal, level=level, cwd=_cwd())
    state.event(f"ночной запуск, уровень {level} ({calibration.LEVEL_NAMES[level]}): {reason}")
    ports = runtime.build_ports(state)
    engine = Engine(state, ports, runtime._budgets(level, profile))
    plan, problem = engine._make_plan()
    if plan is None:
        return f"План составить не удалось: {problem}"
    plan = engine._review_plan(plan)
    commands = plan_commands(plan)

    runtime.say(f"[bold]План ночного запуска[/bold] (уровень {level} — {calibration.LEVEL_NAMES[level]}):\n"
                f"{plan.outline()}")
    listed = "\n".join(f"  {c}" for c in commands) if commands else "  (никаких: шаги без проверок)"
    runtime.say("[bold]Без вас будут выполнены только эти команды-проверки[/bold] (каждая, как написана):\n" + listed)
    import approval
    if not approval.request_approval(
            f"Ultracode: выполнить этот план ночью ({schedule}, {len(plan.steps)} шаг.)? Все остальные действия, "
            f"которым нужно ваше разрешение, будут отклонены и попадут в отчёт.",
            destructive=False, grant_key=None):
        return "План не одобрен — ничего не запланировано."

    state.set_plan(plan)
    state.status = "scheduled"
    state.event(f"одобрено человеком; запуск: {schedule}; команды: {len(commands)}")
    store.save(state)

    budgets = runtime._budgets(level, profile)
    job = astore.Automation(name=task_name(state.id), task=command_for(state.id), schedule=schedule, mode="command",
                            notify="always", allowed_commands=commands,
                            timeout_minutes=int(budgets.max_minutes) + 30, retries=0, max_failures=0)
    problems = astore.validate_automation(job)
    if problems:
        state.status = "cancelled"
        state.result = "Не удалось запланировать: " + "; ".join(problems)
        store.save(state)
        return state.result
    astore.upsert_automation(job)
    return (f"Ночной запуск {state.id} запланирован: {schedule}.\n"
            "Он идёт, пока Argent запущен или работает служба автоматизаций (/tasks service). Отчёт придёт, как у "
            "любой автоматизации; посмотреть ход: /ultracode status, отменить: /ultracode night cancel "
            f"{state.id}.")


def cancel(run_id: str) -> str:
    from src.automation import store as astore
    state = store.load(run_id, _cwd()) if run_id else None
    removed = astore.remove_automation(task_name(run_id)) if run_id else False
    if state is not None and state.status == "scheduled":
        state.status = "cancelled"
        state.result = "Ночной запуск отменён человеком."
        store.save(state)
    if state is None and not removed:
        return f"Ночной запуск «{run_id}» не найден."
    return f"Ночной запуск {run_id} отменён."


def scheduled() -> List[store.RunState]:
    return [s for s in store.list_runs(_cwd()) if s.status == "scheduled"]


def _cwd() -> str:
    import os
    return os.getcwd()


def run(run_id: str) -> int:
    """The unattended half. Exit code: 0 done, 1 not done (report printed), 2 nothing to run."""
    import approval
    from src.automation import store as astore
    from src.automation.runner import UnattendedGate, command_allowed

    state = store.load(run_id, _cwd())
    if state is None:
        print(f"Ночной запуск {run_id}: состояние не найдено в этом проекте.")
        return 2
    if state.status not in ("scheduled", "stopped", "running"):
        print(f"Ночной запуск {run_id}: уже {state.status}, делать нечего.")
        return 2
    plan = state.get_plan()
    if plan is None:
        print(f"Ночной запуск {run_id}: в состоянии нет плана.")
        return 2

    job = astore.get_automation(task_name(run_id))
    allowed = list(job.allowed_commands) if job else []
    unapproved = [c for c in plan_commands(plan) if not command_allowed(c, allowed)]
    if unapproved:
        # The plan on disk is not the plan the person approved: nothing of it runs.
        state.status, state.result = "cancelled", "План изменён после одобрения: " + "; ".join(unapproved[:3])
        state.event(state.result)
        store.save(state)
        print(state.result)
        return 2

    try:
        from src.agent import governor
        governor.lower_process_priority(__import__("os").getpid())
    except Exception:
        pass
    gate = UnattendedGate(allowed)
    approval.set_approval_backend(gate)
    runtime.set_enabled(True)
    try:
        text = runtime.resume(run_id)
    finally:
        approval.reset_approval_backend()
    print(text)
    if gate.denied:
        print("\nОтклонено без человека (нужно ваше решение):")
        for item in gate.denied[:10]:
            print(f"  - {item['action']}")
    final = store.load(run_id, _cwd())
    return 0 if final is not None and final.status == "done" else 1


def main(argv: Optional[list] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m src.agent.workflow.night <run_id>")
        return 2
    return run(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
