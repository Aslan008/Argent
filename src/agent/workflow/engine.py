"""The loop that runs a plan.

It knows nothing about providers, processes or the screen: the world reaches it through `Ports` — plain callables
that `runtime.py` fills with the real thing (a model, a sub-agent process, the approval prompt) and the tests fill
with stand-ins. What lives here is the part that has to be right:

* a step runs only after the steps it depends on are DONE and their work is in the tree;
* two steps that might touch the same file never run together;
* a worker's work is not trusted: its checks run in its private copy, a critic reads the patch, and only then is
  the patch applied; any failure sends the step back with what went wrong, a bounded number of times;
* every limit (steps, tries, minutes, advisor calls) ends in an honest stop with a report, never a silent go-on;
* the state is written after every change, so a crash is a pause (`store.revive`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from logger import get_logger
from src.agent.workflow import plan as P
from src.agent.workflow import roles, store
from src.agent.workflow.advisor import Advice
from src.agent.workflow.plan import DONE, FAILED, PENDING, RUNNING, SKIPPED, Plan, PlanError, Step

log = get_logger("workflow")


@dataclass
class Budgets:
    max_steps: int = 8
    max_attempts: int = 3              # tries per step
    max_minutes: float = 120.0
    max_advisor_calls: int = 6
    parallelism: int = 1


@dataclass
class Outcome:
    """What a finished worker left behind."""
    ok: bool
    report: str = ""
    error: str = ""
    denied: List[str] = field(default_factory=list)   # actions it needed a person for and was refused
    diff: str = ""
    files: List[str] = field(default_factory=list)
    cwd: str = ""                                      # where its checks should run
    in_place: bool = False                             # it worked in the real tree: nothing to apply


class Job:
    """One running worker. `runtime.SubagentJob` is the real one."""

    def finished(self) -> bool: ...
    def outcome(self) -> Outcome: ...
    def apply(self) -> str: ...                        # "" on success, else why not
    def discard(self) -> None: ...
    def stop(self) -> None: ...


@dataclass
class Ports:
    plan: Callable[[list], str]                                    # planner messages -> raw answer
    start: Callable[[Step, str], Job]                              # step, task text -> running job
    check: Callable[[str, str], Tuple[bool, str]]                  # command, cwd -> (passed, output)
    critic: Optional[Callable[[Step, str], Tuple[str, str]]] = None    # step, material -> (verdict, text)
    advise: Optional[Callable[[str, str, str], Optional[Advice]]] = None   # kind, goal, body -> advice
    confirm: Callable[[Plan, int], bool] = lambda plan, level: True
    say: Callable[[str], None] = lambda text: None
    stop_requested: Callable[[], bool] = lambda: False
    repo_hint: Callable[[], str] = lambda: ""
    progress: Callable[[Plan], None] = lambda plan: None             # the plan changed (a step state, an attempt)


def _tail(text: str, limit: int = 1500) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "[…]\n" + text[-limit:]


class Engine:
    def __init__(self, state: store.RunState, ports: Ports, budgets: Budgets,
                 save: Callable[[store.RunState], object] = store.save,
                 clock: Callable[[], float] = time.monotonic, poll: float = 0.25):
        self.state = state
        self.ports = ports
        self.budgets = budgets
        self._save = save
        self.clock = clock
        self.poll = poll
        self._t0 = clock()
        self._base_spent = state.spent_seconds
        self._halt_reason = ""
        self._jobs: Dict[str, Job] = {}

    # -- small things -----------------------------------------------------------------------

    def _say(self, text: str) -> None:
        self.state.event(text)
        try:
            self.ports.say(text)
        except Exception:
            pass

    def _spent(self) -> float:
        return self._base_spent + (self.clock() - self._t0)

    def _persist(self) -> None:
        self.state.spent_seconds = self._spent()
        try:
            self._save(self.state)
        except OSError as e:
            log.warning("workflow state not saved: %s", e)

    def _plan(self) -> Plan:
        return self.state.get_plan()

    def _commit(self, plan: Plan) -> None:
        self.state.set_plan(plan)
        self._persist()
        self._progress(plan)

    def _progress(self, plan: Plan) -> None:
        try:
            self.ports.progress(plan)
        except Exception as e:                           # a panel must never be what stops a workflow
            log.debug("progress report failed: %s", e)

    def _end(self, status: str, text: str) -> store.RunState:
        self.state.status = status
        self.state.result = text
        self._persist()
        return self.state

    # -- advisor ----------------------------------------------------------------------------

    def _advice(self, kind: str, body: str) -> Optional[Advice]:
        if self.ports.advise is None:
            return None
        if self.state.advisor_calls >= self.budgets.max_advisor_calls:
            self._say("Лимит обращений к советнику исчерпан — решаю сам.")
            return None
        self.state.advisor_calls += 1
        try:
            advice = self.ports.advise(kind, self.state.goal, body)
        except Exception as e:
            log.warning("advisor failed: %s", e)
            self._say(f"Советник недоступен ({e}).")
            return None
        if advice is None:
            return None
        suffix = " (ответила основная модель: настроенный советник не ответил)" if advice.fallback else ""
        self._say(f"Советник [{advice.target}]{suffix}: {advice.verdict or 'без вердикта'}")
        return advice

    # -- planning ---------------------------------------------------------------------------

    def _make_plan(self, feedback: str = "") -> Tuple[Optional[Plan], str]:
        problem = feedback
        for _ in range(2):
            messages = roles.planner_messages(self.state.goal, self.budgets.max_steps,
                                              self.ports.repo_hint(), problem)
            text = self.ports.plan(messages)
            try:
                return P.parse_plan(text, self.state.goal, self.budgets.max_steps), ""
            except PlanError as e:
                problem = str(e)
                self._say(f"План не прошёл проверку: {e}")
        return None, problem

    def _review_plan(self, plan: Plan) -> Plan:
        advice = self._advice("plan", plan.outline())
        if advice is None or advice.verdict not in ("REVISE", "REJECT"):
            return plan
        self.state.replans += 1
        better, _ = self._make_plan(f"The advisor reviewed your plan and says:\n{advice.text}")
        if better is None:
            self._say("Исправленный по замечаниям план не получился — иду по исходному.")
            return plan
        self._say("План переписан по замечаниям советника.")
        return better

    # -- running ----------------------------------------------------------------------------

    def run(self) -> store.RunState:
        st = self.state
        try:
            if st.get_plan() is None:
                st.status = "planning"
                self._persist()
                plan, problem = self._make_plan()
                if plan is None:
                    return self._end("failed", f"Не удалось получить пригодный план: {problem}")
                plan = self._review_plan(plan)
                self._commit(plan)
                if not self.ports.confirm(plan, st.level):
                    return self._end("cancelled", "План не одобрен — ничего не запускалось.")
            st.status = "running"
            self._persist()
            self._execute()
            if self._halt_reason:
                return self._end("stopped", self._halt_reason)
            return self._finish()
        finally:
            self._stop_all()
            self._persist()

    def _stop_all(self) -> None:
        plan = self._plan()
        for step_id, job in list(self._jobs.items()):
            try:
                job.stop()
                job.discard()
            except Exception as e:
                log.debug("stopping %s: %s", step_id, e)
            if plan is not None:
                step = plan.by_id().get(step_id)
                if step is not None and step.status == RUNNING:
                    step.status = PENDING
                    step.attempts = max(0, step.attempts - 1)
        self._jobs.clear()
        if plan is not None:
            self.state.set_plan(plan)

    def _must_halt(self) -> str:
        if self.ports.stop_requested():
            return "Остановлено."
        if self._spent() > self.budgets.max_minutes * 60:
            return f"Исчерпан лимит времени ({self.budgets.max_minutes:.0f} мин)."
        return ""

    def _startable(self, plan: Plan) -> List[Step]:
        running = [plan.by_id()[i] for i in self._jobs if i in plan.by_id()]
        out: List[Step] = []
        for step in plan.ready():
            if any(P.overlaps(step, other) for other in running + out):
                continue
            out.append(step)
        return out

    def _execute(self) -> None:
        plan = self._plan()
        while True:
            reason = self._must_halt()
            if reason:
                self._halt_reason = reason
                self._say(reason + " Прогресс сохранён: /ultracode resume.")
                return
            for skipped in plan.skip_blocked():
                self._say(f"Шаг {skipped.id} пропущен: зависит от несделанного.")
            for step in self._startable(plan):
                if len(self._jobs) >= self.budgets.parallelism:
                    break
                self._launch(plan, step)
            if not self._jobs:
                if self._startable(plan):      # a launch failed and the step is waiting for its next try
                    continue
                if plan.unfinished():          # pending but nothing can start: a plan the checks missed
                    for s in plan.unfinished():
                        s.status, s.error = SKIPPED, "no step could be started for it"
                    self._commit(plan)
                return
            finished = [i for i, job in self._jobs.items() if job.finished()]
            if not finished:
                self._progress(plan)                     # the queue and the PC change while a worker thinks
                time.sleep(self.poll)
                continue
            for step_id in finished:
                job = self._jobs.pop(step_id)
                self._settle(plan, plan.by_id()[step_id], job)
                self._commit(plan)

    def _launch(self, plan: Plan, step: Step) -> None:
        step.attempts += 1
        step.status = RUNNING
        task = roles.executor_task(step, plan, self.state.level)
        try:
            job = self.ports.start(step, task)
        except Exception as e:
            self._say(f"Шаг {step.id}: не удалось запустить исполнителя: {e}")
            step.status = PENDING
            self._after_failure(plan, step, f"the worker could not be started: {e}")
            self._commit(plan)
            return
        self._jobs[step.id] = job
        self._say(f"▸ шаг {step.id} «{step.title}» [{step.role}], попытка {step.attempts}")
        self._commit(plan)

    # -- one finished step ------------------------------------------------------------------

    def _after_failure(self, plan: Plan, step: Step, why: str) -> None:
        """A try failed: try again with the lesson, or give the step up."""
        step.notes.append(why)
        step.error = why
        if step.attempts >= self.budgets.max_attempts:
            step.status = FAILED
            self._say(f"✗ шаг {step.id} не получился за {step.attempts} попытк.: {_tail(why, 200)}")
            return
        if step.attempts >= 2:
            body = (f"STEP [{step.id}] {step.title}\n{step.goal}\n\nATTEMPTS SO FAR:\n" +
                    "\n".join(f"{i}. {_tail(n, 800)}" for i, n in enumerate(step.notes[-3:], 1)))
            advice = self._advice("stuck", body)
            if advice is not None:
                if advice.verdict == "GIVE_UP":
                    step.status = FAILED
                    step.notes.append("the advisor: more attempts are pointless — " + _tail(advice.text, 400))
                    self._say(f"✗ шаг {step.id}: советник считает, что продолжать бессмысленно.")
                    return
                if advice.text:
                    step.notes.append("ADVISOR: " + advice.text)
        step.status = PENDING
        self._say(f"↻ шаг {step.id}: повторяю ({_tail(why, 160)})")

    def _settle(self, plan: Plan, step: Step, job: Job) -> None:
        out = job.outcome()
        if not out.ok:
            job.discard()
            self._after_failure(plan, step, f"the worker failed: {out.error or 'it ended without a report'}")
            return
        step.report = (out.report or "").strip()
        if out.denied:
            step.report += "\n[refused actions that need a person: " + "; ".join(out.denied[:6]) + "]"

        if not step.writes:
            step.status, step.error = DONE, ""
            job.discard()
            self._say(f"✓ шаг {step.id} выполнен")
            return

        if not out.in_place and not out.diff.strip():
            step.status, step.error = DONE, ""
            self._say(f"✓ шаг {step.id}: изменений не потребовалось")
            return

        step.changed = list(out.files)
        check_log: List[str] = []
        for command in step.checks:
            passed, output = self.ports.check(command, out.cwd)
            check_log.append(f"$ {command}\n{_tail(output, 600)}")
            if not passed:
                job.discard()
                self._after_failure(plan, step, f"the check `{command}` failed:\n{_tail(output)}")
                return

        verdict = ""
        if not out.in_place and self.ports.critic is not None and self.state.level >= 2:
            material = roles.critic_target(step, out.diff, "\n".join(check_log))
            try:
                verdict, text = self.ports.critic(step, material)
            except Exception as e:
                log.warning("critic failed: %s", e)
                verdict, text = "UNKNOWN", ""
            step.verdict = verdict
            last_try = step.attempts >= self.budgets.max_attempts
            if verdict == "RETHINK" or (verdict == "FIXABLE" and not last_try):
                job.discard()
                self._after_failure(plan, step, f"the critic said {verdict}:\n{_tail(text)}")
                return
            if verdict == "FIXABLE":
                step.notes.append("critic's reservations (accepted on the last try): " + _tail(text, 600))

        if not out.in_place:
            problem = job.apply()
            if problem:
                job.discard()
                self._after_failure(plan, step, f"the patch does not apply to the current tree: {_tail(problem, 600)}")
                return
        step.status, step.error = DONE, ""
        label = f"{verdict or 'без критика'}"
        self._say(f"✓ шаг {step.id} принят ({label}); файлов: {len(step.changed)}")

    # -- the end ----------------------------------------------------------------------------

    def _finish(self) -> store.RunState:
        plan = self._plan()
        broken = [s for s in plan.steps if s.status in (FAILED, SKIPPED)]
        final_log: List[str] = []
        final_ok = True
        if not broken:
            for command in plan.final_checks:
                passed, output = self.ports.check(command, self.state.cwd)
                final_log.append(f"$ {command}\n{_tail(output, 600)}\n=> {'passed' if passed else 'FAILED'}")
                final_ok = final_ok and passed
                self._say(f"{'✓' if passed else '✗'} итоговая проверка: {command}")

        notes = []
        if not broken:
            body = roles.describe_steps(plan) + ("\n\nFINAL CHECKS:\n" + "\n".join(final_log) if final_log else "")
            advice = self._advice("final", body)
            if advice is not None and advice.verdict == "FIX":
                notes.append("Советник просит доработать: " + advice.text)
            elif advice is not None and advice.text:
                notes.append("Советник: " + advice.text)
        self.state.advisor["notes"] = notes
        self._commit(plan)

        from src.agent.workflow import report
        status = "done" if (not broken and final_ok) else "failed"
        return self._end(status, report.render(self.state, final_log=final_log))
