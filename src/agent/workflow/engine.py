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

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from argent.logger import get_logger
from src.agent.workflow import checks as C
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
    def progress(self) -> str:                         # what it is doing now, for the panel
        return ""


@dataclass
class Ports:
    plan: Callable[[list], str]                                    # planner messages -> raw answer
    start: Callable[[Step, str], Job]                              # step, task text -> running job
    check: Callable[[str, str], Tuple[bool, str]]                  # command, cwd -> (passed, output)
    critic: Optional[Callable[[Step, str], Tuple[str, str]]] = None    # step, material -> (verdict, text)
    verify: Optional[Callable[[str], Tuple[bool, str]]] = None          # task text -> (passed, answer)
    advise: Optional[Callable[[str, str, str], Optional[Advice]]] = None   # kind, goal, body -> advice
    confirm: Callable[[Plan, int], bool] = lambda plan, level: True
    say: Callable[[str], None] = lambda text: None
    stop_requested: Callable[[], bool] = lambda: False
    repo_hint: Callable[[], str] = lambda: ""
    # What a worker may be told to run on (provider:model, local/cloud). The planner
    # needs it to be able to choose a cheap target at all, instead of inventing one.
    targets: Callable[[], str] = lambda: ""
    progress: Callable[[Plan], None] = lambda plan: None             # the plan changed (a step state, an attempt)
    # The observers: a command kept running for the whole run, and its newest output.
    watch_start: Optional[Callable[[str], str]] = None
    watch_tail: Optional[Callable[[str], str]] = None
    watch_stop: Callable[[str], None] = lambda pid: None


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
        # The plan the loop is working on, so a step can join it while the run goes.
        self._live: Optional[Plan] = None
        self._lock = threading.Lock()
        self._watchers: Dict[str, str] = {}          # pid -> command
        self._watcher_seen: Dict[str, str] = {}      # pid -> the newest line already reported
        self._scheduled: List[dict] = []

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
                                              self.ports.repo_hint(), problem,
                                              self.ports.targets())
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
            self._start_observers(st.get_plan())
            self._execute()
            if self._halt_reason:
                return self._end("stopped", self._halt_reason)
            return self._finish()
        finally:
            self._stop_all()
            self._stop_observers()
            self._live = None
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

    def _watch(self, plan: Plan) -> None:
        """A running worker's own last words, for the panel.

        A step that says only "running" is a black box for as long as it thinks. The
        worker's log already says which tool it is calling; taking the newest line is
        the whole of this monitor — it costs nothing and answers "what is it doing".
        """
        by_id = plan.by_id()
        for step_id, job in self._jobs.items():
            step = by_id.get(step_id)
            getter = getattr(job, "progress", None)
            if step is None or getter is None:
                continue
            try:
                lines = (getter() or "").strip().splitlines()
            except Exception as e:                     # a monitor must never stop a run
                log.debug("progress of %s failed: %s", step_id, e)
                continue
            if lines and lines[-1] != step.progress:
                step.progress = _tail(lines[-1], 160)

    # -- observers: the world outside the steps ----------------------------------------------

    _ERROR_LINE = re.compile(r"\b(error|failed|failure|exception|traceback|fatal|panic)\b", re.I)

    def _start_observers(self, plan: Optional[Plan]) -> None:
        """Start what should watch the world for the whole run.

        A dev server or a log tail is not a step: it runs from the first minute to the
        last, and the interesting thing is the moment its newest line turns bad. The
        deferred checks are the same idea on a clock — "twenty minutes in, run the smoke
        test again" — which is as close to cron as a run of minutes can honestly get.
        """
        if plan is None:
            return
        self._live = plan
        self._scheduled = [dict(item, done=False) for item in plan.scheduled]
        if not plan.watch:
            return
        if self.ports.watch_start is None:
            self._say("· план просит наблюдателей, но смотреть нечем — пропускаю")
            return
        for command in plan.watch:
            try:
                pid = self.ports.watch_start(command)
            except Exception as e:
                self._say(f"· наблюдатель `{command}` не запустился: {e}")
                continue
            if pid:
                self._watchers[str(pid)] = command
                self._say(f"наблюдатель: `{command}` (pid {pid})")

    def _stop_observers(self) -> None:
        for pid in list(self._watchers):
            try:
                self.ports.watch_stop(pid)
            except Exception as e:                     # a leftover process is not a reason to fail the run
                log.debug("stopping watcher %s: %s", pid, e)
        self._watchers.clear()

    def _poll_observers(self) -> None:
        """Between polls of the workers: did a watcher say something new, is a check due?"""
        for pid, command in list(self._watchers.items()):
            try:
                text = (self.ports.watch_tail(pid) if self.ports.watch_tail else "") or ""
            except Exception as e:
                log.debug("watcher %s: %s", pid, e)
                continue
            lines = [line for line in text.strip().splitlines() if line.strip()]
            last = lines[-1][:200] if lines else ""
            if not last or self._watcher_seen.get(pid) == last:
                continue
            self._watcher_seen[pid] = last
            if self._ERROR_LINE.search(last):
                self._say(f"внимание, наблюдатель `{command}`: {last[:160]}")
                self.state.event(f"watcher {command}: {last}")

        now = self._spent() / 60.0
        for item in self._scheduled:
            if item.get("done") or now < float(item.get("after") or 0):
                continue
            item["done"] = True
            command = str(item.get("command") or "")
            try:
                passed, output = self.ports.check(command, self.state.cwd)
            except Exception as e:
                self._say(f"· отложенная проверка `{command}` не выполнилась: {e}")
                continue
            self._say(f"{'✓' if passed else '✗'} отложенная проверка `{command}`: {_tail(output, 200)}")
            self.state.event(f"scheduled {command}: {'passed' if passed else 'FAILED'}")

    def add_step(self, goal: str, role: str = "coder", title: str = "",
                 files: Tuple[str, ...] = (), checks: Tuple[str, ...] = (),
                 verify: Tuple[str, ...] = ()) -> str:
        """A step discovered while the run goes: it joins the plan at once.

        The loop re-reads the ready steps every time round, so a step added here starts as
        soon as a worker slot and its files allow — the plan's parallelism, usable
        mid-flight instead of only at planning time. Returns the new id, or "" when there
        is no run to add it to.
        """
        with self._lock:
            plan = self._live
            if plan is None:
                return ""
            if len(plan.steps) >= self.budgets.max_steps * 3:
                return ""
            used = plan.by_id()
            index = len(plan.steps) + 1
            while f"x{index}" in used:
                index += 1
            step = Step(id=f"x{index}", title=(title or goal).strip()[:120],
                        goal=goal.strip()[:1200],
                        role=role if role in P.ROLES else "coder",
                        files=[str(f) for f in files][:12],
                        checks=[str(c) for c in checks][:3],
                        verify=[str(v) for v in verify][:2])
            step.notes.append("added while the run was going (found by a worker)")
            plan.steps.append(step)
            self._commit(plan)
        self._say(f"+ шаг {step.id} «{step.title}» добавлен на ходу")
        return step.id

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
        self._live = plan
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
                self._watch(plan)                        # what the workers are doing right now
                self._poll_observers()                   # watchers and deferred checks
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
        task = roles.executor_task(step, plan, self.state.level,
                                   self.ports.repo_hint(), roles.history_of(plan, step))
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

    def _after_failure(self, plan: Plan, step: Step, why: str, report: str = "") -> None:
        """A try failed: try again with the lesson, or give the step up."""
        step.notes.append(why)
        step.error = why
        # What the attempt DID, not only what it got wrong: the next worker starts from
        # this instead of from zero. Its changes are not in the tree, so it is told.
        done = (report or "").strip()
        if done:
            step.attempt_reports.append(done)
        step.progress = ""
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

    # -- the checks a step is judged by ------------------------------------------------------

    def _verify(self, step: Step, instruction: str, out: Outcome, check_log: str) -> Tuple[bool, str]:
        """The semantic check: a second agent does the instruction and answers PASS or FAIL.

        A missing or broken verifier is not a failed patch: the step keeps the verdict of
        its mechanical checks, and the reason is written down rather than guessed.
        """
        if self.ports.verify is None:
            step.notes.append("the semantic check was not run: no verifier at this level")
            self._say(f"· шаг {step.id}: проверку глазами некому сделать — она пропущена")
            return True, ""
        task = roles.verifier_task(step, out.diff, check_log, instruction)
        try:
            ok, text = self.ports.verify(task)
        except Exception as e:
            log.warning("verifier failed: %s", e)
            self._say(f"· шаг {step.id}: проверяющий не ответил ({e}) — сужу по механическим проверкам")
            return True, ""
        return bool(ok), text or ""

    def _commands_for(self, step: Step, out: Outcome) -> List[str]:
        """What this step is judged by: the plan's checks, or the tests for its files."""
        if step.checks:
            return list(step.checks)
        if not step.writes:
            return []
        derived = C.derive_checks(step.changed or step.files, self.state.cwd)
        if not derived:
            return []
        step.notes.append("the plan gave no checks; using the tests for the files it touched: "
                          + "; ".join(derived))
        self._say(f"· шаг {step.id}: проверок не задано — беру тесты по своим файлам: "
                  f"{', '.join(derived)}")
        return derived

    def _judge_check(self, step: Step, command: str, after_output: str, out: Outcome) -> C.Verdict:
        """Is this red check about the patch, or about the repository as it already was?"""
        after = C.Result(ok=False, failed=C.failure_signature(after_output), output=after_output)
        if out.in_place:
            return C.Verdict(
                ok=False, measured=False, new=sorted(after.failed)[:6],
                reason="the worker changed the real tree, so there is no unpatched state left "
                       "to compare this check with")
        base = C.Result.from_dict(step.baseline.get(command))
        if base is None:
            ok_before, before_output = self.ports.check(command, self.state.cwd)
            base = C.Result(ok=ok_before, failed=C.failure_signature(before_output), output=before_output)
            step.baseline[command] = base.to_dict()
            self._say(f"· `{command}` без патча: "
                      f"{'зелёная' if ok_before else 'красная — и это не про патч'}")
        return C.judge(base, after)

    def _settle(self, plan: Plan, step: Step, job: Job) -> None:
        out = job.outcome()
        step.progress = ""
        if not out.ok:
            job.discard()
            self._after_failure(plan, step, f"the worker failed: {out.error or 'it ended without a report'}",
                                report=out.report)
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
        commands = self._commands_for(step, out)
        check_log: List[str] = []
        for command in commands:
            passed, output = self.ports.check(command, out.cwd)
            if passed:
                check_log.append(f"$ {command}\n{_tail(output, 400)}\n=> passed")
                step.verified.append(f"✓ {command}")
                continue
            verdict = self._judge_check(step, command, output, out)
            check_log.append(f"$ {command}\n{_tail(output, 600)}\n=> {'passed' if verdict.ok else 'FAILED'}: {verdict.reason}")
            if verdict.ok:
                step.verified.append(f"≈ {command}: {C.describe(verdict)}")
                continue
            job.discard()
            self._after_failure(plan, step, f"the check `{command}` failed:\n{_tail(output)}\n\n"
                                            f"WHAT THE COMPARISON WITH THE UNPATCHED TREE SHOWS: "
                                            f"{C.describe(verdict)}", report=out.report)
            return

        # An exit code proves that nothing broke; it does not prove the work is right.
        # The plan can name instructions a second agent carries out (open the page, read
        # the report) — they run on the patched copy, before anything reaches the tree.
        for instruction in step.verify:
            ok, text = self._verify(step, instruction, out, "\n".join(check_log))
            if not ok:
                job.discard()
                self._after_failure(plan, step, f"the verifier said FAIL on «{instruction}»:\n{_tail(text)}",
                                    report=out.report)
                return
            if text:
                step.verified.append(f"◐ {instruction}: {_tail(text, 120)}")

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
