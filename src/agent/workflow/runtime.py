"""Plugs the engine into the real world: a model, sub-agent processes, the approval prompt, the screen.

The session switch lives here too. Ultracode is a mode of THIS session, like plan mode: it is not saved, and a
new window starts without it. The model may start a workflow only while the person has it on.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from logger import get_logger
from src.agent.workflow import advisor, calibration, control, report, roles, store
from src.agent.workflow.engine import Budgets, Engine, Job, Outcome, Ports
from src.agent.workflow.plan import Plan, Step

log = get_logger("ultracode")

_state_lock = threading.Lock()
_enabled = False
_active: Optional[store.RunState] = None


# ── the session switch ──────────────────────────────────────────────────────────

def enabled() -> bool:
    return _enabled


def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


def active_run() -> Optional[store.RunState]:
    return _active


INSTRUCTIONS = (
    "## ULTRACODE IS ON\n"
    "The person switched Ultracode on for this session. For a BIG job — several files or modules, work that needs "
    "its own checks — do not improvise: call ultracode(action=\"start\", goal=\"<the whole job, with every "
    "requirement and the checks the person named>\"). It plans, runs workers in isolated copies of the tree, "
    "checks every step, has an independent critic read the patches and an advisor review the plan and the result, "
    "then returns a report. Questions, explanations and small edits you do yourself, as usual. If a workflow was "
    "stopped, ultracode(action=\"resume\") continues it."
)


_keyword = False


def keyword_turn(named: Optional[bool] = None) -> bool:
    """Whether the message of this turn named Ultracode (see trigger.py). With an argument: set it."""
    global _keyword
    if named is not None:
        _keyword = bool(named)
    return _keyword


def instructions() -> Optional[str]:
    if not _enabled:
        return None
    if _keyword:
        from src.agent.workflow import trigger
        return INSTRUCTIONS + "\n" + trigger.NOTE
    return INSTRUCTIONS


# ── the steps on the person's screen ────────────────────────────────────────────

class StepBoard:
    """The run's steps as the checklist the frontends already draw: the TUI's panel above the prompt (it reads
    the list every couple of seconds) and the Telegram status message (it listens to the list).

    The checklist the model had before is put back when the run is done; a run that stopped leaves its steps
    on the screen, because they are what is left to do."""

    #: the queue / PC line is looked up at most this often (it reads lane files and the process table)
    NOTE_EVERY = 5.0

    def __init__(self, clock=None):
        import time
        from src.agent.todos import TODOS
        self._todos = TODOS
        self._clock = clock or time.monotonic
        self._before = TODOS.get("main")
        self._note = ""
        self._note_at = -1e9
        self._last = None

    def _brief(self) -> str:
        now = self._clock()
        if now - self._note_at >= self.NOTE_EVERY:
            self._note_at = now
            try:
                from src.agent import governor
                self._note = governor.brief()
            except Exception:
                self._note = ""
        return self._note

    def update(self, plan: Plan) -> None:
        from src.agent.workflow.plan import DONE, FAILED, RUNNING, SKIPPED
        from src.agent.todos import COMPLETED, IN_PROGRESS, PENDING, Todo
        items = []
        for step in plan.steps:
            text = f"{step.id}. {step.title}"
            if step.status == DONE:
                status = COMPLETED
            elif step.status == RUNNING:
                status = IN_PROGRESS
                text += f" — попытка {step.attempts}" if step.attempts > 1 else " — идёт"
                note = self._brief()
                if note:
                    text += f" ({note})"
            elif step.status == FAILED:
                status, text = COMPLETED, f"✗ {text} — не получился"
            elif step.status == SKIPPED:
                status, text = COMPLETED, f"↷ {text} — пропущен"
            else:
                status = PENDING
            items.append(Todo(text[:200], status))
        if items != self._last:
            self._last = items
            self._todos.set(items, "main")

    def close(self, done: bool) -> None:
        if done and self._before:
            self._todos.set(self._before, "main")
        elif done:
            self._todos.clear("main")


# ── workers ─────────────────────────────────────────────────────────────────────

class SubagentJob(Job):
    """A step's worker: a sub-agent process (src/agent/subagents.py) seen as the engine's `Job`."""

    def __init__(self, run):
        from src.agent.subagents import MANAGER
        self.run = run
        self._manager = MANAGER

    def finished(self) -> bool:
        return not self.run.running

    def outcome(self) -> Outcome:
        run = self.run
        diff = ""
        if run.patch is not None:
            try:
                diff = Path(run.patch).read_text(encoding="utf-8", errors="replace")
            except OSError:
                diff = ""
        isolated = run.isolation == "worktree"
        cwd = str(run.worktree.path) if (isolated and run.worktree is not None) else run.cwd
        if isolated and run.worktree is None and not diff:
            cwd = str(run.repo or run.cwd)              # it changed nothing; the copy was already removed
        error = run.error if run.status != "done" else ""
        return Outcome(ok=run.status == "done", report=run.report, error=error,
                       denied=[d.get("action", "?") for d in (run.denied or [])],
                       diff=diff, files=list(run.files), cwd=cwd, in_place=not isolated)

    def apply(self) -> str:
        return self._manager.apply(self.run)

    def discard(self) -> None:
        if self.run.disposition != "applied":
            self._manager.discard(self.run)

    def stop(self) -> None:
        self._manager.stop(self.run)


def _in_git_repo(cwd: str) -> bool:
    try:
        from src.agent import worktrees
        worktrees.repo_root(Path(cwd))
        return True
    except Exception:
        return False


# ── the real ports ──────────────────────────────────────────────────────────────

_EXIT = re.compile(r"^Exit code:\s*(-?\d+)", re.M)


def run_check(command: str, cwd: str) -> Tuple[bool, str]:
    """A step's check through the same gate as every command the model runs (risk-gate, approval, redaction)."""
    from tools.command_ops import run_command
    output = run_command(command, cwd=cwd or None, timeout=240)
    if output.startswith("Execution aborted") or output.startswith("Command BLOCKED"):
        return False, output
    m = _EXIT.search(output)
    return (m is not None and int(m.group(1)) == 0), output


def _repo_hint(cwd: str) -> str:
    root = Path(cwd)
    lines: List[str] = []
    try:
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in root.iterdir()
                       if not p.name.startswith(".") and p.name not in ("node_modules", "__pycache__", ".venv"))
        lines.append("Top level: " + ", ".join(names[:60]))
    except OSError:
        pass
    probes = (("pytest", ("pytest.ini", "tox.ini", "conftest.py")), ("python -m pytest", ("pyproject.toml",)),
              ("npm test", ("package.json",)), ("cargo test", ("Cargo.toml",)), ("go test ./...", ("go.mod",)))
    found = [cmd for cmd, files in probes if any((root / f).exists() for f in files)]
    if found:
        lines.append("Likely test commands: " + "; ".join(dict.fromkeys(found)))
    for name in ("AGENTS.md", "CLAUDE.md"):
        try:
            text = (root / name).read_text(encoding="utf-8", errors="replace").strip()
            if text:
                lines.append(f"{name} (start):\n{text[:900]}")
                break
        except OSError:
            continue
    return "\n".join(lines)


def _say(text: str) -> None:
    try:
        from ui import print_system
        print_system(text)
    except Exception:
        log.info("%s", text)


say = _say


def build_ports(state: store.RunState, board: Optional[StepBoard] = None) -> Ports:
    import config
    from providers import create_provider

    main_provider, main_model = config.get_provider(), config.get_current_model()
    target = advisor.resolve(main_provider, main_model)
    main_target = advisor.Target(main_provider, main_model, configured=False,
                                 cloud=target.cloud if not target.configured else False, leaves_pc=False)
    note = ""
    if target.leaves_pc and not config.is_advisor_cloud_consented():
        import approval
        asked = approval.request_approval(
            f"Ultracode: советник {target.label} — облачная модель. План и патчи проекта (без секретов) уйдут "
            f"в облако. Разрешить на этот запуск? (навсегда: /advisor consent on)",
            destructive=False, grant_key=None)
        if not asked:
            note = f"облачный советник {target.label} не разрешён — советует основная модель"
            target = main_target
    if not target.configured and not note:
        note = "советник не настроен — советует основная модель (/advisor model <провайдер>:<модель>)"
    state.advisor = {"provider": target.provider, "model": target.model, "cloud": target.cloud,
                     "configured": target.configured, "note": note}
    if note:
        _say(f"[yellow]{note}[/yellow]")

    def plan(messages: list) -> str:
        provider = create_provider(main_provider)
        return provider.sync_chat(main_model, messages, temperature=0.2, json_format=True, timeout=600) or ""

    def start(step: Step, task: str) -> Job:
        from src.agent.subagents import MANAGER
        isolation = "worktree" if (step.writes and _in_git_repo(state.cwd)) else "none"
        run = MANAGER.spawn(step.role, task, isolation=isolation, cwd=Path(state.cwd))
        return SubagentJob(run)

    def critic(step: Step, material: str) -> Tuple[str, str]:
        from agent import run_plan_critique
        from src.agent.critic import parse_critic_verdict, sanitize_and_truncate_diff
        review = run_plan_critique(sanitize_and_truncate_diff(material, 30000), goal=state.goal,
                                   what=f"patch for step {step.id} of a larger job")
        return parse_critic_verdict(review).value, review

    def advise(kind: str, goal: str, body: str):
        return advisor.consult(kind, goal, body, target, fallback=main_target if target.label != main_target.label else None)

    def confirm(plan_obj: Plan, level: int) -> bool:
        import approval
        _say(f"[bold]План Ultracode[/bold] (уровень {level} — {calibration.LEVEL_NAMES.get(level, '?')}):\n"
             f"{plan_obj.outline()}")
        return approval.request_approval(
            f"Ultracode: выполнить этот план ({len(plan_obj.steps)} шаг., уровень {level})?",
            destructive=False, grant_key=None)

    return Ports(plan=plan, start=start, check=run_check, critic=critic, advise=advise, confirm=confirm,
                 say=lambda t: _say(f"[dim]{t}[/dim]"), stop_requested=control.stop_requested,
                 repo_hint=lambda: _repo_hint(state.cwd),
                 progress=board.update if board is not None else (lambda plan: None))


# ── start / resume ──────────────────────────────────────────────────────────────

_BUDGET_STEPS = {3: 12, 2: 8, 1: 6}


def _budgets(level: int, profile: calibration.Profile) -> Budgets:
    import config
    return Budgets(max_steps=_BUDGET_STEPS.get(level, 6), max_attempts=3,
                   max_minutes=float(config.get_ultracode_max_minutes()),
                   max_advisor_calls=6, parallelism=calibration.parallelism_for(level, profile))


def _decide_level() -> Tuple[int, str, calibration.Profile]:
    import config
    provider, model = config.get_provider(), config.get_current_model()
    profile = calibration.calibrate(provider, model)
    level, why = calibration.choose_level(profile, config.get_ultracode_level())
    return level, why, profile


class _ServerPriority:
    """Lowers the model server's priority for the length of a workflow (and puts it back) — unless the person's
    own setting already keeps it lowered. Says so: another program's priority is not changed silently."""

    def __enter__(self):
        self.lowered: List[int] = []
        try:
            import config
            from src.agent.local_backend import TUNER, classify
            backend = classify(config.get_provider(), config.get_current_model())
            if backend.local and not config.is_lower_server_priority_enabled():
                self.lowered = TUNER.lower(backend)
                if self.lowered:
                    _say("[dim]Понижаю приоритет процесса модели на время Ultracode — ПК останется отзывчивым. "
                         "Верну обратно по окончании.[/dim]")
        except Exception as e:
            log.debug("server priority not lowered: %s", e)
        return self

    def __exit__(self, *exc):
        if self.lowered:
            from src.agent.local_backend import TUNER
            TUNER.restore()


def _plan_only(state: store.RunState, ports: Ports, budgets: Budgets) -> str:
    """Level 1: the plan and the advisor's review, then the model carries it out itself."""
    from src.agent import todos
    engine = Engine(state, ports, budgets)
    plan, problem = engine._make_plan()
    if plan is None:
        state.status, state.result = "failed", f"План составить не удалось: {problem}"
        store.save(state)
        return state.result + "\nДелайте задачу обычным способом."
    plan = engine._review_plan(plan)
    state.set_plan(plan)
    state.status = "done"
    items = [todos.Todo(f"{s.title}" + (f" (проверка: {s.checks[0]})" if s.checks else ""))
             for s in plan.steps]
    todos.TODOS.set(items, "main")
    state.result = ("Уровень 1: план составлен и проверен, но исполнять его будете вы сами, шаг за шагом.\n"
                    + plan.outline() +
                    "\n\nПлан занесён в список задач (todo_write). Выполняйте шаги по порядку, запускайте проверки "
                    "каждого шага, в конце запустите: " + (", ".join(plan.final_checks) or "тесты проекта") + ".")
    store.save(state)
    return state.result


def _run(state: store.RunState, level: int, profile: calibration.Profile) -> str:
    board = StepBoard()
    ports = build_ports(state, board)
    budgets = _budgets(level, profile)
    global _active
    with _state_lock:
        _active = state
    final = None
    try:
        with _ServerPriority():
            if level == 1:
                return _plan_only(state, ports, budgets)
            final = Engine(state, ports, budgets).run()
            text = report.render(final)
            return text if (not final.result or final.result in text) else final.result + "\n\n" + text
    finally:
        with _state_lock:
            _active = None
        board.close(done=final is not None and final.status == "done")


def start(goal: str) -> str:
    """Run a workflow for `goal` and return its report. Blocks until the workflow ends."""
    from src.agent import plan_mode
    if plan_mode.PLAN.active:
        return "Ultracode недоступен в режиме планирования: сначала утвердите план."
    if not enabled():
        return ("Ultracode выключен. Он запускается только по решению человека: попросите включить режим "
                "командой /ultracode on. Пока выполняйте задачу обычным способом.")
    goal = (goal or "").strip()
    if len(goal) < 8:
        return "Нужна полная формулировка цели (goal): что именно сделать и как проверить."
    level, why, profile = _decide_level()
    if level == 0:
        return (f"Ultracode не включён для этой задачи: {why}. Выполняйте её обычным способом. "
                "(Принудительно: /ultracode level 1|2|3.)")
    control.reset()
    state = store.RunState(id=store.new_id(), goal=goal, level=level, cwd=os.getcwd())
    state.event(f"уровень {level} ({calibration.LEVEL_NAMES[level]}): {why}")
    _say(f"[bold]Ultracode[/bold]: уровень {level} — {calibration.LEVEL_NAMES[level]} ({why})")
    store.save(state)
    return _run(state, level, profile)


def resume(run_id: str = "") -> str:
    if not enabled():
        return "Ultracode выключен: /ultracode on."
    state = store.load(run_id) if run_id else store.latest(resumable_only=True)
    if state is None or not state.resumable:
        return "Нет незавершённого воркфлоу в этом проекте."
    store.revive(state)
    level, why, profile = _decide_level()
    state.level = level if level > 0 else state.level
    control.reset()
    _say(f"[bold]Ultracode[/bold]: продолжаю {state.id} с последнего завершённого шага")
    return _run(state, state.level, profile)


def stop() -> str:
    control.request_stop()
    return "Остановка запрошена: исполнители будут остановлены, прогресс сохранён."


def status() -> str:
    state = _active or store.latest()
    if state is None:
        return "Воркфлоу в этом проекте ещё не запускались."
    text = report.render(state, brief=True)
    return text if _active is None else "▶ идёт сейчас\n" + text
