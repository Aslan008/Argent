"""The four things a worker used to lack: a brief, a retry that keeps its predecessor's
work, a monitor, and a check that an exit code cannot make.

Each is tested where it lives: the wording in `roles`, the decision in `engine`.
"""

import json

import pytest

from src.agent.workflow import roles, store
from src.agent.workflow.engine import Budgets, Engine, Outcome, Ports
from src.agent.workflow.plan import DONE, FAILED, Step, parse_plan

DIFF = "diff --git a/x.py b/x.py\n+print('x')\n"


# ── stand-ins ────────────────────────────────────────────────────────────────

class Job:
    def __init__(self, out, polls=2, progress=""):
        self.out, self.polls, self._progress = out, polls, progress
        self.applied = self.discarded = False
        self.apply_result = ""
        self.progress_error = None

    def finished(self):
        self.polls -= 1
        return self.polls <= 0

    def outcome(self):
        return self.out

    def apply(self):
        self.applied = True
        return self.apply_result

    def discard(self):
        self.discarded = True

    def stop(self):
        pass

    def progress(self):
        if self.progress_error:
            raise self.progress_error
        return self._progress


class World:
    def __init__(self, plan_text, tmp_path):
        self.plan_text = plan_text
        self.starts = []                 # (step id, attempt, task)
        self.script = {}                 # (step id, attempt) -> Outcome
        self.checks = {}                 # command -> (ok, output)
        self.verify_results = []         # (ok, text), consumed in order
        self.verify_calls = []           # task texts
        self.said = []
        self.cwd = str(tmp_path)
        self.progress = "run_command: pytest -q"
        self.critic_verdict = "SOLID"

    def ports(self, with_verify=True, with_critic=False):
        def plan(messages):
            return self.plan_text

        def start(step_obj, task):
            attempt = sum(1 for s in self.starts if s[0] == step_obj.id) + 1
            self.starts.append((step_obj.id, attempt, task))
            out = self.script.get((step_obj.id, attempt))
            if out is None:
                out = Outcome(ok=True, report=f"report {step_obj.id}",
                              diff=DIFF if step_obj.writes else "",
                              files=["x.py"] if step_obj.writes else [], cwd=self.cwd)
            return Job(out, progress=self.progress)

        def check(command, cwd):
            return self.checks.get(command, (True, "ok"))

        def critic(step_obj, material):
            return self.critic_verdict, "critic text"

        def verify(task):
            self.verify_calls.append(task)
            if not self.verify_results:
                return True, ""
            return self.verify_results.pop(0)

        return Ports(plan=plan, start=start, check=check,
                     critic=critic if with_critic else None,
                     verify=verify if with_verify else None,
                     confirm=lambda plan_obj, level: True,
                     say=self.said.append, stop_requested=lambda: False,
                     repo_hint=lambda: "Top level: src/, tests/")


def plan_json(*steps, final_checks=()):
    return json.dumps({"summary": "approach", "steps": list(steps), "final_checks": list(final_checks)})


@pytest.fixture
def world(tmp_path):
    """A world with no plan yet: each test writes the plan it needs."""
    return World("", tmp_path)


def step(id, role="coder", deps=(), files=(), checks=(), verify=(), goal=None):
    return {"id": id, "title": f"title {id}", "goal": goal or f"goal {id}", "role": role,
            "deps": list(deps), "files": list(files), "checks": list(checks), "verify": list(verify)}


def run(world, tmp_path, level=2):
    state = store.RunState(id="t1", goal="build the thing", level=level, cwd=world.cwd)
    engine = Engine(state, world.ports(), Budgets(parallelism=1), save=lambda s: None, poll=0.001)
    return engine.run()


# ── 1. the brief ─────────────────────────────────────────────────────────────

def test_the_brief_carries_what_the_run_already_did():
    done = Step(id="s1", title="read the config", goal="read it", status=DONE,
                changed=["config.py"], report="found the port in config.py")
    waiting = Step(id="s2", title="use the port", goal="use it")
    plan = __import__("src.agent.workflow.plan", fromlist=["Plan"]).Plan(goal="G", steps=[done, waiting])

    history = roles.history_of(plan, waiting)
    assert "[s1] read the config" in history
    assert "changed config.py" in history
    assert "found the port" in history

    task = roles.executor_task(waiting, plan, 2, "Top level: src/", history)
    assert "WHAT IS KNOWN ABOUT THE PROJECT" in task and "Top level: src/" in task
    assert "WHAT THIS RUN HAS ALREADY DONE" in task and "found the port" in task
    # A step is not its own history.
    assert "s2" not in roles.history_of(plan, waiting)


def test_the_brief_names_the_checks_a_verifier_will_act_out():
    plan_step = Step(id="s1", title="form", goal="make the form submit", verify=["open / and submit the form"])
    plan = __import__("src.agent.workflow.plan", fromlist=["Plan"]).Plan(goal="G", steps=[plan_step])

    task = roles.executor_task(plan_step, plan, 2)

    assert "A VERIFIER ACTS THESE OUT" in task
    assert "open / and submit the form" in task


def test_a_worker_is_told_what_the_project_is(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"]))

    run(world, tmp_path)

    task = world.starts[0][2]
    assert "WHAT IS KNOWN ABOUT THE PROJECT" in task
    assert "Top level: src/, tests/" in task


# ── 2. a retry keeps what the previous attempt did ───────────────────────────

def test_a_retry_is_handed_what_the_previous_attempt_already_did(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"]))
    world.script[("s1", 1)] = Outcome(ok=False, report="I added the parser to x.py",
                                      error="the parser throws on empty input")

    state = run(world, tmp_path)

    assert len(world.starts) == 2, "шаг повторён"
    second = world.starts[1][2]
    assert "THE PREVIOUS ATTEMPT OF THIS STEP ALREADY TRIED THIS" in second
    assert "I added the parser to x.py" in second
    assert "the parser throws on empty input" in second
    plan = state.get_plan()
    assert plan.steps[0].attempt_reports == ["I added the parser to x.py"]


def test_what_a_failed_attempt_did_is_kept_even_when_a_check_failed_it(world, tmp_path):
    # The unpatched tree is green; the worker's copy is red — so the check really is
    # about the patch, and the step is sent back with what it already did.
    world.plan_text = plan_json(step("s1", files=["x.py"], checks=["pytest -q"]))
    copy = str(tmp_path / "wt")
    world.script[("s1", 1)] = Outcome(ok=True, report="wrote the parser", diff=DIFF,
                                      files=["x.py"], cwd=copy)
    world.script[("s1", 2)] = Outcome(ok=True, report="fixed it", diff=DIFF,
                                      files=["x.py"], cwd=copy)
    ports = world.ports()

    def check(command, cwd):
        return (True, "ok") if cwd == str(tmp_path) else (False, "FAILED tests/test_x.py::test_parse")

    ports.check = check
    state = store.RunState(id="t1", goal="G", level=2, cwd=world.cwd)
    result = Engine(state, ports, Budgets(), save=lambda s: None, poll=0.001).run()

    assert "wrote the parser" in result.get_plan().steps[0].attempt_reports[0]
    assert "wrote the parser" in world.starts[1][2]


# ── 3. the monitor ───────────────────────────────────────────────────────────

def test_the_panel_sees_what_a_running_worker_is_doing(tmp_path):
    from src.agent.workflow.plan import Plan

    state = store.RunState(id="t1", goal="G", level=2, cwd=str(tmp_path))
    plan = Plan(goal="G", steps=[Step(id="s1", title="t", goal="g")])
    engine = Engine(state, Ports(plan=lambda m: "{}", start=lambda s, t: None,
                                 check=lambda c, w: (True, "")), Budgets(), save=lambda s: None)
    engine._jobs = {"s1": Job(Outcome(ok=False), progress="run_command: python -m pytest -q tests")}

    engine._watch(plan)

    assert plan.steps[0].progress == "run_command: python -m pytest -q tests"


def test_a_broken_monitor_never_stops_the_run(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"]))
    real_start = world.ports().start

    def start(step_obj, task):
        job = real_start(step_obj, task)
        job.progress_error = RuntimeError("the log is gone")
        return job

    state = store.RunState(id="t1", goal="G", level=2, cwd=world.cwd)
    ports = world.ports()
    ports.start = start
    result = Engine(state, ports, Budgets(), save=lambda s: None, poll=0.001).run()

    assert result.get_plan().steps[0].status == DONE


# ── 4. the semantic check ────────────────────────────────────────────────────

def test_the_verifier_can_fail_a_step_its_checks_accepted(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"], checks=["pytest -q"],
                                     verify=["open / and submit the form"]))
    world.verify_results = [(False, "FAILED: the submit button does nothing\nVERDICT: FAIL")]

    state = run(world, tmp_path)

    plan = state.get_plan()
    assert plan.steps[0].status == DONE, "после повтора шаг прошёл"
    assert any("the verifier said FAIL" in n for n in plan.steps[0].notes)
    assert world.verify_calls, "проверяющего позвали"
    assert "PATCH" in world.verify_calls[0] and "MECHANICAL CHECKS" in world.verify_calls[0]


def test_a_passing_verifier_is_written_into_what_the_step_verified(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"], checks=["pytest -q"],
                                     verify=["open / and submit the form"]))
    world.verify_results = [(True, "clicked submit, got 200")]

    state = run(world, tmp_path)

    step_obj = state.get_plan().steps[0]
    assert step_obj.status == DONE
    assert any(line.startswith("◐ open / and submit the form") for line in step_obj.verified)
    assert any("clicked submit, got 200" in line for line in step_obj.verified)


def test_without_a_verifier_the_step_passes_and_the_reason_is_written_down(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"], verify=["look at it"]))
    state = store.RunState(id="t1", goal="G", level=2, cwd=world.cwd)
    engine = Engine(state, world.ports(with_verify=False), Budgets(), save=lambda s: None, poll=0.001)

    result = engine.run()

    step_obj = result.get_plan().steps[0]
    assert step_obj.status == DONE
    assert any("no verifier" in n for n in step_obj.notes)


def test_a_verifier_that_crashes_does_not_condemn_the_patch(world, tmp_path):
    world.plan_text = plan_json(step("s1", files=["x.py"], verify=["look at it"]))

    def boom(task):
        raise RuntimeError("no browser today")

    ports = world.ports()
    ports.verify = boom
    state = store.RunState(id="t1", goal="G", level=2, cwd=world.cwd)
    result = Engine(state, ports, Budgets(), save=lambda s: None, poll=0.001).run()

    assert result.get_plan().steps[0].status == DONE


# ── the plan reads what it needs ─────────────────────────────────────────────

def test_the_plan_reads_semantic_checks_and_caps_them():
    text = plan_json(step("s1", verify=["one", "two", "three"]))
    plan = parse_plan(text, "G", 8)
    assert plan.steps[0].verify == ["one", "two"]


def test_a_plan_without_semantic_checks_still_parses():
    plan = parse_plan(plan_json(step("s1")), "G", 8)
    assert plan.steps[0].verify == []


def test_the_verifier_verdict_is_read_from_the_answer():
    assert roles.parse_verifier_verdict("all good\nVERDICT: PASS") == "PASS"
    assert roles.parse_verifier_verdict("VERDICT: **FAIL**") == "FAIL"
    assert roles.parse_verifier_verdict("I could not tell") == ""
