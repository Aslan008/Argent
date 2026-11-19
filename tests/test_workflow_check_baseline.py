"""The engine's half of it: a red check is compared with the tree before the patch.

The run this comes from: a step's check was `pytest -k "background or command or
task or process or panel"`, which in this repository can never be green (six
failures belong to the environment). The engine read the red as the worker's
fault and spent three attempts on a check no patch could satisfy.
"""

import json

from src.agent.workflow import store
from src.agent.workflow.engine import Budgets, Engine, Outcome, Ports
from src.agent.workflow.plan import DONE, FAILED

DIFF = "diff --git a/x.py b/x.py\n+print('x')\n"
RED = ("FAILED tests/test_a.py::test_one - AssertionError\n"
       "1 failed, 40 passed\nExit code: 1\n")
RED_PLUS = RED + "FAILED tests/test_a.py::test_new - broke\n"
GREEN = "20 passed in 0.7s\nExit code: 0\n"


class Job:
    def __init__(self, out):
        self.out = out
        self.applied = self.discarded = False

    def finished(self):
        return True

    def outcome(self):
        return self.out

    def apply(self):
        self.applied = True
        return ""

    def discard(self):
        self.discarded = True

    def stop(self):
        pass


class World:
    """One coder step; `script(world)` says how each (command, cwd) answers."""

    def __init__(self, tmp_path, script, files=("x.py",), plan_checks=("check-a",),
                 in_place=False):
        self.cwd = str(tmp_path)
        self.work = str(tmp_path / "work")
        self.script = script(self)
        self.calls = []
        self.said = []
        self.plan_text = json.dumps({"summary": "s", "steps": [{
            "id": "a", "title": "t", "goal": "g", "role": "coder",
            "deps": [], "files": list(files), "checks": list(plan_checks)}]})
        self.out = Outcome(ok=True, report="done", diff=DIFF, files=list(files),
                           cwd=self.work, in_place=in_place)

    def check(self, command, cwd):
        self.calls.append((command, cwd))
        return self.script.get((command, cwd), (True, GREEN))

    def ports(self):
        return Ports(plan=lambda messages: self.plan_text,
                     start=lambda step, task: Job(self.out),
                     check=self.check, say=self.said.append)


def run(world, attempts=3, level=1):
    state = store.RunState(id="b1", goal="g", level=level, cwd=world.cwd)
    engine = Engine(state, world.ports(), Budgets(max_attempts=attempts),
                    save=lambda s: None, poll=0.001)
    engine.run()
    return state


def only_step(state):
    return state.get_plan().steps[0]


def test_a_check_that_was_already_red_does_not_fail_the_step(tmp_path):
    w = World(tmp_path, lambda w: {
        ("check-a", w.work): (False, RED),      # with the patch
        ("check-a", w.cwd): (False, RED),       # without it — the same red
    })

    state = run(w)

    step = only_step(state)
    assert step.status == DONE, step.error
    assert state.status == "done"
    assert any(line.startswith("≈") for line in step.verified), step.verified
    assert "check-a" in step.baseline


def test_a_check_broken_by_the_patch_still_fails_the_step(tmp_path):
    w = World(tmp_path, lambda w: {
        ("check-a", w.work): (False, RED_PLUS),
        ("check-a", w.cwd): (True, GREEN),
    }, )

    state = run(w, attempts=1)

    step = only_step(state)
    assert step.status == FAILED
    assert "broke it" in step.error, step.error
    assert "test_new" in step.error, step.error


def test_a_new_red_on_top_of_an_old_one_is_named(tmp_path):
    w = World(tmp_path, lambda w: {
        ("check-a", w.work): (False, RED_PLUS),
        ("check-a", w.cwd): (False, RED),
    })

    state = run(w, attempts=1)

    step = only_step(state)
    assert step.status == FAILED
    assert "test_new" in step.error
    assert "new failure" in step.error


def test_the_baseline_is_measured_once_even_across_attempts(tmp_path):
    w = World(tmp_path, lambda w: {
        ("check-a", w.work): (False, RED_PLUS),
        ("check-a", w.cwd): (False, RED),
    })

    run(w, attempts=2)

    baseline_runs = [c for c in w.calls if c[1] == w.cwd]
    assert baseline_runs == [("check-a", w.cwd)], "the unpatched run must not be paid for twice"


def test_a_worker_that_changed_the_real_tree_gets_no_benefit_of_the_doubt(tmp_path):
    w = World(tmp_path, lambda w: {
        ("check-a", w.work): (False, RED),
        ("check-a", w.cwd): (False, RED),
    }, in_place=True)

    state = run(w, attempts=1)

    step = only_step(state)
    assert step.status == FAILED
    assert "there is no unpatched state" in step.error
    assert [c for c in w.calls if c[1] == w.cwd] == []


def test_a_step_without_checks_is_judged_by_the_tests_for_its_files(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("", encoding="utf-8")
    w = World(tmp_path, lambda w: {}, plan_checks=())

    state = run(w)

    step = only_step(state)
    assert step.status == DONE, step.error
    ran = [c for c, _ in w.calls if "pytest" in c]
    assert ran and "tests/test_x.py" in ran[0], w.calls
    assert step.verified
