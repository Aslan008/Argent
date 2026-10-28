"""The run's eyes and ears: watchers that live for the whole run, checks due later, and a
step that joins the plan while it is already going."""

import json

import pytest

from src.agent.workflow import store
from src.agent.workflow.engine import Budgets, Engine, Outcome, Ports

DIFF = "diff --git a/x.py b/x.py\n+print('x')\n"


class Job:
    def __init__(self, out, polls=2, on_poll=None):
        self.out, self.polls, self.on_poll = out, polls, on_poll
        self.applied = self.discarded = False

    def finished(self):
        if self.on_poll:
            self.on_poll()
            self.on_poll = None
        self.polls -= 1
        return self.polls <= 0

    def outcome(self):
        return self.out

    def apply(self):
        self.applied = True
        return ""

    def discard(self):
        self.discarded = True

    def stop(self):
        pass

    def progress(self):
        return ""


def plan_json(*steps, watch=(), scheduled=()):
    return json.dumps({"steps": list(steps), "watch": list(watch), "scheduled": list(scheduled)})


def step(id, role="coder", files=(), checks=(), deps=()):
    return {"id": id, "title": f"title {id}", "goal": f"goal {id}", "role": role,
            "deps": list(deps), "files": list(files), "checks": list(checks)}


class World:
    def __init__(self, plan_text, tmp_path):
        self.plan_text = plan_text
        self.starts = []
        self.said = []
        self.cwd = str(tmp_path)
        self.watch_started = []
        self.watch_stopped = []
        self.watch_output = {}
        self.checks_run = []
        self.engine = None

    def ports(self):
        def plan(messages):
            return self.plan_text

        def start(step_obj, task):
            attempt = sum(1 for s in self.starts if s[0] == step_obj.id) + 1
            self.starts.append((step_obj.id, attempt, task))
            out = Outcome(ok=True, report=f"report {step_obj.id}",
                          diff=DIFF if step_obj.writes else "",
                          files=["x.py"] if step_obj.writes else [], cwd=self.cwd)
            return Job(out, polls=2, on_poll=self.maybe_add_step)

        def check(command, cwd):
            self.checks_run.append(command)
            return True, "ok"

        def watch_start(command):
            self.watch_started.append(command)
            return str(100 + len(self.watch_started))

        def watch_tail(pid):
            return self.watch_output.get(pid, "")

        def watch_stop(pid):
            self.watch_stopped.append(pid)

        return Ports(plan=plan, start=start, check=check, confirm=lambda p, l: True,
                     say=self.said.append, stop_requested=lambda: False, repo_hint=lambda: "",
                     watch_start=watch_start, watch_tail=watch_tail, watch_stop=watch_stop)

    def maybe_add_step(self):
        """Called from inside a worker's poll: the run is going, and a sub-task shows up."""
        if self.engine is not None and not self.starts[1:]:
            self.engine.add_step("check what the first step found", files=("y.py",))

    def run(self, budgets=None):
        state = store.RunState(id="t1", goal="G", level=2, cwd=self.cwd)
        self.engine = Engine(state, self.ports(), budgets or Budgets(), save=lambda s: None, poll=0.001)
        return self.engine.run()


# ── watchers ─────────────────────────────────────────────────────────────────

def test_a_watcher_lives_for_the_whole_run_and_is_stopped_at_its_end(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"]), watch=["npm run dev"]), tmp_path)

    world.run()

    assert world.watch_started == ["npm run dev"]
    assert world.watch_stopped == ["101"], "наблюдатель не остаётся сиротой"


def test_a_watcher_that_turns_red_is_reported_while_the_run_goes(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"]), watch=["npm run dev"]), tmp_path)
    world.watch_output["101"] = "vite ready\nError: port 3000 is already in use\n"

    world.run()

    assert any("наблюдатель" in line and "port 3000" in line for line in world.said), world.said


def test_a_quiet_watcher_is_not_reported_every_poll(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"]), watch=["npm run dev"]), tmp_path)
    world.watch_output["101"] = "vite ready\nwatching\n"

    world.run()

    assert not any("наблюдатель `npm run dev`:" in line for line in world.said)


def test_a_plan_without_watchers_starts_nothing(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"])), tmp_path)

    world.run()

    assert world.watch_started == [] and world.watch_stopped == []


# ── checks due later in the run ──────────────────────────────────────────────

def test_a_due_check_runs_while_the_run_goes(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"]),
                            scheduled=[{"after": 0, "command": "pytest -q tests/test_smoke.py"}]), tmp_path)

    world.run()

    assert "pytest -q tests/test_smoke.py" in world.checks_run
    assert any("отложенная проверка" in line for line in world.said), world.said


def test_a_check_that_is_not_due_yet_does_not_run(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"]),
                            scheduled=[{"after": 600, "command": "pytest -q"}]), tmp_path)

    world.run()

    assert "pytest -q" not in world.checks_run


def test_an_overdue_check_runs_once_not_every_poll(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"]),
                            scheduled=[{"after": 0, "command": "echo tick"}]), tmp_path)

    world.run()

    assert world.checks_run.count("echo tick") == 1


# ── a step discovered mid-flight ─────────────────────────────────────────────

def test_a_step_added_while_the_run_goes_is_carried_out(tmp_path):
    world = World(plan_json(step("s1", files=["x.py"])), tmp_path)

    state = world.run()

    plan = state.get_plan()
    assert [s.id for s in plan.steps] == ["s1", "x2"], "шаг добавлен в живой план"
    added = plan.steps[1]
    assert added.status == "done"
    assert "added while the run was going" in " ".join(added.notes)
    assert any(s[0] == "x2" for s in world.starts), "движок запустил новый шаг"


def test_adding_a_step_outside_a_run_says_so(monkeypatch):
    from src.agent.workflow import runtime

    monkeypatch.setattr(runtime, "_engine", None, raising=False)

    answer = runtime.add_step("something")

    assert "не выполняется" in answer


def test_adding_a_step_without_a_goal_is_refused(monkeypatch):
    from src.agent.workflow import runtime

    class FakeEngine:
        def add_step(self, *a, **k):
            raise AssertionError("не должен вызываться")

    monkeypatch.setattr(runtime, "_engine", FakeEngine(), raising=False)

    assert runtime.add_step("   ").startswith("Error")


def test_the_tool_action_reaches_the_runtime(monkeypatch):
    from argent.tools import workflow_tools
    from src.agent.workflow import runtime

    seen = {}

    def fake_add_step(goal, **kwargs):
        seen["goal"] = goal
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(runtime, "add_step", fake_add_step)

    workflow_tools.ultracode(action="add_step", goal="check the log", role="researcher",
                             files="a.txt", checks="pytest -q; echo done")

    assert seen["goal"] == "check the log"
    assert seen["role"] == "researcher"
    assert seen["files"] == "a.txt"
