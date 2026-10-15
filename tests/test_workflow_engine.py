"""The engine: ordering, isolation of work, retries, limits, stops — with stand-ins for every outside part."""

import json
import threading
import time

import pytest

from src.agent.workflow import store
from src.agent.workflow.advisor import Advice
from src.agent.workflow.engine import Budgets, Engine, Outcome, Ports
from src.agent.workflow.plan import DONE, FAILED, PENDING, SKIPPED, Plan

DIFF = "diff --git a/x.py b/x.py\n+print('x')\n"


def plan_json(*steps, final_checks=()):
    return json.dumps({"summary": "approach", "steps": list(steps), "final_checks": list(final_checks)})


def step(id, role="coder", deps=(), files=(), checks=(), goal=None):
    return {"id": id, "title": f"title {id}", "goal": goal or f"goal {id}", "role": role,
            "deps": list(deps), "files": list(files), "checks": list(checks)}


class Job:
    def __init__(self, outcome, polls=1, on_poll=None):
        self.out, self.polls, self.on_poll = outcome, polls, on_poll
        self.applied = self.discarded = self.stopped = False
        self.apply_result = ""

    def finished(self):
        if self.on_poll:
            self.on_poll()
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
        self.stopped = True


class World:
    """Scripts the outside: the planner's answer, each worker's outcome by (step, attempt), checks, critic, advisor."""

    def __init__(self, plan_text, tmp_path):
        self.plan_texts = [plan_text] if isinstance(plan_text, str) else list(plan_text)
        self.planner_calls = []
        self.starts = []                 # (step id, attempt, task)
        self.jobs = []
        self.script = {}                 # (step id, attempt) -> Outcome | callable(task) -> Outcome
        self.check_results = {}          # command -> list of (ok, output), consumed in order
        self.checks_run = []
        self.critic_results = {}         # step id -> list of verdicts
        self.critic_calls = []
        self.advice = []                 # list of Advice | Exception, consumed in order
        self.advice_calls = []
        self.said = []
        self.stop = False
        self.confirm_answer = True
        self.cwd = str(tmp_path)
        self.in_flight = 0
        self.max_in_flight = 0
        self.apply_log = []

    def ports(self, with_critic=True, with_advisor=True):
        def plan(messages):
            self.planner_calls.append(messages)
            index = min(len(self.planner_calls) - 1, len(self.plan_texts) - 1)
            return self.plan_texts[index]

        def start(step_obj, task):
            attempt = sum(1 for s in self.starts if s[0] == step_obj.id) + 1
            self.starts.append((step_obj.id, attempt, task))
            out = self.script.get((step_obj.id, attempt))
            if callable(out):
                out = out(task)
            if out is None:
                out = Outcome(ok=True, report=f"report {step_obj.id}", diff=DIFF if step_obj.writes else "",
                              files=["x.py"] if step_obj.writes else [], cwd=self.cwd)
            job = Job(out, polls=2)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            original = job.outcome

            def outcome():
                return original()
            job.finished_before = False
            real_finished = job.finished

            def finished():
                done = real_finished()
                if done and not job.finished_before:
                    job.finished_before = True
                    self.in_flight -= 1
                return done
            job.finished = finished
            real_apply = job.apply

            def apply():
                self.apply_log.append(step_obj.id)
                return real_apply()
            job.apply = apply
            self.jobs.append(job)
            return job

        def check(command, cwd):
            self.checks_run.append((command, cwd))
            queue = self.check_results.get(command)
            if queue:
                return queue.pop(0) if len(queue) > 1 else queue[0]
            return True, "ok"

        def critic(step_obj, material):
            self.critic_calls.append((step_obj.id, material))
            queue = self.critic_results.get(step_obj.id)
            verdict = (queue.pop(0) if len(queue) > 1 else queue[0]) if queue else "SOLID"
            return verdict, f"critic says {verdict}"

        def advise(kind, goal, body):
            self.advice_calls.append((kind, body))
            if not self.advice:
                return None
            item = self.advice.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return Ports(plan=plan, start=start, check=check,
                     critic=critic if with_critic else None,
                     advise=advise if with_advisor else None,
                     confirm=lambda plan_obj, level: self.confirm_answer,
                     say=self.said.append, stop_requested=lambda: self.stop,
                     repo_hint=lambda: "hint")


def make(world, level=2, parallelism=1, **budget):
    state = store.RunState(id="t1", goal="build the thing", level=level, cwd=world.cwd)
    budgets = Budgets(parallelism=parallelism, **budget)
    saved = []
    engine = Engine(state, world.ports(), budgets, save=lambda s: saved.append(s.status), poll=0.001)
    return engine, state, saved


def statuses(state):
    return {s.id: s.status for s in state.get_plan().steps}


# ── the straight path ────────────────────────────────────────────────────────

def test_steps_run_in_dependency_order_and_everything_is_applied(tmp_path):
    w = World(plan_json(step("a", files=["a.py"]), step("b", deps=["a"], files=["b.py"]),
                        step("c", role="reviewer", deps=["b"]), final_checks=["pytest"]), tmp_path)
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert [s[0] for s in w.starts] == ["a", "b", "c"]
    assert w.apply_log == ["a", "b"]                       # the reviewer changes nothing
    assert ("pytest", w.cwd) in w.checks_run, "the final check runs in the real tree"
    assert "title a" in state.result


def test_a_worker_is_told_what_the_steps_before_it_found(tmp_path):
    w = World(plan_json(step("a", role="researcher"), step("b", deps=["a"])), tmp_path)
    w.script[("a", 1)] = Outcome(ok=True, report="the config lives in settings/base.toml", cwd=w.cwd)
    engine, state, _ = make(w)
    engine.run()
    task_b = [t for sid, _, t in w.starts if sid == "b"][0]
    assert "settings/base.toml" in task_b
    assert "build the thing" in task_b


def test_independent_steps_run_side_by_side_only_when_allowed(tmp_path):
    text = plan_json(step("a", files=["a.py"]), step("b", files=["b.py"]))
    w = World(text, tmp_path)
    engine, state, _ = make(w, parallelism=2)
    engine.run()
    assert w.max_in_flight == 2

    w2 = World(text, tmp_path)
    engine2, state2, _ = make(w2, parallelism=1)
    engine2.run()
    assert w2.max_in_flight == 1


def test_steps_that_may_touch_the_same_file_never_overlap(tmp_path):
    w = World(plan_json(step("a", files=["shared.py"]), step("b", files=["shared.py"]),
                        step("c", files=[])), tmp_path)
    engine, state, _ = make(w, parallelism=3)
    engine.run()
    assert w.max_in_flight == 1, "a, b share a file and c has an unknown scope: none of them may overlap"
    assert state.status == "done"


# ── the work is not trusted ──────────────────────────────────────────────────

def test_a_failing_check_sends_the_step_back_with_the_output(tmp_path):
    w = World(plan_json(step("a", checks=["pytest -q"])), tmp_path)
    w.check_results["pytest -q"] = [(False, "FAILED test_x - assert 1 == 2"), (True, "1 passed")]
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert [s[1] for s in w.starts] == [1, 2]
    assert "assert 1 == 2" in w.starts[1][2], "the second attempt is told what the check said"
    assert w.apply_log == ["a"], "the broken first attempt was never applied"
    assert w.jobs[0].discarded


def test_checks_run_in_the_workers_copy_not_the_real_tree(tmp_path):
    w = World(plan_json(step("a", checks=["pytest"])), tmp_path)
    w.script[("a", 1)] = Outcome(ok=True, report="r", diff=DIFF, files=["x.py"], cwd="/tmp/copy-of-a")
    engine, _, _ = make(w)
    engine.run()
    assert w.checks_run[0] == ("pytest", "/tmp/copy-of-a")


def test_critic_fixable_means_another_try_and_rethink_too(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    w.critic_results["a"] = ["FIXABLE", "RETHINK", "SOLID"]
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert len(w.starts) == 3
    assert "FIXABLE" in w.starts[1][2] and "RETHINK" in w.starts[2][2]
    assert w.apply_log == ["a"]


def test_the_critic_sees_the_patch_and_the_check_log(tmp_path):
    w = World(plan_json(step("a", checks=["pytest"])), tmp_path)
    w.check_results["pytest"] = [(True, "3 passed")]
    engine, _, _ = make(w)
    engine.run()
    material = w.critic_calls[0][1]
    assert "diff --git" in material and "3 passed" in material and "goal a" in material


def test_on_the_last_try_fixable_is_accepted_with_a_reservation(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    w.critic_results["a"] = ["FIXABLE"]
    engine, state, _ = make(w, max_attempts=2)
    engine.run()
    assert state.status == "done"
    assert len(w.starts) == 2
    assert state.get_plan().by_id()["a"].verdict == "FIXABLE"


def test_a_patch_that_does_not_apply_is_retried_on_the_new_tree(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    holder = {}
    original_start = None

    def first_outcome(task):
        return Outcome(ok=True, report="r", diff=DIFF, files=["x.py"], cwd=w.cwd)

    w.script[("a", 1)] = first_outcome
    engine, state, _ = make(w)
    ports = engine.ports
    real_start = ports.start

    def start(step_obj, task):
        job = real_start(step_obj, task)
        if len(w.starts) == 1:
            job.apply_result = "patch failed: x.py does not match"
        return job
    ports.start = start
    engine.run()
    assert len(w.starts) == 2
    assert "does not match" in w.starts[1][2]
    assert state.status == "done"


def test_a_step_that_keeps_failing_is_given_up_and_its_dependents_are_skipped(tmp_path):
    w = World(plan_json(step("a"), step("b", deps=["a"]), step("c")), tmp_path)
    for n in (1, 2, 3):
        w.script[("a", n)] = Outcome(ok=False, error="crashed", cwd=w.cwd)
    engine, state, _ = make(w, max_attempts=3)
    engine.run()
    assert state.status == "failed"
    assert statuses(state) == {"a": FAILED, "b": SKIPPED, "c": DONE}
    assert len([s for s in w.starts if s[0] == "a"]) == 3
    assert "b" not in [s[0] for s in w.starts]


def test_a_worker_that_changed_nothing_is_done_without_a_critic(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    w.script[("a", 1)] = Outcome(ok=True, report="already fine", diff="", cwd=w.cwd)
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert w.critic_calls == [] and w.apply_log == []


def test_in_place_work_is_checked_where_it_happened_and_never_applied(tmp_path):
    w = World(plan_json(step("a", checks=["pytest"])), tmp_path)
    w.script[("a", 1)] = Outcome(ok=True, report="r", in_place=True, cwd=w.cwd, files=["x.py"])
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert w.apply_log == [] and w.critic_calls == []
    assert w.checks_run == [("pytest", w.cwd)]


def test_refused_actions_are_kept_in_the_report(tmp_path):
    w = World(plan_json(step("a", role="researcher")), tmp_path)
    w.script[("a", 1)] = Outcome(ok=True, report="done", denied=["run: rm -rf build"], cwd=w.cwd)
    engine, state, _ = make(w)
    engine.run()
    assert "rm -rf build" in state.get_plan().by_id()["a"].report


# ── planning ────────────────────────────────────────────────────────────────

def test_a_bad_plan_is_sent_back_once_with_the_reason(tmp_path):
    w = World(["not json at all", plan_json(step("a"))], tmp_path)
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert len(w.planner_calls) == 2
    assert "no JSON" in w.planner_calls[1][-1]["content"]


def test_two_bad_plans_end_the_run_honestly(tmp_path):
    w = World(["nope", "still nope"], tmp_path)
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "failed"
    assert w.starts == []
    assert "plan" in state.result.lower() or "план" in state.result.lower()


def test_a_declined_plan_starts_nothing(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    w.confirm_answer = False
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "cancelled"
    assert w.starts == []


def test_the_planner_is_told_the_step_limit(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    engine, _, _ = make(w, max_steps=5)
    engine.run()
    assert "At most 5 steps" in w.planner_calls[0][0]["content"]


# ── the advisor ──────────────────────────────────────────────────────────────

def advice(kind, verdict, text="", target="cloud:big"):
    return Advice(kind=kind, verdict=verdict, text=text, target=target)


def test_the_advisor_reviews_the_plan_and_may_have_it_rewritten(tmp_path):
    w = World([plan_json(step("a")), plan_json(step("a"), step("b", deps=["a"]))], tmp_path)
    w.advice = [advice("plan", "REVISE", "add a verification step"), advice("final", "ACCEPT", "good")]
    engine, state, _ = make(w)
    engine.run()
    assert [s.id for s in state.get_plan().steps] == ["a", "b"]
    assert "add a verification step" in w.planner_calls[1][-1]["content"]
    assert [k for k, _ in w.advice_calls] == ["plan", "final"]
    assert state.advisor_calls == 2
    assert "good" in "\n".join(state.advisor["notes"])


def test_an_unreachable_advisor_does_not_stop_the_work(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    w.advice = [RuntimeError("no network"), RuntimeError("no network")]
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert any("недоступен" in line for line in w.said)


def test_the_advisor_is_asked_when_a_step_fails_twice_and_its_guidance_reaches_the_worker(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    w.script[("a", 1)] = Outcome(ok=False, error="boom one", cwd=w.cwd)
    w.script[("a", 2)] = Outcome(ok=False, error="boom two", cwd=w.cwd)
    w.advice = [advice("plan", "APPROVE"), advice("stuck", "RETRY", "use the other API"),
                advice("final", "ACCEPT")]
    engine, state, _ = make(w)
    engine.run()
    assert state.status == "done"
    assert "use the other API" in w.starts[2][2]


def test_an_advisor_who_says_give_up_ends_the_step(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    for n in (1, 2, 3):
        w.script[("a", n)] = Outcome(ok=False, error="boom", cwd=w.cwd)
    w.advice = [advice("plan", "APPROVE"), advice("stuck", "GIVE_UP", "the premise is wrong")]
    engine, state, _ = make(w, max_attempts=5)
    engine.run()
    assert state.status == "failed"
    assert len(w.starts) == 2, "no third try after GIVE_UP"


def test_the_advisor_budget_is_respected(tmp_path):
    w = World([plan_json(step("a"))], tmp_path)
    w.advice = [advice("plan", "APPROVE"), advice("final", "ACCEPT")]
    engine, state, _ = make(w, max_advisor_calls=1)
    engine.run()
    assert [k for k, _ in w.advice_calls] == ["plan"]
    assert state.status == "done"


# ── limits, stopping, resuming ───────────────────────────────────────────────

def test_stop_ends_cleanly_and_the_run_can_be_resumed_from_the_last_finished_step(tmp_path):
    w = World(plan_json(step("a"), step("b", deps=["a"]), step("c", deps=["b"])), tmp_path)
    engine, state, _ = make(w)
    real_start = engine.ports.start

    def start(step_obj, task):
        job = real_start(step_obj, task)
        if step_obj.id == "b":
            w.stop = True                       # the person pressed Esc while b was working
        return job
    engine.ports.start = start
    engine.run()
    assert state.status == "stopped"
    assert statuses(state) == {"a": DONE, "b": PENDING, "c": PENDING}
    assert w.jobs[1].stopped, "the running worker is stopped"
    assert state.resumable

    # resume: a is not run again
    w.stop = False
    w.starts.clear()
    store.revive(state)
    engine2 = Engine(state, w.ports(), Budgets(), save=lambda s: None, poll=0.001)
    engine2.run()
    assert state.status == "done"
    assert [s[0] for s in w.starts] == ["b", "c"]


def test_the_time_budget_stops_the_run_with_a_reason(tmp_path):
    w = World(plan_json(step("a"), step("b", deps=["a"])), tmp_path)
    now = [0.0]

    def clock():
        now[0] += 400.0                          # every look at the clock is 400 s later
        return now[0]
    state = store.RunState(id="t2", goal="g", level=2, cwd=w.cwd)
    engine = Engine(state, w.ports(), Budgets(max_minutes=10), save=lambda s: None, clock=clock, poll=0.001)
    engine.run()
    assert state.status == "stopped"
    assert "лимит времени" in state.result


def test_state_is_written_after_changes_and_can_be_read_back(tmp_path):
    w = World(plan_json(step("a"), step("b", deps=["a"])), tmp_path)
    state = store.RunState(id="persist-1", goal="g", level=2, cwd=w.cwd)
    Engine(state, w.ports(), Budgets(), poll=0.001).run()          # the real store.save
    loaded = store.load("persist-1", w.cwd)
    assert loaded is not None and loaded.status == "done"
    assert [s.status for s in loaded.get_plan().steps] == [DONE, DONE]
    assert not loaded.resumable
    assert store.latest(w.cwd).id == "persist-1"


def test_level_one_and_no_critic_means_no_critic(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    state = store.RunState(id="t3", goal="g", level=1, cwd=w.cwd)
    Engine(state, w.ports(), Budgets(), save=lambda s: None, poll=0.001).run()
    assert w.critic_calls == []


def test_a_worker_that_cannot_start_is_a_failed_try_not_a_crash(tmp_path):
    w = World(plan_json(step("a")), tmp_path)
    engine, state, _ = make(w, max_attempts=2)

    def start(step_obj, task):
        raise ValueError("4 sub-agents are already running")
    engine.ports.start = start
    engine.run()
    assert state.status == "failed"
    assert statuses(state)["a"] == FAILED
