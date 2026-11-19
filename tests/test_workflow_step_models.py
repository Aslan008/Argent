"""A workflow step may name its own model and thinking level.

The planner is the one place where the cost of a whole run is decided, so the plan
format has to be able to say "this step is routine" — and the run has to carry that
to the worker unchanged.
"""

import json

from src.agent.workflow import roles
from src.agent.workflow.plan import Step, parse_plan
from src.agent.workflow.engine import Ports


def step(id="s1", **extra):
    data = {"id": id, "title": f"title {id}", "goal": f"goal {id}", "role": "coder",
            "deps": [], "files": [], "checks": [], "verify": []}
    data.update(extra)
    return data


def plan_json(*steps):
    return json.dumps({"summary": "approach", "steps": list(steps), "final_checks": []})


def test_a_step_can_name_a_model_and_a_level():
    plan = parse_plan(plan_json(step(model="qwen3:8b", provider="ollama", thinking="low")), "G", 8)
    s = plan.steps[0]
    assert (s.model, s.provider, s.thinking) == ("qwen3:8b", "ollama", "low")


def test_a_step_without_them_inherits():
    plan = parse_plan(plan_json(step()), "G", 8)
    s = plan.steps[0]
    assert (s.model, s.provider, s.thinking) == ("", "", "")


def test_the_level_is_normalised_and_cut_to_a_sane_length():
    plan = parse_plan(plan_json(step(thinking=" LOW ", model="m" * 200)), "G", 8)
    s = plan.steps[0]
    assert s.thinking == "low"
    assert len(s.model) == 80


def test_a_step_survives_a_round_trip_through_the_saved_plan():
    original = Step(id="s1", title="t", goal="g", model="cheap", provider="ollama", thinking="off")
    restored = Step.from_dict(original.to_dict())
    assert (restored.model, restored.provider, restored.thinking) == ("cheap", "ollama", "off")


def test_the_planner_is_told_what_it_may_choose_from():
    messages = roles.planner_messages("do the thing", 6, targets="ollama: qwen3:8b [local]")
    text = messages[1]["content"]
    assert "WHAT THE WORKERS MAY RUN ON" in text
    assert "qwen3:8b" in text


def test_the_planner_is_told_how_to_use_the_levels():
    prompt = roles.PLANNER_SYSTEM
    assert '"thinking"' in prompt
    assert "routine steps" in prompt
    assert "never invent one" in prompt


def test_the_port_a_workflow_reads_targets_from_has_a_safe_default():
    # A run must not fail because the list of models could not be read.
    assert Ports(plan=lambda m: "", start=lambda s, t: None, check=lambda c, w: (True, "")).targets() == ""
