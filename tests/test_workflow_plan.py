"""The plan of a workflow: what a model may write, and what is refused."""

import json

import pytest

from src.agent.workflow import plan as P
from src.agent.workflow.plan import Plan, PlanError, Step, parse_plan


def raw(steps, **extra):
    return json.dumps({"summary": "s", "steps": steps, **extra})


def s(id, goal="do it", **kw):
    return {"id": id, "title": f"title {id}", "goal": goal, **kw}


def test_a_plain_plan_is_read():
    plan = parse_plan(raw([s("a", role="coder", files=["x.py"], checks=["pytest -q"]),
                           s("b", deps=["a"], role="reviewer")], final_checks=["pytest"]), "goal")
    assert [x.id for x in plan.steps] == ["a", "b"]
    assert plan.steps[0].checks == ["pytest -q"]
    assert plan.steps[1].deps == ["a"]
    assert plan.final_checks == ["pytest"]
    assert plan.goal == "goal"


def test_json_inside_prose_and_fences_is_found():
    text = "Sure! Here is the plan:\n```json\n" + raw([s("a")]) + "\n```\nHope it helps {not json}"
    assert parse_plan(text, "g").steps[0].id == "a"


def test_braces_inside_strings_do_not_confuse_the_reader():
    text = raw([s("a", goal='write a function that returns "{}" and "}"')])
    assert parse_plan(text, "g").steps[0].goal.startswith("write a function")


def test_role_spellings_are_forgiven_but_unknown_roles_are_not():
    plan = parse_plan(raw([s("a", role="Developer"), s("b", role="QA"), s("c", role="Research"),
                           s("d", role="docs")]), "g")
    assert [x.role for x in plan.steps] == ["coder", "reviewer", "researcher", "docwriter"]
    with pytest.raises(PlanError, match="unknown role"):
        parse_plan(raw([s("a", role="wizard")]), "g")


def test_missing_ids_are_numbered():
    plan = parse_plan(raw([{"goal": "one"}, {"goal": "two"}]), "g")
    assert [x.id for x in plan.steps] == ["s1", "s2"]


@pytest.mark.parametrize("text, message", [
    ("no json here", "no JSON"),
    ("{broken", "no JSON"),
    ('{"steps": []}', "no `steps`"),
    ('{"steps": "x"}', "no `steps`"),
    (raw([s("a", goal="")]), "`goal` is empty"),
    (raw([s("a"), s("a")]), "duplicate"),
    (raw([s("a", deps=["zzz"])]), "not a step"),
    (raw([s("a", deps=["a"])]), "depends on itself"),
    (raw([s("a", deps=["b"]), s("b", deps=["a"])]), "cycle"),
    (raw([s("bad id!")]), "bad id"),
])
def test_refusals_say_what_is_wrong(text, message):
    with pytest.raises(PlanError, match=message):
        parse_plan(text, "g")


def test_too_many_steps_is_refused_with_the_limit():
    with pytest.raises(PlanError, match="at most 3"):
        parse_plan(raw([s(f"s{i}") for i in range(4)]), "g", max_steps=3)


def test_lists_and_texts_are_bounded():
    plan = parse_plan(raw([s("a", files=[f"f{i}.py" for i in range(40)], checks=["c" * 900] * 9)]), "g")
    assert len(plan.steps[0].files) == P.MAX_FILES
    assert len(plan.steps[0].checks) == P.MAX_CHECKS
    assert len(plan.steps[0].checks[0]) == P.MAX_CHECK_LEN


def test_ready_waits_for_dependencies():
    plan = parse_plan(raw([s("a"), s("b", deps=["a"]), s("c")]), "g")
    assert [x.id for x in plan.ready()] == ["a", "c"]
    plan.by_id()["a"].status = P.DONE
    assert [x.id for x in plan.ready()] == ["b", "c"]


def test_a_failed_step_blocks_what_depends_on_it_transitively():
    plan = parse_plan(raw([s("a"), s("b", deps=["a"]), s("c", deps=["b"]), s("d")]), "g")
    plan.by_id()["a"].status = P.FAILED
    skipped = plan.skip_blocked()
    assert {x.id for x in skipped} == {"b", "c"}
    assert plan.by_id()["d"].status == P.PENDING


def test_layers_group_what_could_run_together():
    plan = parse_plan(raw([s("a"), s("b"), s("c", deps=["a", "b"]), s("d", deps=["c"])]), "g")
    assert P.layers(plan) == [["a", "b"], ["c"], ["d"]]


def test_overlap_rules():
    def step(role, files):
        return Step(id="x", title="t", goal="g", role=role, files=files)
    assert P.overlaps(step("coder", ["a.py"]), step("coder", ["a.py"]))
    assert P.overlaps(step("coder", ["src"]), step("coder", ["src/a.py"])), "a folder covers its files"
    assert not P.overlaps(step("coder", ["a.py"]), step("coder", ["b.py"]))
    assert P.overlaps(step("coder", []), step("coder", ["b.py"])), "unknown scope counts as overlap"
    assert not P.overlaps(step("researcher", []), step("coder", ["b.py"])), "readers never conflict"
    assert P.overlaps(step("coder", ["A.PY"]), step("docwriter", ["a.py"])), "case and slashes are ignored"


def test_the_plan_survives_a_round_trip():
    plan = parse_plan(raw([s("a", files=["x"], checks=["c"]), s("b", deps=["a"])]), "goal")
    plan.by_id()["a"].status = P.DONE
    plan.by_id()["a"].notes.append("n")
    again = Plan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again.to_dict() == plan.to_dict()


def test_the_outline_is_what_a_person_reads():
    plan = parse_plan(raw([s("a", checks=["pytest"]), s("b", deps=["a"])], final_checks=["make"]), "g")
    text = plan.outline()
    assert "a. [coder] title a" in text and "после a" in text and "проверка: pytest" in text
    assert "итоговая проверка: make" in text
