"""The steps of a running workflow are on the person's screen: the checklist panel and the Telegram message."""

from unittest.mock import MagicMock

import pytest

from src.agent import todos
from src.agent.todos import COMPLETED, IN_PROGRESS, PENDING, TODOS, Todo
from src.agent.workflow import runtime
from src.agent.workflow.plan import DONE, FAILED, PENDING as S_PENDING, RUNNING, SKIPPED, Plan, Step


@pytest.fixture(autouse=True)
def _clean():
    TODOS.clear()
    yield
    TODOS.clear()


def _plan(*states):
    steps = []
    for i, state in enumerate(states, 1):
        steps.append(Step(id=f"s{i}", title=f"шаг {i}", goal="g", role="coder", status=state,
                          attempts=2 if state == RUNNING else 0))
    return Plan(goal="цель", steps=steps, summary="s")


def test_the_steps_become_the_checklist(monkeypatch):
    monkeypatch.setattr("src.agent.governor.brief", lambda: "")
    board = runtime.StepBoard()
    board.update(_plan(DONE, RUNNING, S_PENDING, FAILED, SKIPPED))
    items = TODOS.get("main")
    assert [t.status for t in items] == [COMPLETED, IN_PROGRESS, PENDING, COMPLETED, COMPLETED]
    assert items[1].content.startswith("s2. шаг 2") and "попытка 2" in items[1].content
    assert items[3].content.startswith("✗") and items[4].content.startswith("↷")


def test_the_queue_and_the_pc_are_on_the_running_step(monkeypatch):
    monkeypatch.setattr("src.agent.governor.brief", lambda: "модель: 1 работает, 1 ждёт")
    board = runtime.StepBoard()
    board.update(_plan(RUNNING))
    assert "модель: 1 работает, 1 ждёт" in TODOS.get("main")[0].content


def test_the_note_is_not_looked_up_on_every_update(monkeypatch):
    calls = []
    monkeypatch.setattr("src.agent.governor.brief", lambda: calls.append(1) or "")
    t = [0.0]
    board = runtime.StepBoard(clock=lambda: t[0])
    for _ in range(10):
        board.update(_plan(RUNNING))
    assert len(calls) == 1
    t[0] += 6
    board.update(_plan(RUNNING, S_PENDING))
    assert len(calls) == 2


def test_the_models_own_checklist_comes_back_when_the_run_is_done(monkeypatch):
    monkeypatch.setattr("src.agent.governor.brief", lambda: "")
    TODOS.set([Todo("моя задача", IN_PROGRESS)], "main")
    board = runtime.StepBoard()
    board.update(_plan(DONE))
    assert TODOS.get("main")[0].content.startswith("s1.")
    board.close(done=True)
    assert [t.content for t in TODOS.get("main")] == ["моя задача"]


def test_a_stopped_run_leaves_its_steps_on_the_screen(monkeypatch):
    monkeypatch.setattr("src.agent.governor.brief", lambda: "")
    board = runtime.StepBoard()
    board.update(_plan(DONE, S_PENDING))
    board.close(done=False)
    assert len(TODOS.get("main")) == 2


def test_a_listener_hears_every_change_and_one_that_raises_is_harmless():
    heard = []
    TODOS.subscribe(lambda scope, items: heard.append((scope, [t.content for t in items])))
    TODOS.subscribe(lambda scope, items: 1 / 0)
    TODOS.set([Todo("a")], "main")
    TODOS.clear("main")
    assert heard == [("main", ["a"]), ("main", [])]


def test_the_telegram_panel_mirrors_the_checklist_and_lets_go_when_done():
    from src.telegram.live import LiveStatus

    client = MagicMock()
    client.send_message.return_value = {"message_id": 7}
    client.edit_message_text.return_value = {"message_id": 7}
    live = LiveStatus(client, 1, "Большая задача", tick=0)
    live.start()
    TODOS.set([Todo("s1. разобрать", COMPLETED), Todo("s2. править — идёт", IN_PROGRESS)], "main")
    text = live._render()
    assert "s1. разобрать" in text and "s2. править" in text
    live.finish("done")
    TODOS.set([Todo("после конца")], "main")
    assert "после конца" not in live._render(), "a finished panel no longer follows the list"


def test_the_engine_reports_progress_and_survives_a_panel_that_raises(tmp_path):
    from tests.test_workflow_engine import World, make, plan_json, step

    world = World(plan_json(step("a"), step("b", deps=["a"])), tmp_path)
    engine, state, _saved = make(world)
    seen = []

    def progress(plan):
        seen.append({s.id: s.status for s in plan.steps})
        raise RuntimeError("a broken panel")

    engine.ports.progress = progress
    final = engine.run()
    assert final.status == "done"
    assert any(s.get("a") == RUNNING for s in seen) and seen[-1] == {"a": DONE, "b": DONE}
