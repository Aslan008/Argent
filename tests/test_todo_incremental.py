"""Точечная правка списка задач, время шага и напоминание в конце хода.

Раньше единственным способом было переслать весь список: чтобы отметить пятый
пункт из шести, приходилось повторять все шесть — дорого и с риском незаметно
переписать формулировку, которую менять не собирались.
"""

import json
import time

import pytest

from src.agent import todos
from src.agent.todos import TODOS


@pytest.fixture(autouse=True)
def clean():
    TODOS.clear()
    yield
    TODOS.clear()


def _plan():
    todos.write([
        {"content": "прочитать код", "status": "completed"},
        {"content": "починить баг", "status": "in_progress"},
        {"content": "прогнать тесты", "status": "pending"},
    ])


# ── правка одного пункта ───────────────────────────────────────────────────────

def test_a_status_tick_names_one_item_and_leaves_the_rest_alone():
    _plan()

    out = todos.write(index=3, status="completed")

    assert [t.status for t in TODOS.get()] == ["completed", "in_progress", "completed"]
    assert TODOS.get()[0].content == "прочитать код", "формулировки не тронуты"
    assert "2/3 done" in out
    assert "Current step: починить баг" in out


def test_the_reply_numbers_the_items_so_the_next_call_can_point_at_one():
    _plan()

    out = todos.write(index=2, status="completed")

    assert "1. [x] прочитать код" in out
    assert "2. [x] починить баг" in out
    assert "3. [ ] прогнать тесты" in out


def test_content_rewords_the_step_without_touching_its_status():
    _plan()

    todos.write(index=3, content="прогнать ВСЕ тесты, включая e2e")

    item = TODOS.get()[2]
    assert item.content == "прогнать ВСЕ тесты, включая e2e"
    assert item.status == "pending", "статус остался прежним"


def test_a_number_out_of_range_is_refused_and_shows_the_list():
    _plan()

    out = todos.write(index=9, status="completed")

    assert out.startswith("Error:")
    assert "1. [x] прочитать код" in out, "ошибка показывает, из чего выбирать"
    assert len(TODOS.get()) == 3


def test_an_edit_without_status_or_content_is_refused():
    _plan()

    assert todos.write(index=1).startswith("Error:")
    assert TODOS.get()[0].content == "прочитать код"


def test_an_edit_of_an_empty_list_says_what_to_do():
    out = todos.write(index=1, status="completed")

    assert out.startswith("Error:")
    assert "`todos`" in out, "подсказывает, что сначала нужно написать план"


def test_nonsense_status_is_refused():
    _plan()

    assert todos.write(index=2, status="почти").startswith("Error:")
    assert TODOS.get()[1].status == "in_progress"


def test_the_tool_reaches_the_new_shape_through_the_registry():
    from argent.tools.schemas import get_available_tools

    tools = get_available_tools()
    tools["todo_write"](todos=[{"content": "a", "status": "pending"}])
    out = tools["todo_write"](index=1, status="in_progress")

    assert "[~] a" in out
    assert TODOS.get()[0].status == "in_progress"


# ── время шага ────────────────────────────────────────────────────────────────

def test_an_in_progress_item_carries_when_it_started():
    _plan()

    snap = TODOS.snapshot()

    started = snap[1]["since"]
    assert abs(started - time.time()) < 5, "время начала — «сейчас»"
    assert "since" not in snap[0], "у выполненного времени нет"


def test_the_clock_survives_other_items_changing():
    _plan()
    first = TODOS.snapshot()[1]["since"]

    time.sleep(0.01)
    todos.write(index=3, status="completed")       # чужая строка
    second = TODOS.snapshot()[1]["since"]

    assert second == first, "правка другого пункта не перезапускает таймер"


def test_leaving_in_progress_drops_the_clock_and_coming_back_restarts_it():
    _plan()
    first = TODOS.snapshot()[1]["since"]

    todos.write(index=2, status="pending")
    assert "since" not in TODOS.snapshot()[1]

    time.sleep(0.01)
    todos.write(index=2, status="in_progress")
    again = TODOS.snapshot()[1]["since"]

    assert again > first, "вернулись в работу — отсчёт с начала"


def test_the_clock_is_gone_with_the_list():
    _plan()
    TODOS.clear()

    assert TODOS.snapshot() == []


# ── конец хода ────────────────────────────────────────────────────────────────

def test_a_finished_plan_says_nothing_at_the_end_of_a_turn():
    todos.write([{"content": "a", "status": "completed"}])

    assert todos.pending_note(TODOS.get()) == ""


def test_a_stopped_turn_does_not_scold_the_plan():
    _plan()

    assert todos.end_of_turn_note(TODOS.get(), stopped=True) == ""
    assert "Осталось" in todos.end_of_turn_note(TODOS.get(), stopped=False)


def test_an_unfinished_plan_is_named_at_the_end_of_a_turn():
    _plan()

    note = todos.pending_note(TODOS.get())

    assert "Осталось 2 из 3" in note
    assert "[~] починить баг" in note
    assert "[ ] прогнать тесты" in note
    assert "[x] прочитать код" not in note, "сделанное не перечисляется"


def test_a_long_unfinished_tail_is_cut_but_counted():
    todos.write([{"content": f"шаг {i}", "status": "pending"} for i in range(1, 10)])

    note = todos.pending_note(TODOS.get())

    assert "Осталось 9 из 9" in note
    assert "шаг 6" in note
    assert "шаг 7" not in note
    assert "… и ещё 3" in note


def test_the_checklist_still_round_trips_through_json_for_the_frontend():
    _plan()

    raw = json.dumps(TODOS.snapshot(), ensure_ascii=False)
    items = json.loads(raw)

    assert [i["status"] for i in items] == ["completed", "in_progress", "pending"]
    assert isinstance(items[1]["since"], float)
