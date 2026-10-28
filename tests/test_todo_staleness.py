"""A checklist nobody has touched in a while says so.

The list was only ever "the last thing todo_write said": a step left in_progress
lived until the end of the session, the person kept reading "1/4" long after the
work was finished, and the model kept believing it was in the middle of something.
"""

import pytest

from src.agent import todos as T


@pytest.fixture
def store(monkeypatch):
    s = T.TodoStore()
    monkeypatch.setattr(T, "TODOS", s)
    s.set([T.Todo("первый", T.PENDING), T.Todo("второй", T.PENDING)], "main")
    return s


def _block():
    return T.ephemeral_block("main")


def test_a_fresh_list_says_nothing_extra(store):
    store.set([T.Todo("работа", T.IN_PROGRESS)], "main")
    assert "не менялся" not in _block()


def test_an_untouched_in_progress_step_is_called_out(store):
    store.set([T.Todo("работа", T.IN_PROGRESS)], "main")
    for _ in range(T.STALE_TURNS):
        store.begin_turn()
    text = _block()
    assert f"не менялся {T.STALE_TURNS} ход" in text


def test_writing_the_list_clears_the_warning(store):
    store.set([T.Todo("работа", T.IN_PROGRESS)], "main")
    for _ in range(T.STALE_TURNS + 2):
        store.begin_turn()
    assert "не менялся" in _block()

    store.set([T.Todo("работа", T.COMPLETED), T.Todo("следующая", T.PENDING)], "main")
    assert "не менялся" not in _block()


def test_a_stale_list_with_nothing_running_is_quiet(store):
    # Nothing in progress: the plan is simply paused, not forgotten.
    store.set([T.Todo("позже", T.PENDING)], "main")
    for _ in range(T.STALE_TURNS + 2):
        store.begin_turn()
    assert "не менялся" not in _block()


def test_a_sub_agents_turns_do_not_age_the_main_list(store):
    # Delegation is not the supervisor standing still.
    store.set([T.Todo("работа", T.IN_PROGRESS)], "main")
    for _ in range(T.STALE_TURNS + 2):
        store.begin_turn("sub-1")
    assert store.turns_since_change("main") == 0
    for _ in range(T.STALE_TURNS):
        store.begin_turn("main")
    assert store.turns_since_change("main") >= T.STALE_TURNS


def test_a_finished_list_is_still_dropped_at_the_next_turn(store):
    # The older behaviour of begin_turn must survive the counter being added.
    store.set([T.Todo("всё", T.COMPLETED)], "main")
    store.begin_turn()
    assert store.get("main") == []


def test_an_empty_store_says_nothing(store):
    # No list at all is not a stale list: there is nothing to remind anyone of.
    store.clear("main")
    for _ in range(T.STALE_TURNS + 2):
        store.begin_turn()
    assert T.ephemeral_block("main") is None
