"""`/tasks room`: a room on a schedule from the same place as every other automation,
and «Нет комнаты» that names the nearest room."""

from datetime import datetime

import pytest

from src.automation import store, tasks_cli

NOW = datetime(2026, 10, 8, 14, 30)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    return tmp_path


def run(command):
    outs, errs = [], []
    tasks_cli.handle(command, outs.append, errs.append, now=NOW)
    return "\n".join(outs), "\n".join(errs)


def test_a_room_goes_on_a_schedule(project):
    out, err = run("/tasks room каждый день в 3:00 | explain_repo что это за проект depth=short | notify: failures")
    assert err == ""
    [a] = store.load_automations()
    assert a.mode == "room" and a.notify == "failures"
    assert a.task.split()[0] == "explain_repo" and "depth=short" in a.task
    assert "комнат" in out.lower()


def test_the_list_calls_it_a_room_not_an_agent(project):
    run("/tasks room every 1h | explain_repo")
    out, _ = run("/tasks list")
    assert "комната" in out and "агент" not in out


def test_an_unknown_room_is_refused_with_the_nearest_name(project):
    out, err = run("/tasks room every 1h | explain_rep")
    assert store.load_automations() == []
    assert "Нет комнаты" in err and "explain_repo" in err


def test_model_options_are_refused_for_a_room(project):
    out, err = run("/tasks room every 1h | explain_repo | tools: read_file")
    assert store.load_automations() == []
    assert "tools" in err


def test_a_bad_schedule_is_a_sentence_not_a_traceback(project):
    out, err = run("/tasks room когда-нибудь | explain_repo")
    assert store.load_automations() == []
    assert err


def test_no_room_names_close_matches():
    from src.rooms.cli import _no_room
    text = _no_room("clode_code", ["claude_code", "fix_bug", "research"])
    assert "claude_code" in text and "/rooms list" in text
    assert "Может быть" not in _no_room("zzzz", ["claude_code", "fix_bug"])
