"""The person's own rooms are one set for the whole Argent: a room made in one working folder is there
after /cd, the working folder's own .argent/rooms is still shown, and nothing is lost on delete."""

import json

import pytest

from src.rooms import session
from src.rooms.models import Room

TOOLS = {"run_command"}


def room_json(name, text="do it"):
    return {"room": name, "description": text, "tags": [name], "entry": "a",
            "budget": {"max_iterations": 4, "max_tokens": 5000},
            "nodes": [{"id": "a", "type": "agent", "role": "Coder", "instruction": text,
                       "budget": {"max_iterations": 2, "max_tokens": 1000}, "writes": ["r"]}],
            "edges": [{"from": "a", "to": "EXIT:done"}]}


def put(folder, name, text="do it"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(room_json(name, text)), encoding="utf-8")


def test_a_room_saved_in_one_folder_is_there_from_another(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    lib = session.build_library(available_tools=TOOLS)
    assert lib.add_room(Room(**room_json("sandbox")), TOOLS) == []
    assert lib.user_dir == session.user_rooms_dir()
    assert not (first / ".argent" / "rooms" / "sandbox.json").exists()   # not kept in the working folder
    monkeypatch.chdir(second)                                            # /cd
    assert "sandbox" in session.build_library(available_tools=TOOLS).rooms


def test_the_working_folders_own_rooms_are_shown_and_global_wins_on_the_same_name(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    put(project / ".argent" / "rooms", "local_only", "from the project")
    put(project / ".argent" / "rooms", "both", "project copy")
    put(session.user_rooms_dir(), "both", "global copy")
    monkeypatch.chdir(project)
    lib = session.build_library(available_tools=TOOLS)
    assert "local_only" in lib.rooms and lib.sources["local_only"] == "user"
    assert lib.rooms["both"].description == "global copy"


def test_deleting_a_room_also_removes_the_project_copy(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    put(project / ".argent" / "rooms", "gone")
    monkeypatch.chdir(project)
    lib = session.build_library(available_tools=TOOLS)
    assert lib.remove_room("gone") is True
    assert not (project / ".argent" / "rooms" / "gone.json").exists()
    assert "gone" not in session.build_library(available_tools=TOOLS).rooms


def test_pinned_folders_are_exactly_those(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    put(project / ".argent" / "rooms", "local_only")
    monkeypatch.chdir(project)
    lib = session.build_library(available_tools=TOOLS, user_dir=tmp_path / "pinned")
    assert "local_only" not in lib.rooms         # a pinned library does not wander into the working folder


@pytest.mark.parametrize("name", ["stats"])
def test_the_run_counters_are_kept_beside_the_rooms_folder(name):
    assert session.stats_path().parent == session.user_rooms_dir().parent
