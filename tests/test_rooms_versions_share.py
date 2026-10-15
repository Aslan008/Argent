"""Versions of a room, export/import through a file, and the editor's layout of a project-local room."""

import json

import pytest

from src.rooms import cli as rooms_cli
from src.rooms import session
from src.rooms.models import Room

TOOLS = session.available_tool_names()


def room_json(name, text="do it", steps=1):
    nodes, edges = [], []
    for i in range(steps):
        nodes.append({"id": f"s{i}", "type": "agent", "role": "Coder", "instruction": f"{text} {i}",
                      "budget": {"max_iterations": 2, "max_tokens": 1000}, "writes": ["r"]})
        edges.append({"from": f"s{i}", "to": f"s{i + 1}" if i + 1 < steps else "EXIT:done"})
    return {"room": name, "description": text, "tags": [name], "entry": "s0",
            "budget": {"max_iterations": 4, "max_tokens": 5000}, "nodes": nodes, "edges": edges}


class FakeIo:
    def __init__(self, yes=True):
        self.lines, self.errors, self.yes, self.asked = [], [], yes, []

    def out(self, text):
        self.lines.append(text)

    def err(self, text):
        self.errors.append(text)

    def confirm(self, question):
        self.asked.append(question)
        return self.yes

    def choose(self, question, options):
        return options[0]


@pytest.fixture
def lib(tmp_path):
    return session.build_library(available_tools=TOOLS, user_dir=tmp_path / "rooms", stats_file=tmp_path / "s.json")


def save(lib, name, text, steps=1):
    assert lib.add_room(Room(**room_json(name, text, steps)), TOOLS) == []


def test_overwriting_a_room_keeps_the_previous_file(lib):
    save(lib, "r1", "first")
    assert lib.versions("r1") == []                       # nothing before the first save
    save(lib, "r1", "second")
    [v] = lib.versions("r1")
    assert json.loads(v.read_text(encoding="utf-8"))["description"] == "first"


def test_only_the_last_five_versions_are_kept(lib):
    for i in range(9):
        save(lib, "r1", f"v{i}")
    found = lib.versions("r1")
    assert len(found) == 5
    assert json.loads(found[0].read_text(encoding="utf-8"))["description"] == "v7"   # newest first


def test_restore_brings_back_an_earlier_version_and_is_itself_undoable(lib):
    save(lib, "r1", "old")
    save(lib, "r1", "new")
    assert lib.restore_version("r1", 1, TOOLS) == []
    assert lib.get("r1").description == "old"
    assert json.loads(lib.versions("r1")[0].read_text(encoding="utf-8"))["description"] == "new"


def test_a_deleted_room_comes_back(lib):
    save(lib, "r1", "precious")
    lib.remove_room("r1")
    assert lib.get("r1") is None and lib.versions("r1")
    assert lib.restore_version("r1", 1, TOOLS) == []
    assert lib.get("r1").description == "precious"


def test_a_project_local_room_is_copied_before_it_is_deleted(tmp_path):
    local = tmp_path / "proj"
    local.mkdir()
    (local / "only_here.json").write_text(json.dumps(room_json("only_here", "mine")), encoding="utf-8")
    lib = session.build_library(available_tools=TOOLS, user_dir=tmp_path / "rooms", extra_dirs=[local],
                                stats_file=tmp_path / "s.json")
    lib.remove_room("only_here")
    assert not (local / "only_here.json").exists()
    assert lib.restore_version("only_here", 1, TOOLS) == []


def test_version_folders_are_not_loaded_as_rooms(lib, tmp_path):
    save(lib, "r1", "a")
    save(lib, "r1", "b")
    again = session.build_library(available_tools=TOOLS, user_dir=tmp_path / "rooms", stats_file=tmp_path / "s.json")
    assert [n for n in again.rooms if again.sources[n] == "user"] == ["r1"]


def test_versions_and_restore_commands(lib):
    save(lib, "r1", "old")
    save(lib, "r1", "new")
    io = FakeIo()
    rooms_cli.handle("versions r1", io, library=lib)
    assert "1." in "\n".join(io.lines) and not io.errors
    io = FakeIo()
    rooms_cli.handle("restore r1", io, library=lib)
    assert io.asked and lib.get("r1").description == "old"
    io = FakeIo(yes=False)
    rooms_cli.handle("restore r1", io, library=lib)
    assert lib.get("r1").description == "old"             # no is no


def test_export_then_import_on_another_library(lib, tmp_path):
    save(lib, "shared", "carry me", steps=2)
    target = tmp_path / "out" / "shared.room.json"
    io = FakeIo()
    rooms_cli.handle(f'export shared "{target}"', io, library=lib)
    assert target.is_file() and not io.errors
    other = session.build_library(available_tools=TOOLS, user_dir=tmp_path / "rooms2", stats_file=tmp_path / "s2.json")
    io = FakeIo()
    rooms_cli.handle(f'import "{target}"', io, library=other)
    assert not io.errors and other.get("shared").description == "carry me"
    assert (tmp_path / "rooms2" / "shared.json").is_file()


def test_import_is_not_trusted(lib, tmp_path):
    bad = room_json("sneaky")
    bad["nodes"] = [{"id": "x", "type": "tool", "tool": "no_such_tool_at_all", "args": {}}]
    bad["entry"], bad["edges"] = "x", [{"from": "x", "to": "EXIT:done"}]
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad), encoding="utf-8")
    io = FakeIo()
    rooms_cli.handle(f'import "{f}"', io, library=lib)
    assert lib.get("sneaky") is None and io.errors


def test_import_cannot_take_the_name_of_a_starter_room(lib, tmp_path):
    starter = next(n for n, s in lib.sources.items() if s == "starter" and n != "triage")
    f = tmp_path / "x.json"
    f.write_text(json.dumps(room_json(starter, "mine")), encoding="utf-8")
    io = FakeIo()
    rooms_cli.handle(f'import "{f}"', io, library=lib)
    assert lib.sources[starter] == "starter"
    assert lib.get(f"{starter}_custom") is not None


def test_import_says_no_to_garbage(lib, tmp_path):
    f = tmp_path / "junk.json"
    f.write_text("{not json", encoding="utf-8")
    io = FakeIo()
    rooms_cli.handle(f'import "{f}"', io, library=lib)
    assert io.errors


def test_the_editor_finds_the_layout_next_to_a_project_local_room(tmp_path):
    from src.rooms.editor_server import EditorServer
    shared, local = tmp_path / "shared", tmp_path / "proj"
    shared.mkdir()
    local.mkdir()
    (local / "r.layout").write_text(json.dumps({"s0": [10, 20]}), encoding="utf-8")
    server = EditorServer(lambda: None, lambda: TOOLS, user_dir=shared)
    assert server._read_layout("r", [local]) == {"s0": [10, 20]}
    (shared / "r.layout").write_text(json.dumps({"s0": [1, 2]}), encoding="utf-8")
    assert server._read_layout("r", [local]) == {"s0": [1, 2]}      # the shared one wins once saved there
