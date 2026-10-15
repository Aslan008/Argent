"""Chat-built rooms: the builder (a scripted model), the house rules, the drawn flow, and the
/rooms build / edit / diagram commands and the `rooms` tool's build action."""

import json

import pytest

from src.rooms import builder as B
from src.rooms.diagram import diff_rooms, render_flow, render_mermaid
from src.rooms.models import Room
from tests.test_rooms_cli_tool import Term, lib, starter_lib  # noqa: F401  (fixtures and helpers)
from tests.test_rooms_library import STARTER_TOOLS
from src.rooms import cli
from src.rooms.tool import rooms_action

GOOD = {
    "room": "lint_then_tell",
    "description": "Run the linter; tell me on desktop either way.",
    "tags": ["линтер", "lint"],
    "budget": {"max_iterations": 8, "max_tokens": 5000},
    "entry": "lint",
    "defaults": {"lint_command": "ruff check ."},
    "next": {"clean": "END", "dirty": "END"},
    "nodes": [
        {"id": "lint", "type": "tool", "tool": "run_command", "args": {"command": "{{lint_command}}"},
         "sets": {"lint_ok": "ok", "lint_tail": "tail"}, "retries": 1},
        {"id": "good", "type": "notify", "message": "lint is clean", "channel": "desktop"},
        {"id": "bad", "type": "notify", "message": "lint failed: {{lint_tail}}", "channel": "desktop"},
    ],
    "edges": [
        {"from": "lint", "if": "state.lint_ok", "to": "good"},
        {"from": "lint", "else": True, "to": "bad"},
        {"from": "lint", "on_error": True, "to": "bad"},
        {"from": "good", "to": "EXIT:clean"},
        {"from": "bad", "to": "EXIT:dirty"},
    ],
}


def answer(obj, prose="Here is the room:\n"):
    return f"{prose}```json\n{json.dumps(obj, ensure_ascii=False)}\n```\nDone."


class ScriptedModel:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt, max_iterations=None, tools=None, role=None, **kw):
        self.prompts.append((prompt, tools, role))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def builder_for(lib_, *answers, repairs=2):
    model = ScriptedModel(*answers)
    return B.RoomBuilder(model, lib_, STARTER_TOOLS | {"write_file"}, max_repairs=repairs), model


# ── extraction ───────────────────────────────────────────────────────────────

class TestExtract:
    def test_takes_the_outermost_room_object_from_prose_and_fences(self):
        assert B.extract_room_json(answer(GOOD))["room"] == "lint_then_tell"

    def test_the_last_room_wins_and_inner_objects_are_not_mistaken_for_it(self):
        draft = dict(GOOD, room="draft_one")
        text = answer(draft, "first draft:\n") + "\nbut better:\n" + answer(GOOD, "")
        assert B.extract_room_json(text)["room"] == "lint_then_tell"

    def test_nothing_when_there_is_no_room(self):
        assert B.extract_room_json("I could not do it") is None
        assert B.extract_room_json('{"a": 1}') is None
        assert B.extract_room_json("") is None


# ── build ────────────────────────────────────────────────────────────────────

class TestBuild:
    def test_a_valid_answer_becomes_a_proposal_and_the_model_gets_no_tools(self, lib):
        b, model = builder_for(lib, answer(GOOD))
        res = b.build("прогони линтер и скажи мне")
        assert res.ok and res.room.room == "lint_then_tell" and res.attempts == 1
        prompt, tools, role = model.prompts[0]
        assert tools == [] and role == "Researcher"
        assert "прогони линтер и скажи мне" in prompt and "fix_bug" in prompt      # the request and the room catalogue
        assert "run_command" in prompt and "on_error" in prompt                      # tools and the node vocabulary

    def test_the_requested_name_wins(self, lib):
        b, _ = builder_for(lib, answer(GOOD))
        assert b.build("x", name="my_lint").room.room == "my_lint"

    def test_a_rejected_draft_is_repaired_with_the_validators_words(self, lib):
        bad = json.loads(json.dumps(GOOD))
        bad["nodes"][0]["tool"] = "no_such_tool"
        b, model = builder_for(lib, answer(bad), answer(GOOD))
        res = b.build("x")
        assert res.ok and res.attempts == 2
        assert "no_such_tool" in model.prompts[1][0] and "rejected" in model.prompts[1][0]

    def test_gives_up_after_the_repairs_and_says_why(self, lib):
        bad = json.loads(json.dumps(GOOD))
        bad["edges"] = [e for e in bad["edges"] if e["from"] != "bad"]       # 'bad' can no longer reach an EXIT
        b, _ = builder_for(lib, answer(bad), repairs=1)
        res = b.build("x")
        assert not res.ok and res.attempts == 2 and any("cannot reach any EXIT" in e for e in res.errors)

    def test_no_json_at_all(self, lib):
        b, _ = builder_for(lib, "Sorry, I can't.", repairs=0)
        res = b.build("x")
        assert not res.ok and "no room JSON" in res.errors[0]

    def test_a_packaged_rooms_name_is_refused(self, lib):
        b, _ = builder_for(lib, answer(dict(GOOD, room="fix_bug")), repairs=0)
        res = b.build("x")
        assert not res.ok and "taken by a packaged room" in res.errors[0]

    def test_spawn_room_nodes_are_rejected(self, lib):
        bad = json.loads(json.dumps(GOOD))
        bad["nodes"].append({"id": "sneak", "type": "spawn_room"})
        bad["edges"].append({"from": "sneak", "to": "EXIT:clean"})
        b, _ = builder_for(lib, answer(bad), repairs=0)
        assert not b.build("x").ok

    def test_too_many_nodes(self, lib):
        big = json.loads(json.dumps(GOOD))
        big["nodes"] = [{"id": f"s{i}", "type": "set", "values": {"a": i}} for i in range(30)]
        big["entry"] = "s0"
        big["edges"] = [{"from": f"s{i}", "to": f"s{i + 1}"} for i in range(29)] + [{"from": "s29", "to": "EXIT:clean"}]
        b, _ = builder_for(lib, answer(big), repairs=0)
        assert any("too many nodes" in e for e in b.build("x").errors)

    def test_house_rule_a_mutating_tool_must_ask(self, lib):
        draft = json.loads(json.dumps(GOOD))
        draft["nodes"].append({"id": "w", "type": "tool", "tool": "write_file", "args": {"file_path": "a", "content": "b"}})
        draft["edges"].append({"from": "w", "to": "EXIT:clean"})
        b, _ = builder_for(lib, answer(draft))
        res = b.build("x")
        assert res.ok and next(n for n in res.room.nodes if n.id == "w").confirm is True
        assert any("write_file" in n for n in res.notes)

    def test_a_call_to_an_existing_room_is_accepted_and_to_a_missing_one_is_not(self, lib):
        with_room = json.loads(json.dumps(GOOD))
        with_room["nodes"].append({"id": "fix", "type": "room", "room": "debug_loop", "outputs": ["tests_passed"]})
        with_room["edges"].append({"from": "fix", "to": "EXIT:clean"})
        assert builder_for(lib, answer(with_room))[0].build("x").ok
        with_room["nodes"][-1]["room"] = "ghost_room"
        assert not builder_for(lib, answer(with_room), repairs=0)[0].build("x").ok

    def test_edit_keeps_the_old_name_and_reports_the_changes(self, lib):
        old = Room(**GOOD)
        changed = json.loads(json.dumps(GOOD))
        changed["nodes"][1]["message"] = "all clean!"
        b, model = builder_for(lib, answer(changed))
        res = b.edit(old, "change the message")
        assert res.ok and res.room.room == "lint_then_tell"
        assert res.changes == ["~ узел good изменён"]
        assert "CURRENT ROOM" in model.prompts[0][0] and "change the message" in model.prompts[0][0]


# ── drawing ──────────────────────────────────────────────────────────────────

class TestDiagram:
    def test_flow_reads_like_a_recipe(self):
        text = render_flow(Room(**GOOD))
        assert text.startswith("Комната «lint_then_tell»")
        assert "▶ · lint  run_command: {{lint_command}}" in text and "вердикт процесса → lint_ok, lint_tail" in text
        assert "если state.lint_ok → good" in text and "иначе → bad" in text and "при сбое узла → bad" in text
        assert "повторов при сбое: 1" in text and "выход «clean»" in text
        assert "Дальше: clean → END; dirty → END" in text

    def test_mermaid_has_every_node_and_edge(self):
        m = render_mermaid(Room(**GOOD))
        assert m.splitlines()[0] == "flowchart TD"
        for node in ("n_lint", "n_good", "n_bad", "exit_n_clean", "exit_n_dirty"):
            assert node in m
        assert 'n_lint -- "state.lint_ok" --> n_good' in m and '"on error"' in m

    def test_diff_in_words(self):
        old = Room(**GOOD)
        new = json.loads(json.dumps(GOOD))
        new["nodes"].append({"id": "extra", "type": "set", "values": {"a": 1}})
        new["edges"].append({"from": "extra", "to": "EXIT:clean"})
        new["nodes"] = [n for n in new["nodes"] if n["id"] != "bad"]
        new["edges"] = [e for e in new["edges"] if e["to"] != "bad" and e["from"] != "bad"]
        d = diff_rooms(old, Room(**new))
        assert "+ узел extra (set)" in d and "− узел bad" in d and any(x.startswith("− переход lint → bad") for x in d)


# ── /rooms build, edit, diagram ──────────────────────────────────────────────

class TestCliBuild:
    def test_build_shows_the_flow_and_saves_after_a_yes(self, lib, tmp_path):
        b, _ = builder_for(lib, answer(GOOD))
        t = Term(choice="Сохранить")
        cli.handle("build после правки прогони линтер и скажи мне", t.io(), library=lib, builder=b)
        assert "Комната «lint_then_tell»" in t.text and "Сохранена:" in t.text
        assert (tmp_path / "user" / "lint_then_tell.json").exists() and lib.get("lint_then_tell") is not None
        assert "/rooms run lint_then_tell" in t.text and 'lint_command="ruff check ."' in t.text

    def test_nothing_is_saved_without_a_yes(self, lib, tmp_path):
        b, _ = builder_for(lib, answer(GOOD))
        t = Term(choice="Отмена")
        cli.handle("build линтер", t.io(), library=lib, builder=b)
        assert lib.get("lint_then_tell") is None and not (tmp_path / "user").exists()

    def test_the_person_can_refine_in_words_before_saving(self, lib):
        second = json.loads(json.dumps(GOOD))
        second["nodes"][1]["message"] = "refined message"
        b, model = builder_for(lib, answer(GOOD), answer(second))
        t = Term(choice=["Поправить словами", "Сохранить"])
        cli.handle("build линтер", t.io(ask_text=lambda q: "сделай сообщение короче"), library=lib, builder=b)
        assert lib.get("lint_then_tell").nodes[1].message == "refined message"
        assert "сделай сообщение короче" in model.prompts[1][0]

    def test_a_name_prefix_names_the_room(self, lib):
        b, _ = builder_for(lib, answer(GOOD))
        cli.handle("build my_lint: проверь линтер", Term(choice="Сохранить").io(), library=lib, builder=b)
        assert lib.get("my_lint") is not None

    def test_invalid_results_are_explained_and_empty_requests_get_help(self, lib):
        b, _ = builder_for(lib, "no json here", repairs=0)
        t = Term()
        cli.handle("build что-то", t.io(), library=lib, builder=b)
        assert "Не удалось составить" in t.text and "no room JSON" in t.text
        t2 = Term()
        cli.handle("build", t2.io(), library=lib, builder=b)
        assert "Опишите" in t2.text

    def test_edit_of_a_packaged_room_is_saved_as_a_copy(self, lib, tmp_path):
        fixed = json.loads(json.dumps(GOOD))
        fixed["room"] = "whatever"          # the builder is told the new name and overrides it
        b, model = builder_for(lib, answer(fixed))
        t = Term(choice="Сохранить")
        cli.handle("edit release_check добавь уведомление в Telegram", t.io(), library=lib, builder=b)
        assert lib.get("release_check_custom") is not None and lib.get("release_check").room == "release_check"
        assert "release_check_custom" in t.text and "Что изменилось" in t.text
        assert "release_check" in model.prompts[0][0]

    def test_edit_with_an_unknown_first_word_is_a_task(self, lib, tmp_path):
        t = Term(choice=None)
        cli.handle("edit the readme so that it mentions rooms", t.io(), library=lib,
                   builder=builder_for(lib, "x")[0], record_dir=tmp_path / "r")
        assert "Нет комнаты" not in t.text           # went through the router instead of the edit command

    def test_diagram_text_and_mermaid(self, lib):
        t = Term()
        cli.handle("diagram fix_bug", t.io(), library=lib)
        assert "Комната «fix_bug»" in t.text
        t2 = Term()
        cli.handle("diagram fix_bug mermaid", t2.io(), library=lib)
        assert t2.out[0].startswith("flowchart TD")


# ── the tool's build action ──────────────────────────────────────────────────

class TestToolBuild:
    def test_saved_only_when_the_person_approves(self, lib, tmp_path):
        b, _ = builder_for(lib, answer(GOOD))
        seen = {}
        out = rooms_action("build", "", "прогони линтер", library=lib, builder=b,
                           ask_save=lambda flow, name: seen.update(flow=flow, name=name) or True)
        assert "saved" in out and lib.get("lint_then_tell") is not None
        assert seen["name"] == "lint_then_tell" and "▶ · lint" in seen["flow"]

    def test_refusal_is_reported_to_the_model(self, lib):
        b, _ = builder_for(lib, answer(GOOD))
        out = rooms_action("build", "", "прогони линтер", library=lib, builder=b, ask_save=lambda f, n: False)
        assert "did NOT approve" in out and lib.get("lint_then_tell") is None

    def test_missing_task_and_failed_builds_are_errors(self, lib):
        assert rooms_action("build", "", "", library=lib).startswith("Error: `task`")
        b, _ = builder_for(lib, "nothing", repairs=0)
        assert rooms_action("build", "", "x", library=lib, builder=b).startswith("Error: could not make")

    def test_edit_action(self, lib):
        changed = json.loads(json.dumps(GOOD))
        b, _ = builder_for(lib, answer(GOOD), answer(changed))
        rooms_action("build", "", "x", library=lib, builder=b, ask_save=lambda f, n: True)
        out = rooms_action("edit", "lint_then_tell", "make it quieter", library=lib, builder=b,
                           ask_save=lambda f, n: True)
        assert "saved" in out
        assert rooms_action("edit", "ghost", "x", library=lib, builder=b).startswith("Error: no room")

    def test_plan_mode_blocks_build_and_edit(self):
        from src.agent.plan_mode import PLAN, violation
        PLAN.enter()
        try:
            assert violation("rooms", {"action": "build"}) and violation("rooms", {"action": "edit"})
            assert violation("rooms", {"action": "list"}) is None
        finally:
            PLAN.leave()
