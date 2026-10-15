"""The starter rooms end to end (fake executors, real engine), the `/rooms` dispatcher and the `rooms` tool.

No model, no terminal: the person's answers come through `Io`, the room workers are scripted."""

import json
import re
from pathlib import Path

import pytest

from src.rooms import cli, history
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import Room
from src.rooms.runner import NodeRunner
from src.rooms.session import execute_room_run, parse_params, run_rooms
from src.rooms.tool import rooms_action
from tests.test_rooms_library import STARTER_TOOLS


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """The journal, stats and run records are relative to the cwd: keep them out of the repository."""
    monkeypatch.chdir(tmp_path)


def starter_lib(**kw):
    return RoomLibrary(**kw).load_dir(default_starter_dir(), STARTER_TOOLS)


class Script:
    """Scripted workers for the starter rooms: the test command's exit codes come from a queue,
    agents answer from a dict keyed by node id."""

    def __init__(self, exit_codes=(0,), agents=None, git_status="", human=None):
        self.exit_codes = list(exit_codes)
        self.agents = agents or {}
        self.git_status = git_status
        self.commands = []
        self.agent_calls = []
        self.human_answer = human

    def tool(self, node, state):
        if node.tool == "run_command":
            cmd = node.args.get("command", "")
            self.commands.append(cmd)
            if cmd.startswith("git status"):
                return f"Exit code: 0\nOUTPUT:\n{self.git_status}" if self.git_status else "Exit code: 0"
            if cmd.startswith("git commit"):
                return "Exit code: 0\nOUTPUT:\n[master abc123] rooms: tests green"
            code = self.exit_codes.pop(0) if len(self.exit_codes) > 1 else self.exit_codes[0]
            return f"Exit code: {code}\nOUTPUT:\n" + ("all passed" if code == 0 else "FAILED tests/test_x.py::test_a")
        if node.tool == "search_web":
            return "result: use the parser fix"
        if node.tool == "read_git_diff":
            return "diff --git a/x.py b/x.py\n+print('hi')"
        if node.tool == "analyze_project":
            return "python project, 3 packages"
        return "ok"

    def agent(self, node, context):
        self.agent_calls.append(node.id)
        return dict(self.agents.get(node.id, {}))

    def runner(self, confirm=True):
        def human(node, state):
            if self.human_answer is not None:
                state.data[f"{node.id}.output"] = self.human_answer
        return NodeRunner(tool_executor=self.tool, agent_executor=self.agent, human_prompt=human,
                          confirm=(lambda n, s: True) if confirm else None)


# ── the starter rooms as recipes ───────────────────────────────────────────────

class TestStarterRooms:
    def test_every_placeholder_refers_to_a_field_the_room_can_have(self):
        from src.rooms import template
        lib = starter_lib()
        # the catalog of rooms is put into every run's state; triage also reads what other rooms wrote
        written = {f for r in lib.rooms.values() for n in r.nodes for f in list(n.writes) + list(n.sets)}
        for room in lib.rooms.values():
            known = {"task", "rooms_catalog"} | set(room.defaults)
            if room.room == "triage":
                known |= written
            for n in room.nodes:
                known |= set(n.writes) | set(n.sets)
                known |= {f"{n.id}.output"}
                if n.into:
                    known |= {n.into, f"{n.into}_approved"}
            for n in room.nodes:
                for text in [n.instruction, n.prompt, json.dumps(n.args)]:
                    for ph in template.placeholders(text or ""):
                        assert ph in known, (room.room, n.id, ph)
            for ph in template.placeholders(room.report or ""):
                assert ph in known, (room.room, "report", ph)

    def test_starter_tools_exist_in_the_real_registry(self):
        from tools.lazy_registry import DEFAULT_TOOL_LOADERS
        real = set(DEFAULT_TOOL_LOADERS)
        used = set()
        for room in starter_lib().rooms.values():
            for n in room.nodes:
                if n.tool:
                    used.add(n.tool)
                used |= set(n.tools or [])
        assert used <= real, used - real

    def test_read_only_nodes_cannot_write_files(self):
        writers = {"write_file", "replace_in_file", "delete_file", "move_file", "copy_file", "run_command"}
        for room in starter_lib().rooms.values():
            for n in room.nodes:
                if n.type == "agent" and (n.role or "Coder") in ("Researcher", "Reviewer"):
                    assert n.tools is not None and not (set(n.tools) & writers), (room.room, n.id)

    def test_fix_bug_greens_after_one_patch(self):
        s = Script(exit_codes=[1, 0], agents={"diagnose": {"diagnosis": "parser drops the last token"},
                                              "patch": {"fix_summary": "keep the token"},
                                              "review": {"review_verdict": "ok", "review_notes": "minimal"}})
        res = run_rooms("login crashes", start="fix_bug", library=starter_lib(), runner=s.runner())
        assert res.outcome == "done" and res.history == [("fix_bug", "fixed")]
        assert s.agent_calls == ["diagnose", "patch", "review"]
        assert s.commands == ["python -m pytest -q", "python -m pytest -q"]
        assert res.state.data["tests_passed"] is True and res.state.data["review_verdict"] == "ok"

    def test_fix_bug_passes_the_test_command_parameter(self):
        s = Script(exit_codes=[0])
        run_rooms("x", start="fix_bug", library=starter_lib(), runner=s.runner(),
                  params={"test_command": "pytest tests/unit -x"})
        assert s.commands[0] == "pytest tests/unit -x"

    def test_fix_bug_gives_up_after_bounded_retries_and_escalates(self):
        s = Script(exit_codes=[1], agents={"diagnose": {"diagnosis": "?"}, "patch": {"fix_summary": "tried"}})
        res = run_rooms("flaky bug", start="fix_bug", library=starter_lib(), runner=s.runner())
        assert res.outcome == "escalated"
        assert s.agent_calls.count("patch") == 3          # first pass + 2 retries, then it stops
        assert ("fix_bug", "escalate") in res.history

    def test_an_agent_cannot_declare_the_tests_green(self):
        s = Script(exit_codes=[1], agents={"patch": {"fix_summary": "done", "tests_passed": True}})
        res = run_rooms("x", start="fix_bug", library=starter_lib(), runner=s.runner())
        assert res.outcome == "escalated" and res.state.data["tests_passed"] is False

    def test_debug_loop_commit_asks_and_a_refusal_is_reported(self):
        s = Script(exit_codes=[0])
        res = run_rooms("", start="debug_loop", library=starter_lib(), runner=s.runner(confirm=False))
        assert res.outcome == "done" and not any(c.startswith("git commit") for c in s.commands)
        assert res.state.data["committed"] is False and res.state.data["commit.declined"] is True

    def test_debug_loop_commits_when_approved(self):
        s = Script(exit_codes=[0])
        res = run_rooms("", start="debug_loop", library=starter_lib(), runner=s.runner(confirm=True))
        assert any(c == 'git commit -am "rooms: tests green"' for c in s.commands)
        assert res.state.data["committed"] is True

    def test_implement_feature_waits_for_the_persons_plan_approval(self):
        agents = {"plan": {"plan": "1. add exporter"}, "implement": {"impl_summary": "added"},
                  "review": {"review_verdict": "ok", "review_notes": "fine"}}
        s = Script(exit_codes=[0], agents=agents, human="да")
        res = run_rooms("add CSV export", start="implement_feature", library=starter_lib(), runner=s.runner())
        assert res.history == [("implement_feature", "done")]
        assert s.agent_calls == ["plan", "implement", "review"]

    def test_implement_feature_stops_when_the_plan_is_not_approved(self):
        agents = {"plan": {"plan": "1. rewrite everything"}}
        s = Script(exit_codes=[0], agents=agents, human="нет, слишком много")
        res = run_rooms("add CSV export", start="implement_feature", library=starter_lib(), runner=s.runner())
        assert res.history == [("implement_feature", "rejected")] and "implement" not in s.agent_calls

    def test_implement_feature_with_nobody_to_ask_pauses(self):
        s = Script(agents={"plan": {"plan": "p"}}, human=None)
        res = run_rooms("x", start="implement_feature", library=starter_lib(), runner=s.runner())
        assert res.outcome == "escalated" and "implement" not in s.agent_calls

    def test_code_review_verdicts_route_to_distinct_exits(self):
        ok = Script(agents={"review": {"verdict": "approve", "findings": "none"}})
        assert run_rooms("", start="code_review", library=starter_lib(), runner=ok.runner()).history == [
            ("code_review", "approved")]
        bad = Script(agents={"review": {"verdict": "changes_requested", "findings": "x.py:1 — bug"}})
        res = run_rooms("", start="code_review", library=starter_lib(), runner=bad.runner())
        assert res.history == [("code_review", "changes_requested")]

    def test_release_check_needs_green_tests_and_a_clean_tree(self):
        ready = Script(exit_codes=[0], git_status="", agents={"summarize": {"notes": "ship it"}})
        assert run_rooms("", start="release_check", library=starter_lib(), runner=ready.runner()).history == [
            ("release_check", "ready")]
        dirty = Script(exit_codes=[0], git_status=" M src/a.py", agents={"summarize": {"notes": "n"}})
        assert run_rooms("", start="release_check", library=starter_lib(), runner=dirty.runner()).history == [
            ("release_check", "not_ready")]
        red = Script(exit_codes=[1], git_status="", agents={"summarize": {"notes": "n"}})
        assert run_rooms("", start="release_check", library=starter_lib(), runner=red.runner()).history == [
            ("release_check", "not_ready")]

    def test_research_exits_depend_on_confidence(self):
        s = Script(agents={"synthesize": {"answer": "A", "sources": "https://x", "confident": True}})
        assert run_rooms("q", start="research", library=starter_lib(), runner=s.runner()).history == [
            ("research", "answered")]
        s2 = Script(agents={"synthesize": {"answer": "?", "confident": False}})
        assert run_rooms("q", start="research", library=starter_lib(), runner=s2.runner()).history == [
            ("research", "unsure")]

    def test_analyze_ends_after_the_plan_not_in_the_test_loop(self):
        s = Script(agents={"investigate": {"plan_ready": True, "plan": "1. do it"}})
        res = run_rooms("study the cache", start="analyze", library=starter_lib(), runner=s.runner())
        assert res.history == [("analyze", "planned")] and s.commands == []


# ── execute_room_run: the whole experience ───────────────────────────────────

class TestExecuteRoomRun:
    def test_progress_record_and_report(self, tmp_path):
        s = Script(exit_codes=[0], agents={"summarize": {"notes": "all good"}})
        lines = []
        out = execute_room_run("", "release_check", library=starter_lib(), runner=s.runner(),
                               printer=lines.append, record_dir=tmp_path)
        assert out.result.outcome == "done" and "release_check" in out.report and "all good" in out.report
        from rich.text import Text
        plain = [Text.from_markup(l).plain for l in lines]
        assert any("release_check › tests" in l for l in plain) and any("release_check закончена" in l for l in plain)
        rec = history.load_run("last", tmp_path)
        assert rec["start_room"] == "release_check" and [n["node"] for n in rec["nodes"]] == [
            "tests", "tree", "summarize"]

    def test_progress_restores_the_persons_checklist(self, tmp_path):
        from src.agent.todos import TODOS, Todo
        TODOS.set([Todo("my own item")], "main")
        try:
            s = Script(exit_codes=[0], agents={"summarize": {"notes": "n"}})
            execute_room_run("", "release_check", library=starter_lib(), runner=s.runner(),
                             printer=lambda t: None, record_dir=tmp_path)
            assert [t.content for t in TODOS.get("main")] == ["my own item"]
        finally:
            TODOS.clear("main")


# ── parameters ───────────────────────────────────────────────────────────────

class TestParseParams:
    def test_splits_key_values_and_keeps_the_task_verbatim(self):
        task, p = parse_params('fix C:\\proj\\a.py test_command="pytest -x tests" retries=2 now')
        assert task == "fix C:\\proj\\a.py now"
        assert p == {"test_command": "pytest -x tests", "retries": "2"}

    def test_equals_inside_a_word_is_not_a_parameter(self):
        task, p = parse_params("why does a==b fail, x=1+1")
        assert p == {"x": "1+1"} and task == "why does a==b fail,"


# ── /rooms dispatcher ────────────────────────────────────────────────────────

class Term:
    def __init__(self, confirms=(True,), choice=None):
        self.out, self.err = [], []
        self.confirms = list(confirms)
        self.asked = []
        self.choice = choice
        self.choices_seen = None

    def io(self, human=None, ask_text=None):
        def confirm(q):
            self.asked.append(q)
            return self.confirms.pop(0) if len(self.confirms) > 1 else self.confirms[0]

        def choose(q, options):
            self.choices_seen = options
            if isinstance(self.choice, list):          # a script of answers, one per question
                return self.choice.pop(0) if self.choice else None
            return self.choice
        return cli.Io(out=self.out.append, err=self.err.append, confirm=confirm, choose=choose, human=human,
                      confirm_step=lambda n, s: True, ask_text=ask_text)

    @property
    def text(self):
        return "\n".join(self.out + self.err)


@pytest.fixture
def lib(tmp_path):
    return starter_lib(user_dir=tmp_path / "user", stats_path=tmp_path / "stats.json")


class TestCli:
    def test_no_arguments_shows_help(self, lib):
        t = Term()
        cli.handle("", t.io(), library=lib)
        assert "/rooms run" in t.text and "/rooms new" in t.text

    def test_list_show_and_route(self, lib):
        t = Term()
        cli.handle("list", t.io(), library=lib)
        for name in ("fix_bug", "debug_loop", "code_review", "triage"):
            assert name in t.text
        t2 = Term()
        cli.handle("show fix_bug", t2.io(), library=lib)
        assert "▶ · reproduce" in t2.text and "вердикт процесса → tests_passed" in t2.text
        assert "Параметры: test_command=" in t2.text and "пока not state.tests_passed" in t2.text
        t3 = Term()
        cli.handle("route почини падающие тесты", t3.io(), library=lib)
        assert "debug_loop" in t3.text and "уверенно" in t3.text
        t4 = Term()
        cli.handle("show nope", t4.io(), library=lib)
        assert "Нет комнаты" in t4.text

    def test_a_free_task_picks_the_room_asks_once_and_reports(self, lib, tmp_path):
        s = Script(exit_codes=[1, 0], agents={"diagnose": {"diagnosis": "d"}, "patch": {"fix_summary": "f"},
                                              "review": {"review_verdict": "ok", "review_notes": "n"}})
        t = Term(confirms=[True])
        cli.handle("у меня баг: после входа вылетает ошибка", t.io(), library=lib, runner=s.runner(),
                   journal=None, record_dir=tmp_path / "runs")
        assert len(t.asked) == 1 and "fix_bug" in t.asked[0]
        assert "Итог: выполнено" in t.text and "fix_bug:fixed" in t.text
        assert history.load_run("last", tmp_path / "runs")["start_room"] == "fix_bug"

    def test_declining_runs_nothing(self, lib, tmp_path):
        s = Script()
        t = Term(confirms=[False])
        cli.handle("run release_check", t.io(), library=lib, runner=s.runner(), record_dir=tmp_path / "runs")
        assert s.commands == [] and "Отменено" in t.text

    def test_unclear_task_lets_the_person_choose(self, lib, tmp_path):
        s = Script(exit_codes=[0], agents={"investigate": {"plan_ready": True, "plan": "p"}})
        t = Term(confirms=[True], choice="analyze")
        cli.handle("сделай что-нибудь полезное", t.io(), library=lib, runner=s.runner(), record_dir=tmp_path / "r")
        assert "analyze" in t.choices_seen and "Итог" in t.text
        t2 = Term(choice=None)
        cli.handle("сделай что-нибудь полезное", t2.io(), library=lib, runner=s.runner(), record_dir=tmp_path / "r")
        assert "Отменено" in t2.text

    def test_run_takes_parameters(self, lib, tmp_path):
        s = Script(exit_codes=[0], agents={"summarize": {"notes": "n"}})
        t = Term()
        cli.handle('run release_check test_command="pytest -q tests/a"', t.io(), library=lib, runner=s.runner(),
                   record_dir=tmp_path / "r")
        assert s.commands[0] == "pytest -q tests/a"

    def test_quarantined_room_is_refused_until_restored(self, lib, tmp_path):
        lib.sources["release_check"] = "user"
        for _ in range(3):
            lib.record_run("release_check", success=False)
        s = Script()
        t = Term()
        cli.handle("run release_check", t.io(), library=lib, runner=s.runner(), record_dir=tmp_path / "r")
        assert "карантин" in t.text and s.commands == []
        t2 = Term()
        cli.handle("unquarantine release_check", t2.io(), library=lib)
        assert "возвращена" in t2.text and not lib.stats["release_check"]["quarantined"]

    def test_new_validate_delete(self, lib, tmp_path):
        t = Term()
        cli.handle("new weekly_report", t.io(), library=lib)
        f = tmp_path / "user" / "weekly_report.json"
        assert f.exists() and lib.get("weekly_report") is not None
        t2 = Term()
        cli.handle(f"validate {f}", t2.io(), library=lib)
        assert "корректна" in t2.text
        t3 = Term()
        cli.handle("new weekly_report", t3.io(), library=lib)
        assert "уже есть" in t3.text
        t4 = Term()
        cli.handle("new Bad Name", t4.io(), library=lib)
        assert "Имя комнаты" in t4.text
        t5 = Term(confirms=[True])
        cli.handle("delete weekly_report", t5.io(), library=lib)
        assert not f.exists() and lib.get("weekly_report") is None
        t6 = Term()
        cli.handle("delete fix_bug", t6.io(), library=lib)
        assert "только свои" in t6.text and lib.get("fix_bug") is not None

    def test_validate_reports_problems(self, lib, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({
            "room": "bad", "budget": {"max_iterations": 3, "max_tokens": 10},
            "nodes": [{"id": "n", "type": "tool", "tool": "no_such_tool"}],
            "edges": [{"from": "n", "to": "EXIT:done"}]}), encoding="utf-8")
        t = Term()
        cli.handle(f"validate {bad}", t.io(), library=lib)
        assert "no_such_tool" in t.text

    def test_history_last_and_resume_with_nothing_to_resume(self, lib, tmp_path):
        rd = tmp_path / "runs"
        t0 = Term()
        cli.handle("history", t0.io(), library=lib, record_dir=rd)
        assert "ещё не было" in t0.text
        s = Script(exit_codes=[0], agents={"summarize": {"notes": "n"}})
        cli.handle("run release_check", Term().io(), library=lib, runner=s.runner(), record_dir=rd)
        t = Term()
        cli.handle("history", t.io(), library=lib, record_dir=rd)
        assert "release_check:ready" in t.text
        t2 = Term()
        cli.handle("last", t2.io(), library=lib, record_dir=rd)
        assert "Маршрут: release_check:ready" in t2.text and "tests" in t2.text
        from src.rooms.journal import Journal
        t3 = Term()
        cli.handle("resume", t3.io(), library=lib, journal=Journal(tmp_path / "empty.jsonl"))
        assert "Нет прерванного" in t3.text

    def test_a_room_that_failed_to_load_is_reported_but_the_rest_works(self, tmp_path):
        udir = tmp_path / "user"
        udir.mkdir()
        (udir / "broken.json").write_text('{"room": "broken"}', encoding="utf-8")
        lib = starter_lib(user_dir=udir)
        lib.load_dir(udir, STARTER_TOOLS, origin="user")
        t = Term()
        cli.handle("list", t.io(), library=lib)
        assert "broken" in t.text and "fix_bug" in t.text


# ── the tool ─────────────────────────────────────────────────────────────────

class TestRoomsTool:
    def test_list_and_show(self, lib):
        out = rooms_action("list", library=lib)
        assert "fix_bug" in out and "params: test_command" in out
        assert "implement_feature" in rooms_action("show", "implement_feature", library=lib)
        assert rooms_action("show", "nope", library=lib).startswith("Error: no room")

    def test_run_is_refused_when_the_person_says_no(self, lib, tmp_path):
        s = Script()
        out = rooms_action("run", "release_check", "check", library=lib, runner=s.runner(),
                           ask_run=lambda r, t, p: False, record_dir=tmp_path)
        assert "did NOT allow" in out and s.commands == []

    def test_run_executes_and_returns_the_report(self, lib, tmp_path):
        s = Script(exit_codes=[0], agents={"summarize": {"notes": "ready to go"}})
        seen = {}
        out = rooms_action("run", "release_check", "check", {"test_command": "make test"}, library=lib,
                           runner=s.runner(), ask_run=lambda r, t, p: seen.update(room=r.room, params=p) or True,
                           record_dir=tmp_path, journal=None)
        assert seen == {"room": "release_check", "params": {"test_command": "make test"}}
        assert "ready to go" in out and s.commands[0] == "make test"

    def test_run_without_a_room_needs_a_clear_pick(self, lib, tmp_path):
        out = rooms_action("run", "", "fix the failing tests", library=lib, runner=Script().runner(),
                           ask_run=lambda r, t, p: False, record_dir=tmp_path)
        assert "did NOT allow" in out                       # debug_loop was picked, the person refused
        out2 = rooms_action("run", "", "tell me a joke", library=lib, ask_run=lambda r, t, p: True)
        assert out2.startswith("Error: name the room")

    def test_quarantined_and_unknown(self, lib):
        lib.sources["research"] = "user"
        for _ in range(3):
            lib.record_run("research", success=False)
        assert "quarantined" in rooms_action("run", "research", "q", library=lib, ask_run=lambda *a: True)
        assert rooms_action("run", "zzz", "q", library=lib).startswith("Error: no room")
        assert rooms_action("fly", library=lib).startswith("Error: action must be")

    def test_params_may_arrive_as_a_json_string(self, lib, tmp_path):
        s = Script(exit_codes=[0], agents={"summarize": {"notes": "n"}})
        rooms_action("run", "release_check", "", '{"test_command": "tox"}', library=lib, runner=s.runner(),
                     ask_run=lambda *a: True, record_dir=tmp_path)
        assert s.commands[0] == "tox"

    def test_plan_mode_blocks_run_but_not_list(self):
        from src.agent.plan_mode import PLAN, violation
        PLAN.enter()
        try:
            assert violation("rooms", {"action": "run"}) is not None
            assert violation("rooms", {"action": "resume"}) is not None
            assert violation("rooms", {"action": "list"}) is None
        finally:
            PLAN.leave()

    def test_workers_of_a_room_never_get_the_rooms_tool(self):
        from src.agent.subagents import FORBIDDEN_TOOLS
        assert "rooms" in FORBIDDEN_TOOLS

    def test_the_tool_is_registered_with_a_schema(self):
        from tools.lazy_registry import DEFAULT_TOOL_LOADERS
        from tools.schemas import TOOL_SCHEMAS
        assert DEFAULT_TOOL_LOADERS["rooms"] == ("tools.rooms_tools", "rooms")
        entry = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "rooms"]
        assert len(entry) == 1 and "action" in entry[0]["function"]["parameters"]["required"]


# ── triage, interrupted runs ─────────────────────────────────────────────────

class TestTriageAndJournal:
    def test_triage_is_told_which_rooms_exist_and_a_route_is_used_once(self):
        prompts = []

        class Agent(Script):
            def agent(self, node, context):
                return {}

        s = Script(exit_codes=[1], git_status="")
        seen = {}

        def agent(node, context):
            s.agent_calls.append(node.id)
            if node.id == "route":
                seen["instruction"] = node.instruction
                return {"route_to": "release_check"}
            if node.id == "summarize":
                return {"notes": "not ready"}
            return {}
        runner = NodeRunner(tool_executor=s.tool, agent_executor=agent, confirm=lambda n, st: True)
        res = run_rooms("flaky bug", start="fix_bug", library=starter_lib(), runner=runner)
        assert [h for h in res.history] == [("fix_bug", "escalate"), ("triage", "routed"),
                                           ("release_check", "not_ready")]
        assert "- fix_bug:" in seen["instruction"] and "- triage:" not in seen["instruction"]
        assert "flaky bug" in seen["instruction"]
        assert "route_to" not in res.state.data           # consumed by the supervisor

    def test_a_new_run_asks_before_erasing_an_interrupted_one(self, lib, tmp_path):
        from src.rooms.journal import Journal
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent", "room": "debug_loop", "node": "apply_fix", "state": {"data": {}, "loops": {}}})
        assert j.unfinished() is True
        s = Script()
        t = Term(confirms=[False])
        cli.handle("run release_check", t.io(), library=lib, runner=s.runner(), journal=j,
                   record_dir=tmp_path / "runs")
        assert "прерванный прогон" in t.asked[0] and s.commands == [] and j.last()["kind"] == "intent"

    def test_a_finished_journal_is_not_unfinished(self, tmp_path):
        from src.rooms.journal import Journal
        j = Journal(tmp_path / "j.jsonl")
        assert j.unfinished() is False
        j.append({"kind": "commit", "room": "x", "node": "n", "next": None, "exit": "done",
                  "state": {"data": {}, "loops": {}}})
        assert j.unfinished() is False
        j.append({"kind": "commit", "room": "x", "node": "n", "next": "m", "exit": None,
                  "state": {"data": {}, "loops": {}}})
        assert j.unfinished() is True


# ── the nudge toward a room ──────────────────────────────────────────────────

class TestRoomHint:
    def test_a_clear_match_adds_one_block(self):
        from src.rooms.hint import room_hint
        msgs = [{"role": "user", "content": "почини падающие тесты в проекте"}]
        text = room_hint(msgs, library=starter_lib())
        assert text and "debug_loop" in text and 'rooms(action="run"' in text

    def test_nothing_for_chat_short_text_commands_or_the_unrelated(self):
        from src.rooms.hint import room_hint
        lib = starter_lib()
        for content in ("привет, как дела?", "тесты", "/rooms list", "какая сегодня погода в Париже"):
            assert room_hint([{"role": "user", "content": content}], library=lib) is None, content

    def test_uses_the_last_user_message_even_after_tool_results(self):
        from src.rooms.hint import last_user_text
        msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
                {"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image"}]},
                {"role": "tool", "content": "result"}]
        assert last_user_text(msgs) == "second"

    def test_switch_off(self, monkeypatch):
        import config
        from src.rooms.hint import room_hint
        monkeypatch.setattr(config, "get_rooms_hint", lambda: False)
        assert room_hint([{"role": "user", "content": "почини падающие тесты в проекте"}],
                         library=starter_lib()) is None

    def test_command_switches_it(self, lib, monkeypatch):
        import config
        state = {"on": True}
        monkeypatch.setattr(config, "get_rooms_hint", lambda: state["on"])
        monkeypatch.setattr(config, "set_rooms_hint", lambda v: state.update(on=v))
        t = Term()
        cli.handle("hint off", t.io(), library=lib)
        assert state["on"] is False and "OFF" in t.text


class TestCommandWordsInTasks:
    def test_a_task_that_starts_with_a_command_word_is_still_a_task(self, lib, tmp_path):
        from src.rooms.cli import _is_command
        assert _is_command("list", "") and not _is_command("list", "all python files and explain them")
        assert _is_command("last", "") and _is_command("last", "2026") and not _is_command("last", "run of tests")
        assert _is_command("hint", "off") and not _is_command("hint", "me how to fix tests")
        assert _is_command("show", "fix_bug") and _is_command("run", "debug_loop")
        s = Script(exit_codes=[0], agents={"summarize": {"notes": "n"}})
        t = Term(confirms=[True], choice="release_check")
        cli.handle("list what is still uncommitted before the release", t.io(), library=lib, runner=s.runner(),
                   record_dir=tmp_path / "r")
        assert "release_check" in t.text and "[КАРАНТИН]" not in t.text and t.asked   # it went to the router
