"""Rooms, second generation: enforced budgets, process verdicts (`sets`), templates, confirmation,
human answers that continue the run, room-carried routing, persistent stats, the router, run records.

Nothing here calls a model: executors are fakes, the engine and the starter rooms are real."""

import json

import pytest

from src.rooms import history, router, template
from src.rooms.engine import Engine
from src.rooms.library import RoomLibrary, default_starter_dir
from src.rooms.models import Budget, Node, Room, State
from src.rooms.runner import (
    AgentExecutor, NodeRunner, _call_with, _consume_bounded, derive_result_values, estimate_tokens,
    is_affirmative,
)
from src.rooms.supervisor import Supervisor
from src.rooms.validator import validate_room
from tests.test_rooms_library import STARTER_TOOLS


def room(nodes, edges, **kw):
    kw.setdefault("budget", {"max_iterations": 20, "max_tokens": 1000})
    return Room(room="r", nodes=nodes, edges=edges, **kw)


AGENT = {"id": "a", "type": "agent", "budget": {"max_iterations": 2, "max_tokens": 100}}


# ── schema ─────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_sets_only_on_tool_nodes_and_only_known_values(self):
        Node(id="t", type="tool", tool="run_command", sets={"tests_passed": "ok"})
        with pytest.raises(ValueError):
            Node(id="c", type="condition", condition="state.x", sets={"a": "ok"})
        with pytest.raises(ValueError):
            Node(id="t", type="tool", tool="run_command", sets={"a": "vibes"})

    def test_confirm_role_tools_into_are_type_bound(self):
        with pytest.raises(ValueError):
            Node(**AGENT, confirm=True)
        with pytest.raises(ValueError):
            Node(id="t", type="tool", tool="x", role="Coder")
        with pytest.raises(ValueError):
            Node(id="t", type="tool", tool="x", into="go")
        Node(id="h", type="human_pause", into="go")
        Node(**AGENT, role="Reviewer", tools=["read_file"])

    def test_budget_seconds_optional_and_positive(self):
        assert Budget(max_iterations=1, max_tokens=1).max_seconds is None
        with pytest.raises(ValueError):
            Budget(max_iterations=1, max_tokens=1, max_seconds=0)

    def test_old_room_json_still_loads(self):
        r = Room(**{"room": "old", "budget": {"max_iterations": 1, "max_tokens": 1},
                    "nodes": [{"id": "n", "type": "condition", "condition": "state.x"}]})
        assert r.defaults == {} and r.next == {} and r.tags == [] and r.report is None


# ── templates ──────────────────────────────────────────────────────────────────

class TestTemplate:
    def test_fills_fields_and_unknown_becomes_empty(self):
        d = {"task": "fix login", "n.output": "boom"}
        assert template.render("Do: {{task}} / {{ n.output }} / [{{nope}}]", d) == "Do: fix login / boom / []"
        assert template.render("{{state.task}}", d) == "fix login"

    def test_no_evaluation_or_attribute_access(self):
        d = {"task": "x"}
        text = "{{ __import__('os') }} {{task|upper}}"
        assert template.render(text, d) == text          # not placeholders: left alone, never evaluated
        assert template.render("[{{task.__class__}}]", d) == "[]"   # a plain key lookup that finds nothing

    def test_long_value_is_cut_in_the_middle(self):
        out = template.render("{{log}}", {"log": "A" * 5000 + "Z" * 5000}, limit=100)
        assert out.startswith("A") and out.endswith("Z") and "пропущено" in out and len(out) < 300

    def test_nested_values(self):
        assert template.render_value({"cmd": ["run", "{{x}}"], "n": 3}, {"x": "1"}) == {"cmd": ["run", "1"], "n": 3}


# ── runner: process verdicts, confirmation, templates, answers ────────────────

class TestDerive:
    def test_exit_code_zero_is_ok_and_tail_is_empty(self):
        v = derive_result_values("Exit code: 0\nOUTPUT:\n12 passed")
        assert v["ok"] is True and v["exit_code"] == 0 and v["tail"] == "" and v["body"] == "12 passed"

    def test_nonzero_is_failure_with_tail(self):
        out = "Exit code: 1\nOUTPUT:\n" + "\n".join(f"line{i}" for i in range(100))
        v = derive_result_values(out)
        assert v["ok"] is False and v["exit_code"] == 1
        assert v["tail"].splitlines()[-1] == "line99" and len(v["tail"].splitlines()) <= 40

    @pytest.mark.parametrize("text", ["Error running tool 'x': boom", "Execution aborted by user.",
                                      "Command BLOCKED by the risk-gate"])
    def test_refusals_and_errors_are_not_ok(self, text):
        assert derive_result_values(text)["ok"] is False

    def test_plain_text_without_exit_line_is_ok(self):
        assert derive_result_values("here is the diff")["ok"] is True

    def test_empty_output_body_is_empty(self):
        assert derive_result_values("Exit code: 0")["body"] == ""


class TestRunnerNewFeatures:
    def _runner(self, **kw):
        return NodeRunner(tool_executor=lambda n, s: kw.pop("result", "Exit code: 0\nOUTPUT:\nok"), **kw)

    def test_sets_fill_state_from_the_process(self):
        node = Node(id="t", type="tool", tool="run_command",
                    sets={"passed": "ok", "why": "tail", "code": "exit_code"})
        st = State()
        NodeRunner(tool_executor=lambda n, s: "Exit code: 2\nOUTPUT:\nFAILED x").run_node(node, st)
        assert st.data["passed"] is False and st.data["code"] == 2 and st.data["why"] == "FAILED x"

    def test_confirm_node_does_not_run_without_anyone_to_ask(self):
        calls = []
        node = Node(id="c", type="tool", tool="run_command", confirm=True, sets={"done": "ok"})
        r = NodeRunner(tool_executor=lambda n, s: calls.append(1) or "Exit code: 0")
        st = State()
        r.run_node(node, st)
        assert calls == [] and st.data["done"] is False and st.data["c.declined"] is True

    def test_confirm_node_runs_when_approved_and_not_when_refused(self):
        calls = []
        node = Node(id="c", type="tool", tool="run_command", confirm=True, sets={"done": "ok"})
        ex = lambda n, s: calls.append(1) or "Exit code: 0"
        st = State()
        NodeRunner(tool_executor=ex, confirm=lambda n, s: True).run_node(node, st)
        assert calls == [1] and st.data["done"] is True
        st2 = State()
        NodeRunner(tool_executor=ex, confirm=lambda n, s: False).run_node(node, st2)
        assert calls == [1] and st2.data["done"] is False

    def test_a_confirm_callback_that_raises_means_no(self):
        def boom(n, s):
            raise RuntimeError("no terminal")
        st = State()
        NodeRunner(tool_executor=lambda n, s: "Exit code: 0", confirm=boom).run_node(
            Node(id="c", type="tool", tool="x", confirm=True), st)
        assert st.data["c.declined"] is True

    def test_placeholders_reach_the_executor_but_the_room_is_untouched(self):
        seen = {}
        node = Node(id="t", type="tool", tool="run_command", args={"command": "pytest {{path}}"})
        NodeRunner(tool_executor=lambda n, s: seen.update(args=n.args) or "ok").run_node(
            node, State(data={"path": "tests/unit"}))
        assert seen["args"] == {"command": "pytest tests/unit"}
        assert node.args == {"command": "pytest {{path}}"}

    def test_agent_instruction_is_rendered_and_tokens_are_counted(self):
        prompts = []

        def run_agent(prompt, max_iterations=None):
            prompts.append(prompt)
            return '{"x": 1}'
        ex = AgentExecutor(run_agent)
        node = Node(id="a", type="agent", budget={"max_iterations": 2, "max_tokens": 100},
                    instruction="Fix: {{task}}", writes=["x"])
        st = State(data={"task": "the login"})
        NodeRunner(agent_executor=ex).run_node(node, st)
        assert "Fix: the login" in prompts[0] and st.data["x"] == 1
        assert st.usage["tokens"] > 0

    def test_agent_text_tokens_beat_the_estimate(self):
        text = _consume_bounded(iter([{"type": "content", "content": "abc" * 10},
                                      {"type": "tool_end", "result": "r" * 300}]))
        assert text.tokens == estimate_tokens("abc" * 10) + estimate_tokens("r" * 300)
        ex = AgentExecutor(lambda prompt, max_iterations=None: text)
        ex(Node(id="a", type="agent", budget={"max_iterations": 1, "max_tokens": 9}, writes=[]), {})
        assert ex.last_tokens == text.tokens

    def test_node_token_budget_stops_the_loop(self):
        chunks = [{"type": "content", "content": "x" * 30}] * 5 + [{"type": "tool_end", "result": ""}]
        out = _consume_bounded(iter(chunks), max_iterations=9, max_tokens=25)
        assert "token budget (25)" in out and out.tokens >= 25

    def test_role_and_tools_are_passed_only_to_runners_that_take_them(self):
        got = {}

        def new_style(prompt, max_iterations=None, tools=None, role=None, max_tokens=None):
            got.update(tools=tools, role=role, max_tokens=max_tokens)
            return "{}"
        node = Node(id="a", type="agent", budget={"max_iterations": 2, "max_tokens": 77},
                    role="Reviewer", tools=["read_file"], writes=[])
        AgentExecutor(new_style)(node, {})
        assert got == {"tools": ["read_file"], "role": "Reviewer", "max_tokens": 77}
        AgentExecutor(lambda prompt, max_iterations=None: "{}")(node, {})     # the old shape still works

    def test_call_with_passes_everything_to_var_keyword_runners(self):
        assert _call_with(lambda p, **kw: sorted(kw), "p", a=1, b=2) == ["a", "b"]

    def test_human_answer_goes_into_a_field_with_an_approval_flag(self):
        node = Node(id="ask", type="human_pause", into="go")

        def human(n, s):
            s.data["ask.output"] = "Да!"
        st = State()
        NodeRunner(human_prompt=human).run_node(node, st)
        assert st.data["go"] == "Да!" and st.data["go_approved"] is True

    @pytest.mark.parametrize("answer,yes", [("да", True), ("Yes", True), ("ok.", True), ("нет", False),
                                            ("да, но поменяй X", False), ("", False), (None, False)])
    def test_is_affirmative(self, answer, yes):
        assert is_affirmative(answer) is yes


# ── engine: budgets, answered pauses, events ──────────────────────────────────

class _Agent:
    """An agent executor that spends a fixed number of tokens per call."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.last_tokens = 0
        self.calls = 0

    def __call__(self, node, context):
        self.calls += 1
        self.last_tokens = self.tokens
        return {}


class TestEngineBudgets:
    def test_token_budget_ends_the_room_with_budget(self):
        r = room([{**AGENT, "id": "a"}, {**AGENT, "id": "b"}, {**AGENT, "id": "c"}],
                 [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}, {"from": "c", "to": "EXIT:done"}],
                 budget={"max_iterations": 20, "max_tokens": 1000})
        agent = _Agent(600)
        out = Engine().run(r, State(), NodeRunner(agent_executor=agent).run_node)
        assert out.exit == "budget" and "token" in out.reason
        assert agent.calls == 2       # the check runs before the third node, after 1200 >= 1000

    def test_budget_counts_only_this_rooms_tokens(self):
        r = room([{**AGENT, "id": "a"}], [{"from": "a", "to": "EXIT:done"}],
                 budget={"max_iterations": 5, "max_tokens": 1000})
        st = State(usage={"tokens": 900_000})        # spent by earlier rooms of the run
        out = Engine().run(r, st, NodeRunner(agent_executor=_Agent(10)).run_node)
        assert out.exit == "done"

    def test_time_budget(self):
        ticks = iter(range(0, 1000, 30))
        r = room([{"id": "a", "type": "condition", "condition": "state.x"},
                  {"id": "b", "type": "condition", "condition": "state.x"}],
                 [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
                 budget={"max_iterations": 99, "max_tokens": 10, "max_seconds": 100})
        # 'a' <-> 'b' with no exit would be rejected by the validator; the engine's clock still bounds it
        out = Engine().run(r, State(), lambda n, s: None, clock=lambda: next(ticks))
        assert out.exit == "budget" and "time" in out.reason

    def test_answered_human_pause_continues_along_its_edges(self):
        r = room([{"id": "ask", "type": "human_pause", "into": "go"}, {"id": "w", "type": "condition", "condition": "state.x"}],
                 [{"from": "ask", "if": "state.go_approved", "to": "EXIT:yes"},
                  {"from": "ask", "else": True, "to": "EXIT:no"}])
        answering = NodeRunner(human_prompt=lambda n, s: s.data.update({"ask.output": "да"}))
        assert Engine().run(r, State(), answering.run_node).exit == "yes"
        refusing = NodeRunner(human_prompt=lambda n, s: s.data.update({"ask.output": "нет, иначе"}))
        assert Engine().run(r, State(), refusing.run_node).exit == "no"

    def test_unanswered_human_pause_still_pauses(self):
        r = room([{"id": "ask", "type": "human_pause"}], [{"from": "ask", "to": "EXIT:done"}])
        assert Engine().run(r, State(), NodeRunner().run_node).exit == "human_pause"

    def test_stale_answer_from_an_earlier_visit_is_not_reused(self):
        r = room([{"id": "ask", "type": "human_pause", "into": "go"}],
                 [{"from": "ask", "to": "EXIT:done"}])
        st = State(data={"ask.output": "да"})
        assert Engine().run(r, st, NodeRunner().run_node).exit == "human_pause"

    def test_events_describe_the_run_and_a_broken_listener_changes_nothing(self):
        events = []
        r = room([{"id": "c", "type": "condition", "condition": "state.x"}], [{"from": "c", "to": "EXIT:done"}])
        out = Engine().run(r, State(), lambda n, s: None, on_event=events.append)
        assert [e["type"] for e in events] == ["node_start", "node_end", "room_end"]
        assert events[-1]["exit"] == "done" and out.exit == "done"

        def bad(e):
            raise RuntimeError("renderer died")
        assert Engine().run(r, State(), lambda n, s: None, on_event=bad).exit == "done"


# ── supervisor: rooms carry their own routing and defaults ────────────────────

class TestSupervisorFromRoomData:
    def _lib(self):
        lib = RoomLibrary()
        a = room([{"id": "n", "type": "condition", "condition": "state.x"}], [{"from": "n", "to": "EXIT:planned"}],
                 next={"planned": "b"}, defaults={"test_command": "pytest -q"})
        b = Room(room="b", budget={"max_iterations": 3, "max_tokens": 10},
                 nodes=[{"id": "m", "type": "condition", "condition": "state.x"}],
                 edges=[{"from": "m", "to": "EXIT:done"}], next={"done": "END"})
        a.room = "a"
        lib.rooms.update({"a": a, "b": b})
        lib.stats.update({"a": {"successes": 0, "failures": 0, "quarantined": False},
                          "b": {"successes": 0, "failures": 0, "quarantined": False}})
        return lib

    def test_next_chains_rooms_and_end_ends(self):
        lib = self._lib()
        eng = Engine()
        sup = Supervisor(lib, lambda rm, st: eng.run(rm, st, lambda n, s: None))
        res = sup.run("a")
        assert res.outcome == "done" and res.history == [("a", "planned"), ("b", "done")]

    def test_defaults_fill_missing_parameters_only(self):
        lib = self._lib()
        eng = Engine()
        sup = Supervisor(lib, lambda rm, st: eng.run(rm, st, lambda n, s: None))
        st = State(data={})
        sup.run("a", st)
        assert st.data["test_command"] == "pytest -q"
        st2 = State(data={"test_command": "npm test"})
        sup.run("a", st2)
        assert st2.data["test_command"] == "npm test"

    def test_explicit_routes_beat_the_rooms_next(self):
        lib = self._lib()
        eng = Engine()
        sup = Supervisor(lib, lambda rm, st: eng.run(rm, st, lambda n, s: None), routes={("a", "planned"): "END:x"})
        assert sup.run("a").history == [("a", "planned")]


# ── library: persistent statistics ────────────────────────────────────────────

class TestPersistentStats:
    def _load(self, path):
        return RoomLibrary(stats_path=path).load_dir(default_starter_dir(), STARTER_TOOLS)

    def test_counters_and_quarantine_survive_a_new_library(self, tmp_path):
        p = tmp_path / "stats.json"
        lib = self._load(p)
        lib.sources["debug_loop"] = "user"       # the guard is for rooms people and the model add
        for _ in range(3):
            lib.record_run("debug_loop", success=False)
        assert lib.stats["debug_loop"]["quarantined"] is True
        again = self._load(p)
        assert again.stats["debug_loop"] == {"successes": 0, "failures": 3, "quarantined": True}
        assert "debug_loop" not in again.active_rooms()

    def test_starter_rooms_are_counted_but_never_quarantined(self, tmp_path):
        lib = self._load(tmp_path / "stats.json")
        for _ in range(6):
            lib.record_run("debug_loop", success=False)    # the project's tests were red, not the room
        assert lib.stats["debug_loop"] == {"successes": 0, "failures": 6, "quarantined": False}
        assert "debug_loop" in lib.active_rooms()

    def test_unquarantine_restores_and_persists(self, tmp_path):
        p = tmp_path / "stats.json"
        lib = self._load(p)
        lib.sources["analyze"] = "user"
        for _ in range(3):
            lib.record_run("analyze", success=False)
        assert lib.stats["analyze"]["quarantined"] is True
        assert lib.unquarantine("analyze") is True
        assert self._load(p).stats["analyze"]["quarantined"] is False
        assert lib.unquarantine("nope") is False

    def test_damaged_stats_file_is_ignored(self, tmp_path):
        p = tmp_path / "stats.json"
        p.write_text("{not json", encoding="utf-8")
        lib = self._load(p)
        assert lib.stats["analyze"] == {"successes": 0, "failures": 0, "quarantined": False}

    def test_without_a_path_nothing_is_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lib = RoomLibrary().load_dir(default_starter_dir(), STARTER_TOOLS)
        lib.record_run("analyze", success=True)
        assert list(tmp_path.iterdir()) == []


# ── validator ─────────────────────────────────────────────────────────────────

class TestValidatorNew:
    def test_unknown_role_unknown_tool_and_bad_next(self):
        r = room([{**AGENT, "role": "Wizard", "tools": ["read_file", "teleport"]}],
                 [{"from": "a", "to": "EXIT:done"}], next={"typo": "END", "done": "END"})
        errs = validate_room(r, {"read_file"})
        text = " | ".join(errs)
        assert "unknown role 'Wizard'" in text and "'teleport'" in text and "'typo' is not an exit" in text
        assert "'done' is not an exit" not in text

    def test_engine_exits_may_be_routed(self):
        r = room([{**AGENT}], [{"from": "a", "to": "EXIT:done"}], next={"budget": "TRIAGE", "stuck": "TRIAGE"})
        assert validate_room(r, set()) == []


# ── router ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def starter():
    return RoomLibrary().load_dir(default_starter_dir(), STARTER_TOOLS)


class TestRouter:
    @pytest.mark.parametrize("task,expected", [
        ("почини падающие тесты", "debug_loop"),
        ("tests are failing, please fix them", "debug_loop"),
        ("сделай ревью моих изменений", "code_review"),
        ("review my diff before the PR", "code_review"),
        ("добавь функцию экспорта в CSV", "implement_feature"),
        ("implement a new feature: dark theme", "implement_feature"),
        ("проверь, что всё готово к релизу", "release_check"),
        ("объясни, как устроен этот проект", "explain_repo"),
        ("исследуй, чем отличается asyncio от trio", "research"),
        ("у меня баг: после входа вылетает ошибка", "fix_bug"),
    ])
    def test_picks_the_room_from_the_persons_words(self, starter, task, expected):
        pick = router.pick_room(task, starter)
        assert pick.name == expected, pick.ranked

    def test_unrelated_task_matches_nothing(self, starter):
        pick = router.pick_room("какая сегодня погода в Париже", starter)
        assert pick.clear is False

    def test_triage_is_never_picked(self, starter):
        assert all(n != "triage" for n, _, _ in router.pick_room("route failure triage", starter).ranked)

    def test_quarantined_room_is_not_offered(self):
        lib = RoomLibrary().load_dir(default_starter_dir(), STARTER_TOOLS)
        lib.sources["debug_loop"] = "user"
        for _ in range(3):
            lib.record_run("debug_loop", success=False)
        assert all(n != "debug_loop" for n, _, _ in router.pick_room("почини падающие тесты", lib).ranked)

    def test_inflections_match(self):
        assert router._same_word("тесты", "тестов") and router._same_word("баги", "баг")
        assert not router._same_word("проект", "проверь") and not router._same_word("ab", "abc")


# ── history / report ──────────────────────────────────────────────────────────

class _Result:
    def __init__(self, outcome, history_, state):
        self.outcome, self.history, self.state = outcome, history_, state


class TestHistory:
    def test_recorder_writes_a_record_and_prunes(self, tmp_path):
        rec = history.RunRecorder("fix it", "debug_loop", directory=tmp_path, params={"test_command": "pytest"})
        rec.on_event({"type": "node_end", "room": "debug_loop", "node": "run_tests", "kind": "tool",
                      "seconds": 1.5, "tokens": 0})
        st = State(data={"tests_passed": True, "big": "x" * 5000}, usage={"tokens": 321})
        path = rec.finish(_Result("done", [("debug_loop", "success")], st))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["outcome"] == "done" and data["route"] == [["debug_loop", "success"]]
        assert data["nodes"][0]["node"] == "run_tests" and data["usage"] == {"tokens": 321}
        assert len(data["state"]["big"]) < 2000          # long values are cut
        assert history.load_run("last", tmp_path)["task"] == "fix it"
        assert [r["id"] for r in history.list_runs(tmp_path)] == [data["id"]]

    def test_two_runs_in_one_second_do_not_overwrite(self, tmp_path):
        for _ in range(2):
            history.RunRecorder("t", "analyze", directory=tmp_path, clock=lambda: 1_700_000_000.0).finish(
                _Result("done", [], State()))
        assert len(list(tmp_path.glob("*.json"))) == 2

    def test_report_uses_the_rooms_template_and_shows_declined_steps(self, starter):
        st = State(data={"tests_passed": True, "fix_summary": "patched parser", "committed": False,
                         "commit.declined": True}, usage={"tokens": 1500})
        text = history.format_report(_Result("done", [("debug_loop", "success")], st), starter, "fix", seconds=42)
        assert "выполнено" in text and "Тесты: True" in text and "patched parser" in text
        assert "commit" in text and "≈1500 ток" in text and "42 с" in text

    def test_report_without_template_lists_what_the_rooms_wrote(self):
        lib = RoomLibrary()
        r = room([{**AGENT, "writes": ["finding"]}], [{"from": "a", "to": "EXIT:done"}])
        lib.rooms["r"] = r
        st = State(data={"finding": "cache is stale", "a.output": "RAW NOISE", "task": "secret task"})
        text = history.format_report(_Result("escalated", [("r", "done")], st), lib)
        assert "cache is stale" in text and "RAW NOISE" not in text and "нужен человек" in text


# ── resume: which interrupted nodes may be replayed ──────────────────────────

class TestReplayable:
    def _status(self, tmp_path, node):
        from src.rooms.engine import plan_resume
        from src.rooms.journal import Journal
        r = Room(room="r", budget={"max_iterations": 5, "max_tokens": 10}, nodes=[node],
                 edges=[{"from": node["id"], "to": "EXIT:done"}])
        j = Journal(tmp_path / "j.jsonl")
        j.append({"kind": "intent", "room": "r", "node": node["id"], "state": {"data": {}, "loops": {}}})
        return plan_resume(r, j)[2]

    def test_a_read_only_agent_is_replayed(self, tmp_path):
        node = {**AGENT, "role": "Researcher", "tools": ["read_file", "grep_search"]}
        assert self._status(tmp_path, node) == "resume"

    def test_an_agent_that_may_write_goes_to_the_person(self, tmp_path):
        assert self._status(tmp_path, {**AGENT}) == "indeterminate"                       # all tools
        assert self._status(tmp_path, {**AGENT, "tools": ["read_file", "write_file"]}) == "indeterminate"
        assert self._status(tmp_path, {**AGENT, "tools": ["run_command"]}) == "indeterminate"

    def test_tool_nodes_follow_their_idempotent_flag(self, tmp_path):
        assert self._status(tmp_path, {"id": "t", "type": "tool", "tool": "x"}) == "resume"
        assert self._status(tmp_path, {"id": "t", "type": "tool", "tool": "x", "idempotent": False}) == "indeterminate"
