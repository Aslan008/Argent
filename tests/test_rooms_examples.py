"""The shipped editor examples are not just valid — they behave the way their summary says."""

import json
from pathlib import Path

import pytest

from src.rooms.engine import Engine
from src.rooms.integrations import Integrations
from src.rooms.models import Room, State
from src.rooms.runner import NodeRunner

EXAMPLES = Path(__file__).resolve().parents[1] / "src" / "rooms" / "editor" / "examples.json"
TOOLS = {"run_command", "browser_open", "read_file", "search_web"}


def example(name: str) -> Room:
    for ex in json.loads(EXAMPLES.read_text(encoding="utf-8")):
        if ex["name"] == name:
            return Room(**ex["room"])
    raise AssertionError(name)


def run(room: Room, runner: NodeRunner, data=None):
    return Engine().run(room, State(data=dict(data or {})), runner.run_node)


class Researcher:
    """An agent that finds the answer on round `finds_on` (never when None)."""
    accepts_brief = True

    def __init__(self, finds_on=None):
        self.finds_on = finds_on
        self.calls = []
        self.last_tokens = 0
        self.last_text = ""

    def __call__(self, node, context, brief=None):
        self.calls.append((dict(context), brief))
        found = self.finds_on is not None and len(self.calls) >= self.finds_on
        return {"answer": "42"} if found else {}


class TestLoopWithHint:
    def test_gives_up_after_five_rounds_and_nudges_from_the_fourth_round(self):
        agent = Researcher()
        out = run(example("ex_loop_with_hint"), NodeRunner(agent_executor=agent), {"task": "find it"})
        assert out.exit == "gave_up"
        assert out.state.data["try_visits"] == 5
        assert out.state.data["rounds_capped"] is False
        hints = [b["hints"] for _, b in agent.calls if b and b.get("hints")]
        assert len(hints) == 2                                  # said once before round 4 and once before round 5

    def test_break_leaves_through_found(self):
        agent = Researcher(finds_on=2)
        out = run(example("ex_loop_with_hint"), NodeRunner(agent_executor=agent), {"task": "find it"})
        assert out.exit == "found" and out.state.data["answer"] == "42"
        assert len(agent.calls) == 2

    def test_the_room_is_read_only(self):
        assert example("ex_loop_with_hint").rules.read_only is True


class TestNightRun:
    def test_reports_then_dry_runs_the_shutdown(self):
        sent = []
        ran = []
        runner = NodeRunner(
            tool_executor=lambda n, s: ran.append(n.id) or "5 passed in 1.2s",
            compose=lambda ins, data: "Тесты прошли.",
            notify=lambda n, s: sent.append((n.channel, n.message)) or True,
            integrations=Integrations())            # the shutdown is a dry run: nothing is executed
        out = run(example("ex_night_run"), runner)
        assert out.exit == "finished"
        assert sent == [("telegram", "Тесты прошли.")]
        assert out.state.data["off.output"].startswith("Dry run")

    def test_shutdown_leaves_time_to_cancel(self):
        node = next(n for n in example("ex_night_run").nodes if n.type == "power")
        assert node.seconds >= 10


class TestBookmarks:
    SCAN = {"mode": "yesno", "yes": [
        {"post": 1, "text": "Мой кот", "buttons": [{"index": 4, "label": "Нравится"}, {"index": 5, "label": "В закладки"}]}],
        "unsure": []}

    class FakeLeeth:
        def __init__(self, scan):
            self.scan = scan
            self.clicked = []

        def leeth(self, node, state):
            if node.mode == "scan":
                return json.dumps(self.scan, ensure_ascii=False)
            from src.rooms.integrations import click_indices
            idx, problem = click_indices(node, state)
            self.clicked.append(idx)
            return f"Clicked {idx}" if idx else f"Error: {problem}"

    def test_bookmarks_only_the_matching_posts_and_asks_first(self):
        fake = self.FakeLeeth(self.SCAN)
        asked = []
        runner = NodeRunner(tool_executor=lambda n, s: "opened", integrations=fake,
                            confirm=lambda n, s: asked.append(n.id) or True, notify=lambda n, s: True)
        out = run(example("ex_leeth_bookmarks"), runner)
        assert out.exit == "done"
        assert fake.clicked == [[5]] and asked == ["mark"]

    def test_nothing_found_ends_without_clicking(self):
        fake = self.FakeLeeth({"mode": "yesno", "yes": [], "unsure": []})
        runner = NodeRunner(tool_executor=lambda n, s: "opened", integrations=fake, confirm=lambda n, s: True)
        out = run(example("ex_leeth_bookmarks"), runner)
        assert out.exit == "nothing_found" and fake.clicked == []


# ── the teaching rooms: a branching task, a plan–critic loop and the full pipeline ───────────────

import threading

from src.rooms.subroom import SubRoomHandler
from src.rooms.session import build_library
from src.rooms.validator import validate_room


class Scripted:
    """An agent executor that answers each node with the next item of its script (the last one repeats)."""
    accepts_brief = True

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.briefs = []
        self.last_tokens = 0
        self.last_text = ""
        self._lock = threading.Lock()

    def __call__(self, node, context, brief=None):
        with self._lock:
            self.calls.append(node.id)
            self.briefs.append((node.id, brief))
            seq = self.script.get(node.id, [{}])
            n = self.calls.count(node.id) - 1
        return dict(seq[min(n, len(seq) - 1)])


def fake_tests(*passes):
    """A tool executor: run_command answers pass/fail from `passes` in turn, other tools say ok."""
    seen = []

    def tool(node, state):
        if node.tool == "run_command":
            seen.append(node.id)
            ok = passes[min(len(seen) - 1, len(passes) - 1)]
            return "Exit code: 0\nOUTPUT:\n5 passed" if ok else "Exit code: 1\nOUTPUT:\nFAILED test_x - assert 1 == 2"
        return "ok"
    tool.seen = seen
    return tool


def say(answer):
    def human(node, state):
        state.data[f"{node.id}.output"] = answer
    return human


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    d = tmp_path_factory.mktemp("lib")
    from src.rooms.session import available_tool_names
    return build_library(available_tools=available_tool_names(), user_dir=d / "rooms", stats_file=d / "stats.json")


def pipeline_runner(library, agent, tool=None, human=None, integrations=None, notified=None):
    class Skills:
        def skill(self, node, state):
            return "Find the blind spots first."
    return NodeRunner(agent_executor=agent, tool_executor=tool or fake_tests(True),
                      human_prompt=human, integrations=integrations or Skills(),
                      room_handler=SubRoomHandler(library),
                      notify=lambda n, s: (notified.append(n.id) if notified is not None else None) or True)


class TestTeachingRoomsAreSound:
    @pytest.mark.parametrize("name", ["ex_branching_task", "ex_plan_critic_loop", "ex_full_pipeline"])
    def test_valid_against_the_real_tools(self, name, library):
        from src.rooms.session import available_tool_names
        assert validate_room(example(name), available_tool_names()) == []

    def test_every_step_of_the_big_room_explains_itself(self):
        room = example("ex_full_pipeline")
        silent = [n.id for n in room.nodes if not (n.note or "").strip()]
        assert silent == []

    def test_the_big_room_uses_the_things_it_claims_to(self):
        room = example("ex_full_pipeline")
        types = {n.type for n in room.nodes}
        assert {"parallel", "join", "agent", "tool", "skill", "room", "loop", "break", "hint", "forget",
                "assert", "human_pause", "notify", "power", "mcp"} <= types
        assert any(n.disabled for n in room.nodes)
        assert room.rules.memory == "shared" and room.rules.deny_tools


class TestCriticReview:
    def run_review(self, library, count):
        agent = Scripted({"review": [{"findings": "- hole" if count else "", "critical_count": count}]})
        room = library.get("critic_review")
        return Engine().run(room, State(data={**room.defaults, "task": "t"}), NodeRunner(agent_executor=agent).run_node)

    def test_clean_when_nothing_critical(self, library):
        assert self.run_review(library, 0).exit == "clean"

    def test_critical_when_something_is(self, library):
        out = self.run_review(library, 2)
        assert out.exit == "critical" and out.state.data["findings"] == "- hole"

    def test_the_critic_cannot_write(self, library):
        assert library.get("critic_review").rules.read_only is True


class TestBranchingTask:
    def go(self, library, script, tool=None, human=None):
        agent = Scripted(script)
        out = Engine().run(example("ex_branching_task"), State(data={**example("ex_branching_task").defaults, "task": "something is wrong"}),
                           pipeline_runner(library, agent, tool=tool, human=human).run_node)
        return out, agent

    def test_a_bug_is_fixed_in_a_loop(self, library):
        tool = fake_tests(False, True)
        out, agent = self.go(library, {"classify": [{"kind": "bug", "summary": "x"}]}, tool=tool)
        assert out.exit == "fixed"
        assert agent.calls == ["classify", "fix"] and tool.seen == ["run_tests", "run_tests"]

    def test_a_bug_that_stays_is_given_up_after_three_rounds(self, library):
        tool = fake_tests(False)
        out, agent = self.go(library, {"classify": [{"kind": "bug"}]}, tool=tool)
        assert out.exit == "cannot_fix"
        assert agent.calls.count("fix") == 4 and len(tool.seen) == 4

    def test_a_question_is_only_answered(self, library):
        out, agent = self.go(library, {"classify": [{"kind": "question"}], "answer": [{"answer": "42"}]})
        assert out.exit == "answered" and out.state.data["answer"] == "42"
        assert agent.calls == ["classify", "answer"]

    def test_a_feature_waits_for_your_decision(self, library):
        script = {"classify": [{"kind": "feature"}], "plan": [{"plan": "1. do it"}]}
        approved, _ = self.go(library, script, human=say("да"))
        declined, _ = self.go(library, script, human=say("нет"))
        silent, _ = self.go(library, script, human=None)
        assert approved.exit == "plan_approved" and declined.exit == "plan_waiting"
        assert silent.exit == "human_pause"


class TestPlanCriticLoop:
    def go(self, library, verdicts):
        agent = Scripted({"draft": [{"artifact_path": ".argent/artifacts/plan.md"}],
                          "review": [{"findings": "- hole" if c else "", "critical_count": c} for c in verdicts]})
        out = Engine().run(example("ex_plan_critic_loop"), State(data={"task": "build it"}),
                           pipeline_runner(library, agent).run_node)
        return out, agent

    def test_stops_as_soon_as_the_critic_agrees(self, library):
        out, agent = self.go(library, [1, 0])
        assert out.exit == "approved" and agent.calls.count("draft") == 2
        assert out.state.data["rounds_broken"] is True

    def test_gives_up_after_three_rounds(self, library):
        out, agent = self.go(library, [1])
        assert out.exit == "unresolved" and agent.calls.count("draft") == 3
        assert out.state.data["rounds_capped"] is False and out.state.data["rounds_broken"] is False

    def test_the_artifact_path_travels_to_the_end(self, library):
        out, agent = self.go(library, [0])
        assert out.exit == "approved"
        assert out.state.data["artifact_path"].endswith("plan.md")


class TestFullPipeline:
    BASE = {"analyze": [{"project_known": True, "needs_research": False, "summary": "a project", "questions": ""}],
            "plan": [{"artifact_path": ".argent/artifacts/plan.md"}],
            "synthesize": [{"artifact_path": ".argent/artifacts/plan.md"}],
            "web": [{"web_findings": "w"}], "code": [{"code_findings": "c"}],
            "memo": [{"memory_note": "remember", "todo_text": "- a\n- b"}],
            "review": [{"findings": "", "critical_count": 0}],
            "revise": [{"revision_note": "fixed"}], "implement": [{"fix_summary": "did"}],
            "fix": [{"fix_summary": "again"}], "bugfix": [{"fix_summary": "bug"}]}

    def go(self, library, script=None, tool=None, human=None, data=None, notified=None):
        agent = Scripted({**self.BASE, **(script or {})})
        room = example("ex_full_pipeline")
        state = State(data={**room.defaults, "task": "add feature X", **(data or {})})
        runner = pipeline_runner(library, agent, tool=tool, notified=notified,
                                   human=say("да") if human is None else (None if human is False else human))
        events = []
        out = Engine().run(example("ex_full_pipeline"), state, runner.run_node, on_event=events.append)
        return out, agent, events

    def test_known_project_without_research_goes_straight_to_a_plan(self, library):
        notified = []
        out, agent, events = self.go(library, notified=notified)
        assert out.exit == "done" and notified == ["finish"]
        assert "plan" in agent.calls and "synthesize" not in agent.calls
        assert "web" not in agent.calls and "code" not in agent.calls
        assert out.state.data["gathered_arrived"] == ["scan", "recall", "method"]
        assert out.state.data["method"] == "Find the blind spots first."
        assert [e["node"] for e in events if e["type"] == "fork"] == ["gather"]

    def test_research_runs_web_and_code_side_by_side_and_makes_the_artifact(self, library):
        script = {"analyze": [{"project_known": False, "needs_research": True, "summary": "?", "questions": "a\nb"}]}
        out, agent, events = self.go(library, script)
        assert out.exit == "done"
        assert {"web", "code", "synthesize"} <= set(agent.calls) and "plan" not in agent.calls
        assert out.state.data["researched_arrived"] == ["web", "code"]
        assert out.state.data["web_findings"] == "w" and out.state.data["code_findings"] == "c"

    def test_an_unknown_project_is_researched_even_when_the_agent_says_no_research(self, library):
        script = {"analyze": [{"project_known": False, "needs_research": False}]}
        _, agent, _ = self.go(library, script)
        assert "web" in agent.calls

    def test_the_critic_sends_the_plan_back_once_then_agrees(self, library):
        script = {"review": [{"findings": "- no rollback", "critical_count": 1}, {"findings": "", "critical_count": 0}]}
        out, agent, _ = self.go(library, script)
        assert out.exit == "done" and agent.calls.count("revise") == 1
        assert agent.calls.count("review") == 3          # the plan twice, then the implementation once

    def test_a_critic_that_never_agrees_hands_the_decision_to_you(self, library):
        script = {"review": [{"findings": "- still wrong", "critical_count": 1}]}
        no, agent, _ = self.go(library, script, human=say("нет"))
        assert no.exit == "plan_rejected" and agent.calls.count("revise") == 3
        silent, _, _ = self.go(library, script, human=False)
        assert silent.exit == "human_pause"

    def test_memory_and_todo_are_written_from_the_plan(self, library):
        calls = []

        def tool(node, state):
            calls.append((node.tool, dict(node.args)))
            return "Exit code: 0\nOUTPUT:\nok" if node.tool == "run_command" else "ok"
        self.go(library, tool=tool)
        memory = [a for t, a in calls if t == "memory" and a.get("action") == "write"]
        todo = [a for t, a in calls if t == "todo_write"]
        assert memory and memory[0]["body"] == "remember"
        assert todo and todo[0]["todos"] == "- a\n- b"

    def test_it_asks_before_implementing_unless_told_not_to(self, library):
        asked = []

        def human(node, state):
            asked.append(node.id)
            state.data[f"{node.id}.output"] = "да"
        self.go(library, human=human)
        assert asked == ["go_ahead"]
        asked.clear()
        out, agent, _ = self.go(library, human=human, data={"ask_before_implementing": False})
        assert asked == [] and out.exit == "done" and "implement" in agent.calls

    def test_declining_stops_before_any_change(self, library):
        out, agent, _ = self.go(library, human=say("нет"))
        assert out.exit == "stopped_before_implementing" and "implement" not in agent.calls

    def test_red_tests_are_fixed_with_a_hint_and_then_green(self, library):
        tool = fake_tests(False, False, False, True)
        out, agent, _ = self.go(library, tool=tool)
        assert out.exit == "done" and agent.calls.count("fix") == 3
        hinted = [b for n, b in agent.briefs if n == "fix" and any("Две правки не помогли" in h for h in b["hints"])]
        assert len(hinted) == 1                           # said once, to the third attempt

    def test_giving_up_forgets_what_the_agents_remembered(self, library):
        tool = fake_tests(False)
        out, agent, _ = self.go(library, tool=tool)
        assert out.exit == "needs_attention"
        assert out.state.memory == [] and out.state.hints == []

    def test_bugs_found_by_the_implementation_critic_are_fixed_twice_at_most(self, library):
        script = {"review": [{"findings": "", "critical_count": 0}, {"findings": "- bug", "critical_count": 1}]}
        out, agent, _ = self.go(library, script)
        assert out.exit == "needs_attention" and agent.calls.count("bugfix") == 2

    def test_a_plan_that_was_not_saved_stops_the_run(self, library):
        out, agent, _ = self.go(library, {"plan": [{}]})
        assert out.exit == "needs_attention"
        assert "implement" not in agent.calls and "Конвейер остановился" in out.state.data.get("failed_notice.output", "")

    def test_the_switched_off_blocks_stay_off(self, library):
        out, agent, events = self.go(library)
        ran = {e["node"] for e in events if e["type"] == "node_start"}
        assert "shutdown" not in ran and "tracker" not in ran
