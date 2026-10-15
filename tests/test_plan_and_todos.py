"""Plan mode (nothing changes until the plan is approved) and the todo checklist."""

import json
from itertools import islice
from unittest.mock import MagicMock

import pytest

import agent as agent_module
from agent import ArgentAgent
from src.agent import plan_mode, todos
from src.agent.plan_mode import PLAN, readonly_command, violation
from src.agent.strategy import CloudStrategy
from src.agent.todos import TODOS, TodoError, parse_items
from tools.schemas import get_tool_schemas


def names():
    return {s["function"]["name"] for s in get_tool_schemas()}


# ── the checklist ──────────────────────────────────────────────────────────────

class TestTodoParsing:
    def test_plain_strings_become_pending_items(self):
        items = parse_items(["read the code", "write the fix"])
        assert [(t.content, t.status) for t in items] == [
            ("read the code", "pending"), ("write the fix", "pending")]

    def test_objects_json_strings_and_status_aliases(self):
        raw = json.dumps([{"content": "a", "status": "done"}, {"task": "b", "status": "In Progress"},
                          {"title": "c"}])
        assert [t.status for t in parse_items(raw)] == ["completed", "in_progress", "pending"]

    def test_a_bulleted_text_is_read_as_a_list(self):
        assert [t.content for t in parse_items("- one\n* two\n")] == ["one", "two"]

    @pytest.mark.parametrize("bad", [None, 5, [{"status": "pending"}], [{"content": "x", "status": "weird"}],
                                     [3]])
    def test_nonsense_is_refused_with_a_reason(self, bad):
        with pytest.raises(TodoError):
            parse_items(bad)

    def test_the_list_has_a_ceiling(self):
        with pytest.raises(TodoError):
            parse_items([f"step {i}" for i in range(todos.MAX_ITEMS + 1)])


class TestTodoWrite:
    def test_replaces_the_whole_list_and_shows_progress(self):
        out = todos.write([{"content": "a", "status": "completed"}, {"content": "b", "status": "in_progress"},
                           {"content": "c", "status": "pending"}])
        assert "1/3 done" in out and "[x] a" in out and "[~] b" in out and "[ ] c" in out
        assert "Current step: b" in out
        todos.write([{"content": "only", "status": "pending"}])
        assert [t.content for t in TODOS.get()] == ["only"]

    def test_tells_the_model_when_it_forgot_to_mark_a_step_in_progress(self):
        assert "mark the step you are starting" in todos.write([{"content": "a", "status": "pending"}])

    def test_warns_about_two_steps_in_progress(self):
        out = todos.write([{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}])
        assert "One at a time" in out or "one at a time" in out.lower()

    def test_finished_list_says_so_and_is_dropped_by_the_next_turn(self):
        out = todos.write([{"content": "a", "status": "completed"}])
        assert "Every item is completed" in out
        TODOS.begin_turn()
        assert TODOS.get() == []

    def test_an_unfinished_list_survives_the_next_turn(self):
        todos.write([{"content": "a", "status": "pending"}])
        TODOS.begin_turn()
        assert len(TODOS.get()) == 1

    def test_empty_list_clears(self):
        todos.write(["a"])
        assert "cleared" in todos.write([])
        assert TODOS.get() == []

    def test_bad_input_is_an_error_string_not_an_exception(self):
        assert todos.write(None).startswith("Error:")

    def test_a_sub_agents_list_is_its_own_and_is_dropped_with_it(self):
        todos.write(["supervisor step"])
        with todos.use_scope("sub-1"):
            todos.write(["worker step"])
            assert [t.content for t in TODOS.get()] == ["worker step"]
        assert [t.content for t in TODOS.get()] == ["supervisor step"]
        assert TODOS.get("sub-1") == []

    def test_the_model_sees_its_list_at_the_tail_of_the_request(self):
        todos.write([{"content": "step one", "status": "in_progress"}])
        block = todos.ephemeral_block()
        assert "YOUR TODO LIST" in block and "[~] step one" in block


def test_the_tool_is_advertised_and_runs_through_the_registry():
    from tools.schemas import get_available_tools
    assert "todo_write" in names()
    out = get_available_tools()["todo_write"](todos=[{"content": "a", "status": "in_progress"}])
    assert "[~] a" in out


# ── which shell commands only look ─────────────────────────────────────────────

READONLY = {
    "git status": True, "git diff HEAD~1 -- a.py": True, "git log --oneline -5": True, "git -C sub status": True,
    "git branch": True, "git branch -a": True, "git branch --list 'f*'": True, "git branch -d x": False,
    "git branch newname": False, "git tag": True, "git tag v1": False, "git tag -a v1 -m x": False,
    "git remote -v": True, "git remote add o u": False, "git config --get user.name": True,
    "git config user.name x": False, "git stash": False, "git stash list": True, "git push": False,
    "git commit -m x": False, "git checkout x": False, "git reset --hard": False, "git diff --output=f": False,
    "ls -la | grep py": True, "cat a.py > b.py": False, "echo hi >> f": False, "echo hi 2>&1": True,
    "find . -name '*.py'": True, "find . -delete": False, "find . -exec rm {} ;": False, "rm -rf x": False,
    "python -c 'print(1)'": False, "pytest": False, "sort -o f f": False, "sort f | uniq": True,
    "echo $(rm x)": False, "echo `rm x`": False, "echo '>'": True, "dir": True,
    "Get-ChildItem -Recurse": True, "": False, "rg -o foo src": True, "python --version": True,
    "ls && rm x": False, "echo 'unclosed": False,
    "Get-ChildItem -Recurse | Select-Object -First 5": True, "Get-ChildItem | Sort-Object Length | Format-Table": True,
    "Get-ChildItem | Where-Object { $_.Length -gt 1 }": False,
    "Get-ChildItem | Select-Object @{n='x';e={1}}": False, "Get-Content a | Out-File b": False,
}


@pytest.mark.parametrize("command,expected", list(READONLY.items()))
def test_readonly_command(command, expected):
    assert readonly_command(command)[0] is expected, readonly_command(command)


# ── what plan mode refuses ─────────────────────────────────────────────────────

class TestPlanModeGate:
    def test_off_by_default_and_nothing_is_refused(self):
        assert not PLAN.active
        assert violation("write_file", {"file_path": "a"}) is None
        assert violation("run_command", {"command": "rm -rf x"}) is None

    def test_on_refuses_every_mutating_tool(self):
        PLAN.enter()
        for tool in plan_mode.MUTATING_TOOLS:
            assert violation(tool, {}) and "Plan mode is ON" in violation(tool, {}), tool
        # the tools that exist for planning are never refused
        assert violation("exit_plan_mode", {}) is None and violation("todo_write", {}) is None

    def test_on_allows_reading_and_readonly_commands_only(self):
        PLAN.enter()
        assert violation("read_file", {"file_path": "a"}) is None
        assert violation("grep_search", {"pattern": "x"}) is None
        assert violation("run_command", {"command": "git status"}) is None
        assert "not read-only" in violation("run_command", {"command": "npm install"})
        assert violation("run_command", {"command": "echo x > f.txt"})

    def test_mutating_tools_are_not_advertised_while_planning_and_exit_is(self):
        assert "exit_plan_mode" not in names() and "write_file" in names()
        PLAN.enter()
        shown = names()
        assert "exit_plan_mode" in shown
        assert not (plan_mode.MUTATING_TOOLS & shown)
        assert {"read_file", "grep_search", "run_command", "todo_write"} <= shown
        PLAN.leave()
        assert "exit_plan_mode" not in names() and "write_file" in names()

    def test_every_mutating_tool_name_is_a_real_tool(self):
        """A typo here would leave a real tool unguarded."""
        from tools.schemas import TOOL_SCHEMAS
        real = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert plan_mode.MUTATING_TOOLS <= real, plan_mode.MUTATING_TOOLS - real

    def test_the_local_agent_is_refused_too_although_it_is_registered_only_when_enabled(self):
        PLAN.enter()
        assert violation("delegate_local", {"task": "x"}) and plan_mode.hides("delegate_local")


# ── leaving plan mode ──────────────────────────────────────────────────────────

class TestExitPlanMode:
    PLAN_TEXT = "Goal: add retry.\nFiles: fetch.py (add backoff), tests/test_fetch.py.\nSteps: 1 2 3. Verify: pytest."

    @pytest.fixture
    def answer(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        reply = {"value": True}

        class Q:
            def ask(self):
                return reply["value"]
        import questionary
        monkeypatch.setattr(questionary, "confirm", lambda *a, **k: Q())
        return reply

    def test_outside_plan_mode_it_does_nothing(self, answer):
        from tools.plan_tools import exit_plan_mode
        assert "not on" in exit_plan_mode(self.PLAN_TEXT)

    def test_approval_turns_plan_mode_off_and_saves_the_plan(self, answer, tmp_path):
        from tools.plan_tools import exit_plan_mode
        PLAN.enter()
        out = exit_plan_mode(self.PLAN_TEXT)
        assert "APPROVED" in out and not PLAN.active
        saved = list((tmp_path / ".argent" / "artifacts").glob("plan-*.md"))
        assert len(saved) == 1 and "add retry" in saved[0].read_text(encoding="utf-8")

    def test_a_no_keeps_plan_mode_on(self, answer):
        from tools.plan_tools import exit_plan_mode
        PLAN.enter()
        answer["value"] = False
        assert "did NOT approve" in exit_plan_mode(self.PLAN_TEXT) and PLAN.active

    def test_dismissing_the_question_is_not_an_approval(self, answer):
        from tools.plan_tools import exit_plan_mode
        PLAN.enter()
        answer["value"] = None
        assert "NOT approved" in exit_plan_mode(self.PLAN_TEXT) and PLAN.active

    def test_a_stub_plan_is_sent_back(self, answer):
        from tools.plan_tools import exit_plan_mode
        PLAN.enter()
        assert exit_plan_mode("do it").startswith("Error")
        assert PLAN.active


# ── through the real agent loop ────────────────────────────────────────────────

class ScriptedProvider:
    """Serves scripted chunk lists, one per call; a model that asks for native tools gets a 404 once,
    so the agent falls back to raw-JSON tool calls."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        if kwargs.get("tools"):
            from providers import ProviderError
            raise ProviderError("HTTP 404: No endpoints found that support tool use.")
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        self.last_messages = kwargs.get("messages")
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def _make(provider):
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        a = ArgentAgent()
        a.model_name = "gpt-4o"
        a.strategy = CloudStrategy()
        a._native_tools_unsupported = False
        return a
    return _make


def run_turn(agent, text):
    return list(islice(agent.process_user_input(text), 500))


def test_a_write_in_plan_mode_is_refused_before_it_runs(make_agent, tmp_path):
    PLAN.enter()
    provider = ScriptedProvider([
        [{"content": '{"name": "write_file", "arguments": {"file_path": "hacked.txt", "content": "x"}}'}],
        [{"content": "I will plan instead."}],
    ])
    chunks = run_turn(make_agent(provider), "create hacked.txt")
    assert not (tmp_path / "hacked.txt").exists()
    ends = [c for c in chunks if c["type"] == "tool_end"]
    assert ends and "Plan mode is ON" in str(ends[0]["result"])


def test_a_shell_write_in_plan_mode_is_refused_but_a_read_is_not(make_agent, tmp_path):
    PLAN.enter()
    provider = ScriptedProvider([
        [{"content": '{"name": "run_command", "arguments": {"command": "echo x > made.txt"}}'}],
        [{"content": '{"name": "run_command", "arguments": {"command": "git status"}}'}],
        [{"content": "done"}],
    ])
    chunks = run_turn(make_agent(provider), "look around")
    assert not (tmp_path / "made.txt").exists()
    results = [str(c["result"]) for c in chunks if c["type"] == "tool_end"]
    assert "Plan mode is ON" in results[0]
    assert "Plan mode is ON" not in results[1]


def test_the_plan_reminder_and_the_checklist_reach_the_model(make_agent):
    PLAN.enter()
    todos.write([{"content": "survey the code", "status": "in_progress"}])
    provider = ScriptedProvider([[{"content": "ok"}]])
    run_turn(make_agent(provider), "plan a refactor")
    seen = "\n".join(str(m.get("content")) for m in provider.last_messages)
    assert "PLAN MODE IS ON" in seen and "survey the code" in seen


def test_the_checklist_is_cleared_with_the_conversation(make_agent):
    agent = make_agent(ScriptedProvider([[{"content": "ok"}]]))
    todos.write(["a"])
    agent.clear_history()
    assert TODOS.get() == []
