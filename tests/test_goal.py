"""The /goal feature: set_goal tool + its registration across all tiers."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep .argent/memory.json out of the repo


class TestSetGoalTool:
    def test_sets_objective_and_task(self):
        from tools.misc_tools import set_goal
        from memory_manager import memory
        memory.clear()
        out = set_goal(objective="Build X", current_task="write tests")
        assert "Goal updated" in out
        assert memory.data["objective"] == "Build X"
        assert memory.data["current_task"] == "write tests"

    def test_objective_only(self):
        from tools.misc_tools import set_goal
        from memory_manager import memory
        memory.clear()
        set_goal(objective="Just the goal")
        assert memory.data["objective"] == "Just the goal"
        assert memory.data["current_task"] == ""

    def test_requires_an_argument(self):
        from tools.misc_tools import set_goal
        assert set_goal().startswith("Error")
        assert set_goal(objective="   ").startswith("Error")


class TestSetGoalRegistration:
    def test_in_dispatch(self):
        from tools.schemas import AVAILABLE_TOOLS
        assert AVAILABLE_TOOLS.get("set_goal") is not None

    def test_has_schema(self):
        from tools.schemas import TOOL_SCHEMAS
        names = {t["function"]["name"] for t in TOOL_SCHEMAS if "function" in t}
        assert "set_goal" in names

    def test_available_to_weak_models(self):
        from tool_profiles import CORE_CHAT_TOOLS, slim_tools_for_category
        assert "set_goal" in CORE_CHAT_TOOLS
        assert "set_goal" in slim_tools_for_category(["set_goal", "browser_open"], "tiny")


class TestObjectiveAnchor:
    """The trailing anchor re-pins the goal AND a compact working-memory slice
    (already-done, known-failures) so weak models on long conversations don't
    redo finished work or retry failed approaches."""

    def _agent(self, monkeypatch):
        import agent as agent_module
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        return agent_module.ArgentAgent()

    def test_anchor_includes_done_and_failures(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.set_objective("Build parser")
        memory.set_current_task("write lexer")
        memory.add_completed("Wrote lexer.py")
        memory.add_completed("Ran tests")
        memory.add_error("pytest failed: import error")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        assert "OBJECTIVE: Build parser" in anchor
        assert "CURRENT TASK: write lexer" in anchor
        assert "ALREADY DONE" in anchor and "Ran tests" in anchor
        assert "KNOWN FAILURES" in anchor and "import error" in anchor

    def test_anchor_none_when_memory_empty(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        assert self._agent(monkeypatch)._build_objective_anchor() is None

    def test_anchor_present_with_only_working_memory(self, monkeypatch):
        # No explicit goal, but work has happened — the anchor still fires so the
        # "don't redo / don't retry" reminders reach the model.
        from memory_manager import memory
        memory.clear()
        memory.add_completed("Created project skeleton")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        assert anchor is not None
        assert "ALREADY DONE" in anchor and "skeleton" in anchor

    def test_anchor_is_bounded(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.set_objective("Goal")
        for i in range(10):
            memory.add_completed(f"did {i}")
        for i in range(6):
            memory.add_error(f"err {i}")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        # only the last 3 completions and last 2 failures are re-pinned
        assert "did 9" in anchor and "did 7" in anchor and "did 6" not in anchor
        assert "err 5" in anchor and "err 4" in anchor and "err 3" not in anchor
