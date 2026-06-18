"""The /goal feature: set_goal tool + its registration across all tiers."""

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
