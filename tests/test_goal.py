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
    def test_not_in_tool_schemas(self):
        """set_goal is not in LLM schemas to avoid goal confusion and token waste."""
        from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
        assert "set_goal" not in AVAILABLE_TOOLS
        names = {t["function"]["name"] for t in TOOL_SCHEMAS if "function" in t}
        assert "set_goal" not in names


class TestObjectiveAnchor:
    """The trailing anchor re-pins a compact working-memory slice
    (completed, recent errors, edited files) so weak models on long conversations don't
    redo finished work or retry failed approaches, without pinning a stale goal."""

    def _agent(self, monkeypatch):
        import agent as agent_module
        import config
        monkeypatch.setattr(config, "get_current_model", lambda: "qwen2.5-coder:7b")
        monkeypatch.setattr(agent_module, "get_current_model", lambda: "qwen2.5-coder:7b")
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: MagicMock())
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        return agent_module.ArgentAgent()

    def test_anchor_includes_done_and_failures(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.add_completed("Wrote lexer.py")
        memory.add_completed("Ran tests")
        memory.add_error("pytest failed: import error")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        assert "OBJECTIVE" not in anchor
        assert "COMPLETED" in anchor and "Ran tests" in anchor
        assert "RECENT ERRORS" in anchor and "import error" in anchor

    def test_anchor_none_when_memory_empty(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        assert self._agent(monkeypatch)._build_objective_anchor() is None

    def test_anchor_present_with_only_working_memory(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.add_completed("Created project skeleton")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        assert anchor is not None
        assert "COMPLETED" in anchor and "skeleton" in anchor

    def test_anchor_is_bounded(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        for i in range(10):
            memory.add_completed(f"did {i}")
        for i in range(6):
            memory.add_error(f"err {i}")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        # only the last 3 completions and last 2 failures are re-pinned
        assert "did 9" in anchor and "did 7" in anchor and "did 6" not in anchor
        assert "err 5" in anchor and "err 4" in anchor and "err 3" not in anchor

    def test_anchor_pins_edited_files(self, monkeypatch):
        # On a long task the early edits scroll out of history; without this the
        # model re-edits a file it already rewrote.
        from memory_manager import memory
        memory.clear()
        memory.add_file_modified("src/board/MatchDetector.cs")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        assert "FILES ALREADY EDITED" in anchor
        assert "MatchDetector.cs" in anchor

    def test_edited_files_are_deduplicated_and_bounded(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        for i in range(12):
            memory.add_file_modified(f"f{i}.py")
        for _ in range(5):
            memory.add_file_modified("f11.py")        # repeatedly edited
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        files_line = [ln for ln in anchor.splitlines() if "ALREADY EDITED" in ln][0]
        assert files_line.count("f11.py") == 1        # repeats collapsed
        assert "f11.py" in files_line and "f0.py" not in files_line   # newest kept

    def test_anchor_fires_with_only_touched_files(self, monkeypatch):
        from memory_manager import memory
        memory.clear()
        memory.add_file_modified("only.py")
        anchor = self._agent(monkeypatch)._build_objective_anchor()
        assert anchor is not None and "only.py" in anchor
