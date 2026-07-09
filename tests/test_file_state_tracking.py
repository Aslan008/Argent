"""External-change detection on edits + the Prove-It verification rule."""

from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent


@pytest.fixture
def agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "cloud")
    return ArgentAgent()


def _seed_read(agent, path):
    content = path.read_text(encoding="utf-8")
    agent.messages.append({"role": "tool", "content": content})
    agent._record_read({"file_path": str(path)}, content)


class TestExternalChangeNote:
    def test_edit_after_external_change_is_flagged(self, agent, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        _seed_read(agent, f)
        f.write_text("x = 1\n# edited in Unity\n", encoding="utf-8")   # external change

        note = agent._external_change_note("replace_in_file", {"file_path": str(f)})
        assert note is not None and "changed on disk AFTER" in note

    def test_unchanged_file_is_silent(self, agent, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        _seed_read(agent, f)
        assert agent._external_change_note("replace_in_file", {"file_path": str(f)}) is None

    def test_never_read_file_is_silent(self, agent, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        assert agent._external_change_note("replace_in_file", {"file_path": str(f)}) is None

    def test_non_edit_tool_is_silent(self, agent, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        _seed_read(agent, f)
        f.write_text("changed\n", encoding="utf-8")
        assert agent._external_change_note("grep_search", {"file_path": str(f)}) is None

    def test_own_edit_drops_cache_so_no_false_flag(self, agent, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\n", encoding="utf-8")
        _seed_read(agent, f)

        # Simulate what the dispatch tail does after "Successfully replaced...".
        agent._drop_read_cache({"file_path": str(f)})
        f.write_text("x = 2\n", encoding="utf-8")                      # our own change

        assert agent._external_change_note("replace_in_file", {"file_path": str(f)}) is None


class TestProveItRule:
    def test_capable_models_get_prove_it(self, agent):
        assert "Prove It" in agent.build_system_prompt()

    def test_tiny_models_keep_short_protocol(self, monkeypatch, agent):
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: "tiny")
        a = ArgentAgent()
        assert "Prove It" not in a.build_system_prompt()
