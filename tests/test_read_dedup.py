"""read_file dedup: short-circuit re-reads of an unchanged whole file still in history."""

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
    a = ArgentAgent()
    a.messages = [{"role": "system", "content": "sys"}]
    return a


def _seed(agent, path, content):
    """Simulate a real whole-file read: content lands in history + gets cached."""
    agent.messages.append({"role": "tool", "content": content})
    agent._record_read({"file_path": str(path)}, content)


def test_unchanged_file_in_history_is_short_circuited(agent, tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def a():\n    return 1\n", encoding="utf-8")
    content = f.read_text(encoding="utf-8")
    _seed(agent, f, content)

    note = agent._read_dedup_note({"file_path": str(f)})
    assert note is not None
    assert "UNCHANGED" in note and "mod.py" in note


def test_modified_file_falls_through_to_real_read(agent, tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("old\n", encoding="utf-8")
    _seed(agent, f, "old\n")

    f.write_text("old\nnew line\n", encoding="utf-8")  # mtime + size change
    assert agent._read_dedup_note({"file_path": str(f)}) is None


def test_trimmed_content_falls_through(agent, tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("body\n", encoding="utf-8")
    agent._record_read({"file_path": str(f)}, "body\n")  # cached but NOT in messages

    # History no longer contains the content -> must not lie, must re-read.
    assert agent._read_dedup_note({"file_path": str(f)}) is None


def test_range_read_is_never_deduped(agent, tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("l1\nl2\nl3\n", encoding="utf-8")
    _seed(agent, f, "l1\nl2\nl3\n")

    assert agent._read_dedup_note({"file_path": str(f), "start_line": 1, "end_line": 2}) is None


def test_uncached_file_falls_through(agent, tmp_path):
    f = tmp_path / "fresh.py"
    f.write_text("x\n", encoding="utf-8")
    assert agent._read_dedup_note({"file_path": str(f)}) is None


def test_missing_file_is_safe(agent, tmp_path):
    assert agent._read_dedup_note({"file_path": str(tmp_path / "nope.py")}) is None


def test_record_ignores_range_reads(agent, tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("data\n", encoding="utf-8")
    agent._record_read({"file_path": str(f), "start_line": 1}, "[Lines 1-1 of 1]\ndata\n")
    assert str(f.resolve()) not in agent._read_cache
