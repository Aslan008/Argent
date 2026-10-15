"""Back to before the last message: the conversation AND the files (real git repos)."""

import subprocess
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from src.agent.checkpoints import set_auto_checkpoint


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


@pytest.fixture(autouse=True)
def _time_machine_on():
    set_auto_checkpoint(True)
    yield
    set_auto_checkpoint(False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    _git("config", "user.email", "t@argent.local")
    _git("config", "user.name", "T")
    (tmp_path / "app.py").write_text("v1\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", "initial")
    return tmp_path


@pytest.fixture
def agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    return ArgentAgent()


def start_turn(agent, text):
    """What _run_turn does first, then the user message and a reply."""
    agent._begin_turn_record(text)
    agent.messages.append({"role": "user", "content": text})
    agent._turn_checkpoint_done = False
    agent._turn_label = text[:60]


def finish_turn(agent, reply="готово"):
    agent.messages.append({"role": "assistant", "content": reply})


def edit(agent, repo, name, content, tool="write_file"):
    """The agent's file tool: checkpoint first (as the loop does), then the write."""
    agent._maybe_turn_checkpoint(tool)
    (repo / name).write_text(content, encoding="utf-8")


class TestRewindLastTurn:
    def test_clean_tree_goes_back_to_head_and_the_edits_are_stashed(self, agent, repo):
        before = list(agent.messages)
        start_turn(agent, "поправь app.py")
        edit(agent, repo, "app.py", "v2\n")
        edit(agent, repo, "new.py", "created\n")
        finish_turn(agent)
        head = _git("rev-parse", "HEAD").stdout.strip()

        plan = agent.plan_last_turn_rewind()
        assert plan["ok"] and plan["files"] == "restore" and plan["text"] == "поправь app.py"
        res = agent.rewind_last_turn()

        assert res["ok"] and res["text"] == "поправь app.py"
        assert agent.messages == before                                   # the conversation, exactly
        assert (repo / "app.py").read_text(encoding="utf-8") == "v1\n"    # the files
        assert not (repo / "new.py").exists()
        assert _git("rev-parse", "HEAD").stdout.strip() == head
        assert "Argent Auto-Save before Rewind" in _git("stash", "list").stdout   # nothing lost silently

    def test_dirty_tree_goes_back_to_the_state_the_message_found(self, agent, repo):
        (repo / "app.py").write_text("my own unsaved work\n", encoding="utf-8")       # uncommitted before the message
        start_turn(agent, "дополни")
        edit(agent, repo, "app.py", "my own unsaved work\nplus the agent\n")
        finish_turn(agent)

        res = agent.rewind_last_turn()
        assert res["ok"]
        assert (repo / "app.py").read_text(encoding="utf-8") == "my own unsaved work\n"

    def test_a_turn_that_changed_no_files_only_rewinds_the_conversation(self, agent, repo):
        before = list(agent.messages)
        start_turn(agent, "что в app.py?")
        finish_turn(agent)
        plan = agent.plan_last_turn_rewind()
        assert plan["files"] == "none"
        assert agent.rewind_last_turn()["ok"]
        assert agent.messages == before
        assert (repo / "app.py").read_text(encoding="utf-8") == "v1\n"

    def test_it_walks_back_one_message_at_a_time(self, agent, repo):
        start_turn(agent, "первое")
        edit(agent, repo, "app.py", "v2\n")
        finish_turn(agent)
        _git("add", "-A")                                                   # the person keeps the first result...
        _git("commit", "-q", "-m", "first result")                          # ...as a real commit
        after_first = list(agent.messages)
        start_turn(agent, "второе")
        edit(agent, repo, "app.py", "v3\n")
        finish_turn(agent)

        assert agent.rewind_last_turn()["text"] == "второе"
        assert agent.messages == after_first and (repo / "app.py").read_text(encoding="utf-8") == "v2\n"
        # going further back would have to undo the person's own commit: refused, nothing touched
        res = agent.rewind_last_turn()
        assert not res["ok"] and "ничего не изменено" in res["error"]
        assert agent.messages == after_first and (repo / "app.py").read_text(encoding="utf-8") == "v2\n"
        assert len(agent._turn_records) == 1                                # still there to try again

    def test_a_real_commit_since_the_message_blocks_the_file_rewind(self, agent, repo):
        start_turn(agent, "правка")
        edit(agent, repo, "app.py", "v2\n")
        finish_turn(agent)
        _git("add", "-A")
        _git("commit", "-q", "-m", "the person committed it")
        snapshot = list(agent.messages)
        res = agent.rewind_last_turn()
        assert not res["ok"] and "ничего не изменено" in res["error"]
        assert agent.messages == snapshot                                    # all or nothing
        assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"

    def test_without_a_checkpoint_the_files_stay_and_the_result_says_so(self, agent, repo):
        set_auto_checkpoint(False)
        before = list(agent.messages)
        start_turn(agent, "правка без страховки")
        edit(agent, repo, "app.py", "v2\n")
        finish_turn(agent)
        plan = agent.plan_last_turn_rewind()
        assert plan["ok"] and plan["files"] == "unavailable" and "Файлы вернуть нельзя" in plan["detail"]
        assert agent.rewind_last_turn()["ok"]
        assert agent.messages == before
        assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"       # honest: not touched

    def test_delete_and_move_are_covered_by_the_checkpoint(self, agent, repo):
        (repo / "keep.txt").write_text("keep\n", encoding="utf-8")
        _git("add", "-A")
        _git("commit", "-q", "-m", "add keep")
        start_turn(agent, "удали keep")
        agent._maybe_turn_checkpoint("delete_file")
        (repo / "keep.txt").unlink()
        finish_turn(agent)
        assert agent.rewind_last_turn()["ok"]
        assert (repo / "keep.txt").read_text(encoding="utf-8") == "keep\n"

    def test_nothing_to_go_back_to(self, agent, repo):
        res = agent.plan_last_turn_rewind()
        assert not res["ok"] and "Нет сообщения" in res["error"]

    def test_records_are_bounded_and_reset_with_the_conversation(self, agent, repo):
        for i in range(agent.MAX_TURN_RECORDS + 5):
            start_turn(agent, f"m{i}")
            finish_turn(agent)
        assert len(agent._turn_records) == agent.MAX_TURN_RECORDS
        agent.clear_history()
        assert agent.plan_last_turn_rewind()["ok"] is False
