"""Time machine: per-turn auto-checkpoints and safe rewind (real git repos)."""

import subprocess
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from src.agent import checkpoints
from src.agent.checkpoints import (
    CheckpointError, auto_checkpoint, blocking_commits, create_checkpoint,
    list_checkpoints, rewind_to, set_auto_checkpoint,
)


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)


@pytest.fixture(autouse=True)
def _enable_time_machine():
    """conftest disables auto-checkpoints suite-wide; this file exercises them
    for real, safely inside tmp_path repos."""
    set_auto_checkpoint(True)
    yield
    set_auto_checkpoint(False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo with one initial commit, as the current directory."""
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    _git("config", "user.email", "test@argent.local")
    _git("config", "user.name", "Argent Test")
    (tmp_path / "app.py").write_text("v1\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", "initial")
    return tmp_path


class TestCreateAndList:
    def test_dirty_tree_creates_checkpoint(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        sha = create_checkpoint("before: fix menu")
        assert sha
        assert _git("log", "-1", "--pretty=%s").stdout.strip() == \
            "Argent Checkpoint: before: fix menu"

    def test_clean_tree_returns_none(self, repo):
        assert create_checkpoint("nothing") is None

    def test_list_is_newest_first_and_filters_real_commits(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("first")
        (repo / "app.py").write_text("v3\n", encoding="utf-8")
        _git("add", "-A")
        _git("commit", "-q", "-m", "real user commit")
        (repo / "app.py").write_text("v4\n", encoding="utf-8")
        create_checkpoint("second")

        cps = list_checkpoints()
        assert [c["label"] for c in cps] == ["second", "first"]
        assert all(c["sha"] for c in cps)


class TestRewind:
    def test_rewind_restores_the_pre_turn_state(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("before: turn 1")     # snapshot contains v2
        (repo / "app.py").write_text("broken\n", encoding="utf-8")

        msg = rewind_to(list_checkpoints()[0]["sha"])
        assert "Rewound to checkpoint" in msg
        assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"

    def test_uncommitted_changes_are_stashed_not_lost(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("cp")
        (repo / "app.py").write_text("work in progress\n", encoding="utf-8")

        msg = rewind_to(list_checkpoints()[0]["sha"])
        assert "stashed" in msg
        assert "Argent Auto-Save before Rewind" in _git("stash", "list").stdout

    def test_discards_newer_checkpoints_and_reports_them(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("older")
        target = list_checkpoints()[0]["sha"]
        (repo / "app.py").write_text("v3\n", encoding="utf-8")
        create_checkpoint("newer")

        msg = rewind_to(target)
        assert "Discarded 1 newer checkpoint" in msg
        assert [c["label"] for c in list_checkpoints()] == ["older"]

    def test_real_commit_in_between_blocks_rewind(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("cp")
        target = list_checkpoints()[0]["sha"]
        (repo / "app.py").write_text("v3\n", encoding="utf-8")
        _git("add", "-A")
        _git("commit", "-q", "-m", "precious user work")

        assert blocking_commits(target)
        with pytest.raises(CheckpointError, match="precious user work"):
            rewind_to(target)

    def test_non_checkpoint_target_is_refused(self, repo):
        sha = _git("rev-parse", "--short", "HEAD").stdout.strip()   # 'initial'
        with pytest.raises(CheckpointError, match="not an Argent Checkpoint"):
            rewind_to(sha)


class TestAutoCheckpoint:
    def test_creates_labelled_checkpoint(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        assert auto_checkpoint("почини меню")
        assert list_checkpoints()[0]["label"] == "before: почини меню"

    def test_disabled_flag_skips(self, repo):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        set_auto_checkpoint(False)
        try:
            assert auto_checkpoint("x") is None
        finally:
            set_auto_checkpoint(True)

    def test_staged_changes_are_left_alone(self, repo):
        # A hand-crafted index means the user is mid-commit; auto add -A
        # would destroy it. The skip must also leave the index untouched.
        (repo / "staged.py").write_text("mine\n", encoding="utf-8")
        _git("add", "staged.py")
        assert auto_checkpoint("x") is None
        staged = _git("diff", "--cached", "--name-only").stdout.strip()
        assert staged == "staged.py"

    def test_staged_skip_warns_the_user(self, repo):
        """Skipping is right; skipping SILENTLY is the trap — the user would
        believe /rewind covers them while the agent edits unprotected."""
        from src.agent.checkpoints import auto_checkpoint_detailed
        (repo / "staged.py").write_text("mine\n", encoding="utf-8")
        _git("add", "staged.py")

        result = auto_checkpoint_detailed("почини меню")
        assert result.sha is None
        assert result.warn is True
        assert "staged" in result.reason and "/rewind" in result.reason

    def test_clean_tree_skip_is_not_a_warning(self, repo):
        from src.agent.checkpoints import auto_checkpoint_detailed
        result = auto_checkpoint_detailed("x")
        assert result.sha is None and result.warn is False

    def test_successful_checkpoint_reports_no_reason(self, repo):
        from src.agent.checkpoints import auto_checkpoint_detailed
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        result = auto_checkpoint_detailed("x")
        assert result.sha and result.reason is None and result.warn is False

    def test_outside_git_is_silent(self, tmp_path, monkeypatch):
        outside = tmp_path / "no_repo"
        outside.mkdir()
        monkeypatch.chdir(outside)
        monkeypatch.setattr(checkpoints, "is_git_repo", lambda: False)
        assert auto_checkpoint("x") is None

    def test_never_raises(self, repo, monkeypatch):
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        monkeypatch.setattr(checkpoints, "create_checkpoint",
                            MagicMock(side_effect=RuntimeError("boom")))
        assert auto_checkpoint("x") is None


class TestGitRollbackTool:
    def _approve(self, monkeypatch):
        import approval
        monkeypatch.setattr(approval, "request_approval", lambda *a, **k: True)

    def test_no_arg_rolls_back_to_latest(self, repo, monkeypatch):
        self._approve(monkeypatch)
        from tools.misc_tools import git_rollback
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("latest state")
        (repo / "app.py").write_text("broken\n", encoding="utf-8")

        assert "Rewound to checkpoint" in git_rollback()
        assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"

    def test_substring_picks_older_checkpoint(self, repo, monkeypatch):
        self._approve(monkeypatch)
        from tools.misc_tools import git_rollback
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("before: сломал меню")
        (repo / "app.py").write_text("v3\n", encoding="utf-8")
        create_checkpoint("before: другое")

        assert "Rewound to checkpoint" in git_rollback(to_checkpoint="меню")
        assert (repo / "app.py").read_text(encoding="utf-8") == "v2\n"

    def test_no_match_lists_available(self, repo, monkeypatch):
        self._approve(monkeypatch)
        from tools.misc_tools import git_rollback
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("before: правка")

        out = git_rollback(to_checkpoint="nonexistent")
        assert out.startswith("Error: no checkpoint matches")
        assert "before: правка" in out

    def test_denied_approval_aborts(self, repo, monkeypatch):
        import approval
        monkeypatch.setattr(approval, "request_approval", lambda *a, **k: False)
        from tools.misc_tools import git_rollback
        (repo / "app.py").write_text("v2\n", encoding="utf-8")
        create_checkpoint("cp")
        assert git_rollback() == "Rollback aborted by user."

    def test_no_checkpoints_is_explicit(self, repo, monkeypatch):
        self._approve(monkeypatch)
        from tools.misc_tools import git_rollback
        assert "no Argent Checkpoints found" in git_rollback()


@pytest.fixture
def agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)
    return ArgentAgent()


class TestAgentTurnCheckpoint:
    def test_first_edit_of_turn_yields_checkpoint_chunk(self, agent, monkeypatch):
        monkeypatch.setattr(checkpoints, "auto_checkpoint_detailed",
                            lambda label: checkpoints.CheckpointResult("abc1234"))
        agent._turn_checkpoint_done = False
        agent._turn_label = "почини меню"

        chunk = agent._maybe_turn_checkpoint("write_file")
        assert chunk == {"type": "checkpoint", "sha": "abc1234", "label": "почини меню"}

    def test_only_once_per_turn(self, agent, monkeypatch):
        calls = []
        monkeypatch.setattr(
            checkpoints, "auto_checkpoint_detailed",
            lambda label: calls.append(label) or checkpoints.CheckpointResult("abc1234"))
        agent._turn_checkpoint_done = False
        agent._turn_label = "задача"

        assert agent._maybe_turn_checkpoint("write_file") is not None
        assert agent._maybe_turn_checkpoint("replace_in_file") is None
        assert calls == ["задача"]

    def test_read_only_tools_never_checkpoint(self, agent, monkeypatch):
        monkeypatch.setattr(checkpoints, "auto_checkpoint_detailed",
                            MagicMock(side_effect=AssertionError("must not be called")))
        agent._turn_checkpoint_done = False
        assert agent._maybe_turn_checkpoint("read_file") is None

    def test_failed_snapshot_yields_nothing(self, agent, monkeypatch):
        monkeypatch.setattr(checkpoints, "auto_checkpoint_detailed",
                            lambda label: checkpoints.CheckpointResult(None, "clean tree"))
        agent._turn_checkpoint_done = False
        agent._turn_label = "x"
        assert agent._maybe_turn_checkpoint("write_file") is None
        assert agent._turn_checkpoint_done is True

    def test_unprotected_edit_surfaces_a_notice(self, agent, monkeypatch):
        monkeypatch.setattr(
            checkpoints, "auto_checkpoint_detailed",
            lambda label: checkpoints.CheckpointResult(
                None, "staged files — not covered by /rewind", warn=True))
        agent._turn_checkpoint_done = False
        agent._turn_label = "x"

        chunk = agent._maybe_turn_checkpoint("write_file")
        assert chunk["type"] == "error"
        assert "[System:" in chunk["content"] and "/rewind" in chunk["content"]


class TestCheckpointEvent:
    def test_chunk_maps_to_structured_event(self):
        from src.server.events import to_event
        ev = to_event({"type": "checkpoint", "sha": "abc1234", "label": "почини меню"})
        assert ev == {"type": "checkpoint", "sha": "abc1234", "label": "почини меню"}
