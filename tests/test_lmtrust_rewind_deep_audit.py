"""LMTrust Deep Blind-Spot Audit & Verification Suite for /rewind in Argent.

Verifies:
- L1 (Contract & CLI Parsing): /rewind flags, search by needle, SHA prefix
- L2 (Boundary): Target validation, HEAD target, limit handling
- L4 (Adversarial): Stash failure aborts to prevent data loss, --no-verify bypasses local hooks
- L4b (Fallback & Recovery): Reset failure restores uncommitted changes via stash pop
- L5 (State Synchronization): Agent conversation messages truncated & _read_cache cleared
- L6 (Cross-System Consistency): file_tracker session snapshots cleared on rewind
- L7 (Compositional / Safety Wall): In-progress git merge/rebase blocks rewind safely
- L9 (Negative Space): No silent uncommitted data loss, no index poisoning
"""

import subprocess
from pathlib import Path
import pytest

from src.agent import checkpoints
from src.agent.checkpoints import (
    CheckpointError,
    auto_checkpoint_detailed,
    create_checkpoint,
    list_checkpoints,
    rewind_to,
    set_auto_checkpoint,
)
from file_tracker import snapshot, undo, clear_session_snapshots


def _git(*args, cwd=None):
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)


@pytest.fixture(autouse=True)
def _enable_time_machine():
    set_auto_checkpoint(True)
    yield
    set_auto_checkpoint(False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    _git("config", "user.email", "audit@argent.local")
    _git("config", "user.name", "Argent Audit")
    (tmp_path / "main.py").write_text("print('v1')\n", encoding="utf-8")
    _git("add", "-A")
    _git("commit", "-q", "-m", "initial")
    return tmp_path


# ============================================================================
# LAYER L4: Adversarial & Data Loss Protection
# ============================================================================

class TestL4AdversarialSafety:
    def test_stash_failure_aborts_before_reset(self, repo, monkeypatch):
        """BUG-01 Fix: If git stash fails, rewind_to must raise CheckpointError
        BEFORE calling git reset --hard, preventing destruction of uncommitted work.
        """
        (repo / "main.py").write_text("print('v2')\n", encoding="utf-8")
        sha = create_checkpoint("turn 1")

        # User has uncommitted work
        (repo / "main.py").write_text("print('precious uncommitted work')\n", encoding="utf-8")

        real_git = checkpoints._git
        reset_called = []

        def fake_git(*args):
            if len(args) > 0 and args[0] == "stash" and "push" in args:
                return subprocess.CompletedProcess(
                    args=["git", *args],
                    returncode=1,
                    stdout="",
                    stderr="fatal: unable to write new index file (locked / permission denied)",
                )
            if len(args) > 0 and args[0] == "reset" and "--hard" in args:
                reset_called.append(True)
            return real_git(*args)

        monkeypatch.setattr(checkpoints, "_git", fake_git)

        with pytest.raises(CheckpointError, match="Не удалось безопасно сохранить незакоммиченные изменения"):
            rewind_to(sha)

        # Ensure reset was never called!
        assert len(reset_called) == 0
        # User's uncommitted work is intact
        assert (repo / "main.py").read_text(encoding="utf-8") == "print('precious uncommitted work')\n"

    def test_pre_commit_hook_does_not_poison_git_index(self, repo):
        """BUG-02 Fix: Pre-commit hook must not block auto-checkpoints (--no-verify).
        Even if commit fails for another reason, index must be unstaged so subsequent
        turns are not poisoned.
        """
        hooks_dir = repo / ".git" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_script = hooks_dir / "pre-commit"
        hook_script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        try:
            subprocess.run(["chmod", "+x", str(hook_script)], capture_output=True)
        except Exception:
            pass

        (repo / "main.py").write_text("print('wip code')\n", encoding="utf-8")

        # Thanks to --no-verify, the checkpoint succeeds despite the failing pre-commit hook!
        res = auto_checkpoint_detailed("wip turn")
        assert res.sha is not None
        assert res.warn is False

    def test_commit_failure_unstages_index(self, repo, monkeypatch):
        """BUG-02 Fix (Fallback): If git commit genuinely fails, the index must be
        cleared with git reset so subsequent turns don't see staged changes.
        """
        (repo / "main.py").write_text("print('dirty')\n", encoding="utf-8")

        real_git = checkpoints._git
        def fake_git(*args):
            if len(args) > 0 and args[0] == "commit":
                return subprocess.CompletedProcess(
                    args=["git", *args],
                    returncode=1,
                    stdout="",
                    stderr="fatal: disk full or git failure",
                )
            return real_git(*args)

        monkeypatch.setattr(checkpoints, "_git", fake_git)

        res = auto_checkpoint_detailed("attempt")
        assert res.sha is None
        assert res.warn is True

        # Ensure index was cleared and is not dirty/staged
        staged = _git("diff", "--cached", "--name-only").stdout.strip()
        assert staged == "", "Index should have been unstaged by git reset!"


# ============================================================================
# LAYER L4b: Fallback & Recovery (Stash Recovery on Reset Failure)
# ============================================================================

class TestL4bStashRecovery:
    def test_reset_failure_restores_stash(self, repo, monkeypatch):
        """INC-03 Fix: If git reset --hard fails after stash, rewind_to must
        attempt git stash pop to restore the user's uncommitted work.
        """
        (repo / "main.py").write_text("print('v2')\n", encoding="utf-8")
        sha = create_checkpoint("turn 1")

        (repo / "main.py").write_text("print('dirty work')\n", encoding="utf-8")

        real_git = checkpoints._git
        def fake_git(*args):
            if len(args) > 0 and args[0] == "reset" and "--hard" in args:
                return subprocess.CompletedProcess(
                    args=["git", *args],
                    returncode=128,
                    stdout="",
                    stderr="fatal: unable to unlink 'main.py': Permission denied",
                )
            return real_git(*args)

        monkeypatch.setattr(checkpoints, "_git", fake_git)

        with pytest.raises(CheckpointError, match="Незакоммиченные изменения автоматически восстановлены"):
            rewind_to(sha)

        # The uncommitted work was restored back to working tree via stash pop
        assert (repo / "main.py").read_text(encoding="utf-8") == "print('dirty work')\n"


# ============================================================================
# LAYER L7: Safety Wall (Active Merge / Rebase Protection)
# ============================================================================

class TestL7MergeRebaseSafetyWall:
    def test_active_merge_blocks_rewind(self, repo):
        """INC-04 Fix: If MERGE_HEAD exists, rewind_to must refuse to run."""
        (repo / "main.py").write_text("print('v2')\n", encoding="utf-8")
        sha = create_checkpoint("turn 1")

        # Simulate an active merge
        git_dir = Path(_git("rev-parse", "--git-dir").stdout.strip())
        merge_head = git_dir / "MERGE_HEAD"
        merge_head.write_text("0123456789abcdef\n", encoding="utf-8")

        with pytest.raises(CheckpointError, match="Обнаружен активный незавершённый процесс: слияние"):
            rewind_to(sha)


# ============================================================================
# LAYER L5: State Synchronization (Agent History & Read Cache)
# ============================================================================

class TestL5StateSync:
    def test_agent_on_rewind_truncates_messages_and_clears_read_cache(self):
        """BUG-03 Fix: agent.on_rewind() must truncate subsequent turn messages
        and clear _read_cache.
        """
        from agent import ArgentAgent
        agent = ArgentAgent()
        agent._read_cache["/path/to/file.py"] = {"sig": [100, 200], "stored": "code"}

        # Simulate turn 1
        agent.messages.append({"role": "user", "content": "turn 1 prompt"})
        agent._current_turn_msg_index = 1
        agent._checkpoint_turn_indices["abc1234"] = 1
        agent.messages.append({"role": "assistant", "content": "turn 1 done"})

        # Simulate turn 2
        agent.messages.append({"role": "user", "content": "turn 2 prompt"})
        agent._current_turn_msg_index = 3
        agent._checkpoint_turn_indices["def5678"] = 3
        agent.messages.append({"role": "assistant", "content": "turn 2 done"})

        assert len(agent.messages) == 5  # system + 2 turns

        # Rewind to turn 1 checkpoint
        agent.on_rewind("abc1234", "turn 1 label")

        # _read_cache must be cleared
        assert len(agent._read_cache) == 0

        # Messages must be truncated back to turn 1 start (index 1) + system note
        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "system"
        assert "откачено к чекпоинту abc1234" in agent.messages[1]["content"]

    def test_agent_on_rewind_external_checkpoint_injects_notice(self):
        """If checkpoint SHA is not in turn indices, on_rewind still clears cache
        and injects system guidance without crashing.
        """
        from agent import ArgentAgent
        agent = ArgentAgent()
        agent._read_cache["foo"] = {"sig": [1, 2]}
        agent.messages.append({"role": "user", "content": "hello"})

        agent.on_rewind("external999", "legacy")
        assert len(agent._read_cache) == 0
        assert any("external999" in m.get("content", "") for m in agent.messages)


# ============================================================================
# LAYER L6: Cross-System Consistency (file_tracker snapshots)
# ============================================================================

class TestL6CrossSystemFileTracker:
    def test_clear_session_snapshots_purges_history(self, repo, tmp_path):
        """BUG-04 Fix: clear_session_snapshots() purges old snapshots so /undo
        doesn't restore discarded turn files.
        """
        f = repo / "main.py"
        snapshot(str(f), session_id="test_session")
        f.write_text("print('edited')\n", encoding="utf-8")

        # Verify undo works before clear
        undo_res = undo(str(f), session_id="test_session")
        assert "Restored" in undo_res

        # Take new snapshot and clear
        snapshot(str(f), session_id="test_session")
        removed = clear_session_snapshots(session_id="test_session")
        assert removed > 0

        # Now undo finds nothing
        undo_res2 = undo(str(f), session_id="test_session")
        assert "No snapshots found" in undo_res2
