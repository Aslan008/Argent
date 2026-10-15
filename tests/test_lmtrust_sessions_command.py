"""LMTrust blind spot test suite for /sessions . and positional load/delete commands.

Covers:
- L1 Contract (D7): /sessions . / ./ / .\\ filtering by current working directory
- L2 Boundary (D1): Slashes normalization (forward / vs backward \\) between cwd and stored sessions
- L4 Adversarial (D4/D10): Corrupted/non-string cwd in index does NOT fail open or crash
- L6 Cross-System (D3/D7): Positional index synchronization: /load 1 and /sessions rm 1 after /sessions .
- L9 Negative Space: No cross-project session leakage during local directory operations
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import session as session_mod
from src.cli.commands.base import CommandContext
from src.cli.commands.session_commands import handle_sessions, handle_load
import src.cli.commands.session_commands as sc_module


@pytest.fixture
def isolated_sessions_env(tmp_path, monkeypatch):
    """Set up isolated session storage and clear last displayed sessions."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_path = sessions_dir / "index.json"

    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_mod, "INDEX_PATH", index_path)
    sc_module._last_displayed_sessions = None
    return sessions_dir


class DummyAgent:
    def __init__(self):
        self.messages = []
        self.session_id = None
        self.model_name = "test-model"


class TestLMTrustSessionsCommand:
    """LMTrust Matrix tests for /sessions . and session commands."""

    # -----------------------------------------------------------------------
    # L1 x D7: Directory filtering handles '.', './', '.\\' and 'here'
    # -----------------------------------------------------------------------
    def test_l1_sessions_dot_variations_match_current_directory(self, tmp_path, isolated_sessions_env, monkeypatch):
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        other_dir = tmp_path / "other_project"
        other_dir.mkdir()

        monkeypatch.chdir(project_dir)
        session_mod.save_session([{"role": "user", "content": "Project message"}], metadata={"cwd": str(project_dir)})
        session_mod.save_session([{"role": "user", "content": "Other message"}], metadata={"cwd": str(other_dir)})

        for query in (".", "./", ".\\", "here", "cwd"):
            results = session_mod.list_sessions(query)
            assert len(results) == 1, f"Failed for query '{query}'"
            assert results[0]["preview"] == "Project message"

    # -----------------------------------------------------------------------
    # L2 x D1: Slashes normalization across forward and backward slashes
    # -----------------------------------------------------------------------
    def test_l2_slashes_normalization_matches_regardless_of_slash_style(self, tmp_path, isolated_sessions_env, monkeypatch):
        project_dir = tmp_path / "app_folder"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        # Store with forward slashes explicitly
        fwd_cwd = str(project_dir).replace("\\", "/")
        session_mod.save_session([{"role": "user", "content": "Slash test"}], metadata={"cwd": fwd_cwd})

        results = session_mod.list_sessions(".")
        assert len(results) == 1
        assert results[0]["preview"] == "Slash test"

        # Also get_last_session with os.getcwd() must find it
        last = session_mod.get_last_session(os.getcwd())
        assert last is not None
        assert last["preview"] == "Slash test"

    # -----------------------------------------------------------------------
    # L4 x D4/D10: Corrupted cwd does NOT fail open
    # -----------------------------------------------------------------------
    def test_l4_corrupted_cwd_does_not_leak_global_sessions(self, tmp_path, isolated_sessions_env, monkeypatch):
        project_dir = tmp_path / "safe_proj"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        # 1 valid session in project, 1 session in another dir, 1 session with corrupted cwd
        s_valid = {"id": "s_valid", "cwd": str(project_dir), "saved_at": "2026-01-01T10:00:00", "preview": "valid"}
        s_other = {"id": "s_other", "cwd": "C:/Other/Secret", "saved_at": "2026-01-01T09:00:00", "preview": "secret"}
        s_corrupt = {"id": "s_corrupt", "cwd": 12345, "saved_at": "2026-01-01T08:00:00", "preview": "corrupt"}

        monkeypatch.setattr(session_mod, "_sync_index", lambda: {s["id"]: s for s in (s_valid, s_other, s_corrupt)})

        res = session_mod.list_sessions(".")
        # Must only return the valid session matching current directory, never all 3!
        assert len(res) == 1
        assert res[0]["id"] == "s_valid"

    # -----------------------------------------------------------------------
    # L6 x D3/D7: Positional index sync between /sessions . and /load 1 / /sessions rm 1
    # -----------------------------------------------------------------------
    def test_l6_positional_sync_load_first_session_after_sessions_dot(self, tmp_path, isolated_sessions_env, monkeypatch):
        dir_here = tmp_path / "here_dir"
        dir_here.mkdir()
        dir_other = tmp_path / "other_dir"
        dir_other.mkdir()

        monkeypatch.chdir(dir_here)

        # Other project has the newest session globally
        sid_other = session_mod.save_session(
            [{"role": "user", "content": "Other project"}],
            metadata={"cwd": str(dir_other), "model": "m"},
            label="OTHER"
        )
        # Current project has an older session
        sid_here = session_mod.save_session(
            [{"role": "user", "content": "Here project"}],
            metadata={"cwd": str(dir_here), "model": "m"},
            label="HERE"
        )

        agent = DummyAgent()
        ctx_sessions = CommandContext(agent=agent, user_input="/sessions .")
        handle_sessions(ctx_sessions)

        # Row 1 shown in /sessions . was 'HERE' (sid_here).
        # /load 1 must restore 'sid_here', NOT 'sid_other' from the other directory!
        with patch("src.cli.commands.session_commands.restore_session") as mock_restore:
            ctx_load = CommandContext(agent=agent, user_input="/load 1")
            handle_load(ctx_load)

            assert mock_restore.called
            target_restored = mock_restore.call_args[0][1]
            assert target_restored["id"] == sid_here
            assert target_restored["label"] == "HERE"

    def test_l6_positional_sync_rm_session_after_sessions_dot(self, tmp_path, isolated_sessions_env, monkeypatch):
        dir_here = tmp_path / "here_dir_rm"
        dir_here.mkdir()
        dir_other = tmp_path / "other_dir_rm"
        dir_other.mkdir()

        monkeypatch.chdir(dir_here)

        sid_other = session_mod.save_session(
            [{"role": "user", "content": "Other content"}],
            metadata={"cwd": str(dir_other), "model": "other_model"},
            label="OTHER_LBL"
        )
        sid_here = session_mod.save_session(
            [{"role": "user", "content": "Here content"}],
            metadata={"cwd": str(dir_here), "model": "here_model"},
            label="HERE_LBL"
        )

        agent = DummyAgent()
        handle_sessions(CommandContext(agent=agent, user_input="/sessions ."))

        # /sessions rm 1 must target sid_here, never sid_other
        with patch("questionary.confirm") as mock_confirm:
            mock_confirm.return_value.ask.return_value = False  # just check target
            handle_sessions(CommandContext(agent=agent, user_input="/sessions rm 1"))

            # Confirm prompt must mention sid_here and HERE_LBL
            prompt_text = mock_confirm.call_args[0][0]
            assert sid_here in prompt_text
            assert "HERE_LBL" in prompt_text
            assert "OTHER_LBL" not in prompt_text

    # -----------------------------------------------------------------------
    # L9 x D7: Empty directory message & custom title formatting
    # -----------------------------------------------------------------------
    def test_l9_empty_directory_and_title_ui(self, tmp_path, isolated_sessions_env, monkeypatch):
        empty_dir = tmp_path / "empty_proj"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        agent = DummyAgent()
        with patch("src.cli.commands.session_commands.print_system") as mock_print:
            handle_sessions(CommandContext(agent=agent, user_input="/sessions ."))
            # Must inform specifically about the current directory
            called_text = mock_print.call_args[0][0]
            assert "текущей директории" in called_text
