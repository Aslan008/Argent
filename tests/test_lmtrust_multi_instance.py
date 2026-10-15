"""LMTrust Multi-Instance Concurrency & Blind Spot Audit Tests.

Probes multi-process concurrency, cache invalidation, session race conditions,
resource contention (ports, locks, configs), and project isolation when
two or more Argent instances run concurrently.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import config
import file_tracker
import session
from memory_manager import MemoryManager
from session import save_session, list_sessions, get_last_session, _sync_index


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Isolate config, session, and history directories to temporary paths."""
    cfg_file = tmp_path / ".argent_config.json"
    sessions_dir = tmp_path / "sessions"
    history_dir = tmp_path / "file_history"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "CONFIG_FILE", cfg_file)
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(session, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session, "INDEX_PATH", sessions_dir / "index.json")
    monkeypatch.setattr(file_tracker, "HISTORY_DIR", history_dir)

    return {
        "cfg": cfg_file,
        "sessions": sessions_dir,
        "history": history_dir,
        "tmp": tmp_path,
    }


class TestConfigMultiProcessBlindSpots:
    """L6 Cross-System × D6 Concurrency: Config caching across separate processes."""

    def test_in_memory_cache_automatically_refreshes_on_external_process_updates(self, isolated_dirs):
        """Verified: Instance B immediately detects Instance A's disk writes.

        The detection compares the file's identity (mtime, size, inode), so it
        works even when both writes land inside one filesystem timestamp tick.
        The model is read through load_config(): get_current_model() prefers the
        per-provider entry, which this external edit deliberately leaves alone."""
        # Instance A starts and sets model to "claude-3.7-sonnet"
        config.set_current_model("claude-3.7-sonnet")
        assert config.get_current_model() == "claude-3.7-sonnet"

        # Simulate Instance B running in another process: it reads config and caches it
        cached_val = config.load_config()
        assert cached_val["model"] == "claude-3.7-sonnet"

        # Now external process (Instance A) changes the file directly on disk
        with open(isolated_dirs["cfg"], "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        disk_data["model"] = "gpt-4o"
        disk_data["xkiro_api_key"] = "sk-new-key"
        with open(isolated_dirs["cfg"], "w", encoding="utf-8") as f:
            json.dump(disk_data, f)

        # Instance B automatically sees the update without restarting or a manual cache reset!
        assert config.load_config()["model"] == "gpt-4o"
        assert config.get_xkiro_api_key() == "sk-new-key"

    def test_concurrent_write_does_not_clobber_external_modifications(self, isolated_dirs):
        """Verified: Instance B updating a key preserves keys written by Instance A."""
        config.set_current_model("model-A")
        # Instance B has loaded cache
        _ = config.load_config()

        # Instance A (external) adds a new key on disk
        with open(isolated_dirs["cfg"], "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        disk_data["brave_api_key"] = "brave-secret-key-123"
        with open(isolated_dirs["cfg"], "w", encoding="utf-8") as f:
            json.dump(disk_data, f)

        # Instance B updates an unrelated key: debug_mode
        config.set_debug_mode(True)

        with open(isolated_dirs["cfg"], "r", encoding="utf-8") as f:
            final_disk = json.load(f)

        # brave_api_key is preserved because load_config re-synced from disk!
        assert final_disk.get("brave_api_key") == "brave-secret-key-123"
        assert final_disk.get("debug_mode") is True


class TestSessionPersistenceMultiInstance:
    """L1 Contract × D3 Data Integrity: Multiple instances saving and restoring sessions."""

    def test_both_instances_save_distinct_sessions_without_data_loss(self, isolated_dirs):
        """Both instances write their own session files and both appear in the session index."""
        # Instance 1 session
        msg1 = [{"role": "user", "content": "Task in Instance 1: write game backend"}]
        sid1 = save_session(msg1, {"model": "claude-3.5-sonnet", "cwd": "D:/Proj1"})

        # Instance 2 session
        msg2 = [{"role": "user", "content": "Task in Instance 2: debug shaders"}]
        sid2 = save_session(msg2, {"model": "deepseek-chat", "cwd": "D:/Proj2"})

        assert sid1 is not None and sid2 is not None
        assert sid1 != sid2

        # Verify both files exist on disk
        sess_files = list(isolated_dirs["sessions"].glob("*.json.gz"))
        assert len(sess_files) == 2

        # Verify index lists both sessions
        all_sessions = list_sessions()
        ids = [s["id"] for s in all_sessions]
        assert sid1 in ids
        assert sid2 in ids

    def test_last_session_ordered_by_saved_at_timestamp(self, isolated_dirs):
        """get_last_session returns the session that was closed/saved most recently."""
        msg1 = [{"role": "user", "content": "First conversation"}]
        sid1 = save_session(msg1, {"model": "m1"})

        time.sleep(0.05)

        msg2 = [{"role": "user", "content": "Second conversation (closed later)"}]
        sid2 = save_session(msg2, {"model": "m2"})

        last = get_last_session()
        assert last is not None
        assert last["id"] == sid2
        assert last["preview"] == "Second conversation (closed later)"

    def test_directory_filtering_isolates_parallel_projects(self, isolated_dirs):
        """get_last_session(cwd=...) retrieves the correct session for the specific project."""
        proj_a = "C:/Workspace/ProjectA"
        proj_b = "C:/Workspace/ProjectB"

        save_session([{"role": "user", "content": "Project A work"}], {"cwd": proj_a, "model": "m1"})
        time.sleep(0.02)
        save_session([{"role": "user", "content": "Project B work"}], {"cwd": proj_b, "model": "m2"})

        # If Argent is started in ProjectA, /resume picks ProjectA's session, NOT ProjectB
        last_a = get_last_session(proj_a)
        assert last_a is not None
        assert last_a["preview"] == "Project A work"

        last_b = get_last_session(proj_b)
        assert last_b is not None
        assert last_b["preview"] == "Project B work"


class TestFileTrackerMultiInstanceIsolation:
    """L4 Adversarial × D3 Data Integrity: File history collisions."""

    def test_distinct_file_paths_never_collide_in_snapshots(self, isolated_dirs):
        """Files in different projects get unique sha1 keys and never overwrite each other."""
        file_a = isolated_dirs["tmp"] / "projA" / "main.py"
        file_b = isolated_dirs["tmp"] / "projB" / "main.py"
        file_a.parent.mkdir(parents=True, exist_ok=True)
        file_b.parent.mkdir(parents=True, exist_ok=True)

        file_a.write_text("print('projA')", encoding="utf-8")
        file_b.write_text("print('projB')", encoding="utf-8")

        # Snapshot both
        ok_a = file_tracker.snapshot(str(file_a))
        ok_b = file_tracker.snapshot(str(file_b))

        assert ok_a is True and ok_b is True

        key_a = file_tracker._key(file_a.resolve())
        key_b = file_tracker._key(file_b.resolve())

        # Keys must be distinct because normalized absolute paths differ
        assert key_a != key_b

        # Modifying file_a and diffing only affects file_a
        file_a.write_text("print('projA modified')", encoding="utf-8")
        diff_a = file_tracker.get_diff(str(file_a))
        diff_b = file_tracker.get_diff(str(file_b))

        assert "projA modified" in diff_a
        assert "No changes detected" in diff_b  # file_b has no uncommitted changes


class TestMemoryManagerIsolation:
    """L5 State × D2 State/Transition: Project-level memory isolation."""

    def test_two_different_projects_have_independent_memory(self, isolated_dirs, monkeypatch):
        root_1 = isolated_dirs["tmp"] / "ProjectAlpha"
        root_2 = isolated_dirs["tmp"] / "ProjectBeta"
        root_1.mkdir()
        root_2.mkdir()

        # Instance 1 in ProjectAlpha
        monkeypatch.setattr("project_paths.project_root_or_cwd", lambda: root_1)
        mem1 = MemoryManager()
        mem1.set_objective("Build 3D game engine")

        # Instance 2 in ProjectBeta
        monkeypatch.setattr("project_paths.project_root_or_cwd", lambda: root_2)
        mem2 = MemoryManager()
        mem2.set_objective("Create REST API backend")

        # Verify they wrote to different files and maintain separate state
        assert (root_1 / ".argent" / "memory.json").exists()
        assert (root_2 / ".argent" / "memory.json").exists()

        assert mem1.data["objective"] == "Build 3D game engine"
        assert mem2.data["objective"] == "Create REST API backend"


class TestBrowserPortFallbackBlindSpot:
    """L4b Fallback × D5 Resource: Browser engine CDP port 9222 collision handling."""

    @pytest.mark.asyncio
    async def test_fallback_to_isolated_when_browser_already_running(self):
        """When user browser is already open without CDP, engine must fallback to isolated mode."""
        from browser_engine import BrowserEngine
        from browser_detect import BrowserInfo

        engine = BrowserEngine()
        fake_info = BrowserInfo(
            name="Chrome",
            key="chrome",
            exe_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
            user_data_dir="C:/Users/test/AppData/Local/Google/Chrome/User Data",
        )

        with patch.object(engine, "_is_browser_running", return_value=True), \
             patch.object(engine, "_launch_isolated") as mock_isolated, \
             patch("browser_detect.get_default_browser", return_value=fake_info), \
             patch("config.get_browser_name", return_value="auto"), \
             patch.object(engine, "_browser", None):

            # Simulate attempting to launch user browser while it's running
            # In _launch_user_browser, if _is_browser_running is True:
            # it falls back to _launch_isolated
            with patch("asyncio.open_connection", side_effect=OSError("Port 9222 unreachable")):
                await engine._connect_user_browser(headed=True, initial_url="https://example.com")

            mock_isolated.assert_called_once_with(True)
