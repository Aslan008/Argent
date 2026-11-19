"""Logging must not touch the filesystem at import time.

`logger` used to mkdir `~/.argent/logs` while being imported, so every Argent
module — they all log — failed to import on a read-only home. These tests pin
the lazy behaviour and the graceful degradation.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

import argent.logger as logger_mod

ROOT = Path(__file__).resolve().parents[1]


def test_importing_logger_does_not_create_the_home_directory(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "ARGENT_TESTS_USE_REAL_HOME"}
    env["USERPROFILE"] = str(home)   # what Path.home() reads on Windows
    env["HOME"] = str(home)          # POSIX
    # Keep the developer's site-packages out of the way of a clean import.
    env.pop("PYTHONPATH", None)

    proc = subprocess.run(
        [sys.executable, "-c", "import argent.logger; print(argent.logger.LOG_DIR)"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert not (home / ".argent").exists(), (
        "importing logger created ~/.argent; the mkdir must happen on first use"
    )


def test_an_unwritable_log_dir_degrades_to_a_null_handler(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(logger_mod, "LOG_DIR", blocker / "logs")

    log = logger_mod.get_logger("probe_readonly_home")

    assert any(isinstance(h, logging.NullHandler) for h in log.handlers)
    log.info("this must not raise")          # used to raise PermissionError
    log.error("nor must this")


def test_an_existing_but_unwritable_log_dir_also_degrades(tmp_path, monkeypatch):
    """Existence is not writability: the directory may exist and still be closed
    to this process (a locked profile, a full disk). The FileHandler failure must
    be caught, not raised at every import that logs."""
    monkeypatch.setattr(logger_mod, "LOG_DIR", tmp_path)

    def refuse(*args, **kwargs):
        raise PermissionError(13, "access denied")

    monkeypatch.setattr(logger_mod.logging, "FileHandler", refuse)

    log = logger_mod.get_logger("probe_locked_dir")

    assert any(isinstance(h, logging.NullHandler) for h in log.handlers)
    log.info("this must not raise")


def test_a_writable_log_dir_still_gets_a_real_file(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_mod, "LOG_DIR", tmp_path)
    log = logger_mod.get_logger("probe_writable_home")

    log.info("hello")

    assert (tmp_path / "probe_writable_home.log").read_text(encoding="utf-8").find("hello") != -1
