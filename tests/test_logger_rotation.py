"""Logs are bounded: a file past the limit is set aside when a process opens it.

Every subsystem writes DEBUG to its own file under ~/.argent/logs and nothing
ever pruned them. The fix has to work with several Argent processes appending to
the same file at once (terminal, TUI, scheduled jobs), which is why it renames at
open time instead of using RotatingFileHandler.
"""

import logging
import os

import pytest

import logger as logger_mod


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(logger_mod, "MAX_LOG_BYTES", 1000)
    yield tmp_path


def _fresh(name):
    logging.getLogger(name).handlers.clear()
    return logger_mod.get_logger(name)


def _flush(log):
    for handler in log.handlers:
        handler.flush()
        handler.close()


class TestRotation:
    def test_a_small_log_is_left_alone(self, log_dir):
        (log_dir / "probe_small.log").write_text("x" * 500, encoding="utf-8")
        log = _fresh("probe_small")
        log.warning("hello")
        _flush(log)
        text = (log_dir / "probe_small.log").read_text(encoding="utf-8")
        assert text.startswith("x" * 500) and text.rstrip().endswith("hello")
        assert not (log_dir / "probe_small.log.1").exists()

    def test_a_large_log_is_set_aside_and_a_fresh_one_started(self, log_dir):
        (log_dir / "probe_big.log").write_text("old line\n" * 500, encoding="utf-8")
        log = _fresh("probe_big")
        log.warning("new line")
        _flush(log)
        assert (log_dir / "probe_big.log.1").read_text(encoding="utf-8").startswith("old line")
        fresh = (log_dir / "probe_big.log").read_text(encoding="utf-8")
        assert "old line" not in fresh and fresh.rstrip().endswith("new line")

    def test_only_one_generation_is_kept(self, log_dir):
        (log_dir / "probe_gen.log.1").write_text("ancient\n", encoding="utf-8")
        (log_dir / "probe_gen.log").write_text("recent\n" * 500, encoding="utf-8")
        _flush(_fresh("probe_gen"))
        assert "recent" in (log_dir / "probe_gen.log.1").read_text(encoding="utf-8")
        assert not (log_dir / "probe_gen.log.2").exists()

    def test_a_log_held_open_elsewhere_still_logs(self, log_dir, monkeypatch):
        """Windows refuses to rename a file another process has open; the logger
        must carry on appending, not raise at import time."""
        (log_dir / "probe_busy.log").write_text("busy\n" * 500, encoding="utf-8")

        def refuse(*args, **kwargs):
            raise PermissionError("in use by another process")

        monkeypatch.setattr(os, "replace", refuse)
        log = _fresh("probe_busy")
        log.warning("still here")
        _flush(log)
        assert (log_dir / "probe_busy.log").read_text(encoding="utf-8").rstrip().endswith("still here")

    def test_a_new_log_needs_nothing(self, log_dir):
        log = _fresh("probe_new")
        log.warning("first")
        _flush(log)
        assert (log_dir / "probe_new.log").exists()

    def test_the_limit_is_sane(self):
        assert 1024 * 1024 <= logger_mod.MAX_LOG_BYTES <= 100 * 1024 * 1024
