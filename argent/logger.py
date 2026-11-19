"""
File logging system for Argent.
All modules use this for debug-level logging to ~/.argent/logs/
"""

import logging
import os
import threading
from pathlib import Path

LOG_DIR = Path.home() / ".argent" / "logs"


def _ensure_log_dir() -> bool:
    """Create the log directory on first use, never at import time.

    Importing a module must not touch the filesystem. This used to mkdir at
    import: a machine with a read-only home (a container, a locked-down
    profile, a service account) could not import ANY Argent module — the
    failure surfaced as a PermissionError from an unrelated import, far from
    the cause. Now the directory is created when the first log record needs a
    file; if that is impossible, logging degrades to a NullHandler instead of
    taking the program down.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


# Every subsystem logs at DEBUG to its own file, and nothing ever pruned them: a
# year of use is a multi-gigabyte directory, and a log you cannot open is not a
# diagnostic. A file past this size is set aside once, when a process first opens
# it, keeping a single `.1` generation: bounded (about twice the limit per file)
# and cheap, with none of RotatingFileHandler's trouble on Windows, where several
# Argent processes (a terminal, the TUI, scheduled jobs) append to the same file
# and a rename-under-write raises in the one that loses the race.
MAX_LOG_BYTES = 10 * 1024 * 1024

# Suppress noisy internal asyncio socket warnings (e.g. on closed LSP/subprocess pipes)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def _set_aside_if_large(path: Path) -> None:
    try:
        if path.stat().st_size <= MAX_LOG_BYTES:
            return
        os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        # Missing (nothing to do) or held open by another process (try again at
        # the next start): never a reason to lose the ability to log.
        pass


class _SecretFilter(logging.Filter):
    """A key that reached a log line (a request header in a traceback, a URL with a token) stays out of the file.

    The filter itself reads the config, which logs; the guard keeps that from recursing."""

    _busy = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(self._busy, "on", False):
            return True
        self._busy.on = True
        try:
            from src.agent.secret_redactor import redact
            message = record.getMessage()
            cleaned = redact(message)
            if cleaned is not message and cleaned != message:
                record.msg, record.args = cleaned, ()
        except Exception:
            pass
        finally:
            self._busy.on = False
        return True


def get_logger(name: str) -> logging.Logger:
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        try:
            if not _ensure_log_dir():
                raise OSError("the log directory cannot be created")
            target = LOG_DIR / f"{name}.log"
            _set_aside_if_large(target)
            fh = logging.FileHandler(target, encoding="utf-8")
        except OSError:
            # Read-only home, a locked profile, a full disk, a directory that
            # exists but cannot be written to: losing log files is a
            # degradation, not a reason for every module that logs to fail to
            # import. One NullHandler marks the logger configured so this is
            # not retried for every call.
            _logger.addHandler(logging.NullHandler())
            _logger.setLevel(logging.CRITICAL)
            return _logger
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        fh.addFilter(_SecretFilter())
        _logger.addHandler(fh)
        _logger.setLevel(logging.DEBUG)
    return _logger
