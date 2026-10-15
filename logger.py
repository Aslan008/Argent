"""
File logging system for Argent.
All modules use this for debug-level logging to ~/.argent/logs/
"""

import logging
import os
import threading
from pathlib import Path

LOG_DIR = Path.home() / ".argent" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
        target = LOG_DIR / f"{name}.log"
        _set_aside_if_large(target)
        fh = logging.FileHandler(target, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        fh.addFilter(_SecretFilter())
        _logger.addHandler(fh)
        _logger.setLevel(logging.DEBUG)
    return _logger
