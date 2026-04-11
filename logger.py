"""
File logging system for Argent.
All modules use this for debug-level logging to ~/.argent/logs/
"""

import logging
from pathlib import Path

LOG_DIR = Path.home() / ".argent" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        fh = logging.FileHandler(
            LOG_DIR / f"{name}.log", encoding="utf-8"
        )
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        _logger.addHandler(fh)
        _logger.setLevel(logging.DEBUG)
    return _logger
