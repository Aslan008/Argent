"""The whole output of a command, kept where the model can read the part it needs.

A long build log is cut to its head and tail before it reaches the model (`compress_tool_result`): the
window is small, and a thousand lines of "Compiling x" help nobody. What was cut used to be gone — the
model could only run the command again and hope for a shorter answer.

When a command's output is longer than the window this model gets, the full text is written to
`.argent/run/output/` and the message ends with a line that names the file, so that the middle is one
`read_file(start_line, end_line)` or `grep_search` away. The text is saved AFTER secret redaction, so the file
holds nothing the model was not allowed to see.

The folder is bounded: the newest KEEP_FILES files or KEEP_BYTES bytes, whichever is smaller.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger("output_store")

KEEP_FILES = 40
KEEP_BYTES = 64 * 1024 * 1024
#: A single saved output is cut at this size (the command layer already caps what it collects at ~1 MB).
MAX_FILE_BYTES = 8 * 1024 * 1024

_LOCK = threading.Lock()
_COUNTER = [0]


def output_dir() -> Path:
    from project_paths import project_root_or_cwd
    return Path(project_root_or_cwd()) / ".argent" / "run" / "output"


def _prune(folder: Path) -> None:
    try:
        files = sorted((p for p in folder.glob("*.log") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    total = 0
    for index, path in enumerate(files):
        try:
            total += path.stat().st_size
        except OSError:
            continue
        if index >= KEEP_FILES or total > KEEP_BYTES:
            try:
                path.unlink()
            except OSError:
                pass


def save(text: str, label: str = "command") -> Optional[Path]:
    """Write `text` to a new file in the output folder and return its path, or None when that fails."""
    if not text:
        return None
    try:
        folder = output_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            _COUNTER[0] += 1
            stamp = time.strftime("%Y%m%d-%H%M%S")
            safe = "".join(c if c.isalnum() else "-" for c in label.lower())[:24].strip("-") or "command"
            path = folder / f"{stamp}-{os.getpid()}-{_COUNTER[0]}-{safe}.log"
        data = text.encode("utf-8", errors="replace")
        if len(data) > MAX_FILE_BYTES:
            data = data[:MAX_FILE_BYTES]
        path.write_bytes(data)
        _prune(folder)
        return path
    except Exception as e:                         # keeping the log is a convenience; never fail the command
        log.warning("could not save the full command output: %s", e)
        return None


def footer(path: Path, text: str) -> str:
    """The line appended to a result whose middle may be cut."""
    lines = text.count("\n") + 1
    try:
        from project_paths import project_root_or_cwd
        shown = os.path.relpath(path, project_root_or_cwd())
    except Exception:
        shown = str(path)
    return (f"\n[Full output: {lines} lines, {max(1, len(text) // 1024)} KB, saved to {shown}. "
            f"The text above may be shortened for your context window; read an exact part with "
            f"read_file(file_path, start_line, end_line) or find it with grep_search(pattern, path).]")


def keep_if_long(text: str, label: str, model_name: str, tool_name: str = "run_command") -> str:
    """`text` plus a footer pointing at the saved full output, when this model's window would cut it."""
    if not text:
        return text
    try:
        from prompt_compressor import would_truncate
        if not would_truncate(text, model_name or "", tool_name):
            return text
    except Exception:
        return text
    path = save(text, label)
    return text + footer(path, text) if path else text
