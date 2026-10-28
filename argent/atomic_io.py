"""Crash- and race-safe file writes.

A plain `open(path, "w")` truncates the target first, then writes. If the
process dies mid-write — or two Argent instances in different terminals write
the same file at once — the reader is left with a truncated or interleaved
file. For small JSON state files (config, project brain) that means a corrupt
config on next launch.

atomic_write_text writes to a uniquely-named temp file in the SAME directory
(so the final rename stays on one filesystem) and then os.replace()s it over
the target. os.replace is atomic on POSIX and on Windows (ReplaceFile
semantics): a reader sees either the whole old file or the whole new one,
never a half-written mix.
"""

import os
import threading
import uuid
from pathlib import Path


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write(path, lambda f: f.write(text), "w", encoding=encoding)


def atomic_write_bytes(path, writer) -> None:
    """Same guarantee for binary payloads. ``writer`` receives the open file,
    or can be raw bytes/bytearray.
    """
    if not callable(writer):
        data = writer
        writer = lambda f: f.write(data)
    _atomic_write(path, writer, "wb")


def _atomic_write(path, writer, mode: str, encoding: str = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Include PID, thread ID and UUID so concurrent writers don't clobber each other's temp.
    stem = path.name[:180]
    tmp = path.with_name(f".{stem}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, mode, encoding=encoding) as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())      # durability: the bytes are on disk pre-rename
        os.replace(tmp, path)         # atomic swap
    finally:
        # If the rename never happened (write failed), don't leave a temp behind.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
