"""Streaming markdown segmenter for append-only terminal rendering.

Argent used to redraw the *whole* growing answer each frame inside a Rich
``Live`` and then print it once more at the end. That breaks the moment the
answer is taller than the terminal: Rich can't erase content that has scrolled
off-screen, so a partial copy is stranded in the scrollback and the final
re-render adds a second, full copy.

The fix mirrors how Ink-based CLIs (e.g. Claude Code) stream: commit COMPLETE
markdown blocks to the terminal once (they scroll naturally and are never
redrawn), and keep only the incomplete trailing block in a small live region.
This module is the pure brain of that: given a growing buffer, it decides how
much is safe to commit and what's still pending. No Rich, no I/O — unit-testable.
"""

import re

# A fenced code block opens/closes with >=3 backticks or tildes.
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def _last_safe_boundary(text: str) -> int:
    """Offset up to which `text` is a run of COMPLETE markdown blocks.

    A safe boundary is a blank line *outside* a code fence, or the end of a
    closed fence. Everything after the returned offset is an incomplete
    trailing block that must stay pending (committing it could render a broken
    markdown fragment — e.g. the inside of an unterminated code fence).
    """
    in_fence = False
    fence_char = ""
    offset = 0
    last_safe = 0
    for line in text.splitlines(keepends=True):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                offset += len(line)
                continue
            if marker[0] == fence_char:  # closing fence (same char)
                in_fence = False
                offset += len(line)
                last_safe = offset
                continue
        offset += len(line)
        if not in_fence and line.strip() == "":
            last_safe = offset
    return last_safe


class MarkdownStreamSplitter:
    """Splits a growing markdown buffer into committed blocks + a pending tail."""

    def __init__(self):
        self._buffer = ""
        self._committed = 0

    def feed(self, text: str) -> str:
        """Append `text`; return any newly-committable markdown ('' if none yet)."""
        self._buffer += text
        region = self._buffer[self._committed:]
        boundary = _last_safe_boundary(region)
        if boundary <= 0:
            return ""
        self._committed += boundary
        return region[:boundary]

    def replace(self, full_text: str) -> None:
        """The stream now declares the whole content is `full_text` (e.g. a raw
        tool-call JSON was stripped out of it). Keep already-committed output if
        it is still a prefix of the new text; otherwise drop the pending tail
        (we can't un-print what was already committed — a rare, best-effort case).
        """
        committed_text = self._buffer[:self._committed]
        if full_text.startswith(committed_text):
            self._buffer = full_text
        else:
            self._buffer = committed_text

    def pending(self) -> str:
        """The current uncommitted tail (incomplete block)."""
        return self._buffer[self._committed:]

    def finalize(self) -> str:
        """Mark everything committed and return the remaining tail."""
        rest = self._buffer[self._committed:]
        self._committed = len(self._buffer)
        return rest
