"""The TUI's side of live command output and of a screenshot pasted from the clipboard.

Two separate contracts live here:

* `StdoutTrap` must hand the frontend the *raw* shape of a write, newlines
  included. The REPL streams a command's output one write at a time with
  `console.print(..., end="")`, so a write that does not end with a newline is
  continued by the next one; stripping the newline told the frontend that every
  write finished its line, and `git status --short` broke after its status letter.
* `ReplBridge.queue_clipboard_image` turns the raw RGBA the Rust side can read
  from the clipboard into the same queued payload `view_image` produces.
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bridge.repl_bridge import ReplBridge, StdoutTrap

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _trap():
    seen = []
    trap = StdoutTrap(on_output=seen.append)
    # `start()` swaps the process streams; the write path only needs the flag.
    trap._active = True
    return trap, seen


class TestStreamedOutput:
    def test_a_write_that_ends_a_line_keeps_its_newline(self):
        trap, seen = _trap()
        trap.write(" M")
        trap.write(" .github/workflows/ci.yml\n")
        assert seen == [" M", " .github/workflows/ci.yml\n"]

    def test_the_status_column_survives(self):
        # Rich wraps each print in its own style; the leading space is part of
        # porcelain's two-column layout and must reach the frontend intact.
        trap, seen = _trap()
        trap.write("\x1b[2m M README.md\x1b[0m\n")
        assert " M README.md" in seen[0]
        assert seen[0].endswith("\n")

    def test_cursor_and_mode_sequences_never_reach_the_frontend(self):
        trap, seen = _trap()
        trap.write("\x1b[2K\x1b[1Gdone\n")
        assert seen == ["done\n"]

    def test_an_inactive_trap_stays_silent(self):
        seen = []
        trap = StdoutTrap(on_output=seen.append)
        trap.write("nothing yet\n")
        assert seen == []


class TestClipboardImage:
    def _capture_queue(self, monkeypatch):
        from argent import vision

        queued = []
        # `path` rides along so the chat can also offer "open at full size" in the
        # system viewer; the stub takes it rather than pinning the production signature.
        monkeypatch.setattr(
            vision, "queue_image",
            lambda payload, media, label, path="": queued.append((payload, media, label)),
        )
        return queued

    def test_an_image_is_encoded_and_queued(self, monkeypatch):
        queued = self._capture_queue(monkeypatch)
        w, h = 4, 3
        rgba = bytes([200, 30, 30, 255]) * (w * h)

        reply = ReplBridge.queue_clipboard_image(None, w, h, rgba)

        assert reply == f"{w}×{h}", reply
        assert len(queued) == 1
        payload, media, label = queued[0]
        assert media == "image/png"
        assert base64.b64decode(payload)[:8] == PNG_MAGIC
        assert str(w) in label and str(h) in label

    def test_a_transparent_grab_is_flattened_onto_white(self, monkeypatch):
        from PIL import Image
        import io

        queued = self._capture_queue(monkeypatch)
        w, h = 3, 3
        rgba = bytes([0, 0, 0, 0]) * (w * h)  # fully transparent

        assert ReplBridge.queue_clipboard_image(None, w, h, rgba) == f"{w}×{h}"
        img = Image.open(io.BytesIO(base64.b64decode(queued[0][0])))
        assert img.mode == "RGB", img.mode
        assert img.getpixel((0, 0)) == (255, 255, 255)

    def test_an_empty_clipboard_reports_instead_of_queueing(self, monkeypatch):
        queued = self._capture_queue(monkeypatch)
        reply = ReplBridge.queue_clipboard_image(None, 0, 0, b"")
        assert "пуст" in reply.lower(), reply
        assert queued == []

    def test_short_data_is_refused(self, monkeypatch):
        queued = self._capture_queue(monkeypatch)
        reply = ReplBridge.queue_clipboard_image(None, 10, 10, b"\x00\x00\x00\xff")
        assert "поврежд" in reply.lower(), reply
        assert queued == []
