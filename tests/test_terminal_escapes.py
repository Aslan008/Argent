"""What a program prints for a terminal never reaches the model as garbage."""

import pytest

import tools.command_ops as command_ops
from text_safety import strip_terminal_escapes as strip
from tests.test_command_ops_streaming import _MockProc, _mock_approve


@pytest.mark.parametrize("raw,clean", [
    ("\x1b[31mred\x1b[0m", "red"),
    ("\x1b[?25lhidden cursor\x1b[?25h", "hidden cursor"),                 # progress bars
    ("\x1b[2K\x1b[1Gdone", "done"),                                       # erase line, column 1
    ("\x1b]0;window title\x07text", "text"),                              # OSC terminated by BEL
    ("\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\", "link"),      # hyperlink, ST terminated
    ("\x1b(Bplain", "plain"),                                             # charset selection
    ("\x1b7saved\x1b8", "saved"),                                         # save / restore cursor
    ("\x9b32mcsi8", "csi8"),                                              # 8-bit CSI
    ("a\x1bPq#0;2;0;0;0\x1b\\b", "ab"),                                   # DCS
])
def test_sequences_are_removed(raw, clean):
    assert strip(raw) == clean


def test_plain_text_and_unicode_are_untouched():
    text = "Привет, мир! [1/5] 100% done — ok\n\ttab"
    assert strip(text) is text


def test_a_lone_escape_at_the_end_of_text_is_left_for_the_next_chunk():
    assert strip("abc\x1b") == "abc\x1b"


def test_run_command_returns_clean_text(monkeypatch):
    _mock_approve(monkeypatch)
    raw = b"\x1b[?25l\x1b[32mBuilding\x1b[0m\r\n\x1b]0;npm\x07done\r\n\x1b[?25h"
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: _MockProc([raw]))
    out = command_ops.run_command("npm run build")
    assert "Building" in out and "done" in out
    assert "\x1b" not in out and "[?25" not in out
