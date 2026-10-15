"""Argent.bat and run_tui.py agree on how a start is confirmed.

The launcher cannot be double-clicked from a test, so this pins the parts that broke it
before: the batch file's encoding/line ends, and the mark file both sides use.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAT = ROOT / "Argent.bat"
TUI = ROOT / "run_tui.py"


def test_the_batch_file_is_ascii_with_crlf_line_ends():
    raw = BAT.read_bytes()
    assert all(b < 128 for b in raw), "cmd.exe reads non-ASCII text under an OEM code page wrongly"
    assert raw.count(b"\r\n") == raw.count(b"\n"), "every line must end with CRLF"


def test_the_launcher_and_the_entry_point_use_the_same_mark_file():
    bat = BAT.read_text(encoding="ascii")
    tui = TUI.read_text(encoding="utf-8")
    assert re.search(r'set "MARK=%CD%\\\.argent\\launch\.ok"', bat)
    assert 'os.path.join(_ROOT, ".argent", "launch.ok")' in tui
    # the mark is written before anything heavy is imported
    assert tui.index("_announce_start()") < tui.index("import argent_tui")


def test_the_launcher_has_fallbacks_and_never_just_closes():
    bat = BAT.read_text(encoding="ascii")
    for needle in ("wt.exe", "conhost.exe", ":wait_started", "pause"):
        assert needle in bat
    # the final stage reports why instead of flashing a window
    assert bat.rindex("pause") > bat.index("no way of starting Argent worked")
