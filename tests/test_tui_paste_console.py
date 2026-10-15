"""The Rust TUI in a real Windows console (ConPTY): a pasted multi-line text is ONE message.

A Windows console does not hand a paste over as a paste: it arrives as key presses, and every Enter in it
used to send the line. The test types the text in one write, like a terminal does on Ctrl+V."""

import os
import sys
import time
from pathlib import Path

import pytest

winpty = pytest.importorskip("winpty")
pytest.importorskip("argent_tui")
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="needs a Windows console")

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = Path(__file__).with_name("_fake_tui_bridge.py")


def _submitted(log: Path) -> str:
    return log.read_text(encoding="utf-8") if log.exists() else ""


def _wait_for(log: Path, text: str, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if text in _submitted(log):
            return True
        time.sleep(0.2)
    return False


def test_a_pasted_multiline_text_is_one_message_sent_by_enter(tmp_path):
    log = tmp_path / "prompts.log"
    env = dict(os.environ, FAKE_TUI_LOG=str(log), PYTHONIOENCODING="utf-8")
    proc = winpty.PtyProcess.spawn([sys.executable, str(BRIDGE)], dimensions=(40, 120), env=env, cwd=str(ROOT))
    try:
        time.sleep(4)                                         # the TUI starts
        proc.write("/rooms run x Создай песочницу:\rКисть (песок, вода)\rФизика песка\rКнопка очистки")
        time.sleep(2.5)
        assert _submitted(log) == "", "a line of the paste was sent by itself"
        proc.write("\r")                                      # the person's own Enter
        assert _wait_for(log, "Кнопка очистки", 6), _submitted(log)
        sent = _submitted(log)
        assert sent.count("PROMPT<<") == 1
        one_line = "Создай песочницу:\nКисть (песок, вода)\nФизика песка\nКнопка очистки".replace("\n", "\\n")
        assert one_line in sent                               # the bridge logs line breaks as \n
    finally:
        try:
            proc.terminate(force=True)
        except Exception:
            pass


def test_typing_then_enter_still_sends(tmp_path):
    log = tmp_path / "prompts.log"
    env = dict(os.environ, FAKE_TUI_LOG=str(log), PYTHONIOENCODING="utf-8")
    proc = winpty.PtyProcess.spawn([sys.executable, str(BRIDGE)], dimensions=(40, 120), env=env, cwd=str(ROOT))
    try:
        time.sleep(4)
        for ch in "hi":
            proc.write(ch)
            time.sleep(0.15)
        proc.write("\r")
        assert _wait_for(log, "PROMPT<<hi>>", 6), _submitted(log)
    finally:
        try:
            proc.terminate(force=True)
        except Exception:
            pass
