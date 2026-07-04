"""run_text: crash-proof text-mode subprocess capture.

Regression for a UnicodeDecodeError raised inside subprocess's reader thread
when a child (tasklist, git, dotnet, node, ...) emits non-UTF-8 (OEM/locale)
bytes and Python is in UTF-8 mode. run_text forces errors='replace' so a stray
byte degrades to a replacement char instead of crashing the reader thread.
"""

import sys

import src.agent.shell as sh
from src.agent.shell import run_text


def test_injects_safe_decoding_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(sh.subprocess, "run", lambda *a, **k: captured.update(k) or "OK")
    run_text(["echo", "hi"], capture_output=True)
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["capture_output"] is True


def test_explicit_encoding_and_errors_respected(monkeypatch):
    captured = {}
    monkeypatch.setattr(sh.subprocess, "run", lambda *a, **k: captured.update(k))
    run_text(["x"], errors="ignore", encoding="cp866")
    assert captured["errors"] == "ignore"      # caller's choice kept
    assert captured["encoding"] == "cp866"
    assert captured["text"] is True            # text mode is always forced


def test_survives_non_utf8_output():
    # Reproduces the exact crash condition: a child writing byte 0xFF to stdout.
    # With plain text=True + UTF-8 this kills the reader thread; run_text must
    # return cleanly with the bad bytes replaced.
    r = run_text(
        [sys.executable, "-c", r"import sys; sys.stdout.buffer.write(b'\xff\xfe hi there')"],
        capture_output=True,
    )
    assert r.returncode == 0
    assert "hi there" in r.stdout  # surrounding text survives; 0xFF/0xFE replaced


def test_returns_completed_process():
    r = run_text([sys.executable, "-c", "print('ok')"], capture_output=True)
    assert r.returncode == 0
    assert r.stdout.strip() == "ok"
