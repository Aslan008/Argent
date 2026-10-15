"""Tests for real-time subprocess streaming, prompt display, and UTF-8 handling in run_command."""

import os
import threading
import pytest
import tools.command_ops as command_ops
from tools.command_ops import run_command


class _MockProc:
    """Mock process feeding arbitrary byte chunks through a real OS pipe."""

    def __init__(self, chunks, returncode=0, close_at_end=True):
        read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb")
        self.returncode = returncode
        self._done = False
        self._closed = False
        self.pid = -1  # Fake PID so _kill_process_tree can proceed

        def _writer():
            try:
                for chunk in chunks:
                    os.write(self._write_fd, chunk)
                if close_at_end:
                    self._close()
            except OSError:
                pass

        threading.Thread(target=_writer, daemon=True).start()

    def _close(self):
        if not self._closed:
            self._closed = True
            try:
                os.close(self._write_fd)
            except OSError:
                pass

    def poll(self):
        return None if not self._done else self.returncode

    def terminate(self):
        self._done = True
        self.returncode = -15
        self._close()

    def kill(self):
        """Called by _kill_process_tree fallback when taskkill fails."""
        self._done = True
        self.returncode = -9
        self._close()

    def wait(self):
        return self.returncode


def _mock_approve(monkeypatch):
    monkeypatch.setattr(command_ops, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(command_ops, "assess_command_risk", lambda c: ("safe", []))


@pytest.mark.lmtrust(layer="L2", direction="D1")
def test_split_utf8_multibyte_across_chunks(monkeypatch):
    """A multi-byte UTF-8 character (e.g. Cyrillic 'Привет') split across chunks must not fail."""
    _mock_approve(monkeypatch)
    # 'Привет' in UTF-8: \xd0\x9f \xd1\x80 \xd0\xb8 \xd0\xb2 \xd0\xb5 \xd1\x82
    raw = "Привет, мир!\n".encode("utf-8")
    # Split right in the middle of the second character
    chunks = [raw[:3], raw[3:7], raw[7:]]
    proc = _MockProc(chunks)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("echo Привет")
    assert "Привет, мир!" in out


@pytest.mark.lmtrust(layer="L4", direction="D4")
def test_rich_markup_in_output_does_not_crash(monkeypatch):
    """Output containing Rich markup tags like [dim], [bold red], [1/5] must print safely."""
    _mock_approve(monkeypatch)
    chunks = [b"[1/5] Compiling source.cpp\n", b"[ERROR] Unmatched [bracket in output\n", b"Done [y/n]: "]
    proc = _MockProc(chunks)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("fake_compiler")
    assert "[1/5] Compiling source.cpp" in out
    assert "[ERROR] Unmatched [bracket in output" in out
    assert "Done [y/n]:" in out


@pytest.mark.lmtrust(layer="L1", direction="D2")
def test_prompt_visible_immediately_without_newline(monkeypatch):
    """An interactive prompt without a trailing newline must be decoded and present in output."""
    _mock_approve(monkeypatch)
    chunks = [b"Need to install package foo\n", b"Ok to proceed? (y) "]
    proc = _MockProc(chunks)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("npx foo")
    assert "Need to install package foo" in out
    assert "Ok to proceed? (y)" in out


# --- L1: Contract ----------------------------------------------------------
@pytest.mark.lmtrust(layer="L1", direction="D7")
def test_l1_contract_empty_output_process(monkeypatch):
    """Process that produces no output at all must return clean exit code structure."""
    _mock_approve(monkeypatch)
    proc = _MockProc([])
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("true")
    assert "Exit code: 0" in out


# --- L2: Boundary (Single byte streaming) ----------------------------------
@pytest.mark.lmtrust(layer="L2", direction="D3")
def test_l2_boundary_single_byte_trickle(monkeypatch):
    """Streaming 1 byte at a time through the OS pipe must concatenate seamlessly."""
    _mock_approve(monkeypatch)
    text = "Hello, world!"
    chunks = [bytes([b]) for b in text.encode("utf-8")]
    proc = _MockProc(chunks)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("slow_echo")
    assert "Hello, world!" in out


# --- L4b: Fallback (Invalid byte sequence) ----------------------------------
@pytest.mark.lmtrust(layer="L4b", direction="D4")
def test_l4b_fallback_invalid_bytes_replaced(monkeypatch):
    """Invalid UTF-8 byte sequences must be safely replaced without raising UnicodeDecodeError."""
    _mock_approve(monkeypatch)
    bad_bytes = b"\xff\xfe\xfd\x80\x81 valid text\n"
    proc = _MockProc([bad_bytes])
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("binary_dump")
    assert "valid text" in out
    assert "Exit code: 0" in out


# --- L8: Temporal (Diagnosis on silence timeout) ----------------------------
@pytest.mark.lmtrust(layer="L8", direction="D2")
def test_l8_temporal_silent_timeout_captures_trailing_prompt(monkeypatch):
    """When a process times out waiting for input, the prompt is captured in [DIAGNOSIS]."""
    _mock_approve(monkeypatch)
    monkeypatch.setattr(command_ops, "COMMAND_SILENCE_LIMIT", 0.2)
    proc = _MockProc([b"Are you sure? [y/N] "], close_at_end=False)
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: proc)

    out = run_command("interactive_cmd")
    assert "[DIAGNOSIS]" in out
    assert "Are you sure? [y/N]" in out
    assert proc._done



