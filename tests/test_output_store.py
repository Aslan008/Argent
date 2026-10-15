"""A long command output is kept whole in a file; the message tells the model where."""

import os
import time

import pytest

import tools.command_ops as command_ops
from src.agent import output_store as store
from tests.test_command_ops_streaming import _MockProc, _mock_approve

TINY = "qwen2.5:1.5b"          # tiny: 40 lines
CLOUD = "claude-sonnet-5-5"    # cloud: 2000 lines


@pytest.fixture(autouse=True)
def _folder(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "output_dir", lambda: tmp_path / "out")
    yield tmp_path / "out"


def _lines(n):
    return "\n".join(f"line {i}" for i in range(n))


def test_short_output_is_returned_as_is(_folder):
    text = _lines(10)
    assert store.keep_if_long(text, "ls", TINY) is text
    assert not _folder.exists()


def test_long_output_is_saved_whole_and_the_footer_names_the_file(_folder):
    text = _lines(500)
    out = store.keep_if_long(text, "npm run build", TINY)
    files = list(_folder.glob("*.log"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == text
    assert out.startswith(text) and "[Full output: 500 lines" in out
    assert files[0].name in out.replace("\\", "/") or str(files[0].name) in out


def test_the_window_depends_on_the_model(_folder):
    text = _lines(500)
    assert store.keep_if_long(text, "x", CLOUD) is text            # fits a cloud model's window
    assert store.keep_if_long(text, "x", TINY) != text             # does not fit a tiny one


def test_the_folder_is_bounded(_folder, monkeypatch):
    monkeypatch.setattr(store, "KEEP_FILES", 3)
    for i in range(7):
        store.save(f"output {i}\n" * 5, "cmd")
        time.sleep(0.01)
    names = sorted(p.name for p in _folder.glob("*.log"))
    assert len(names) == 3


def test_saving_failure_never_fails_the_command(monkeypatch):
    monkeypatch.setattr(store, "output_dir", lambda: (_ for _ in ()).throw(OSError("disk full")))
    text = _lines(500)
    assert store.keep_if_long(text, "x", TINY) == text


def test_run_command_points_at_the_saved_output(monkeypatch, _folder):
    _mock_approve(monkeypatch)
    body = (_lines(300) + "\n").encode()
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: _MockProc([body]))
    monkeypatch.setattr("config.get_current_model", lambda: TINY)

    out = command_ops.run_command("make all")
    assert "[Full output: " in out
    saved = list(_folder.glob("*.log"))
    assert len(saved) == 1 and "line 150" in saved[0].read_text(encoding="utf-8")


def test_secrets_are_not_written_to_the_saved_file(monkeypatch, _folder):
    _mock_approve(monkeypatch)
    key = "sk-" + "Ab1" * 12
    body = (_lines(300) + f"\nKEY={key}\n").encode()
    monkeypatch.setattr(command_ops, "_spawn", lambda *a, **k: _MockProc([body]))
    monkeypatch.setattr("config.get_current_model", lambda: TINY)

    command_ops.run_command("env")
    assert all(key not in p.read_text(encoding="utf-8") for p in _folder.glob("*.log"))
