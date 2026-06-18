"""Regression: input history must survive /cd and disk errors.

Bug: history_path was relative (.argent/input_history); prompt_toolkit re-opens
that path against the *current* cwd on every append, so after the user changed
directory with /cd into a folder without a .argent/, appending history crashed
the whole event loop with FileNotFoundError.
"""

from pathlib import Path

from src.cli.cli_prompt import _SafeFileHistory


def test_history_survives_cwd_change(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)

    # Mimic build_prompt_session: a relative path resolved to absolute at start.
    hp = (Path(".argent") / "input_history").expanduser().resolve()
    hp.parent.mkdir(parents=True, exist_ok=True)
    hist = _SafeFileHistory(str(hp))
    hist.store_string("first command")

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)            # new cwd has no .argent/
    hist.store_string("after cd")       # must NOT raise

    saved = hp.read_text(encoding="utf-8")
    assert "first command" in saved
    assert "after cd" in saved
    assert not (other / ".argent").exists()  # nothing leaked into the new cwd


def test_store_swallows_disk_errors(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")
    bad = blocker / "nested" / "input_history"  # parent traverses a file
    hist = _SafeFileHistory(str(bad))
    hist.store_string("x")  # must not raise


def test_load_swallows_errors(tmp_path):
    bad = tmp_path / "does_not_exist" / "input_history"
    hist = _SafeFileHistory(str(bad))
    assert list(hist.load_history_strings()) == []  # no crash, just empty
