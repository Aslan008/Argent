"""read_git_diff caps a huge diff into a summary + bounded slice."""

import tools.command_ops as command_ops
from tools.command_ops import read_git_diff


class R:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_large_diff_is_capped(monkeypatch):
    big = "+" + ("x" * 20000)

    def fake(cmd, **k):
        if "is-inside-work-tree" in cmd:
            return R("true", 0)
        if "--cached --stat" in cmd:
            return R("")
        if "--stat" in cmd:
            return R(" Player.cs | 500 +++++\n")
        if "--cached" in cmd:
            return R("")                      # nothing staged
        if cmd.strip() == "git diff":
            return R(big)
        return R("")

    monkeypatch.setattr(command_ops, "run_text", fake)
    out = read_git_diff()
    assert "Diff is large" in out
    assert "FILES CHANGED" in out and "Player.cs" in out
    assert "git diff -- <file>" in out
    assert len(out) < len(big)                # genuinely truncated


def test_small_diff_is_untouched(monkeypatch):
    def fake(cmd, **k):
        if "is-inside-work-tree" in cmd:
            return R("true", 0)
        if "--cached" in cmd:
            return R("")
        if cmd.strip() == "git diff":
            return R("+small change\n")
        return R("")

    monkeypatch.setattr(command_ops, "run_text", fake)
    out = read_git_diff()
    assert "Diff is large" not in out
    assert "small change" in out


def test_no_changes(monkeypatch):
    def fake(cmd, **k):
        if "is-inside-work-tree" in cmd:
            return R("true", 0)
        return R("")

    monkeypatch.setattr(command_ops, "run_text", fake)
    assert read_git_diff() == "No changes detected in Git."
