"""run_command must not eat leading whitespace on Windows line endings.

Every line a Windows child prints ends with CRLF. The post-processing treated
that carriage return as a progress-bar overwrite and stripped each piece, so
`git diff`, a Python listing or a YAML manifest reached the model flattened to
column zero (only the last line kept its indentation, because the final
`.strip()` had already removed its `\\r`).
"""

import subprocess
import sys

import pytest

import argent.tools.command_ops as co
from argent.tools.command_ops import _collapse_carriage_returns


@pytest.fixture
def run_emitting(monkeypatch):
    """`run_command`, but the child is a python one-liner that prints `payload`.

    Spawning a real shell would cost a second per test and differ per platform;
    this keeps the whole pipeline (pump thread, decoder, post-processing) real.
    """

    def _run(payload: str) -> str:
        script = "import sys;sys.stdout.write(%r)" % payload
        real_popen = subprocess.Popen

        def fake_spawn(command, cwd=None, **kwargs):
            return real_popen([sys.executable, "-c", script], cwd=cwd, **kwargs)

        monkeypatch.setattr(co, "_spawn", fake_spawn)
        monkeypatch.setattr(co, "request_approval", lambda *a, **k: True)
        monkeypatch.setattr(co, "is_destructive_command", lambda c: False)
        monkeypatch.setattr(co, "assess_command_risk", lambda c: ("safe", []))
        return co.run_command("ignored")

    return _run


class TestCrlfLines:
    def test_indentation_survives(self, run_emitting):
        out = run_emitting("def f():\r\n    return 1\r\n\r\nclass A:\r\n    x = 2\r\n")
        assert "    return 1" in out
        assert "    x = 2" in out
        assert "\nreturn 1" not in out

    def test_a_diff_keeps_its_leading_space(self, run_emitting):
        out = run_emitting("@@ -1,2 +1,2 @@\r\n context\r\n-    old\r\n+    new\r\n")
        assert " context" in out
        assert "-    old" in out
        assert "+    new" in out

    def test_progress_bar_keeps_only_the_last_frame(self, run_emitting):
        out = run_emitting("10%\r50%\r100% done\r\nnext line\r\n")
        assert "100% done" in out
        assert "10%" not in out
        assert "next line" in out


class TestCollapseHelper:
    def test_no_carriage_return_is_a_no_op(self):
        text = "  a\n\tb\n"
        assert _collapse_carriage_returns(text) is text

    def test_crlf_is_a_line_ending_not_an_overwrite(self):
        assert _collapse_carriage_returns("  a\r\n  b\r\n") == "  a\n  b\n"

    def test_lone_cr_rewrites_the_row(self):
        assert _collapse_carriage_returns("loading 10%\rloading 100%\nok") == "loading 100%\nok"

    def test_a_progress_bar_mixed_with_crlf(self):
        text = "  keep\r\nstep 1\rdone\r\n  keep2\r\n"
        assert _collapse_carriage_returns(text) == "  keep\ndone\n  keep2\n"
