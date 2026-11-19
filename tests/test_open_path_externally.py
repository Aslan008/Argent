"""Handing a file to the system must work everywhere, and say why when it cannot.

`os.startfile` exists only on Windows. On macOS and Linux the call raised
AttributeError, the caller turned it into "Не удалось открыть", and the picture /
artifact button simply never worked outside Windows.
"""

import os
import subprocess
import sys

import pytest

from src.bridge import repl_bridge


def a_bridge_without_a_session():
    """`open_path_externally` touches no state, so no session is needed."""
    return repl_bridge.ReplBridge.__new__(repl_bridge.ReplBridge)


class TestThePlatformDispatch:
    def test_windows_uses_startfile(self, monkeypatch):
        opened = []
        monkeypatch.setattr(os, "startfile", lambda target: opened.append(target), raising=False)
        monkeypatch.setattr(os, "name", "nt")

        repl_bridge._open_with_os("C:/proj/plan.md")

        assert opened == ["C:/proj/plan.md"]

    def test_macos_uses_open(self, monkeypatch):
        spawned = []
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: spawned.append(argv))

        repl_bridge._open_with_os("/tmp/plan.md")

        assert spawned == [["open", "/tmp/plan.md"]]

    def test_linux_uses_xdg_open(self, monkeypatch):
        spawned = []
        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(subprocess, "Popen", lambda argv, **kw: spawned.append(argv))

        repl_bridge._open_with_os("/tmp/plan.md")

        assert spawned == [["xdg-open", "/tmp/plan.md"]]

    def test_the_helper_never_waits_for_the_viewer(self, monkeypatch):
        # A blocking call would freeze the agent until the user closes the app.
        seen = {}

        def fake_popen(argv, **kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(subprocess, "Popen", fake_popen)

        repl_bridge._open_with_os("/tmp/plan.md")
        assert seen["stdin"] is subprocess.DEVNULL
        assert seen["stdout"] is subprocess.DEVNULL


class TestOpenPathExternally:
    def test_a_missing_file_is_reported(self, tmp_path):
        answer = a_bridge_without_a_session().open_path_externally(str(tmp_path / "nope.md"))
        assert "не найден" in answer.lower()

    def test_an_empty_path_is_reported(self):
        assert "не найден" in a_bridge_without_a_session().open_path_externally("").lower()

    def test_a_real_file_is_handed_to_the_system(self, tmp_path, monkeypatch):
        artifact = tmp_path / "plan.md"
        artifact.write_text("# plan", encoding="utf-8")
        opened = []
        monkeypatch.setattr(repl_bridge, "_open_with_os", lambda target: opened.append(target))

        answer = a_bridge_without_a_session().open_path_externally(str(artifact))

        assert answer == "", "success says nothing"
        assert opened == [str(artifact)]

    def test_a_failure_from_the_system_is_reported_not_swallowed(self, tmp_path, monkeypatch):
        artifact = tmp_path / "plan.md"
        artifact.write_text("# plan", encoding="utf-8")

        def refuse(target):
            raise OSError("no handler")

        monkeypatch.setattr(repl_bridge, "_open_with_os", refuse)
        answer = a_bridge_without_a_session().open_path_externally(str(artifact))
        assert "Не удалось открыть" in answer
        assert "no handler" in answer
