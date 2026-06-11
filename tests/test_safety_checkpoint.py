from types import SimpleNamespace
from unittest.mock import MagicMock

import main as main_module
import tools.misc_tools as misc_tools


def fake_git(is_repo=True, dirty=True):
    def runner(cmd, **kwargs):
        if "rev-parse" in cmd:
            return SimpleNamespace(returncode=0 if is_repo else 128, stdout="")
        if "status" in cmd:
            return SimpleNamespace(returncode=0, stdout=" M file.py\n" if dirty else "")
        raise AssertionError(f"unexpected command: {cmd}")
    return runner


def fake_confirm(answer):
    prompt = MagicMock()
    prompt.ask.return_value = answer
    return MagicMock(return_value=prompt)


class TestOfferSafetyCheckpoint:
    def test_dirty_tree_and_yes_creates_checkpoint(self, monkeypatch):
        monkeypatch.setattr(main_module.subprocess, "run", fake_git(dirty=True))
        monkeypatch.setattr(main_module.questionary, "confirm", fake_confirm(True))
        cp = MagicMock(return_value="Checkpoint created")
        monkeypatch.setattr(misc_tools, "git_checkpoint", cp)

        main_module.offer_safety_checkpoint("написать игру")

        cp.assert_called_once()
        assert "перед /auto: написать игру" in cp.call_args[0][0]

    def test_dirty_tree_and_no_skips_checkpoint(self, monkeypatch):
        monkeypatch.setattr(main_module.subprocess, "run", fake_git(dirty=True))
        monkeypatch.setattr(main_module.questionary, "confirm", fake_confirm(False))
        cp = MagicMock()
        monkeypatch.setattr(misc_tools, "git_checkpoint", cp)

        main_module.offer_safety_checkpoint("задача")

        cp.assert_not_called()

    def test_clean_tree_never_prompts(self, monkeypatch):
        monkeypatch.setattr(main_module.subprocess, "run", fake_git(dirty=False))
        confirm = MagicMock()
        monkeypatch.setattr(main_module.questionary, "confirm", confirm)

        main_module.offer_safety_checkpoint("задача")

        confirm.assert_not_called()

    def test_outside_git_never_prompts(self, monkeypatch):
        monkeypatch.setattr(main_module.subprocess, "run", fake_git(is_repo=False))
        confirm = MagicMock()
        monkeypatch.setattr(main_module.questionary, "confirm", confirm)

        main_module.offer_safety_checkpoint("задача")

        confirm.assert_not_called()

    def test_git_failure_is_swallowed(self, monkeypatch):
        def boom(cmd, **kwargs):
            raise OSError("git missing")
        monkeypatch.setattr(main_module.subprocess, "run", boom)
        main_module.offer_safety_checkpoint("задача")  # must not raise
