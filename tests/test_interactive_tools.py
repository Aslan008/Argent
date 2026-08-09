"""A tool that asks the user must not run under the spinner.

Rich's `console.status` owns the same lines questionary draws on, so a prompt
raised inside it is rendered and immediately overwritten. The user sees the
preamble, no question, and a program that will not continue — reported for
request_user_approval:

    Agent requests approval: План live-sync … Подтвердите — начну кодить.
    (и всё, при этом Argent ждёт ввода)

The set of such tools lived in the CLI and fell behind every tool added since
it was written. It is declared beside the registry now, and derived here from
the sources so it cannot drift again.
"""

import inspect

import pytest

from tools.schemas import AVAILABLE_TOOLS, INTERACTIVE_TOOLS

# Primitives that take the terminal away from the renderer.
_PROMPT_MARKERS = ("questionary", "ptk_prompt", "prompt_toolkit",
                   "request_approval", "input(")


def _tools_that_prompt() -> set:
    found = set()
    for name, func in AVAILABLE_TOOLS.items():
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            continue
        if any(marker in source for marker in _PROMPT_MARKERS):
            found.add(name)
    return found


class TestTheSetIsComplete:
    def test_every_prompting_tool_is_declared(self):
        """The exact regression: request_user_approval prompted, was not in the
        set, and its question was painted over by the spinner."""
        missing = sorted(_tools_that_prompt() - set(INTERACTIVE_TOOLS))
        assert not missing, (
            f"эти инструменты спрашивают пользователя, но не объявлены "
            f"интерактивными — спиннер затрёт вопрос: {missing}")

    def test_nothing_is_declared_that_never_asks(self):
        """Skipping the spinner for a silent tool costs the user the only
        feedback that anything is happening."""
        spurious = sorted(set(INTERACTIVE_TOOLS) - _tools_that_prompt())
        assert not spurious, f"объявлены интерактивными, но ничего не спрашивают: {spurious}"

    def test_the_names_are_real_tools(self):
        assert set(INTERACTIVE_TOOLS) <= set(AVAILABLE_TOOLS)

    @pytest.mark.parametrize("name", ["request_user_approval", "ask_user_questions",
                                      "run_command", "git_rollback", "browser_input"])
    def test_the_ones_that_bit_us(self, name):
        assert name in INTERACTIVE_TOOLS


class TestTheCliUsesIt:
    def test_the_cli_does_not_keep_its_own_copy(self):
        """It did, and that copy is what fell behind."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src" / "cli" / "cli_ui.py"
                  ).read_text(encoding="utf-8")
        assert "INTERACTIVE_TOOLS" in source
        assert "interactive_tools = {" not in source

    def test_a_prompting_tool_gets_no_spinner(self):
        from tools.schemas import INTERACTIVE_TOOLS as declared
        assert "request_user_approval" in declared      # -> use_spinner is False


class TestDismissalIsNotConsent:
    def test_escaping_the_prompt_does_not_approve(self, monkeypatch):
        """Ctrl+C at an approval prompt must not start the very work the user
        was declining to authorise."""
        from tools import misc_tools

        class _Q:
            @staticmethod
            def confirm(*a, **k):
                return type("A", (), {"ask": lambda s: None})()

        monkeypatch.setitem(__import__("sys").modules, "questionary", _Q)
        monkeypatch.setattr(misc_tools.memory, "add_completed", lambda *a: None)
        out = misc_tools.request_user_approval("удалить всё")
        assert "NO" in out and "do not proceed" in out

    def test_yes_and_no_still_work(self, monkeypatch):
        from tools import misc_tools

        def _answering(value):
            class _Q:
                @staticmethod
                def confirm(*a, **k):
                    return type("A", (), {"ask": lambda s: value})()
            return _Q

        monkeypatch.setattr(misc_tools.memory, "add_completed", lambda *a: None)
        import sys
        monkeypatch.setitem(sys.modules, "questionary", _answering(True))
        assert "approved" in misc_tools.request_user_approval("план")
        monkeypatch.setitem(sys.modules, "questionary", _answering(False))
        assert "REJECTED" in misc_tools.request_user_approval("план")
