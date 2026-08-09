"""What the user reads when something goes wrong, and when they mistype.

Two measured problems. The interface spoke two languages at once — 118
user-facing strings in English in a tool whose user writes Russian. And a real
failure arrived as

    Error: Error: 'utf-8' codec can't encode characters in position 53000-53001

a raw Python exception wearing two labels, naming neither what broke nor what
to do. Argent already fuzzy-matches tool names for the MODEL while answering a
human's typo with a flat "Unknown command" — the person is the one who cannot
see the list.
"""

import re

import pytest

import ui
from command_handler import known_commands, unknown_command_message
from error_help import explain_error


class TestErrorPrefix:
    def test_a_caller_that_already_said_error_is_not_doubled(self, monkeypatch):
        printed = []
        monkeypatch.setattr(ui, "safe_print", lambda t, *a, **k: printed.append(str(t)))
        ui.print_error("Error: something broke")
        assert "Error: Error:" not in printed[0]
        assert "Ошибка: Error:" not in printed[0]

    def test_a_bare_message_gets_exactly_one_prefix(self, monkeypatch):
        printed = []
        monkeypatch.setattr(ui, "safe_print", lambda t, *a, **k: printed.append(str(t)))
        ui.print_error("что-то сломалось")
        assert printed[0].count("Ошибка:") == 1

    def test_an_already_russian_prefix_is_respected(self, monkeypatch):
        printed = []
        monkeypatch.setattr(ui, "safe_print", lambda t, *a, **k: printed.append(str(t)))
        ui.print_error("Ошибка: файл не найден")
        assert printed[0].count("Ошибка:") == 1

    def test_the_raw_text_survives(self, monkeypatch):
        """It is what the user quotes in a bug report; the explanation is added
        below it, never instead of it."""
        printed = []
        monkeypatch.setattr(ui, "safe_print", lambda t, *a, **k: printed.append(str(t)))
        ui.print_error("'utf-8' codec can't encode: surrogates not allowed")
        assert "surrogates not allowed" in printed[0]
        assert len(printed) == 2 and "Unicode" in printed[1]

    def test_an_unknown_failure_gets_no_invented_advice(self, monkeypatch):
        """A guessed explanation sends the reader down a path that may have
        nothing to do with their problem."""
        printed = []
        monkeypatch.setattr(ui, "safe_print", lambda t, *a, **k: printed.append(str(t)))
        ui.print_error("нечто совершенно новое")
        assert len(printed) == 1


class TestExplanations:
    @pytest.mark.parametrize("message,expected", [
        ("'utf-8' codec can't encode characters: surrogates not allowed", "Unicode"),
        ("HTTP 404: No endpoints found that support tool use.", "инструментов"),
        ("MCP server request timed out after 120 seconds", "не дождались"),
        ("[Errno 13] Permission denied", "прав"),
        ("HTTPConnectionPool: Max retries exceeded", "подключиться"),
        ("HTTP 401 Unauthorized", "ключ"),
        ("HTTP 429 rate limit exceeded", "частоту"),
    ])
    def test_failures_that_actually_happened_are_explained(self, message, expected):
        hint = explain_error(message)
        assert hint and expected in hint

    def test_silence_when_we_do_not_know(self):
        assert explain_error("совершенно новая беда") is None
        assert explain_error("") is None
        assert explain_error(None) is None

    def test_every_explanation_is_in_one_language(self):
        """The point of the exercise: no mixed-language advice."""
        import error_help
        for _, text in error_help._EXPLANATIONS:
            assert re.search(r"[а-яА-ЯёЁ]", text), text


class TestUnknownCommand:
    def test_a_typo_gets_the_real_command(self):
        for typo, expected in [("/rewnd", "/rewind"), ("/moddel", "/model"),
                               ("/sesions", "/sessions")]:
            assert expected in unknown_command_message(typo)

    def test_nonsense_gets_the_list_instead_of_a_wrong_guess(self):
        out = unknown_command_message("/qqqqq")
        assert "/help" in out and "имели в виду" not in out

    def test_the_typed_text_is_echoed(self):
        assert "/rewnd" in unknown_command_message("/rewnd")

    def test_arguments_do_not_confuse_the_match(self):
        assert "/rewind" in unknown_command_message("/rewnd до правки меню")

    def test_the_suggestions_are_real_commands(self):
        """Derived from the dispatch source, so a suggestion cannot name a
        command that no longer exists."""
        commands = known_commands()
        assert "/rewind" in commands and "/model" in commands
        assert len(commands) > 30

    def test_empty_input_does_not_crash(self):
        assert unknown_command_message("") and unknown_command_message(None)


class TestOneLanguage:
    def test_no_english_left_in_user_facing_messages(self):
        """Measured before the change: 118 English strings shown to a Russian
        speaker, sometimes two languages in one screen."""
        from pathlib import Path

        english = re.compile(
            r"\b(the|is|are|for|and|not|found|failed|updated|saved|select|current|"
            r"available|enter|invalid|usage|already|please|cleared|unknown|nothing)\b",
            re.I)
        cyrillic = re.compile(r"[а-яА-ЯёЁ]")
        root = Path(__file__).resolve().parents[1]

        offenders = []
        for name in ("main.py", "command_handler.py"):
            text = (root / name).read_text(encoding="utf-8")
            for m in re.finditer(r'print_(?:system|error)\(\s*f?(["\'])([^"\']{8,120})\1', text):
                s = m.group(2)
                if not cyrillic.search(s) and english.search(s):
                    offenders.append(f"{name}: {s[:60]}")
        assert not offenders, "английские сообщения пользователю: " + "; ".join(offenders[:5])
